"""MTP training losses.

Target alignment
----------------
``targets[:, t] = idx[:, t + 1]`` is the usual next-token target. Depth ``k`` predicts the token
``k`` further out, so its target at slot ``t`` is ``targets[:, t + k]`` and only the first
``T - k`` slots have a target::

    L_k = CE( logits_k[:, :T-k] , targets[:, k:] )

The total loss follows DeepSeek-V3::

    L = L_ntp + lambda / D * sum_{k=1..D} L_k

Memory
------
Every depth produces a ``(B, T, V)`` logits tensor. With a 128k vocabulary that is the dominant
activation, and naively keeping ``D + 1`` of them alive for one backward pass multiplies peak
memory. Gloeckle et al. avoid it by running the forward *and backward* of each head in turn,
accumulating the gradient at the trunk output, and only then back-propagating through the
trunk. ``train_step(memory_efficient=True)`` does exactly that; it produces the same gradients
as the naive path (``tests/test_loss.py`` checks this) while holding one logits tensor at a time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

IGNORE_INDEX = -100


@dataclass
class Losses:
    total: torch.Tensor
    ntp: torch.Tensor
    mtp: list[torch.Tensor] = field(default_factory=list)

    def as_floats(self) -> dict[str, float]:
        out = {"total": float(self.total.detach()), "ntp": float(self.ntp.detach())}
        for k, l in enumerate(self.mtp, start=1):
            out[f"mtp{k}"] = float(l.detach())
        return out


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), targets.reshape(-1), ignore_index=IGNORE_INDEX)


def depth_loss(logits_k: torch.Tensor, targets: torch.Tensor, k: int) -> torch.Tensor:
    """Loss for depth ``k`` given logits over slots ``0..n-1`` (``n`` may be ``T`` or ``T - k``)."""
    T = targets.shape[1]
    n = T - k
    if n <= 0:
        raise ValueError(f"sequence length {T} is too short for depth {k}")
    return cross_entropy(logits_k[:, :n], targets[:, k:])


def combine(ntp: torch.Tensor, mtp: list[torch.Tensor], loss_weight: float) -> torch.Tensor:
    if not mtp:
        return ntp
    return ntp + loss_weight / len(mtp) * torch.stack(mtp).sum()


def compute_losses(model, idx: torch.Tensor, targets: torch.Tensor) -> Losses:
    """Plain forward pass keeping the whole graph (used for evaluation and as the reference
    implementation for ``train_step``)."""
    trunk, heads, mcfg = model.trunk, model.heads, model.mtp_cfg
    h0 = trunk(idx)
    ntp = cross_entropy(trunk.logits(h0), targets)
    mtp: list[torch.Tensor] = []
    if heads is not None:
        h_in = h0.detach() if mcfg.detach_trunk else h0
        for k, h_k in enumerate(heads.forward_train(h_in, idx, trunk), start=1):
            mtp.append(depth_loss(heads.logits(k, h_k, trunk), targets, k))
    return Losses(combine(ntp, mtp, mcfg.loss_weight), ntp, mtp)


def train_step(model, idx: torch.Tensor, targets: torch.Tensor, memory_efficient: bool = True) -> Losses:
    """Forward + backward. Gradients are left in ``.grad``; the caller steps the optimizer."""
    if not memory_efficient or model.heads is None:
        losses = compute_losses(model, idx, targets)
        losses.total.backward()
        return losses

    trunk, heads, mcfg = model.trunk, model.heads, model.mtp_cfg
    D, T = mcfg.n_future, idx.shape[1]
    scale = mcfg.loss_weight / D

    h0 = trunk(idx)
    # Bridge tensor: everything downstream reads this detached copy; its .grad collects the
    # contributions of every head, and a single trunk backward finishes the job.
    h0_d = h0.detach().requires_grad_(True)
    ntp = cross_entropy(trunk.logits(h0_d), targets)
    ntp.backward()

    h_in = h0.detach() if mcfg.detach_trunk else h0_d
    mtp: list[torch.Tensor] = []
    if mcfg.kind == "parallel":
        cos, sin = trunk.rope(torch.arange(T, device=idx.device))
        for k in range(1, D + 1):
            h_k = heads.head_forward(k, h_in, cos, sin)
            loss_k = depth_loss(heads.logits(k, h_k, trunk), targets, k)
            (scale * loss_k).backward()
            mtp.append(loss_k.detach())
    else:
        # Sequential depths form a chain h0 -> h1 -> ... -> hD. Forward all depths with a detached
        # bridge between consecutive depths, then walk the chain backwards: depth k receives its
        # own loss gradient plus whatever depth k+1 sent into its bridge.
        outs: list[torch.Tensor] = []
        bridges: list[torch.Tensor] = []
        prev = h_in
        for k in range(1, D + 1):
            emb = trunk.embed(idx[:, k:])
            cos, sin = trunk.rope(torch.arange(k, T, device=idx.device))
            h_k = heads.step(k, prev[:, : T - k], emb, cos, sin)
            outs.append(h_k)
            bridge = h_k.detach().requires_grad_(True)
            bridges.append(bridge)
            prev = bridge
        for k in range(D, 0, -1):
            loss_k = depth_loss(heads.logits(k, outs[k - 1], trunk), targets, k)
            tensors, grads = [scale * loss_k], [None]
            if k < D and bridges[k - 1].grad is not None:
                tensors.append(outs[k - 1])
                grads.append(bridges[k - 1].grad)
            torch.autograd.backward(tensors, grads)
            mtp.insert(0, loss_k.detach())
    if h0_d.grad is not None:
        h0.backward(h0_d.grad)
    total = combine(ntp.detach(), mtp, mcfg.loss_weight)
    return Losses(total, ntp.detach(), mtp)
