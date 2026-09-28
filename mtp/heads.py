"""MTP head designs.

Both designs share the trunk's unembedding matrix (and its token embedding, for the sequential
variant), exactly like the papers they come from. What differs is *what the head reads*:

``ParallelHeads``  (Gloeckle et al. 2024; Medusa)
    head_k : trunk state at position t  ->  distribution over token t + 1 + k

    Each head is a small function of the trunk output only. Heads are independent of each other
    and of the tokens the other heads predict, so at inference all ``D`` drafts come out of one
    pass over the trunk states. ``head_arch="block"`` uses one transformer block per head (the
    original paper); ``head_arch="mlp"`` uses a zero-initialised residual MLP per head (Medusa).

``SequentialMTP``  (DeepSeek-V3; Nemotron 3 Super / GLM-5 / FastMTP when ``share_weights``)
    h_k[t] = Block_k( W_k [ RMSNorm(h_{k-1}[t]) ; RMSNorm(Emb(x[t + k])) ] )
    head_k : h_k[t]  ->  distribution over token t + 1 + k

    Depth ``k`` is conditioned on the *actual* token at ``t + k`` (teacher forcing during training,
    the previously drafted token during inference), so the chain keeps the full causal
    factorisation instead of assuming the future tokens are conditionally independent.

Positions: depth ``k`` handles the token at ``t + k`` in slot ``t``, so its rotary position is
``t + k``. Keeping this consistent between training (``forward_train``) and drafting
(``mtp.decode``) is the single most common source of MTP bugs; ``tests/test_decode.py`` checks it.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, MTPConfig
from .layers import Block, KVCache, ResidualMLP, RMSNorm, init_weights


class ParallelHeads(nn.Module):
    def __init__(self, cfg: ModelConfig, mtp_cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        self.n_future = mtp_cfg.n_future
        self.head_arch = mtp_cfg.head_arch
        self.share_weights = False
        if self.head_arch == "block":
            self.heads = nn.ModuleList(
                nn.ModuleList(Block(cfg) for _ in range(mtp_cfg.head_layers)) for _ in range(self.n_future)
            )
        else:
            self.heads = nn.ModuleList(
                nn.ModuleList(ResidualMLP(cfg.d_model) for _ in range(mtp_cfg.head_layers)) for _ in range(self.n_future)
            )
        self.in_norm = RMSNorm(cfg.d_model, cfg.norm_eps)  # only used by the mlp arch (Medusa reads normed states)
        self.out_norms = nn.ModuleList(RMSNorm(cfg.d_model, cfg.norm_eps) for _ in range(self.n_future))
        init_weights(self, cfg.n_layers)

    def check_depth(self, k: int) -> None:
        if not 1 <= k <= self.n_future:
            raise ValueError(f"parallel heads have fixed offsets 1..{self.n_future}, got depth {k}")

    def new_caches(self, k: int, batch: int, max_len: int, trunk) -> list[KVCache] | None:
        self.check_depth(k)
        if self.head_arch != "block":
            return None
        return [KVCache(batch, self.cfg.n_heads, max_len, self.cfg.head_dim, trunk.device, trunk.dtype) for _ in self.heads[k - 1]]

    def head_forward(
        self,
        k: int,
        h0: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        caches: list[KVCache] | None = None,
    ) -> torch.Tensor:
        """Depth-``k`` hidden states for trunk states ``h0`` of shape ``(B, n, d)`` at positions
        described by ``cos``/``sin``."""
        self.check_depth(k)
        layers = self.heads[k - 1]
        if self.head_arch == "block":
            x = h0
            for i, layer in enumerate(layers):
                x = layer(x, cos, sin, None if caches is None else caches[i])
            return x
        x = self.in_norm(h0)
        for layer in layers:
            x = layer(x)
        return x

    def logits(self, k: int, h_k: torch.Tensor, trunk) -> torch.Tensor:
        self.check_depth(k)
        return trunk.unembed(self.out_norms[k - 1](h_k))

    def forward_train(self, h0: torch.Tensor, idx: torch.Tensor, trunk, n_depths: int | None = None) -> list[torch.Tensor]:
        """Hidden states ``[h_1, ..., h_D]`` for a full training sequence; each is ``(B, T, d)``.

        Slot ``t`` of ``h_k`` is a prediction for token ``t + 1 + k``; the loss slices off the last
        ``k`` slots, which have no target.
        """
        n_depths = self.n_future if n_depths is None else n_depths
        T = idx.shape[1]
        cos, sin = trunk.rope(torch.arange(T, device=idx.device))
        return [self.head_forward(k, h0, cos, sin) for k in range(1, n_depths + 1)]


class MTPModule(nn.Module):
    """One DeepSeek-V3 MTP module: two norms, a ``2d -> d`` projection, transformer block(s) and an
    output norm feeding the shared unembedding."""

    def __init__(self, cfg: ModelConfig, n_layers: int):
        super().__init__()
        self.norm_h = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.norm_e = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model, bias=False)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(n_layers))
        self.norm_out = RMSNorm(cfg.d_model, cfg.norm_eps)

    def forward(
        self,
        prev_h: torch.Tensor,
        tok_emb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        caches: list[KVCache] | None = None,
    ) -> torch.Tensor:
        x = self.proj(torch.cat((self.norm_h(prev_h), self.norm_e(tok_emb)), dim=-1))
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        return x


class SequentialMTP(nn.Module):
    def __init__(self, cfg: ModelConfig, mtp_cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        self.n_future = mtp_cfg.n_future
        self.share_weights = mtp_cfg.share_weights
        n_modules = 1 if self.share_weights else self.n_future
        self.modules_ = nn.ModuleList(MTPModule(cfg, mtp_cfg.head_layers) for _ in range(n_modules))
        init_weights(self, cfg.n_layers)

    def module(self, k: int, recursive: bool = False) -> MTPModule:
        """Parameters used at depth ``k``.

        With ``share_weights`` every depth uses the single module. Otherwise depth ``k`` has its own
        module up to ``n_future``; beyond that, ``recursive=True`` reuses the deepest module (what
        DeepSeek-V3 does when its single module drafts several steps), and ``recursive=False`` raises
        so that the mismatch is opt-in rather than silent.
        """
        if k < 1:
            raise ValueError("depth must be >= 1")
        if self.share_weights:
            return self.modules_[0]
        if k > self.n_future:
            if recursive:
                return self.modules_[-1]
            raise ValueError(
                f"depth {k} > n_future={self.n_future}; drafting beyond the trained depth needs "
                "share_weights=True (trained for it) or recursive=True (untrained reuse of the deepest module)"
            )
        return self.modules_[k - 1]

    def new_caches(self, k: int, batch: int, max_len: int, trunk, recursive: bool = False) -> list[KVCache]:
        return [
            KVCache(batch, self.cfg.n_heads, max_len, self.cfg.head_dim, trunk.device, trunk.dtype)
            for _ in self.module(k, recursive).blocks
        ]

    def step(
        self,
        k: int,
        prev_h: torch.Tensor,
        tok_emb: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        caches: list[KVCache] | None = None,
        recursive: bool = False,
    ) -> torch.Tensor:
        """Run depth ``k`` over a chunk of slots. ``prev_h`` are depth ``k-1`` states (depth 0 = trunk),
        ``tok_emb`` the embeddings of the tokens ``k`` positions ahead of those slots."""
        return self.module(k, recursive)(prev_h, tok_emb, cos, sin, caches)

    def logits(self, k: int, h_k: torch.Tensor, trunk, recursive: bool = False) -> torch.Tensor:
        return trunk.unembed(self.module(k, recursive).norm_out(h_k))

    def forward_train(
        self,
        h0: torch.Tensor,
        idx: torch.Tensor,
        trunk,
        n_depths: int | None = None,
        recursive: bool = False,
    ) -> list[torch.Tensor]:
        """Teacher-forced hidden states ``[h_1, ..., h_D]``; ``h_k`` has shape ``(B, T - k, d)`` and
        slot ``t`` predicts token ``t + 1 + k``."""
        n_depths = self.n_future if n_depths is None else n_depths
        T = idx.shape[1]
        outs: list[torch.Tensor] = []
        prev = h0
        for k in range(1, n_depths + 1):
            prev = prev[:, : T - k]
            emb = trunk.embed(idx[:, k:])
            cos, sin = trunk.rope(torch.arange(k, T, device=idx.device))
            prev = self.step(k, prev, emb, cos, sin, recursive=recursive)
            outs.append(prev)
        return outs


def build_heads(cfg: ModelConfig, mtp_cfg: MTPConfig) -> ParallelHeads | SequentialMTP | None:
    if mtp_cfg.kind == "none":
        return None
    if mtp_cfg.kind == "parallel":
        return ParallelHeads(cfg, mtp_cfg)
    return SequentialMTP(cfg, mtp_cfg)
