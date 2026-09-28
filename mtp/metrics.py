"""Evaluation helpers that connect training-time numbers to decoding-time speedups."""

from __future__ import annotations

import torch

from .loss import IGNORE_INDEX


@torch.no_grad()
def depth_accuracies(model, idx: torch.Tensor, targets: torch.Tensor) -> list[float]:
    """Top-1 accuracy of the next-token head (index 0) and of every MTP depth (index k).

    Under greedy verification the depth-``k`` draft is accepted iff the head's argmax equals the
    trunk's argmax one step later. Teacher-forced accuracy against the *data* is the quantity we
    can measure without decoding, and it tracks acceptance closely for a well-trained model.
    """
    trunk, heads = model.trunk, model.heads
    h0 = trunk(idx)
    accs = [_top1(trunk.logits(h0), targets)]
    if heads is not None:
        for k, h_k in enumerate(heads.forward_train(h0, idx, trunk), start=1):
            T = targets.shape[1]
            accs.append(_top1(heads.logits(k, h_k, trunk)[:, : T - k], targets[:, k:]))
    return accs


def _top1(logits: torch.Tensor, targets: torch.Tensor) -> float:
    mask = targets != IGNORE_INDEX
    pred = logits.argmax(-1)
    return float((pred[mask] == targets[mask]).float().mean())


def expected_accepted_length(acceptance_by_depth: list[float]) -> float:
    """E[accepted drafts per round] from conditional per-depth acceptance rates.

    Drafts are accepted left to right, so the expected count is
    ``sum_k prod_{j<=k} alpha_j``. Tokens per verify pass is this plus one (bonus token).
    """
    total, running = 0.0, 1.0
    for a in acceptance_by_depth:
        running *= a
        total += running
    return total


def speedup_estimate(acceptance_by_depth: list[float], draft_cost: float) -> float:
    """Idealised speedup: tokens per round divided by the relative cost of a round.

    ``draft_cost`` is the cost of drafting ``K`` tokens relative to one trunk forward pass
    (e.g. ``K * n_head_layers / n_trunk_layers`` for sequential heads); the verify pass is taken
    as one trunk forward regardless of ``K`` (the memory-bound regime).
    """
    return (1.0 + expected_accepted_length(acceptance_by_depth)) / (1.0 + draft_cost)
