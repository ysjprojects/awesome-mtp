"""Configuration dataclasses.

Two configs describe a model:

* ``ModelConfig`` -- the shared transformer trunk (a small Llama-style decoder).
* ``MTPConfig`` -- how many *extra* future tokens are predicted and by which head design.

Naming convention (used throughout the repo and the tutorials): ``n_future = D`` is the
number of additional prediction depths beyond ordinary next-token prediction. Depth ``k``
(``1 <= k <= D``) reads the representation at position ``t`` and predicts token ``t + 1 + k``.
"MTP-1" in a model card therefore means ``n_future = 1`` (two targets per position). The
original Gloeckle et al. (2024) paper counts differently: their ``n = 4`` means four heads in
total, i.e. ``n_future = 3`` here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ModelConfig:
    vocab_size: int = 65
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int | None = None  # SwiGLU hidden size; defaults to ~8/3 * d_model rounded to a multiple of 64
    max_seq_len: int = 256
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.d_ff is None:
            self.d_ff = 64 * ((8 * self.d_model // 3 + 63) // 64)
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads


@dataclass
class MTPConfig:
    """How the model predicts extra future tokens.

    kind:
        ``"none"``       -- plain next-token prediction (the baseline).
        ``"parallel"``   -- independent heads on top of the trunk output (Gloeckle et al. 2024 /
                            Medusa). Head ``k`` sees only trunk states and predicts ``t + 1 + k``.
        ``"sequential"`` -- a causal chain of MTP modules (DeepSeek-V3). Module ``k`` combines the
                            previous depth's hidden state with the embedding of the *true* token
                            ``t + k`` and predicts ``t + 1 + k``.
    n_future:
        ``D``, the number of extra depths. ``0`` is only valid with ``kind="none"``.
    loss_weight:
        ``lambda`` in ``L = L_ntp + lambda / D * sum_k L_k`` (DeepSeek-V3 convention; they use 0.3
        for most of pre-training and 0.1 at the end, Ling 2.0 uses 0.1).
    head_arch:
        For ``kind="parallel"``: ``"block"`` = one transformer block per head (Gloeckle),
        ``"mlp"`` = one residual MLP per head (Medusa). Ignored otherwise.
    head_layers:
        Number of transformer blocks in each head/module (1 in both Gloeckle and DeepSeek-V3).
    share_weights:
        For ``kind="sequential"``: reuse a single module at every depth (Nemotron 3 Super, GLM-5,
        FastMTP). Lets the drafter be applied recursively beyond ``n_future`` at inference.
    detach_trunk:
        Do not send MTP gradients into the trunk (Medusa-1 style training of heads on a frozen
        model). Also used by the loss ablations in the tutorials.
    """

    kind: str = "none"
    n_future: int = 0
    loss_weight: float = 0.3
    head_arch: str = "block"
    head_layers: int = 1
    share_weights: bool = False
    detach_trunk: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"none", "parallel", "sequential"}:
            raise ValueError(f"unknown MTP kind {self.kind!r}")
        if self.head_arch not in {"block", "mlp"}:
            raise ValueError(f"unknown head_arch {self.head_arch!r}")
        if self.kind == "none" and self.n_future != 0:
            raise ValueError("kind='none' requires n_future=0")
        if self.kind != "none" and self.n_future < 1:
            raise ValueError("MTP requires n_future >= 1")
        if self.share_weights and self.kind != "sequential":
            raise ValueError("share_weights only applies to kind='sequential'")


def to_dict(cfg: Any) -> dict[str, Any]:
    return asdict(cfg)


@dataclass
class TrainConfig:
    steps: int = 1000
    batch_size: int = 32
    seq_len: int = 128
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_every: int = 200
    eval_batches: int = 10
    log_every: int = 50
    seed: int = 0
    memory_efficient: bool = True
    device: str = "auto"
    out_dir: str = "runs/default"
    extra: dict[str, Any] = field(default_factory=dict)
