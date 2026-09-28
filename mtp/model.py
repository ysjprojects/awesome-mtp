"""The shared trunk and the ``MTPModel`` wrapper that attaches MTP heads to it."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig, MTPConfig
from .layers import Block, KVCache, RMSNorm, init_weights, precompute_rope


class Trunk(nn.Module):
    """A small Llama-style decoder: embeddings -> ``n_layers`` pre-norm blocks -> final norm ->
    (tied) unembedding.

    ``forward`` returns the *pre-final-norm* residual stream ``h`` of shape ``(B, n, d_model)``.
    This is the representation MTP heads consume; ``logits(h)`` applies the final norm and the
    unembedding for ordinary next-token prediction.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        init_weights(self, cfg.n_layers)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight
        cos, sin = precompute_rope(cfg.max_seq_len, cfg.head_dim, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.tok_emb.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.tok_emb.weight.dtype

    def rope(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotary tables for explicit absolute ``positions`` (a 1-D LongTensor)."""
        if int(positions.max()) >= self.cfg.max_seq_len:
            raise ValueError(f"position {int(positions.max())} exceeds max_seq_len={self.cfg.max_seq_len}")
        return self.rope_cos[positions], self.rope_sin[positions]

    def embed(self, idx: torch.Tensor) -> torch.Tensor:
        return self.tok_emb(idx)

    def unembed(self, x: torch.Tensor) -> torch.Tensor:
        return self.lm_head(x)

    def logits(self, h: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.norm(h))

    def new_caches(self, batch: int, max_len: int) -> list[KVCache]:
        return [KVCache(batch, self.cfg.n_heads, max_len, self.cfg.head_dim, self.device, self.dtype) for _ in self.blocks]

    def forward(
        self,
        idx: torch.Tensor,
        positions: torch.Tensor | None = None,
        caches: list[KVCache] | None = None,
    ) -> torch.Tensor:
        """Run the trunk over ``idx`` of shape ``(B, n)``.

        Without caches, ``idx`` is a full sequence starting at position 0. With caches, ``idx`` is a
        chunk appended after whatever the caches already hold; ``positions`` defaults to the
        contiguous range following the cache length.
        """
        B, n = idx.shape
        if positions is None:
            start = caches[0].length if caches else 0
            positions = torch.arange(start, start + n, device=idx.device)
        cos, sin = self.rope(positions)
        x = self.tok_emb(idx)
        for i, block in enumerate(self.blocks):
            x = block(x, cos, sin, None if caches is None else caches[i])
        return x


class MTPModel(nn.Module):
    """Trunk + optional MTP heads.

    ``forward(idx, targets)`` returns a ``Losses`` object (see ``mtp.loss``); ``forward(idx)``
    returns next-token logits only. Training code should prefer ``mtp.loss.train_step`` which
    also implements the memory-efficient per-depth backward pass.
    """

    def __init__(self, cfg: ModelConfig, mtp_cfg: MTPConfig | None = None):
        super().__init__()
        from .heads import build_heads  # local import: heads depend on layers only, but keep module graph acyclic

        self.cfg = cfg
        self.mtp_cfg = mtp_cfg or MTPConfig()
        self.trunk = Trunk(cfg)
        self.heads = build_heads(cfg, self.mtp_cfg)

    @property
    def n_future(self) -> int:
        return self.mtp_cfg.n_future

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        if targets is None:
            return self.trunk.logits(self.trunk(idx))
        from .loss import compute_losses

        return compute_losses(self, idx, targets)

    def num_params(self, trainable_only: bool = False) -> dict[str, int]:
        def count(m: nn.Module | None) -> int:
            if m is None:
                return 0
            seen: set[int] = set()
            total = 0
            for p in m.parameters():
                if id(p) in seen or (trainable_only and not p.requires_grad):
                    continue
                seen.add(id(p))
                total += p.numel()
            return total

        return {"trunk": count(self.trunk), "heads": count(self.heads), "total": count(self)}
