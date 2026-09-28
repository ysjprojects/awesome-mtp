"""Transformer building blocks: RMSNorm, rotary embeddings, a KV cache, attention, SwiGLU.

Everything here is deliberately plain PyTorch so the MTP-specific code in ``heads.py`` and
``decode.py`` has no hidden machinery behind it. The only slightly unusual piece is that
attention takes explicit rotary tables for the positions of the tokens it is processing: MTP
modules process *shifted* sequences (depth ``k`` handles token ``t + k`` at slot ``t``) and
drafters process short chunks appended to a cache, so positions cannot be inferred from the
tensor shape.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def precompute_rope(max_len: int, head_dim: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``cos, sin`` tables of shape ``(max_len, head_dim // 2)``."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    pos = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate ``x`` of shape ``(B, H, n, head_dim)`` with tables of shape ``(n, head_dim // 2)``."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.to(x.dtype)[None, None]
    sin = sin.to(x.dtype)[None, None]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1)


class KVCache:
    """A preallocated key/value cache for one attention layer.

    ``append`` writes ``n`` new slots and returns views over everything cached so far;
    ``truncate`` rewinds the cache after a speculative round rejected some drafts.
    """

    def __init__(self, batch: int, n_heads: int, max_len: int, head_dim: int, device, dtype):
        self.k = torch.zeros(batch, n_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    @property
    def max_len(self) -> int:
        return self.k.shape[2]

    def append(self, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = k.shape[2]
        if self.length + n > self.max_len:
            raise RuntimeError(f"KV cache overflow: {self.length} + {n} > {self.max_len}")
        self.k[:, :, self.length : self.length + n] = k
        self.v[:, :, self.length : self.length + n] = v
        self.length += n
        return self.k[:, :, : self.length], self.v[:, :, : self.length]

    def truncate(self, length: int) -> None:
        if not 0 <= length <= self.length:
            raise ValueError(f"cannot truncate cache of length {self.length} to {length}")
        self.length = length


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        B, n, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q = q.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, n, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        if cache is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            m = cache.length
            k, v = cache.append(k, v)
            if n == 1:
                y = F.scaled_dot_product_attention(q, k, v)
            else:
                # query i sits at absolute slot m + i and may attend to keys j <= m + i
                mask = torch.ones(n, m + n, dtype=torch.bool, device=x.device).tril(diagonal=m)
                y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        y = y.transpose(1, 2).reshape(B, n, d)
        return self.dropout(self.proj(y))


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w3 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.w2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    """Pre-norm transformer block."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, cache: KVCache | None = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), cos, sin, cache)
        return x + self.mlp(self.mlp_norm(x))


class ResidualMLP(nn.Module):
    """Medusa-style head block: ``x + silu(W x)`` with ``W`` zero-initialised so the head starts as
    the identity map on the trunk representation."""

    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + F.silu(self.linear(x))


def init_weights(module: nn.Module, n_layers: int) -> None:
    """GPT-2 style init: N(0, 0.02) with residual projections scaled by 1/sqrt(2 * n_layers).

    ``ResidualMLP`` blocks are left alone so they keep their zero initialisation.
    """
    skip = {id(p) for m in module.modules() if isinstance(m, ResidualMLP) for p in m.parameters()}
    for name, p in module.named_parameters():
        if p.dim() < 2 or id(p) in skip:
            continue
        std = 0.02
        if name.endswith("proj.weight") or name.endswith("w2.weight"):
            std = 0.02 / math.sqrt(2 * n_layers)
        nn.init.normal_(p, mean=0.0, std=std)
