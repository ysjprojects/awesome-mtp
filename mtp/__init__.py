"""mtp -- a small, readable framework for multi-token prediction.

Layout:
    config.py   ModelConfig / MTPConfig / TrainConfig
    layers.py   RMSNorm, RoPE, KVCache, Attention, SwiGLU, Block
    model.py    Trunk (decoder) and MTPModel (trunk + heads)
    heads.py    ParallelHeads (Gloeckle / Medusa) and SequentialMTP (DeepSeek-V3 / shared-weight)
    loss.py     target alignment, DeepSeek-style loss combination, memory-efficient train step
    decode.py   greedy/sampled generation and self-speculative decoding with cache rewind
    metrics.py  per-depth accuracy, acceptance -> speedup arithmetic
    data.py     char-level TinyShakespeare
    train.py    CLI training loop
"""

from .config import ModelConfig, MTPConfig, TrainConfig
from .decode import SpecStats, SpeculativeDecoder, generate, speculative_generate
from .loss import Losses, compute_losses, train_step
from .model import MTPModel, Trunk

__all__ = [
    "ModelConfig",
    "MTPConfig",
    "TrainConfig",
    "MTPModel",
    "Trunk",
    "Losses",
    "compute_losses",
    "train_step",
    "generate",
    "speculative_generate",
    "SpeculativeDecoder",
    "SpecStats",
]
