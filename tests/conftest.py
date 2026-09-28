"""Shared fixtures: a tiny model and a synthetic Markov-chain corpus it can learn in seconds.

The chain has a dominant successor for every state (probability ``PEAK``) so that a trained model
accepts drafts often but not always -- both the accept and the reject paths of speculative
decoding get exercised, and the greedy-equivalence tests are meaningful.
"""

from __future__ import annotations

import functools

import pytest
import torch

from mtp import ModelConfig, MTPConfig, MTPModel, train_step

VOCAB = 8
PEAK = 0.75


def tiny_model_cfg(**overrides) -> ModelConfig:
    base = dict(vocab_size=VOCAB, d_model=32, n_layers=2, n_heads=4, max_seq_len=128)
    base.update(overrides)
    return ModelConfig(**base)


def transition_matrix(seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(VOCAB, generator=g)
    P = torch.full((VOCAB, VOCAB), (1 - PEAK) / (VOCAB - 1))
    P[torch.arange(VOCAB), perm] = PEAK
    return P


def sample_chain(P: torch.Tensor, batch: int, length: int, generator: torch.Generator) -> torch.Tensor:
    x = torch.empty(batch, length + 1, dtype=torch.long)
    x[:, 0] = torch.randint(0, VOCAB, (batch,), generator=generator)
    for t in range(length):
        x[:, t + 1] = torch.multinomial(P[x[:, t]], 1, generator=generator).squeeze(-1)
    return x


def make_model(kind: str = "none", n_future: int = 0, seed: int = 0, **mtp_kw) -> MTPModel:
    torch.manual_seed(seed)
    return MTPModel(tiny_model_cfg(), MTPConfig(kind=kind, n_future=n_future, **mtp_kw))


@functools.lru_cache(maxsize=None)
def _trained(kind: str, n_future: int, head_arch: str, share_weights: bool, steps: int) -> MTPModel:
    model = make_model(kind, n_future, head_arch=head_arch, share_weights=share_weights)
    P = transition_matrix()
    g = torch.Generator().manual_seed(1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        x = sample_chain(P, 16, 32, g)
        train_step(model, x[:, :-1], x[:, 1:])
        opt.step()
        opt.zero_grad(set_to_none=True)
    return model.eval()


def trained_model(kind: str, n_future: int, head_arch: str = "block", share_weights: bool = False, steps: int = 200) -> MTPModel:
    return _trained(kind, n_future, head_arch, share_weights, steps)


@pytest.fixture
def chain_prompt() -> torch.Tensor:
    g = torch.Generator().manual_seed(7)
    return sample_chain(transition_matrix(), 1, 11, g)[:, :12]
