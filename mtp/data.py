"""Datasets for the tutorials.

Character-level TinyShakespeare is the default: 1.1 MB of text, a 65-symbol vocabulary, no
tokenizer to install, and a model that trains to something readable in a few minutes on a laptop.
Everything about MTP (target alignment, heads, acceptance rates, speculative decoding) is
independent of the tokenizer, so the small vocabulary only changes the numbers, not the code.
"""

from __future__ import annotations

import os
import urllib.request

import torch

TINY_SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_tinyshakespeare(root: str = "data") -> str:
    path = os.path.join(root, "tinyshakespeare.txt")
    if not os.path.exists(path):
        os.makedirs(root, exist_ok=True)
        print(f"downloading TinyShakespeare to {path}")
        urllib.request.urlretrieve(TINY_SHAKESPEARE_URL, path)
    with open(path, encoding="utf-8") as f:
        return f.read()


class CharDataset:
    """Holds an encoded corpus split 90/10 and serves random contiguous windows."""

    def __init__(self, text: str, train_frac: float = 0.9):
        chars = sorted(set(text))
        self.itos = chars
        self.stoi = {c: i for i, c in enumerate(chars)}
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(len(data) * train_frac)
        self.train, self.val = data[:n], data[n:]

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def get_batch(
        self,
        split: str,
        batch_size: int,
        seq_len: int,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(idx, targets)`` with ``targets[:, t] = idx[:, t + 1]``."""
        data = self.train if split == "train" else self.val
        starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,), generator=generator)
        idx = torch.stack([data[s : s + seq_len] for s in starts])
        targets = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
        return idx.to(device), targets.to(device)
