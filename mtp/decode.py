"""Decoding: ordinary autoregressive generation and MTP self-speculative decoding.

Self-speculative decoding turns the MTP heads into a built-in draft model:

1. **Draft.** From the current state propose ``K`` tokens ``d_1..d_K`` with distributions
   ``q_1..q_K`` (one per depth).
2. **Verify.** Run the trunk over ``[t1, d_1, ..., d_K]`` in a single forward pass. Position ``n+j``
   yields the target distribution ``p_j`` for the token that follows.
3. **Accept.** Walk the drafts left to right; accept ``d_{j+1}`` with probability
   ``min(1, p_j[d] / q_j[d])`` (Leviathan et al. 2023; Chen et al. 2023). On the first rejection
   sample a replacement from ``norm(max(p_j - q_{j+1}, 0))``; if everything was accepted sample a
   bonus token from ``p_K``. Either way exactly one extra token comes out of the verify pass.
4. **Rewind.** Truncate the trunk KV cache and the drafter state to the accepted prefix.

Throughout, ``t1`` denotes a token that is already decided (sampled from the correct
distribution) but not yet fed through the trunk: the very first one comes from the prompt's
last logits, later ones are the bonus/replacement tokens. With ``temperature=0`` every
distribution is a one-hot, the acceptance test reduces to ``d == argmax p``, and the output is
token-for-token identical to greedy decoding (``tests/test_decode.py``).

The drafters live here rather than in ``heads.py`` because they are decoding algorithms with
state (KV caches, slot bookkeeping, rewind), not model parameters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .model import MTPModel


def probs_from_logits(logits: torch.Tensor, temperature: float, top_k: int | None = None) -> torch.Tensor:
    """Distribution over the vocabulary; ``temperature == 0`` gives a one-hot argmax."""
    logits = logits.float()
    if temperature <= 0:
        return torch.nn.functional.one_hot(logits.argmax(-1), logits.shape[-1]).to(logits.dtype)
    logits = logits / temperature
    if top_k is not None and top_k < logits.shape[-1]:
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    return torch.softmax(logits, dim=-1)


def sample_from_probs(probs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Sample one index per row of ``probs`` (``(..., V)``) -> ``(...)``."""
    flat = probs.reshape(-1, probs.shape[-1])
    if generator is not None and generator.device != flat.device:
        flat = flat.cpu()
    idx = torch.multinomial(flat, 1, generator=generator).squeeze(-1)
    return idx.to(probs.device).reshape(probs.shape[:-1])


@torch.no_grad()
def generate(
    model: MTPModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Standard one-token-per-step decoding with a KV cache. ``idx`` is ``(B, T)``."""
    trunk = model.trunk
    B, T = idx.shape
    caches = trunk.new_caches(B, T + max_new_tokens)
    h = trunk(idx, caches=caches)
    logits = trunk.logits(h[:, -1])
    out = [idx]
    for i in range(max_new_tokens):
        nxt = sample_from_probs(probs_from_logits(logits, temperature, top_k), generator)
        out.append(nxt[:, None])
        if i + 1 == max_new_tokens:
            break
        h = trunk(nxt[:, None], caches=caches)
        logits = trunk.logits(h[:, -1])
    return torch.cat(out, dim=1)


@dataclass
class SpecStats:
    draft_len: int
    rounds: int = 0
    generated: int = 0
    drafted: list[int] = field(default_factory=list)
    accepted: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.drafted = self.drafted or [0] * self.draft_len
        self.accepted = self.accepted or [0] * self.draft_len

    @property
    def mean_accepted(self) -> float:
        """Average number of accepted draft tokens per verify pass."""
        return sum(self.accepted) / max(self.rounds, 1)

    @property
    def tokens_per_round(self) -> float:
        """Tokens decided per trunk forward pass (accepted drafts + the bonus token)."""
        return self.mean_accepted + 1.0

    def acceptance_by_depth(self) -> list[float]:
        """P(draft at depth k accepted | all shallower drafts accepted)."""
        rates = []
        for k in range(self.draft_len):
            attempts = self.rounds if k == 0 else self.accepted[k - 1]
            rates.append(self.accepted[k] / attempts if attempts else 0.0)
        return rates

    def summary(self) -> str:
        rates = ", ".join(f"{r:.2f}" for r in self.acceptance_by_depth())
        return (
            f"rounds={self.rounds} generated={self.generated} "
            f"mean_accepted={self.mean_accepted:.2f} tokens/round={self.tokens_per_round:.2f} "
            f"acceptance_by_depth=[{rates}]"
        )


class _ParallelDrafter:
    """Drafts with ``ParallelHeads``. Heads only ever read committed trunk states, so there is
    nothing to rewind: each round feeds the newly committed states through the heads and takes
    the last slot's predictions."""

    def __init__(self, dec: "SpeculativeDecoder"):
        self.heads = dec.model.heads
        self.heads.check_depth(dec.draft_len)
        self.caches = [self.heads.new_caches(k, 1, dec.capacity, dec.trunk) for k in range(1, dec.draft_len + 1)]
        self.processed = 0
        self.last_logits: list[torch.Tensor] = []

    def draft(self, dec: "SpeculativeDecoder") -> tuple[torch.Tensor, torch.Tensor]:
        n = dec.n
        chunk = dec.h0[:, self.processed : n]
        cos, sin = dec.trunk.rope(torch.arange(self.processed, n, device=dec.device))
        tokens, probs, self.last_logits = [], [], []
        for k in range(1, dec.draft_len + 1):
            h_k = self.heads.head_forward(k, chunk, cos, sin, self.caches[k - 1])
            logits = self.heads.logits(k, h_k[:, -1], dec.trunk)  # (1, V)
            self.last_logits.append(logits)
            q = dec.probs(logits)
            probs.append(q)
            tokens.append(dec.sample(q))
        self.processed = n
        return torch.stack(tokens, dim=1), torch.stack(probs, dim=1)  # (1, K), (1, K, V)

    def rollback(self, n: int) -> None:
        pass  # heads only consumed committed states; nothing to undo


class _SequentialDrafter:
    """Drafts with ``SequentialMTP``.

    Depth ``k`` keeps its own KV cache over *slots*: slot ``t`` of depth ``k`` combines depth
    ``k-1``'s state at slot ``t`` with the token at position ``t + k`` and predicts ``t + 1 + k``.
    With ``n`` committed tokens, depth ``k``'s cache is valid up to slot ``n - 1 - k`` (all inputs
    committed). Drafting extends every depth to slot ``n - 1`` using the proposals made so far,
    and ``rollback`` truncates each depth back to what the verify pass actually accepted.
    """

    def __init__(self, dec: "SpeculativeDecoder"):
        self.heads = dec.model.heads
        K = dec.draft_len
        self.heads.module(K)  # raises early if K exceeds what the heads can do
        self.caches = [self.heads.new_caches(k, 1, dec.capacity, dec.trunk) for k in range(1, K + 1)]
        self.outs = [torch.empty(1, dec.capacity, dec.model.cfg.d_model, device=dec.device, dtype=dec.trunk.dtype) for _ in range(K)]
        self.len = [0] * K  # committed valid slots per depth
        self.last_logits: list[torch.Tensor] = []

    def draft(self, dec: "SpeculativeDecoder") -> tuple[torch.Tensor, torch.Tensor]:
        n = dec.n
        proposals = [dec.t1]  # 0-d tokens at positions n, n+1, ... (t1, then the drafts)
        probs, self.last_logits = [], []
        for k in range(1, dec.draft_len + 1):
            start = self.len[k - 1]
            prev = (dec.h0 if k == 1 else self.outs[k - 2])[:, start:n]
            # Tokens k positions ahead of slots start..n-1, i.e. positions start+k .. n-1+k:
            # committed tokens below n, then proposals[0 .. k-1] (only the tail that is needed).
            n_proposed = n + k - max(start + k, n)
            ahead = torch.cat((dec.tokens[start + k : n], torch.stack(proposals[:n_proposed])))
            emb = dec.trunk.embed(ahead[None])
            cos, sin = dec.trunk.rope(torch.arange(start + k, n + k, device=dec.device))
            h_k = self.heads.step(k, prev, emb, cos, sin, self.caches[k - 1])
            self.outs[k - 1][:, start:n] = h_k
            self.len[k - 1] = n
            logits = self.heads.logits(k, h_k[:, -1], dec.trunk)  # (1, V)
            self.last_logits.append(logits)
            q = dec.probs(logits)
            probs.append(q)
            proposals.append(dec.sample(q)[0])
        return torch.stack(proposals[1:])[None], torch.stack(probs, dim=1)  # (1, K), (1, K, V)

    def rollback(self, n: int) -> None:
        for k in range(1, len(self.len) + 1):
            valid = min(self.len[k - 1], max(0, n - k))
            self.len[k - 1] = valid
            for cache in self.caches[k - 1]:
                cache.truncate(valid)


class SpeculativeDecoder:
    """Stateful self-speculative decoder for a single sequence (``idx`` of shape ``(1, T)``)."""

    def __init__(
        self,
        model: MTPModel,
        idx: torch.Tensor,
        max_new_tokens: int,
        draft_len: int,
        temperature: float = 0.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ):
        if model.heads is None:
            raise ValueError("speculative decoding needs a model with MTP heads")
        if idx.shape[0] != 1:
            raise ValueError("SpeculativeDecoder handles one sequence at a time")
        if draft_len < 1:
            raise ValueError("draft_len must be >= 1")
        self.model, self.trunk = model, model.trunk
        self.device = self.trunk.device
        self.draft_len, self.temperature, self.top_k, self.generator = draft_len, temperature, top_k, generator
        self.max_new_tokens = max_new_tokens
        T = idx.shape[1]
        self.capacity = T + max_new_tokens + draft_len + 1
        if self.capacity > model.cfg.max_seq_len:
            raise ValueError(
                f"prompt ({T}) + max_new_tokens ({max_new_tokens}) + draft_len + 1 exceeds max_seq_len={model.cfg.max_seq_len}"
            )
        self.tokens = torch.empty(self.capacity, dtype=torch.long, device=self.device)
        self.h0 = torch.empty(1, self.capacity, model.cfg.d_model, device=self.device, dtype=self.trunk.dtype)
        self.trunk_caches = self.trunk.new_caches(1, self.capacity)
        self.stats = SpecStats(draft_len)

        with torch.no_grad():
            h = self.trunk(idx, caches=self.trunk_caches)
        self.tokens[:T] = idx[0]
        self.h0[:, :T] = h
        self.prompt_len = T
        self.n = T  # committed tokens processed by the trunk
        self.t1 = self.sample(self.probs(self.trunk.logits(h[:, -1])))[0]
        self.stats.generated = 1
        self.drafter = _ParallelDrafter(self) if model.mtp_cfg.kind == "parallel" else _SequentialDrafter(self)

    # -- helpers --------------------------------------------------------------------------
    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return probs_from_logits(logits, self.temperature, self.top_k)

    def sample(self, probs: torch.Tensor) -> torch.Tensor:
        return sample_from_probs(probs, self.generator)

    def uniform(self) -> float:
        return float(torch.rand(1, generator=self.generator))

    @property
    def done(self) -> bool:
        return self.stats.generated >= self.max_new_tokens

    def output(self) -> torch.Tensor:
        """Prompt + generated tokens, trimmed to ``max_new_tokens``."""
        seq = torch.cat((self.tokens[: self.n], self.t1[None]))
        return seq[: self.prompt_len + self.max_new_tokens][None]

    # -- one speculative round --------------------------------------------------------------
    @torch.no_grad()
    def round(self) -> int:
        """Draft, verify, accept, rewind. Returns the number of newly decided tokens."""
        n, K = self.n, self.draft_len
        drafts, q = self.drafter.draft(self)  # (1, K), (1, K, V)
        block = torch.cat((self.t1[None], drafts[0]))[None]  # (1, K+1): tokens at positions n..n+K
        h = self.trunk(block, positions=torch.arange(n, n + K + 1, device=self.device), caches=self.trunk_caches)
        p = self.probs(self.trunk.logits(h))  # (1, K+1, V); p[:, j] is for position n+j+1

        accepted = 0
        for j in range(K):
            d = int(drafts[0, j])
            ratio = float(p[0, j, d]) / max(float(q[0, j, d]), 1e-20)
            if self.uniform() < min(1.0, ratio):
                accepted += 1
            else:
                break
        if accepted == K:
            bonus_probs = p[0, K]
        else:
            residual = (p[0, accepted] - q[0, accepted]).clamp_(min=0.0)
            total = float(residual.sum())
            bonus_probs = residual / total if total > 0 else p[0, accepted]
        bonus = self.sample(bonus_probs[None])[0]

        # commit t1 and the accepted drafts; rewind everything else
        new_n = n + accepted + 1
        self.tokens[n:new_n] = block[0, : accepted + 1]
        self.h0[:, n:new_n] = h[:, : accepted + 1]
        for cache in self.trunk_caches:
            cache.truncate(new_n)
        self.n = new_n
        self.drafter.rollback(new_n)
        self.t1 = bonus

        self.stats.rounds += 1
        for j in range(K):
            self.stats.drafted[j] += 1
        for j in range(accepted):
            self.stats.accepted[j] += 1
        self.stats.generated += accepted + 1
        return accepted + 1


@torch.no_grad()
def speculative_generate(
    model: MTPModel,
    idx: torch.Tensor,
    max_new_tokens: int,
    draft_len: int,
    temperature: float = 0.0,
    top_k: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, SpecStats]:
    dec = SpeculativeDecoder(model, idx, max_new_tokens, draft_len, temperature, top_k, generator)
    while not dec.done:
        dec.round()
    return dec.output(), dec.stats
