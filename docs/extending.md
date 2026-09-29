# Extending the framework

The code is small on purpose; the contracts below are what keep it correct when you add a head
design, a drafter, or a training objective. Every planned chapter (EAGLE-style drafters, mask
tokens, registers, tree verification) fits one of these three extension points.

## 1. A new head family

A head object is any `nn.Module` with these methods (see `ParallelHeads` and `SequentialMTP` in
[`mtp/heads.py`](../mtp/heads.py)):

| method | contract |
|---|---|
| `n_future: int`, `share_weights: bool` | how many depths are trained; whether one module serves all depths |
| `forward_train(h0, idx, trunk, n_depths=None, ...) -> list[Tensor]` | teacher-forced hidden states per depth; slot `t` of depth `k` must predict `idx[:, t + 1 + k]`. Length may be `T` or `T - k`; the loss slices to `T - k` |
| `logits(k, h_k, trunk) -> Tensor` | per-depth norm + the *shared* unembedding (`trunk.unembed`); works on `(B, n, d)` and `(B, d)` |
| `new_caches(k, batch, max_len, trunk) -> list[KVCache] | None` | one cache per attention layer inside depth `k`, or `None` if the depth is stateless |
| a chunk-wise forward (`head_forward` / `step`) | the same computation as `forward_train` for a contiguous chunk of slots, with explicit rotary tables and optional caches. Drafters call this |

Rules that the tests enforce and you should keep:

- **Positions are token positions, not slot indices.** A depth that handles the token at `t + k`
  in slot `t` uses rotary position `t + k`. `tests/test_decode.py::test_drafter_state_matches_teacher_forced_forward`
  compares the incremental path against `forward_train` and fails on any drift.
- **Gradients must be reproducible through `train_step`.** If your depths form a chain, mirror
  the detached-bridge scheme in `mtp/loss.py`; then add your kind to
  `tests/test_loss.py::test_memory_efficient_train_step_matches_naive_backward`.
- **Share the unembedding** unless the paper you are reproducing does not; separate output
  matrices are the single largest parameter cost and change comparisons.
- Register the kind in `MTPConfig.__post_init__` and `heads.build_heads`.

## 2. A new drafter

Drafters live in [`mtp/decode.py`](../mtp/decode.py); sequential-head drafters are registered
in the `DRAFTERS` dict (`"depth"` = per-depth caches, `"eagle"` = single cache) and selected
with `SpeculativeDecoder(..., drafter=...)` / `bench --drafter`. A new one implements:

```python
class Drafter:
    def __init__(self, dec: SpeculativeDecoder): ...
    def draft(self, dec) -> tuple[Tensor, Tensor]:   # (1, K) tokens, (1, K, V) proposal distributions
    def rollback(self, n_committed: int) -> None:    # forget everything that depended on rejected drafts
    last_logits: list[Tensor]                        # per-depth logits of the last draft (tests read this)
```

The decoder guarantees, when `draft` is called: `dec.n` committed tokens processed by the trunk
(`dec.tokens[:n]`, trunk states `dec.h0[:, :n]`, trunk caches of length `n`) and one decided
token `dec.t1` for position `n`. It guarantees, after `round`, that `rollback(n')` is called with
the new committed length before the next `draft`. The proposal distributions must be exactly
the distributions the tokens were sampled from (`dec.probs` / `dec.sample`), or rejection
sampling stops being exact - `test_speculative_sampling_matches_target_distribution` will notice
a systematic error, `test_speculative_greedy_decoding_is_lossless` an argmax one.

Invariant to preserve for anything with per-depth state: a slot of depth `k` is valid iff every
token it consumed is committed, i.e. `t + k <= n' - 1`.

## 3. A new training objective

`mtp/loss.py` composes `L_ntp + lambda / D * sum_k L_k`. Objectives that change *what* a depth
predicts (token order, future summaries, leap targets, masked spans) should replace `depth_loss`
for their kind and keep the rest; objectives that change *how* the depths are trained (curricula,
schedules, self-distillation targets) belong in `train.py` as a function of `step` that adjusts
`model.mtp_cfg.loss_weight` or the batch, so `train_step` stays a pure forward/backward.

## Checklist for a pull request that adds a kind

1. `MTPConfig` validation and `build_heads` registration.
2. `forward_train` + chunk-wise forward with explicit positions; `logits`; `new_caches`.
3. `train_step` support (naive path works automatically; add the memory-efficient path if the
   naive one would materialise more than one logits tensor).
4. A drafter, or a statement in the docstring that the kind is training-only.
5. Tests: gradient equivalence, lossless greedy decoding with trained and scrambled heads,
   drafter-vs-teacher-forcing after rewinds.
6. A tutorial section with the command that trains it and the numbers it produced.
