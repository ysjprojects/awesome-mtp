# 01 - The next-token baseline

Everything in this repository is a delta on top of an ordinary decoder-only language model, so
chapter 01 builds that model, trains it, and fixes the vocabulary of shapes and names the later
chapters rely on. If you have read a GPT implementation before, skim the *conventions* section
and run the training command.

Code: [`mtp/layers.py`](../mtp/layers.py), [`mtp/model.py`](../mtp/model.py),
[`mtp/data.py`](../mtp/data.py), [`mtp/train.py`](../mtp/train.py).

## The trunk

`Trunk` is a small Llama-style decoder: token embedding, `n_layers` pre-norm blocks (RMSNorm,
multi-head attention with rotary position embeddings, SwiGLU MLP), a final RMSNorm and an
unembedding tied to the input embedding.

```python
h = trunk(idx)            # (B, T, d_model): the residual stream *before* the final norm
logits = trunk.logits(h)  # (B, T, vocab): lm_head(norm(h))
```

Two deliberate choices matter for what comes later:

1. `forward` returns the pre-norm residual stream `h`, not logits. MTP heads consume `h` (the
   DeepSeek-V3 module normalises it itself; Gloeckle heads treat it like the input to one more
   layer), and the ordinary next-token head is just `logits(h)`.
2. Attention takes explicit rotary tables for the positions it is processing
   (`trunk.rope(positions)`) instead of assuming `0..T-1`. MTP modules run over *shifted*
   sequences and drafters append short chunks to caches, so positions must be passed in.

## The KV cache

`KVCache` preallocates `(B, H, max_len, head_dim)` keys and values per layer and exposes exactly
two operations:

```python
k_all, v_all = cache.append(k_new, v_new)   # write n new slots, return views over everything so far
cache.truncate(length)                      # rewind after a rejected speculative draft
```

When a chunk of `n` queries is appended after `m` cached slots, query `i` may attend to keys
`j <= m + i`, so the mask is `ones(n, m + n).tril(diagonal=m)`. Chapter 04 leans on `truncate`:
speculative decoding writes `K + 1` tokens into the cache during verification and then rewinds
to however many were accepted. `tests/test_layers.py` checks that chunked, cached forwards
equal a full forward, including after a truncate.

## Conventions used everywhere

- A batch is `idx` of shape `(B, T)` and `targets` with `targets[:, t] = idx[:, t + 1]`.
- `n_future = D` is the number of *extra* depths. Depth `k` reads position `t` and predicts
  `x[t + 1 + k]`. Depth 0 is the ordinary next-token head. (Gloeckle et al.'s "`n = 4`" is
  `D = 3` here; a model card's "MTP-1" is `D = 1`.)
- Losses follow DeepSeek-V3: `L = L_ntp + lambda / D * sum_k L_k`.

## Train it

```bash
python -m mtp.train --kind none --out runs/ntp
```

Defaults: 6 layers, `d_model = 256`, 8 heads (4.8M parameters), sequence length 256, batch 32,
1200 steps of AdamW with warmup + cosine decay, character-level TinyShakespeare (downloaded to
`data/` on first use). On an M1 Pro (MPS) this takes about six minutes; on CPU roughly three
times that. The log prints train loss every 50 steps and, every 250 steps, validation loss and
next-token top-1 accuracy (`val_acc0`); the checkpoint is saved to `runs/ntp/ckpt.pt`.

Expected: training loss around 1.0 nats/char and validation loss around 1.5 by the end (the
model overfits the 1 MB corpus in 1200 steps; that is fine for our purposes, but it is why the
later chapters compare *acceptance rates* rather than claim quality wins from MTP on this
dataset).

Generate from it:

```bash
python -m mtp.bench --ckpt runs/ntp/ckpt.pt --prompt "ROMEO:" --max-new-tokens 200 --show --draft-len 1
```

(`bench` will refuse the speculative part because this model has no heads; the autoregressive
line still prints.)

## What to look at in the code

- `Attention.forward`: the three cases (no cache / cache with one query / cache with a chunk)
  are all the attention logic this repository ever needs.
- `Trunk.forward`: note the `positions` argument and the default when a cache is present.
- `train.py::train`: the loop calls `mtp.loss.train_step`, which for `kind="none"` is just
  `loss.backward()`; chapter 02 replaces it.

## Exercises

1. Set `--seq-len 64` and confirm generation past position 64 still works (RoPE extrapolates,
   badly). Where would you cap `max_new_tokens` in `bench.py`?
2. Replace `is_causal=True` with an explicit mask in `Attention` and verify with the cache tests
   that nothing changes.
