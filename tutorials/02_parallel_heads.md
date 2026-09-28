# 02 - Parallel heads: Gloeckle et al. and Medusa

The simplest way to make a language model predict several tokens at once is to give it several
output heads. All of them read the same trunk state `h[t]`; head `k` is trained to predict
`x[t + 1 + k]`. This chapter implements that design in its two most-cited forms, gets the loss
right, and deals with the memory problem that every real MTP implementation has to solve.

Code: [`mtp/heads.py::ParallelHeads`](../mtp/heads.py), [`mtp/loss.py`](../mtp/loss.py).

## Two head architectures

**Gloeckle et al. (2024)**: each head is one full transformer block applied to the trunk output
sequence, followed by a norm and the *shared* unembedding. The heads have their own attention,
so head `k` at slot `t` can look back over `h[<= t]` and is not limited to the information the
trunk chose to put into one vector.

**Medusa (2024)**: each head is a residual MLP `x + SiLU(W x)` on the final normed state,
zero-initialised so every head starts out predicting exactly what the base model predicts, and
trained on a frozen backbone (Medusa-1) or jointly (Medusa-2). Medusa gives each head its own
unembedding; we share it, which loses nothing at this scale and keeps the parameter count honest.

```bash
python -m mtp.train --kind parallel --n-future 3                 --out runs/parallel3        # Gloeckle
python -m mtp.train --kind parallel --n-future 3 --head-arch mlp --out runs/medusa3          # Medusa heads
python -m mtp.train --kind parallel --n-future 3 --head-arch mlp --detach-trunk --out runs/medusa3_frozen
```

`--detach-trunk` stops MTP gradients at the trunk output (the heads still learn, the trunk only
sees the next-token loss) - Medusa-1's regime, and the switch you flip when you want MTP purely
as a drafter without touching the base model's quality.

In code, `ParallelHeads.head_forward(k, h0, cos, sin, caches)` produces depth-`k` states for a
chunk of trunk states, and `ParallelHeads.logits(k, h_k, trunk)` applies the head's norm and the
shared unembedding. The same two functions serve training (`forward_train`, all positions at
once) and drafting (chapter 04, one chunk of newly committed positions at a time).

## Target alignment, once and for all

With `targets[:, t] = idx[:, t + 1]`, depth `k`'s target at slot `t` is `targets[:, t + k]`, and
the last `k` slots have nothing to predict:

```python
def depth_loss(logits_k, targets, k):
    n = targets.shape[1] - k
    return cross_entropy(logits_k[:, :n], targets[:, k:])
```

That is the entire alignment logic. `tests/test_loss.py::test_depth_loss_targets_the_token_k_steps_ahead`
feeds a head that is one-hot on the true token `k` steps ahead and checks the loss is ~0, and a
head that is one-hot on the *next* token and checks the loss is large - the off-by-one that a
misaligned implementation would silently reward.

The total loss is `L_ntp + lambda / D * sum_k L_k` (`--loss-weight`, default 0.3). Equal
weighting (`lambda = D`) is a common mistake that measurably hurts the next-token head; the
production range is 0.1-0.3.

## The memory problem and the per-head backward

Each depth produces logits of shape `(B, T, V)`. With `V = 65` characters that is nothing; with a
128k-token vocabulary, `D + 1` such tensors kept alive for one backward pass become the dominant
activation and the reason people report MTP as "too expensive". Gloeckle et al.'s fix, which we
implement in `train_step(memory_efficient=True)`:

```python
h0 = trunk(idx)
h0_d = h0.detach().requires_grad_(True)    # a bridge tensor
ntp_loss(trunk.logits(h0_d)).backward()    # gradient lands in h0_d.grad, logits freed
for k in 1..D:
    (lambda / D * depth_loss(head_k(h0_d))).backward()   # accumulates into h0_d.grad, logits freed
h0.backward(h0_d.grad)                     # one trunk backward at the end
```

Only one logits tensor exists at any time, and the gradients are identical to the naive
all-at-once path: `tests/test_loss.py::test_memory_efficient_train_step_matches_naive_backward`
compares every parameter's gradient between the two paths for both head families.

Run with `--no-memory-efficient` to use the naive path; on this toy model you will not see a
difference in speed or memory, which is the point of doing the comparison here rather than on
a model where you cannot afford the naive path.

## What the numbers mean

The eval line prints `val_acc0` (next-token accuracy) and `val_acc1..val_accD`: the teacher-forced
top-1 accuracy of each depth against the data. Two things to expect:

- Accuracy drops with depth. Predicting `x[t + 2]` without knowing `x[t + 1]` is a marginal
  prediction; on text the second character after a given prefix is much less determined than
  the first. This is the structural weakness of parallel heads and the motivation for chapter 03.
- Depth accuracy is (approximately) the greedy acceptance rate you will measure in chapter 04:
  under greedy verification the depth-`k` draft is accepted iff it equals the trunk's own argmax,
  and a well-trained trunk's argmax agrees with the data about `val_acc0` of the time.

## Exercises

1. Train `--n-future 1` and `--n-future 6` and compare `val_acc1`. Does adding deeper heads
   hurt the shallow one? (Gloeckle et al. found `n = 4` best for 7B code models; there is no
   universal answer.)
2. Give Medusa heads their own unembedding (a `(V, d)` matrix per head initialised from the
   trunk's) and see whether `val_acc1` moves. Count the parameters you added.
3. Implement `lambda` warmup (Medusa-2 trains with the head weight ramped up over the first
   steps) and check whether `val_ntp` at step 300 improves.
