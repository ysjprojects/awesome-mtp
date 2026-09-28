# 03 - Sequential MTP: the DeepSeek-V3 module and its shared-weight descendants

Parallel heads predict `x[t + 2]` without knowing `x[t + 1]`. DeepSeek-V3's design fixes that
with a chain: depth 1 sees the true next token and predicts the one after; depth 2 sees depth 1's
state and the true token two ahead; and so on. The chain keeps the full causal factorisation
`P(x[t+2] | x[<= t+1])`, which is why every production MTP model since has used it. This chapter
implements it exactly, including the position bookkeeping that most explanations gloss over,
and the shared-weight variant that GLM-5, Nemotron 3 Super and FastMTP use.

Code: [`mtp/heads.py::SequentialMTP`](../mtp/heads.py), [`mtp/loss.py::train_step`](../mtp/loss.py).

## The module

For depth `k` at slot `t`:

```
h'_k[t] = M_k [ RMSNorm(h_{k-1}[t]) ; RMSNorm(Emb(x[t + k])) ]     M_k : 2d -> d, no bias
h_k     = Block_k(h'_k)                                            causal attention over slots
logits  = Unembed( RMSNorm_k(h_k[t]) )                             predicts x[t + 1 + k]
```

`h_0` is the trunk output. `Emb` and `Unembed` are the trunk's own (shared) matrices; each depth
owns the two input norms, `M_k`, one transformer block (`--head-layers`) and an output norm,
mirroring the checkpoint layout of DeepSeek-V3 (`enorm`, `hnorm`, `eh_proj`, `mtp_block`,
`shared_head.norm`).

```python
class MTPModule(nn.Module):
    def forward(self, prev_h, tok_emb, cos, sin, caches=None):
        x = self.proj(torch.cat((self.norm_h(prev_h), self.norm_e(tok_emb)), dim=-1))
        for block in self.blocks:
            x = block(x, cos, sin, caches)
        return x
```

## Teacher forcing and positions

During training `x[t + k]` is the ground-truth token, so depth `k` runs over slots
`0 .. T-1-k` with the token sequence shifted by `k`:

```python
prev = h0
for k in 1..D:
    prev = prev[:, :T-k]                      # depth k-1 states for slots 0..T-1-k
    emb  = trunk.embed(idx[:, k:])            # tokens x[k .. T-1]
    cos, sin = trunk.rope(arange(k, T))       # slot t holds the token at position t+k
    prev = heads.step(k, prev, emb, cos, sin) # -> h_k, shape (B, T-k, d)
```

Two details are easy to get wrong and are the reason `tests/test_decode.py` exists:

- **Rotary positions.** Slot `t` of depth `k` is about the token at position `t + k`, so its
  rotary position is `t + k`, not `t`. Getting this wrong still trains (the model adapts) but
  breaks the equivalence between teacher-forced training and incremental drafting.
- **Which sequence the block attends over.** Depth `k`'s attention runs over depth `k`'s own
  sequence `h'_k`, not over the trunk's. At inference each depth therefore needs its own KV
  cache over its own slots (chapter 04).

The loss is the same `depth_loss` as chapter 02; the logits for depth `k` already have length
`T - k`, so the slice is a no-op.

## Training: the chain needs a chained backward

The memory-efficient trick of chapter 02 assumed heads were independent given `h0`. In the
chain, depth `k + 1`'s loss must also flow back through depth `k`. `train_step` handles it by
placing a detached bridge between consecutive depths on the way forward and walking the chain
backwards on the way back:

```python
for k in D..1:
    tensors, grads = [lambda/D * loss_k], [None]
    if k < D:                                   # gradient that depth k+1 sent into its input bridge
        tensors.append(h_k); grads.append(bridge_k.grad)
    torch.autograd.backward(tensors, grads)
h0.backward(h0_d.grad)
```

At most one depth's logits are alive at a time, and the gradients match the naive path
(`tests/test_loss.py`, cases `sequential-3` and `sequential-2-shared`).

```bash
python -m mtp.train --kind sequential --n-future 2 --loss-weight 0.3 --out runs/seq2
```

DeepSeek-V3 used `D = 1` with `lambda = 0.3` for the first 10T tokens and `0.1` for the last
4.8T; Ling 2.0 uses `D = 1`, `lambda = 0.1`; MiMo-V2-Flash trains three modules. Compare
`val_acc1` with the parallel model from chapter 02 trained for the same number of steps: the
chained depth-1 head is conditioned on the true next token and should be clearly more accurate,
and the gap widens at depth 2.

## Shared weights: training the module the way you will use it

At inference a `D = 1` module is routinely applied *recursively* to draft 2, 3, 4 tokens: feed its
own output state back in as `h_{k-1}`. That input is out of distribution - training only ever
showed the module trunk states - and acceptance at depth 2+ suffers (GLM-5 measured DeepSeek-V3.2
at 2.55 accepted tokens per step). The fix used by GLM-5 (three depths), Nemotron 3 Super (two)
and FastMTP is to train *one* set of parameters at several depths, so during training depth 2's
input is exactly the module's own depth-1 output:

```bash
python -m mtp.train --kind sequential --n-future 3 --share-weights --out runs/shared3
```

`SequentialMTP.module(k)` returns the single module for every `k` when `share_weights=True`,
and the drafter in chapter 04 is then allowed to draft beyond `n_future` (`--draft-len 4` on a
`--n-future 3` model), which is what "recursive MTP" means in the model cards. Memory and KV
cost stay at the one-module level; the price is that one module has to be good at every depth.

## Exercises

1. Train `--n-future 1` sequential and decode it at `--draft-len 3 --recursive` (chapter 04),
   then do the same with `--n-future 3 --share-weights`. That gap is the train/inference
   mismatch; chapter 05 reports it for the reference sweep.
2. Add a second block per module (`--head-layers 2`); does `val_acc2` improve more than
   `val_acc1`? What did it cost in drafting time?
3. DeepSeek-V3 feeds the *token embedding* of `x[t + k]`; EAGLE feeds the trunk's *feature* for
   it instead. Sketch what changes in `MTPModule.forward` and in the drafter's state (chapter 06
   does this properly).
