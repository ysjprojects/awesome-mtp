# 04 - Self-speculative decoding: draft, verify, accept, rewind

The heads from chapters 02 and 03 predict future tokens; this chapter turns them into a draft
model and uses the trunk to verify the drafts, so that several tokens come out of every trunk
forward pass with *no change to the output distribution*. This is the mechanism behind every
"MTP speedup" number in the README, and the place where MTP implementations are most often
subtly wrong.

Code: [`mtp/decode.py`](../mtp/decode.py) (`SpeculativeDecoder`, `_ParallelDrafter`,
`_SequentialDrafter`, `speculative_generate`), [`mtp/metrics.py`](../mtp/metrics.py),
[`mtp/bench.py`](../mtp/bench.py).

## The round

State: `n` committed tokens that the trunk has processed (trunk KV cache holds positions
`0..n-1`, and we keep the trunk states `h0[0..n-1]`), plus one token `t1` that is already
*decided* but not yet fed through the trunk. `t1` starts as a sample from the prompt's last
logits.

1. **Draft.** Ask the heads for `K` tokens `d_1..d_K` for positions `n+1..n+K`, each with its
   proposal distribution `q_j` (a one-hot when decoding greedily).
2. **Verify.** Run the trunk once over the block `[t1, d_1, ..., d_K]` at positions `n..n+K`.
   The logits at position `n+j` give the target distribution `p_j` for the token at `n+j+1`.
3. **Accept.** For `j = 0..K-1`, accept `d_{j+1}` with probability `min(1, p_j[d] / q_{j+1}[d])`;
   stop at the first rejection. If rejected at `j`, sample the replacement from
   `norm(max(p_j - q_{j+1}, 0))`; if all `K` were accepted, sample a bonus token from `p_K`.
   Either way the round decides `accepted + 1` new tokens: `t1` (already decided), the accepted
   drafts, and one more token that becomes the next round's `t1`.
4. **Rewind.** Truncate the trunk cache to `n + accepted + 1` positions, store the trunk states
   of the newly committed tokens, and rewind the drafter (below).

With `temperature = 0` all distributions are one-hots: acceptance reduces to `d == argmax p`,
the replacement is `argmax p`, and the output is *exactly* the greedy sequence. This gives us a
test that has no tolerance in it:
`tests/test_decode.py::test_speculative_greedy_decoding_is_lossless` compares
`speculative_generate` with `generate` token for token, for both head families, for trained
heads (mostly accepts) and scrambled heads (mostly rejects and rewinds). With `temperature > 0`
the accept/replace rule (Leviathan et al. 2023; Chen et al. 2023) makes every emitted token an
exact sample from `p`, whatever `q` was; `test_speculative_sampling_matches_target_distribution`
checks the joint distribution of the first two generated tokens against the trunk's own
(total variation about 0.04 at 3000 samples, with roughly half the drafts rejected).

## Drafting with parallel heads

Head `k` reads trunk states only, so drafting is one pass of each head over the trunk states
committed since the last round; the last slot's outputs are the `K` drafts. Block heads keep a
KV cache over the committed positions (so the pass is incremental); MLP heads need none.
Nothing ever has to be rewound, because the heads never see a draft. The cost per round is
`K` head evaluations on `accepted + 1` new positions.

Note what this means: `d_2` does not know `d_1`. Acceptance at depth 2 is bounded by how
predictable `x[t+3]` is from `x[<= t]` alone.

## Drafting with sequential modules (the part worth reading twice)

Depth `k` at slot `t` consumes depth `k-1`'s state at slot `t` and the token at position `t+k`.
Two consequences:

- The drafts form a chain: `d_1` comes from depth 1 at slot `n-1` given `t1`; `d_2` from depth 2
  at slot `n-1` given depth 1's output there and `d_1`; and so on.
- Depth `k`'s block attends over depth `k`'s own slots, so it needs its own KV cache, and that
  cache is only *valid* for slots whose inputs were all committed tokens. With `n` committed
  tokens, slot `t` of depth `k` used the token at `t+k`, so the valid prefix is slots
  `0 .. n-1-k`.

`_SequentialDrafter` keeps, per depth, a KV cache and the depth's output states, plus `len[k]`,
the number of valid slots. A round extends every depth from `len[k]` up to slot `n-1`
(`k` slots in steady state, using committed tokens where the position is below `n` and the
proposals `t1, d_1, ...` above it), reads the drafts off slot `n-1`, and after verification
`rollback(n')` truncates depth `k` to `max(0, n' - k)` slots. The invariant that makes this
correct: a slot of depth `k` is valid iff its token position `t+k` is below the new committed
length `n'`, i.e. `t <= n' - 1 - k`.

Because the drafter is stateful and incremental, the natural bug is a state that drifts from
what teacher-forced training saw. `test_drafter_state_matches_teacher_forced_forward` runs six
rounds (with rejections and rewinds), then compares every depth's logits from the incremental
drafter against a from-scratch `forward_train` over the same committed tokens and proposals,
to 1e-4. If you change anything about positions or caches, that is the test that will tell you.

The first round pays the "MTP prefill": each depth runs over the whole prompt. That is the
draft-KV cost Windowed-MTP measures at million-token context; at our sizes it is invisible.

Recursive drafting beyond the trained depth (`--draft-len 4` on a `--n-future 3 --share-weights`
model) uses the same code path with the shared module at every depth; without `--share-weights`
the decoder raises, because depth `D+1` has no parameters.

## From acceptance to speed

`SpecStats` records, per depth, how many drafts were accepted *given the shallower ones were*.
If those conditional rates are `alpha_1..alpha_K`, the expected number of decided tokens per
round is

```
1 + alpha_1 + alpha_1*alpha_2 + ... + alpha_1*...*alpha_K
```

(`metrics.expected_accepted_length`), and the idealised speedup divides that by the relative
cost of a round, `1 + draft_cost`, where the verify pass counts as one decode step (true when
decoding is memory-bound) and `draft_cost` is `K` head evaluations relative to a trunk pass
(`K * head_layers / n_layers` for sequential heads, plus overheads). Wall-clock on a laptop
with a 6-layer model is dominated by Python and kernel-launch overhead and will not match this;
the acceptance numbers are what transfers to a real deployment, where a 1-layer drafter on a
60-layer trunk costs a few percent of a step.

## Run it

```bash
python -m mtp.bench --ckpt runs/parallel3/ckpt.pt --draft-len 1 2 3
python -m mtp.bench --ckpt runs/seq2/ckpt.pt      --draft-len 1 2
python -m mtp.bench --ckpt runs/shared3/ckpt.pt   --draft-len 1 2 3 4
python -m mtp.bench --ckpt runs/seq2/ckpt.pt --draft-len 2 --temperature 0.8 --seed 3 --show
```

`bench` decodes 200 characters greedily the ordinary way, then speculatively for each `K`,
asserts the greedy outputs are identical, and prints acceptance per depth, mean accepted
drafts per round and tokens per trunk pass.

## Results

RESULTS_PLACEHOLDER

## Exercises

1. Plot conditional acceptance per depth for `runs/seq2` at `K = 1, 2` and for `runs/shared3` at
   `K = 1..4`. Where does the extra depth stop paying for itself under the cost model above?
2. Change the accept rule to "accept if `p[d] > tau`" (a Medusa-style typical-acceptance
   heuristic) and measure both the speedup and the divergence from greedy output. That is the
   trade every non-lossless scheme is making.
3. Batching: `SpeculativeDecoder` handles one sequence. Sketch what `rollback` needs when each
   sequence in a batch accepts a different number of drafts (hint: per-row cache lengths, or
   padding plus masks). Chapter 10 does this.
