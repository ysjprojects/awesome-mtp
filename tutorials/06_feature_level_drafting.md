# 06 - Feature-level drafting: EAGLE, and how engines actually run MTP heads

Chapter 04's sequential drafter kept one KV cache *per depth*, because that is what a stack of
separately trained DeepSeek-V3 modules saw during training. It is not what vLLM or SGLang do
when they serve DeepSeek, Qwen3-Next, GLM or Kimi MTP heads. They run the single MTP module the
way EAGLE runs its drafter: one cache, and the module fed its own output as the "previous state"
for every draft step after the first. This chapter implements that drafter, adds EAGLE-1's
feature-regression loss, and measures both against the chapter 04 machinery on the same models.

Code: [`mtp/decode.py::_EagleDrafter`](../mtp/decode.py) (`drafter="eagle"`),
[`mtp/loss.py::feature_loss`](../mtp/loss.py) (`feature_loss_weight`),
[`scripts/ch06_drafters.py`](../scripts/ch06_drafters.py).

## EAGLE in one paragraph

EAGLE (Li et al. 2024) observed two things. Autoregression is easier at the *feature* level
than at the token level: the trunk's last hidden state `f_t` is a smooth, information-rich
object, while the token `x_{t+1}` sampled from it throws that information away. And the
uncertainty that remains after seeing `f_t` is exactly the sampling randomness, which is
resolved by feeding the drafter the token that was actually sampled. So the EAGLE drafter is a
one-layer decoder that consumes pairs `(f_t, Emb(x_{t+1}))`, predicts `f_{t+1}`, and reads the
next token from the shared unembedding of that predicted feature. Training regresses the
predicted feature onto the true one (Smooth-L1) and adds a small token cross-entropy
(`w = 0.1`); data is the model's own regenerated answers, not corpus text. EAGLE-2 grows a draft
*tree* whose shape follows the drafter's confidence; EAGLE-3 drops the regression (it caps how
much the drafter can improve with data), fuses low/mid/high-layer trunk features as input, and
trains with simulated multi-step drafting ("training-time test"). Kimi K3 pre-trains a
DeepSeek-style MTP layer and fine-tunes it into an EAGLE-3 drafter, which says how close the two
families are.

Look at the input side: `[RMSNorm(h) ; RMSNorm(Emb(x))] -> W -> Block` is *exactly* the
DeepSeek-V3 MTP module of chapter 03. The differences are what the module is trained on, what
loss it gets, and - the subject of this chapter - what cache it drafts with.

## Two cache disciplines for one module

**Per-depth** (`drafter="depth"`, chapter 04). Depth `k` owns a cache over slots; slot `t` of
depth `k` attends over depth-`k` slots `<= t`, all of which were computed from depth-`(k-1)`
states. This reproduces the attention context of teacher-forced training for a chain of
modules, at the cost of `K` caches and `K(K+1)/2` slot evaluations per round in steady state.

**Single / EAGLE** (`drafter="eagle"`). One cache. Committed slots pair the trunk's true state
at `t` with the token at `t+1` (identical to depth 1 above); each draft slot pairs the
module's *own previous output* with the previously drafted token. A draft at step `j` therefore
attends over `n` true-state slots followed by `j-1` self-predicted ones - a context no
teacher-forced training run ever produced. After verification the draft slots are discarded
and the accepted positions are re-processed with their true trunk states (this is what EAGLE's
reference implementation does), so the committed prefix never contains self-predicted
features. Cost: one cache, `a + 1` true slots plus `K - 1` draft slots per round.

`_EagleDrafter.draft` is twenty lines: extend the true-state prefix through slot `n-1` (whose
token is the decided `t1`), then loop `K-1` times feeding `(f_hat, Emb(d_j))` at position
`n + j`. `tests/test_decode.py::test_eagle_drafter_cache_matches_uncached_forward` runs six
rounds with rewinds, then replays the mixed sequence of true and predicted states through the
module without a cache and checks every draft's logits; the lossless test runs it with trained
and scrambled heads.

```bash
python -m mtp.bench --ckpt runs/shared3/ckpt.pt --draft-len 1 2 3 4 --drafter eagle
python -m mtp.bench --ckpt runs/seq2/ckpt.pt --draft-len 3 --recursive --drafter eagle   # refused: two modules
```

The eagle drafter refuses multi-module models (it reuses one module at every step; a `D = 2`
model would silently skip its trained depth-2 module) and, like the per-depth drafter, refuses
to go past a single module's trained depth without `--recursive`.

## Feature regression

`MTPConfig(feature_loss_weight=w)` adds, per depth, a Smooth-L1 term pulling depth `k`'s state at
slot `t` toward the trunk's state at position `t + k` - the state that predicts the same token.
Both sides are divided by the target's per-position RMS so the loss is O(1) regardless of the
trunk's residual-stream scale (raw Smooth-L1 on a freshly initialised 6-layer trunk is ~0.01
and would be inert next to a cross-entropy of ~4). It rides along in both loss paths, including
the chained memory-efficient backward, and `tests/test_loss.py` checks the gradients match the
naive path with it on.

The hypothesis it tests: if the module's outputs are trained to *look like* trunk states, then
feeding those outputs back in (recursive drafting) is less out-of-distribution, and acceptance
at depths 2+ should suffer less. EAGLE-1 relied on this; EAGLE-3 found it limiting.

```bash
python -m mtp.train --kind sequential --n-future 1 --feature-loss-weight 1.0 --out runs/seq1_feat
```

## Experiment

Four models, same trunk and 800-step budget as chapter 05 (`seq1_l0.3` and `shared3` are the
chapter 05 checkpoints; `seq1_feat` and `shared3_feat` add `feature_loss_weight = 1.0`), each
decoded with both drafters from four prompts, 200 greedy characters, output verified equal to
plain greedy.

```bash
python scripts/ch06_drafters.py            # trains the two feature-loss models if missing, then decodes
```

**Training-time metrics** (validation)

| model | val NTP loss | acc_1 | acc_2 | acc_3 | feature loss (depth 1) |
|---|---|---|---|---|---|
| `seq1_l0.3` | 1.510 | 0.540 | - | - | - |
| `seq1_feat` | 1.513 | 0.538 | - | - | 0.087 |
| `shared3` | 1.501 | 0.524 | 0.535 | 0.535 | - |
| `shared3_feat` | 1.506 | 0.521 | 0.526 | 0.526 | 0.091 |

**Decoding** (greedy, 4 prompts x 200 characters)

| model | K | drafter | conditional acceptance by depth | tokens per trunk pass |
|---|---|---|---|---|
| `seq1_l0.3` | 1 | depth | 0.81 | 1.81 |
| `seq1_l0.3` | 2 (recursive) | depth | 0.83, 0.53 | 2.27 |
| `seq1_l0.3` | 3 (recursive) | depth | 0.85, 0.54, 0.32 | 2.45 |
| `seq1_l0.3` | 1 | eagle | 0.81 | 1.81 |
| `seq1_l0.3` | 2 (recursive) | eagle | 0.81, 0.59 | 2.28 |
| `seq1_l0.3` | 3 (recursive) | eagle | 0.83, 0.56, 0.47 | 2.52 |
| `seq1_feat` | 1 | depth | 0.80 | 1.80 |
| `seq1_feat` | 2 (recursive) | depth | 0.81, 0.84 | 2.50 |
| `seq1_feat` | 3 (recursive) | depth | 0.84, 0.76, 0.74 | 2.94 |
| `seq1_feat` | 1 | eagle | 0.80 | 1.80 |
| `seq1_feat` | 2 (recursive) | eagle | 0.80, 0.83 | 2.45 |
| `seq1_feat` | 3 (recursive) | eagle | 0.83, 0.73, 0.75 | 2.89 |
| `shared3` | 1 | depth | 0.83 | 1.83 |
| `shared3` | 2 | depth | 0.80, 0.87 | 2.50 |
| `shared3` | 3 | depth | 0.79, 0.86, 0.82 | 3.03 |
| `shared3` | 4 | depth | 0.82, 0.87, 0.76, 0.85 | 3.52 |
| `shared3` | 1 | eagle | 0.83 | 1.83 |
| `shared3` | 2 | eagle | 0.81, 0.86 | 2.52 |
| `shared3` | 3 | eagle | 0.79, 0.86, 0.84 | 3.05 |
| `shared3` | 4 | eagle | 0.83, 0.88, 0.77, 0.88 | 3.60 |
| `shared3_feat` | 1 | depth | 0.76 | 1.76 |
| `shared3_feat` | 2 | depth | 0.75, 0.80 | 2.36 |
| `shared3_feat` | 3 | depth | 0.73, 0.89, 0.79 | 2.89 |
| `shared3_feat` | 4 | depth | 0.73, 0.83, 0.84, 0.87 | 3.28 |
| `shared3_feat` | 1 | eagle | 0.76 | 1.76 |
| `shared3_feat` | 2 | eagle | 0.75, 0.81 | 2.37 |
| `shared3_feat` | 3 | eagle | 0.74, 0.90, 0.78 | 2.92 |
| `shared3_feat` | 4 | eagle | 0.73, 0.83, 0.84, 0.87 | 3.28 |

## What it says

1. **Feature regression rescues recursive drafting of a single module.** `seq1_l0.3`, drafted
   two and three steps past its trained depth, accepts 53% and 32% of those drafts;
   `seq1_feat` - same architecture, same budget, one extra loss term - accepts 84% and 74%,
   and its tokens per pass at `K = 3` (2.94) match `shared3`, which was explicitly trained at
   three depths (3.03). Depth-1 accuracy (0.538 vs 0.540) and next-token loss (1.513 vs 1.510)
   are untouched. This is EAGLE-1's design decision confirmed in miniature: if the module's
   output is trained to look like a trunk state, feeding it back in is no longer out of
   distribution. For a DeepSeek-style `D = 1` head that will be used recursively, this is the
   cheapest fix available - no extra parameters, no multi-depth training.
2. **The same loss hurts a module that was already trained recursively.** `shared3_feat`
   loses ~7 points of depth-1 acceptance (0.76 vs 0.83) and drops from 3.52 to 3.28 tokens per
   pass at `K = 4`, with slightly lower teacher-forced accuracy at every depth. When the module
   already sees its own outputs during training, the regression is a constraint with no
   remaining mismatch to fix - the observation behind EAGLE-3 dropping it. Rule of thumb:
   regress features if you train at one depth and draft at several; do not if you train at
   the depths you draft.
3. **The cache discipline does not matter for acceptance.** Per-depth and single-cache drafting
   agree to within noise on every model and draft length (3.52 vs 3.60, 2.45 vs 2.52, 2.94 vs
   2.89, 3.28 vs 3.28 tokens per pass). Attending over the trunk's true states followed by the
   module's own predictions is not worse than attending over a self-consistent per-depth history,
   even for `shared3`, whose training only ever produced the latter. Since the single cache costs
   one cache and `a + K` slot evaluations per round instead of `K` caches and `K(K+1)/2`, the
   engines' choice is the right one, and `--drafter eagle` is the mode to reach for at `K >= 3`.
4. **Where the per-depth drafter still matters.** Separately trained multi-module models (`seq3`
   in chapter 05) have a distinct module per depth; the single-cache drafter cannot use them
   without collapsing them to one, which is why it refuses. Their acceptance at trained depths is
   as good as `shared3`'s (chapter 05), so the choice between the two families is about parameter
   count and recursion, not about the cache.

## Where this leaves the roadmap

- **Tree verification** (Medusa, EAGLE-2, ESP): several candidates per depth verified in one
  trunk pass with a tree attention mask. The single-cache drafter is the right base for it
  because a tree is a set of self-predicted branches over one true-state prefix. Planned as
  chapter 07 together with typical acceptance.
- **Multi-layer features and training-time test** (EAGLE-3): feed the drafter a projection of
  several trunk layers, and train it on its own outputs inside a single cache, which is the
  training-side twin of `drafter="eagle"`. The `share_weights` multi-depth training of chapter
  03 already does the "own outputs" half with per-depth contexts.
- **Data**: every strong drafter in the literature is trained on the model's own samples. The
  self-distillation chapter (08) supplies that.

## Exercises

1. Add `feature_loss_weight` to `scripts/sweep.py` and sweep `{0.1, 1, 10}` on `seq1`. Is there
   a weight at which depth-1 accuracy drops while recursive acceptance rises?
2. In `_EagleDrafter.rollback`, keep the accepted draft slots (self-predicted states) instead
   of re-processing with true states. Measure acceptance after 200 tokens. This is the cheaper,
   drifting variant that some early implementations shipped.
3. Count slot evaluations per round for both drafters at `K = 4` and confirm the `K(K+1)/2` vs
   `a + K` accounting above from `KVCache.length` deltas.
