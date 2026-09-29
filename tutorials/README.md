# Tutorial index

Each chapter is short, points at the code it introduces, and ends with exercises. Run them in
order the first time; the later ones assume the vocabulary of the earlier ones (depth `k`,
`n_future = D`, draft length `K`, "committed" vs "decided" tokens).

| # | Chapter | Code introduced | Runs |
|---|---|---|---|
| 01 | [Next-token baseline](01_next_token_baseline.md) | `layers.py`, `model.py`, `data.py`, `train.py` | `python -m mtp.train --kind none` |
| 02 | [Parallel heads](02_parallel_heads.md) | `heads.py::ParallelHeads`, `loss.py` | `--kind parallel --n-future 3 [--head-arch mlp] [--detach-trunk]` |
| 03 | [Sequential MTP](03_sequential_mtp.md) | `heads.py::SequentialMTP`, chained backward in `loss.py` | `--kind sequential --n-future 2`, `--n-future 3 --share-weights` |
| 04 | [Self-speculative decoding](04_self_speculative_decoding.md) | `decode.py`, `metrics.py`, `bench.py` | `python -m mtp.bench --ckpt ... --draft-len 1 2 3` |
| 05 | [Measuring MTP](05_measuring_mtp.md) | `scripts/sweep.py`, `scripts/plot_results.py`, `--recursive` | `python scripts/sweep.py && python scripts/plot_results.py` |
| 06 | [Feature-level drafting](06_feature_level_drafting.md) | `decode.py::_EagleDrafter` (`--drafter eagle`), `loss.py::feature_loss` (`--feature-loss-weight`) | `python scripts/ch06_drafters.py` |

Planned: 07 tree verification and non-lossless acceptance; 08 mask tokens, gated LoRA,
registers and training recipes; 09 MTP and RL; 10 serving and real checkpoints; 11 beyond
next-`k` tokens; 12 a scaling study. The [README roadmap](../README.md#12-the-tutorial-roadmap-and-status)
tracks status.

Setup for all chapters:

```bash
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"
pytest -q
```
