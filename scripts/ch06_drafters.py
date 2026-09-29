"""Chapter 06 experiments: cache discipline (per-depth vs single/EAGLE) and EAGLE-1 feature regression.

    python scripts/ch06_drafters.py            # trains seq1_feat / shared3_feat if missing, then decodes
    python scripts/ch06_drafters.py --device cpu

Uses the chapter 05 sweep checkpoints for the no-feature-loss baselines (run scripts/sweep.py
first, or pass --sweep to another sweep directory). Writes runs/ch06/results.{json,md}.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

from sweep import bench  # noqa: E402

from mtp.config import ModelConfig, MTPConfig, TrainConfig, to_dict  # noqa: E402
from mtp.data import CharDataset, load_tinyshakespeare  # noqa: E402
from mtp.train import load_checkpoint, train  # noqa: E402

TRAIN = {
    "seq1_feat": {"kind": "sequential", "n_future": 1, "loss_weight": 0.3, "feature_loss_weight": 1.0},
    "shared3_feat": {"kind": "sequential", "n_future": 3, "share_weights": True, "feature_loss_weight": 1.0},
}
SEQ1 = [(1, False), (2, True), (3, True)]
SHARED3 = [(1, False), (2, False), (3, False), (4, False)]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", default="runs/sweep")
    p.add_argument("--out", default="runs/ch06")
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-new-tokens", type=int, default=200)
    a = p.parse_args()

    ds = CharDataset(load_tinyshakespeare("data"))
    for name, mtp_kw in TRAIN.items():
        ckpt = os.path.join(a.out, name, "ckpt.pt")
        if os.path.exists(ckpt):
            continue
        print(f"\n=== training {name}: {mtp_kw}")
        train(
            ModelConfig(vocab_size=ds.vocab_size, max_seq_len=512),
            MTPConfig(**mtp_kw),
            TrainConfig(steps=a.steps, seq_len=a.seq_len, batch_size=32, eval_every=a.steps // 4, log_every=100, device=a.device, out_dir=os.path.join(a.out, name)),
            ds,
        )

    compare = [
        ("seq1_l0.3", os.path.join(a.sweep, "seq1_l0.3", "ckpt.pt"), SEQ1),
        ("seq1_feat", os.path.join(a.out, "seq1_feat", "ckpt.pt"), SEQ1),
        ("shared3", os.path.join(a.sweep, "shared3", "ckpt.pt"), SHARED3),
        ("shared3_feat", os.path.join(a.out, "shared3_feat", "ckpt.pt"), SHARED3),
    ]
    results: dict = {}
    for name, ckpt, settings in compare:
        model, _, meta = load_checkpoint(ckpt, "cpu")
        rows = []
        for drafter in ("depth", "eagle"):
            print(f"=== decoding {name} with drafter={drafter}")
            for b in bench(model, ds, settings, a.max_new_tokens, drafter):
                print(f"  K={b['K']}{' rec' if b['recursive'] else ''}: acc={['%.2f' % x for x in b['acceptance_by_depth']]} tokens/round={b['tokens_per_round']:.2f}")
                rows.append(b)
        results[name] = {"mtp": to_dict(model.mtp_cfg), "eval": meta["extra"], "bench": rows}
    os.makedirs(a.out, exist_ok=True)
    json.dump(results, open(os.path.join(a.out, "results.json"), "w"), indent=1)

    lines = ["| model | val NTP loss | acc_1 | acc_2 | acc_3 | feature loss (depth 1) |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        ev = r["eval"]
        accs = [f"{ev[f'val_acc{k}']:.3f}" if f"val_acc{k}" in ev else "-" for k in (1, 2, 3)]
        feat = f"{ev['val_feat1']:.3f}" if "val_feat1" in ev else "-"
        lines.append(f"| `{name}` | {ev['val_ntp']:.3f} | " + " | ".join(accs) + f" | {feat} |")
    lines += ["", "| model | K | drafter | conditional acceptance by depth | tokens per trunk pass |", "|---|---|---|---|---|"]
    for name, r in results.items():
        for b in r["bench"]:
            k = f"{b['K']}" + (" (recursive)" if b["recursive"] else "")
            rates = ", ".join(f"{x:.2f}" for x in b["acceptance_by_depth"])
            lines.append(f"| `{name}` | {k} | {b['drafter']} | {rates} | {b['tokens_per_round']:.2f} |")
    with open(os.path.join(a.out, "results.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nwrote {a.out}/results.json and results.md")


if __name__ == "__main__":
    main()
