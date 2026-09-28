"""Chapter 05 sweep: train a family of small MTP variants and measure what each choice buys.

    python scripts/sweep.py                       # ~35 min on an M1 Pro (MPS); results in runs/sweep/
    python scripts/sweep.py --only seq1_l0.3 seq3 # subset
    python scripts/sweep.py --skip-train          # re-run only the decoding measurements

Each variant is trained with the same budget (default 800 steps, sequence 128, batch 32) and
then decoded greedily from four prompts with every draft length it supports. Writes
``runs/sweep/results.json`` (consumed by ``scripts/plot_results.py``) and ``results.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mtp.config import ModelConfig, MTPConfig, TrainConfig, to_dict  # noqa: E402
from mtp.data import CharDataset, load_tinyshakespeare  # noqa: E402
from mtp.decode import generate, speculative_generate  # noqa: E402
from mtp.train import load_checkpoint, train  # noqa: E402

# name -> (MTPConfig kwargs, draft settings [(K, recursive), ...])
VARIANTS: dict[str, tuple[dict, list[tuple[int, bool]]]] = {
    "ntp": ({"kind": "none"}, []),
    # how much auxiliary loss? (DeepSeek-V3: 0.3 then 0.1; Ling 2.0: 0.1)
    "seq1_l0.1": ({"kind": "sequential", "n_future": 1, "loss_weight": 0.1}, [(1, False), (2, True), (3, True)]),
    "seq1_l0.3": ({"kind": "sequential", "n_future": 1, "loss_weight": 0.3}, [(1, False), (2, True), (3, True)]),
    "seq1_l1.0": ({"kind": "sequential", "n_future": 1, "loss_weight": 1.0}, [(1, False), (2, True), (3, True)]),
    # Medusa-1 regime: the trunk never sees the MTP gradient
    "seq1_detach": ({"kind": "sequential", "n_future": 1, "loss_weight": 0.3, "detach_trunk": True}, [(1, False), (2, True), (3, True)]),
    # parallel heads: transformer block vs residual MLP
    "par3_block": ({"kind": "parallel", "n_future": 3, "head_arch": "block"}, [(1, False), (2, False), (3, False)]),
    "par3_mlp": ({"kind": "parallel", "n_future": 3, "head_arch": "mlp"}, [(1, False), (2, False), (3, False)]),
    # three depths: separate modules vs one shared module trained at every depth
    "seq3": ({"kind": "sequential", "n_future": 3}, [(1, False), (2, False), (3, False), (4, True)]),
    "shared3": ({"kind": "sequential", "n_future": 3, "share_weights": True}, [(1, False), (2, False), (3, False), (4, False)]),
}

PROMPTS = ["ROMEO:", "JULIET:", "First Citizen:", "KING RICHARD III:"]


def bench(model, ds: CharDataset, settings: list[tuple[int, bool]], max_new_tokens: int) -> list[dict]:
    rows = []
    for K, recursive in settings:
        accepted = [0] * K
        rounds = 0
        for prompt in PROMPTS:
            idx = ds.encode(prompt)[None]
            base = generate(model, idx, max_new_tokens)
            out, stats = speculative_generate(model, idx, max_new_tokens, K, recursive=recursive)
            if not torch.equal(out, base):
                raise RuntimeError(f"speculative output diverged from greedy for K={K}")
            rounds += stats.rounds
            for j in range(K):
                accepted[j] += stats.accepted[j]
        by_depth = []
        for j in range(K):
            attempts = rounds if j == 0 else accepted[j - 1]
            by_depth.append(accepted[j] / attempts if attempts else 0.0)
        rows.append(
            {
                "K": K,
                "recursive": recursive,
                "rounds": rounds,
                "acceptance_by_depth": by_depth,
                "mean_accepted": sum(accepted) / rounds,
                "tokens_per_round": 1 + sum(accepted) / rounds,
            }
        )
    return rows


def write_markdown(results: dict, path: str) -> None:
    lines = ["| run | val NTP loss | acc_0 | acc_1 | acc_2 | acc_3 |", "|---|---|---|---|---|---|"]
    for name, r in results.items():
        ev = r["eval"]
        accs = [f"{ev[f'val_acc{k}']:.3f}" if f"val_acc{k}" in ev else "-" for k in range(4)]
        lines.append(f"| `{name}` | {ev['val_ntp']:.3f} | " + " | ".join(accs) + " |")
    lines += ["", "| run | K | conditional acceptance by depth | tokens per trunk pass |", "|---|---|---|---|"]
    for name, r in results.items():
        for b in r["bench"]:
            k = f"{b['K']}" + (" (recursive)" if b["recursive"] else "")
            rates = ", ".join(f"{a:.2f}" for a in b["acceptance_by_depth"])
            lines.append(f"| `{name}` | {k} | {rates} | {b['tokens_per_round']:.2f} |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--only", nargs="*", default=None)
    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default="runs/sweep")
    p.add_argument("--skip-train", action="store_true")
    a = p.parse_args()

    ds = CharDataset(load_tinyshakespeare("data"))
    names = a.only or list(VARIANTS)
    results_path = os.path.join(a.out, "results.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}
    for name in names:
        mtp_kw, settings = VARIANTS[name]
        out_dir = os.path.join(a.out, name)
        ckpt = os.path.join(out_dir, "ckpt.pt")
        t0 = time.time()
        if not a.skip_train or not os.path.exists(ckpt):
            print(f"\n=== training {name}: {mtp_kw}")
            train(
                ModelConfig(vocab_size=ds.vocab_size, max_seq_len=512),
                MTPConfig(**mtp_kw),
                TrainConfig(
                    steps=a.steps, seq_len=a.seq_len, batch_size=a.batch_size, eval_every=a.steps // 4,
                    log_every=100, device=a.device, out_dir=out_dir,
                ),
                ds,
            )
        model, _, meta = load_checkpoint(ckpt, "cpu")
        print(f"=== decoding {name}")
        rows = bench(model, ds, settings, a.max_new_tokens)
        for b in rows:
            print(f"  K={b['K']}{' rec' if b['recursive'] else ''}: acc={['%.2f' % x for x in b['acceptance_by_depth']]} tokens/round={b['tokens_per_round']:.2f}")
        results[name] = {"mtp": to_dict(model.mtp_cfg), "eval": meta["extra"], "bench": rows, "seconds": time.time() - t0}
        os.makedirs(a.out, exist_ok=True)
        json.dump(results, open(results_path, "w"), indent=1)
    write_markdown({n: results[n] for n in VARIANTS if n in results}, os.path.join(a.out, "results.md"))
    print(f"\nwrote {results_path} and results.md")


if __name__ == "__main__":
    main()
