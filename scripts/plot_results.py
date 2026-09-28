"""Figures for chapter 05 from ``runs/sweep/results.json`` (and the reference runs' metrics logs).

    python scripts/plot_results.py            # writes assets/*.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mtp.metrics import expected_accepted_length  # noqa: E402


def acceptance_by_depth(results: dict, out: str) -> None:
    """Conditional acceptance per depth for the longest draft each variant supports."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, r in results.items():
        if not r["bench"]:
            continue
        b = max(r["bench"], key=lambda x: x["K"])
        depths = range(1, b["K"] + 1)
        style = "--" if b["recursive"] else "-"
        ax.plot(depths, b["acceptance_by_depth"], style, marker="o", label=f"{name} (K={b['K']}{', recursive' if b['recursive'] else ''})")
    ax.set_xlabel("draft depth k")
    ax.set_ylabel("P(accept depth k | shallower accepted)")
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Greedy acceptance by depth (char-level TinyShakespeare, 800-step sweep)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def tokens_per_round(results: dict, out: str) -> None:
    """Tokens decided per trunk pass as a function of draft length, per variant."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, r in results.items():
        if not r["bench"]:
            continue
        ks = [b["K"] for b in r["bench"]]
        ax.plot(ks, [b["tokens_per_round"] for b in r["bench"]], marker="o", label=name)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("draft length K")
    ax.set_ylabel("tokens per trunk forward pass")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    ax.set_title("Expected tokens per verify pass")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def lambda_sweep(results: dict, out: str) -> None:
    """Auxiliary-loss weight vs next-token loss and depth-1 accuracy (D = 1 sequential models)."""
    rows = [(r["mtp"]["loss_weight"], r["eval"]) for n, r in results.items() if n.startswith("seq1_l")]
    if not rows:
        return
    rows.sort()
    lams = [l for l, _ in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(lams, [e["val_ntp"] for _, e in rows], marker="o", color="C0", label="val NTP loss")
    if "ntp" in results:
        ax1.axhline(results["ntp"]["eval"]["val_ntp"], color="C0", ls=":", label="NTP baseline")
    ax1.set_xscale("log")
    ax1.set_xlabel("lambda (MTP loss weight)")
    ax1.set_ylabel("val NTP loss", color="C0")
    ax2 = ax1.twinx()
    ax2.plot(lams, [e["val_acc1"] for _, e in rows], marker="s", color="C1", label="depth-1 accuracy")
    ax2.set_ylabel("teacher-forced depth-1 accuracy", color="C1")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    ax1.set_title("Loss-weight sweep, D = 1 sequential module")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def speedup_model(results: dict, out: str, n_layers: int = 6) -> None:
    """Idealised memory-bound speedup vs draft length, for a 6-layer trunk and for a 60-layer one."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for name, r in results.items():
        if not r["bench"] or name.startswith("par3_mlp"):
            continue
        ks, s6, s60 = [], [], []
        for b in r["bench"]:
            e = expected_accepted_length(b["acceptance_by_depth"])
            ks.append(b["K"])
            s6.append((1 + e) / (1 + b["K"] / n_layers))
            s60.append((1 + e) / (1 + b["K"] / 60))
        (line,) = ax.plot(ks, s60, marker="o", label=f"{name} (60-layer trunk)")
        ax.plot(ks, s6, ls=":", marker=".", color=line.get_color(), label=f"{name} (6-layer trunk)")
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("draft length K")
    ax.set_ylabel("idealised speedup (memory-bound)")
    ax.set_xticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Same acceptance, different head/trunk cost ratio")
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="runs/sweep/results.json")
    p.add_argument("--out", default="assets")
    a = p.parse_args()
    results = json.load(open(a.results))
    os.makedirs(a.out, exist_ok=True)
    acceptance_by_depth(results, os.path.join(a.out, "acceptance_by_depth.png"))
    tokens_per_round(results, os.path.join(a.out, "tokens_per_round.png"))
    lambda_sweep(results, os.path.join(a.out, "lambda_sweep.png"))
    speedup_model(results, os.path.join(a.out, "speedup_model.png"))
    print(f"wrote figures to {a.out}/")


if __name__ == "__main__":
    main()
