"""Merge several chapter-05 sweeps (different seeds) into mean +- half-range tables.

    python scripts/aggregate_sweeps.py runs/sweep runs/sweep_seed1 > runs/sweep_aggregate.md
"""

from __future__ import annotations

import json
import os
import sys


def fmt(values: list[float], digits: int = 3) -> str:
    if len(values) == 1:
        return f"{values[0]:.{digits}f}"
    mean = sum(values) / len(values)
    half = (max(values) - min(values)) / 2
    return f"{mean:.{digits}f} +- {half:.{digits}f}"


def main(dirs: list[str]) -> None:
    sweeps = [json.load(open(os.path.join(d, "results.json"))) for d in dirs]
    names = [n for n in sweeps[0] if all(n in s for s in sweeps)]
    print(f"Aggregated over {len(sweeps)} seed(s): {', '.join(dirs)}. Cells are mean +- half the range.\n")
    print("| run | val NTP loss | acc_0 | acc_1 | acc_2 | acc_3 |")
    print("|---|---|---|---|---|---|")
    for n in names:
        evs = [s[n]["eval"] for s in sweeps]
        cells = [fmt([e["val_ntp"] for e in evs])]
        for k in range(4):
            key = f"val_acc{k}"
            cells.append(fmt([e[key] for e in evs]) if key in evs[0] else "-")
        print(f"| `{n}` | " + " | ".join(cells) + " |")
    print("\n| run | K | conditional acceptance by depth | tokens per trunk pass |")
    print("|---|---|---|---|")
    for n in names:
        rows = [s[n]["bench"] for s in sweeps]
        for i, b0 in enumerate(rows[0]):
            bs = [r[i] for r in rows]
            k = f"{b0['K']}" + (" (recursive)" if b0["recursive"] else "")
            rates = ", ".join(fmt([b["acceptance_by_depth"][j] for b in bs], 2) for j in range(b0["K"]))
            print(f"| `{n}` | {k} | {rates} | {fmt([b['tokens_per_round'] for b in bs], 2)} |")


if __name__ == "__main__":
    main(sys.argv[1:] or ["runs/sweep", "runs/sweep_seed1"])
