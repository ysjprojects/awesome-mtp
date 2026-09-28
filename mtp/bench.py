"""Decode a trained checkpoint two ways and compare: plain autoregressive vs. self-speculative.

    python -m mtp.bench --ckpt runs/seq2/ckpt.pt --prompt "ROMEO:" --max-new-tokens 200 --draft-len 1 2 3

With ``--temperature 0`` the speculative output must equal the greedy output exactly; the script
asserts it. Wall-clock speedups on a laptop CPU/MPS are dominated by Python overhead for a
model this small; the acceptance statistics are the numbers that transfer to real deployments.
"""

from __future__ import annotations

import argparse
import time

import torch

from .data import CharDataset
from .decode import generate, speculative_generate
from .train import load_checkpoint, pick_device, sync


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", default="ROMEO:")
    p.add_argument("--max-new-tokens", type=int, default=200)
    p.add_argument("--draft-len", type=int, nargs="+", default=[1])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu", help="cpu is fastest for these tiny models on Apple silicon; pass cuda if you have it")
    p.add_argument("--show", action="store_true", help="print the generated text")
    p.add_argument(
        "--recursive",
        action="store_true",
        help="let a sequential model draft beyond its trained depth by reusing its deepest module (DeepSeek-V3 practice)",
    )
    a = p.parse_args(argv)

    device = pick_device(a.device)
    model, itos, _ = load_checkpoint(a.ckpt, device)
    ds_stub = CharDataset.__new__(CharDataset)
    ds_stub.itos, ds_stub.stoi = itos, {c: i for i, c in enumerate(itos)}
    idx = ds_stub.encode(a.prompt)[None].to(device)
    print(f"device={device} kind={model.mtp_cfg.kind} n_future={model.n_future} share_weights={model.mtp_cfg.share_weights}")

    gen = torch.Generator().manual_seed(a.seed)
    sync(device)
    t0 = time.perf_counter()
    base = generate(model, idx, a.max_new_tokens, a.temperature, a.top_k, gen)
    sync(device)
    t_base = time.perf_counter() - t0
    print(f"autoregressive : {a.max_new_tokens / t_base:7.1f} tok/s  ({t_base:.2f}s)")
    if a.show:
        print(ds_stub.decode(base[0]))
        print("-" * 60)
    if model.heads is None:
        print("model has no MTP heads; skipping speculative decoding")
        return
    for K in a.draft_len:
        gen = torch.Generator().manual_seed(a.seed)
        sync(device)
        t0 = time.perf_counter()
        out, stats = speculative_generate(model, idx, a.max_new_tokens, K, a.temperature, a.top_k, gen, a.recursive)
        sync(device)
        t_spec = time.perf_counter() - t0
        same = torch.equal(out, base)
        print(
            f"speculative K={K}: {a.max_new_tokens / t_spec:7.1f} tok/s  ({t_spec:.2f}s, {t_base / t_spec:.2f}x) | "
            f"{stats.summary()}" + ("" if a.temperature > 0 else f" | identical_to_greedy={same}")
        )
        if a.temperature == 0 and not same:
            raise SystemExit("greedy speculative decoding diverged from autoregressive decoding")
        if a.show:
            print(ds_stub.decode(out[0]))
            print("-" * 60)


if __name__ == "__main__":
    main()
