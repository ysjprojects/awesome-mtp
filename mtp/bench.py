"""Decode a trained checkpoint two ways and compare: plain autoregressive vs. self-speculative.

    python -m mtp.bench --ckpt runs/seq2/ckpt.pt --prompt "ROMEO:" "JULIET:" --max-new-tokens 200 --draft-len 1 2
    python -m mtp.bench --ckpt runs/shared3/ckpt.pt --draft-len 1 2 3 4 --drafter eagle

With ``--temperature 0`` the speculative output must equal the greedy output exactly; the script
asserts it. Acceptance statistics are pooled over all prompts. Wall-clock speedups on a laptop
CPU/MPS are dominated by Python overhead for a model this small; the acceptance statistics are
the numbers that transfer to real deployments.
"""

from __future__ import annotations

import argparse
import time

import torch

from .data import CharDataset
from .decode import DRAFTERS, SpecStats, generate, speculative_generate
from .train import load_checkpoint, pick_device, sync


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--prompt", nargs="+", default=["ROMEO:"])
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
    p.add_argument(
        "--drafter",
        default="depth",
        choices=sorted(DRAFTERS),
        help="sequential heads only: 'depth' = one cache per depth (training-consistent), 'eagle' = single cache, module fed its own outputs (vLLM/SGLang pattern)",
    )
    a = p.parse_args(argv)

    device = pick_device(a.device)
    model, itos, _ = load_checkpoint(a.ckpt, device)
    ds_stub = CharDataset.__new__(CharDataset)
    ds_stub.itos, ds_stub.stoi = itos, {c: i for i, c in enumerate(itos)}
    prompts = [ds_stub.encode(s)[None].to(device) for s in a.prompt]
    print(
        f"device={device} kind={model.mtp_cfg.kind} n_future={model.n_future} share_weights={model.mtp_cfg.share_weights} "
        f"prompts={len(prompts)} max_new_tokens={a.max_new_tokens}"
    )

    bases = []
    sync(device)
    t0 = time.perf_counter()
    for idx in prompts:
        gen = torch.Generator().manual_seed(a.seed)
        bases.append(generate(model, idx, a.max_new_tokens, a.temperature, a.top_k, gen))
    sync(device)
    t_base = time.perf_counter() - t0
    n_tokens = a.max_new_tokens * len(prompts)
    print(f"autoregressive : {n_tokens / t_base:7.1f} tok/s  ({t_base:.2f}s)")
    if a.show:
        for base in bases:
            print(ds_stub.decode(base[0]))
            print("-" * 60)
    if model.heads is None:
        print("model has no MTP heads; skipping speculative decoding")
        return
    for K in a.draft_len:
        pooled = SpecStats(K)
        same = True
        outs = []
        sync(device)
        t0 = time.perf_counter()
        for idx, base in zip(prompts, bases):
            gen = torch.Generator().manual_seed(a.seed)
            out, stats = speculative_generate(model, idx, a.max_new_tokens, K, a.temperature, a.top_k, gen, a.recursive, a.drafter)
            pooled = pooled.merge(stats)
            same = same and torch.equal(out, base)
            outs.append(out)
        sync(device)
        t_spec = time.perf_counter() - t0
        tag = f"K={K}" + (" recursive" if a.recursive else "") + (f" {a.drafter}" if a.drafter != "depth" else "")
        print(
            f"speculative {tag}: {n_tokens / t_spec:7.1f} tok/s  ({t_spec:.2f}s, {t_base / t_spec:.2f}x) | "
            f"{pooled.summary()}" + ("" if a.temperature > 0 else f" | identical_to_greedy={same}")
        )
        if a.temperature == 0 and not same:
            raise SystemExit("greedy speculative decoding diverged from autoregressive decoding")
        if a.show:
            for out in outs:
                print(ds_stub.decode(out[0]))
                print("-" * 60)


if __name__ == "__main__":
    main()
