"""Training CLI.

Examples::

    # next-token baseline
    python -m mtp.train --kind none --out runs/ntp

    # Gloeckle-style parallel heads, 3 extra depths
    python -m mtp.train --kind parallel --n-future 3 --out runs/parallel3

    # DeepSeek-V3-style sequential modules, 2 depths, lambda = 0.3
    python -m mtp.train --kind sequential --n-future 2 --loss-weight 0.3 --out runs/seq2

    # one shared module trained at 3 depths (Nemotron 3 Super / GLM-5 recipe)
    python -m mtp.train --kind sequential --n-future 3 --share-weights --out runs/shared3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from .config import ModelConfig, MTPConfig, TrainConfig, to_dict
from .data import CharDataset, load_tinyshakespeare
from .loss import compute_losses, train_step
from .metrics import depth_accuracies
from .model import MTPModel


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def lr_at(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (cfg.lr - cfg.min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model: MTPModel, cfg: TrainConfig) -> torch.optim.Optimizer:
    seen: set[int] = set()
    decay, no_decay = [], []
    for p in model.parameters():
        if id(p) in seen or not p.requires_grad:
            continue
        seen.add(id(p))
        (decay if p.dim() >= 2 else no_decay).append(p)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def save_checkpoint(path: str, model: MTPModel, itos: list[str], step: int, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(
        {
            "model_cfg": to_dict(model.cfg),
            "mtp_cfg": to_dict(model.mtp_cfg),
            "state_dict": model.state_dict(),
            "itos": itos,
            "step": step,
            "extra": extra or {},
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device | str = "cpu") -> tuple[MTPModel, list[str], dict]:
    ckpt = torch.load(path, map_location="cpu")
    model = MTPModel(ModelConfig(**ckpt["model_cfg"]), MTPConfig(**ckpt["mtp_cfg"]))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    return model, ckpt["itos"], ckpt


@torch.no_grad()
def evaluate(model: MTPModel, ds: CharDataset, cfg: TrainConfig, device: torch.device, generator: torch.Generator) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    accs = None
    for _ in range(cfg.eval_batches):
        idx, targets = ds.get_batch("val", cfg.batch_size, cfg.seq_len, device, generator)
        for k, v in compute_losses(model, idx, targets).as_floats().items():
            sums[f"val_{k}"] = sums.get(f"val_{k}", 0.0) + v
        a = depth_accuracies(model, idx, targets)
        accs = a if accs is None else [x + y for x, y in zip(accs, a)]
    model.train()
    out = {k: v / cfg.eval_batches for k, v in sums.items()}
    for k, a in enumerate(accs or []):
        out[f"val_acc{k}"] = a / cfg.eval_batches
    return out


def train(model_cfg: ModelConfig, mtp_cfg: MTPConfig, cfg: TrainConfig, ds: CharDataset) -> MTPModel:
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = MTPModel(model_cfg, mtp_cfg).to(device)
    opt = make_optimizer(model, cfg)
    counts = model.num_params()
    print(f"device={device} params: trunk={counts['trunk']:,} heads={counts['heads']:,} total={counts['total']:,}")
    print(f"mtp: {to_dict(mtp_cfg)}")

    os.makedirs(cfg.out_dir, exist_ok=True)
    log_path = os.path.join(cfg.out_dir, "metrics.jsonl")
    log = open(log_path, "a")
    t0 = time.time()
    model.train()
    for step in range(cfg.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, cfg)
        idx, targets = ds.get_batch("train", cfg.batch_size, cfg.seq_len, device, gen)
        losses = train_step(model, idx, targets, memory_efficient=cfg.memory_efficient)
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        record = {"step": step, "lr": lr_at(step, cfg), **losses.as_floats()}
        if step % cfg.log_every == 0 or step + 1 == cfg.steps:
            sync(device)
            elapsed = time.time() - t0
            parts = " ".join(f"{k}={v:.3f}" for k, v in losses.as_floats().items())
            print(f"step {step:5d} | {parts} | {elapsed:6.1f}s")
        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            ev = evaluate(model, ds, cfg, device, gen)
            record.update(ev)
            parts = " ".join(f"{k}={v:.3f}" for k, v in ev.items())
            print(f"  eval | {parts}")
            save_checkpoint(os.path.join(cfg.out_dir, "ckpt.pt"), model, ds.itos, step + 1, ev)
        log.write(json.dumps(record) + "\n")
        log.flush()
    log.close()
    return model


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a small MTP model on TinyShakespeare")
    p.add_argument("--kind", default="none", choices=["none", "parallel", "sequential"])
    p.add_argument("--n-future", type=int, default=0)
    p.add_argument("--loss-weight", type=float, default=0.3)
    p.add_argument("--head-arch", default="block", choices=["block", "mlp"])
    p.add_argument("--head-layers", type=int, default=1)
    p.add_argument("--share-weights", action="store_true")
    p.add_argument("--detach-trunk", action="store_true")
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--n-layers", type=int, default=6)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--max-seq-len", type=int, default=512)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--no-memory-efficient", action="store_true", help="use the naive all-heads-at-once backward")
    p.add_argument("--data-root", default="data")
    p.add_argument("--out", default="runs/default")
    return p


def main(argv: list[str] | None = None) -> None:
    a = build_parser().parse_args(argv)
    ds = CharDataset(load_tinyshakespeare(a.data_root))
    model_cfg = ModelConfig(
        vocab_size=ds.vocab_size, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads, max_seq_len=a.max_seq_len
    )
    mtp_cfg = MTPConfig(
        kind=a.kind,
        n_future=a.n_future,
        loss_weight=a.loss_weight,
        head_arch=a.head_arch,
        head_layers=a.head_layers,
        share_weights=a.share_weights,
        detach_trunk=a.detach_trunk,
    )
    cfg = TrainConfig(
        steps=a.steps,
        batch_size=a.batch_size,
        seq_len=a.seq_len,
        lr=a.lr,
        warmup_steps=a.warmup_steps,
        eval_every=a.eval_every,
        log_every=a.log_every,
        seed=a.seed,
        memory_efficient=not a.no_memory_efficient,
        device=a.device,
        out_dir=a.out,
    )
    train(model_cfg, mtp_cfg, cfg, ds)


if __name__ == "__main__":
    main()
