"""Minimal single-process trainer for the RVQ depth decoder.

Runs for a fixed wall-clock budget, then scores the val split.  Everything
lands in one run directory::

    tmp/train_<timestamp>/
        config.json         resolved architecture + hyperparameters + git sha
        train.jsonl         one record per logged step
        model.safetensors   final weights
        result.json         before/after val NLL, same keys as `cli.py eval`

The default architecture is *not* the full Miso decoder: at ~8.3 s/step on MPS
the real 674.8M model cannot move in a five-minute budget. ``smoke_config()``
shrinks depth/width and keeps only the first 16 codebooks (~240 ms/step).

It cuts ``num_codebooks`` rather than ``audio_vocab_size`` so codes stay valid
indices into the shared embedding table and NLL stays comparable to
ln(2051) = 7.626.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import DepthDecoderConfig, MisoRVQDepthDecoder, codebook_nll, loss_fn

DEFAULT_FRAME_RATE = 12.5
NATS_TO_BITS = 1.0 / math.log(2.0)


def smoke_config(
    *, num_codebooks: int = 16, hidden_size: int = 384, num_hidden_layers: int = 2
) -> DepthDecoderConfig:
    """Reduced architecture that trains meaningfully in minutes on MPS.

    head_dim stays 64 so the RoPE geometry matches the full model.
    """
    heads = max(1, hidden_size // 64)
    return DepthDecoderConfig(
        num_codebooks=num_codebooks,
        hidden_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=heads,
        num_key_value_heads=max(1, heads // 3),
        head_dim=64,
        intermediate_size=hidden_size * 8 // 3,
    )


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def lr_lambda(step: int, *, warmup_steps: int, total_steps: int) -> float:
    """Linear warmup then cosine decay to 10% of peak.

    ``total_steps`` is a wall-clock estimate, so progress is clamped rather than
    allowed to run past the end of the cosine.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return 1.0
    progress = min(1.0, (step - warmup_steps) / (total_steps - warmup_steps))
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


@torch.no_grad()
def evaluate(model, source, *, batch_frames: int, device, num_codebooks: int) -> dict:
    """Single teacher-forced pass over a split, frame-weighted per codebook.

    Mirrors ``cli.evaluate_nll``.
    """
    from dataset import FramePackingIterableDataset

    was_training = model.training
    model.eval()
    dataset = FramePackingIterableDataset(
        source,
        batch_frames=batch_frames,
        eval_mode=True,
    )

    totals = torch.zeros(num_codebooks - 1, dtype=torch.float64)
    frames = 0
    for batch in dataset:
        hiddens = batch["hiddens"].to(device)
        targets = batch["targets"][:, :num_codebooks].to(device)
        nll = codebook_nll(model(hiddens, targets), targets)
        # .cpu() before .double(): MPS has no float64.
        totals += nll.sum(dim=0).cpu().double()
        frames += nll.shape[0]

    if was_training:
        model.train()
    if frames == 0:
        raise ValueError("no frames evaluated; check shard urls")

    per_codebook = (totals / frames).tolist()
    avg = float(sum(per_codebook) / len(per_codebook))
    count = len(per_codebook)
    return {
        "frames": frames,
        "num_codebooks": count,
        "avg_nll_per_codebook": avg,
        "kbits_per_second": avg * NATS_TO_BITS * DEFAULT_FRAME_RATE * count / 1000.0,
        "per_codebook_nll_nats": per_codebook,
    }


def run_train(
    *,
    source,
    val_source,
    out_dir: Path,
    config: DepthDecoderConfig | None = None,
    minutes: float = 5.0,
    batch_frames: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    warmup_steps: int = 50,
    grad_clip: float = 1.0,
    max_steps: int | None = None,
    log_every: int = 20,
    device: str | None = None,
    seed: int = 0,
    progress: bool = True,
) -> dict:
    """Train for ``minutes`` of wall clock, then score val. Returns the summary."""
    from safetensors.torch import save_file
    from tqdm import tqdm

    from dataset import FramePackingIterableDataset

    cfg = config or smoke_config()
    dev = resolve_device(device)
    budget = minutes * 60.0
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model = MisoRVQDepthDecoder(cfg).to(dev).train()
    params = sum(p.numel() for p in model.parameters())

    tqdm.write(f"device     : {dev}")
    tqdm.write(
        f"model      : {params / 1e6:.1f}M params, {cfg.num_hidden_layers}L "
        f"d={cfg.hidden_size}, codebooks 1..{cfg.num_residual_levels}"
    )
    tqdm.write(f"budget     : {minutes:g} min, batch_frames={batch_frames}, lr={lr:g}")

    # Only feeds the LR schedule; wall clock is the real stop condition.
    est_total = max_steps if max_steps is not None else max(1, int(budget / 0.25))
    opt = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.95)
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: lr_lambda(s, warmup_steps=warmup_steps, total_steps=est_total)
    )

    val_before = evaluate(
        model, val_source, batch_frames=batch_frames, device=dev,
        num_codebooks=cfg.num_codebooks,
    )
    chance = math.log(cfg.audio_vocab_size)
    tqdm.write(
        f"val (init) : {val_before['avg_nll_per_codebook']:.4f} nats "
        f"over {val_before['frames']:,} frames  (chance {chance:.4f})"
    )

    dataset = FramePackingIterableDataset(
        source, batch_frames=batch_frames, shuffle_buffer=1000, resampled=True
    )

    log_path = out_dir / "train.jsonl"
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "architecture": asdict(cfg),
                "params": params,
                "minutes": minutes,
                "batch_frames": batch_frames,
                "lr": lr,
                "weight_decay": weight_decay,
                "warmup_steps": warmup_steps,
                "grad_clip": grad_clip,
                "max_steps": max_steps,
                "seed": seed,
                "device": str(dev),
                "git_sha": _git_sha(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    step = 0
    frames_seen = 0
    ema = None
    last_draw = 0.0
    start = time.monotonic()
    bar = tqdm(
        total=round(budget), desc="training", unit="s",
        disable=not progress, leave=False,
    )

    with bar, log_path.open("w", encoding="utf-8") as log:
        for batch in dataset:
            elapsed = time.monotonic() - start
            if elapsed >= budget or (max_steps is not None and step >= max_steps):
                break

            hiddens = batch["hiddens"].to(dev)
            targets = batch["targets"][:, : cfg.num_codebooks].to(dev)

            loss, _ = loss_fn(model(hiddens, targets), targets)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

            value = loss.item()
            ema = value if ema is None else 0.9 * ema + 0.1 * value
            step += 1
            frames_seen += targets.shape[0]

            elapsed = time.monotonic() - start
            # Refresh on an interval, not every step, to keep redirected logs small.
            if elapsed - last_draw >= 0.5:
                last_draw = elapsed
                bar.n = min(round(elapsed), round(budget))
                bar.set_postfix(
                    step=step, loss=f"{ema:.3f}", lr=f"{sched.get_last_lr()[0]:.2e}"
                )
                bar.refresh()

            if step % log_every == 0 or step == 1:
                log.write(
                    json.dumps(
                        {
                            "step": step,
                            "elapsed_s": round(elapsed, 2),
                            "loss": round(value, 5),
                            "loss_ema": round(ema, 5),
                            "lr": sched.get_last_lr()[0],
                            "frames": frames_seen,
                        }
                    )
                    + "\n"
                )
                log.flush()

    wall = time.monotonic() - start
    save_file(
        {k: v.contiguous() for k, v in model.state_dict().items()},
        str(out_dir / "model.safetensors"),
    )

    val_after = evaluate(
        model, val_source, batch_frames=batch_frames, device=dev,
        num_codebooks=cfg.num_codebooks,
    )

    result = {
        "steps": step,
        "frames_seen": frames_seen,
        "wall_seconds": round(wall, 1),
        "params": params,
        "num_codebooks": val_after["num_codebooks"],
        "chance_nll": chance,
        "final_train_loss_ema": ema,
        "val_before": val_before,
        "val_after": val_after,
        # Promoted for direct comparison with tmp/eval_*.json.
        "avg_nll_per_codebook": val_after["avg_nll_per_codebook"],
        "kbits_per_second": val_after["kbits_per_second"],
        "per_codebook_nll_nats": val_after["per_codebook_nll_nats"],
        "out_dir": str(out_dir),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
