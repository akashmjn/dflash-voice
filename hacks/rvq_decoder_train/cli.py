"""Convert, train, and score the Miso depth decoder. Unmaintained -- see __init__.

    python cli.py convert
    python cli.py eval --checkpoint tmp/miso_depth_decoder.safetensors
    python cli.py train --preset smoke --minutes 5

Random init lands near ln(2051) = 7.63 nats; the checkpoint near 4.26 nats /
2.38 kbit/s over codebooks 1..31. Stuck near 7.6 means the converter
mismatched; below ~2 means the causal mask is not applied and later codebooks
are leaking. Same convention as ``dataprep.common.nll_summary``.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Optional

import torch
import typer

DEFAULT_DATASET_ROOT = Path("data") / "sharded_wds"
DEFAULT_SHARD_SLUG = "expresso-full-logits-bucketed-0804"
DEFAULT_CHECKPOINT = Path("tmp/miso_depth_decoder.safetensors")
DEFAULT_FRAME_RATE = 12.5
NATS_TO_BITS = 1.0 / math.log(2.0)

app = typer.Typer(
    add_completion=False,
    help="Convert, train, and evaluate the Miso depth decoder.",
)


def resolve_device(device: str | None = None) -> torch.device:
    if device:
        return torch.device(device)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint: Path | None, *, seed: int = 0, device=None):
    """Random init when ``checkpoint`` is None, else load converted weights.

    Accepts a converted checkpoint or a raw published one, so the cache is an
    optimization rather than a prerequisite.
    """
    from safetensors.torch import load_file

    from convert import convert_miso_decoder_state_dict, decoder_source_keys
    from model import MisoRVQDepthDecoder

    torch.manual_seed(seed)
    model = MisoRVQDepthDecoder()

    if checkpoint is not None:
        state = load_file(str(checkpoint))
        if any(k.startswith("decoder.layers.0.attn.") for k in state):
            state = convert_miso_decoder_state_dict(
                {k: state[k] for k in decoder_source_keys()}
            )
        # Cached checkpoints are bf16; the model runs in fp32, and load_state_dict
        # would otherwise copy bf16 values in and change eval numerics.
        state = {k: v.float() for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # embed_tokens is the unused vocab_size=1 dummy (we feed inputs_embeds).
        if unexpected:
            raise typer.BadParameter(f"unexpected keys in {checkpoint}: {unexpected}")
        if list(missing) != ["decoder.embed_tokens.weight"]:
            raise typer.BadParameter(f"missing keys in {checkpoint}: {missing}")

    return model.to(resolve_device(device) if device is None else device).eval()


def evaluate_nll(
    *,
    source,
    checkpoint: Path | None = None,
    batch_frames: int = 2048,
    device: str | None = None,
    frame_rate: float = DEFAULT_FRAME_RATE,
    progress: bool = True,
) -> dict:
    """Frame-weighted mean NLL over codebooks 1..K-1 of a single pass."""
    from tqdm import tqdm

    from dataset import FramePackingIterableDataset
    from model import codebook_nll

    dev = resolve_device(device)

    # Before the model: loading 674.8M params takes seconds, so fail a bad
    # shard path first.
    try:
        dataset = FramePackingIterableDataset(
            source,
            batch_frames=batch_frames,
            eval_mode=True,  # single pass, shard order, keep every val frame
        )
    except FileNotFoundError as exc:
        raise typer.BadParameter(
            f"{exc}; run 'python -m dataprep.cli prepare' or pass --dataset-slug"
        ) from exc

    model = load_model(checkpoint, seed=0, device=dev)
    tqdm.write(
        f"model      : {checkpoint.name if checkpoint else 'random-init'} on {dev}, "
        f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params"
    )

    totals = torch.zeros(model.config.num_residual_levels, dtype=torch.float64)
    frames = 0
    with tqdm(desc="scoring", unit="frame", unit_scale=True,
              disable=not progress, leave=False) as bar:
        for batch in dataset:
            hiddens = batch["hiddens"].to(dev)
            targets = batch["targets"].to(dev)
            with torch.no_grad():
                nll = codebook_nll(model(hiddens, targets), targets)  # (n, K-1)
            # .cpu() before .double(): MPS has no float64.
            totals += nll.sum(dim=0).cpu().double()
            frames += nll.shape[0]
            bar.update(nll.shape[0])

    if frames == 0:
        raise typer.BadParameter("no frames evaluated; check shard urls")

    per_codebook = (totals / frames).tolist()
    avg = float(sum(per_codebook) / len(per_codebook))
    count = len(per_codebook)
    return {
        "checkpoint": str(checkpoint) if checkpoint else "random-init",
        "frames": frames,
        "num_codebooks": count,
        "avg_nll_per_codebook": avg,
        "kbits_per_second": avg * NATS_TO_BITS * frame_rate * count / 1000.0,
        "per_codebook_nll_nats": per_codebook,
    }


def _check_shards(*dirs: Path) -> None:
    for d in dirs:
        if not d.is_dir() or not any(d.glob("*.tar")):
            raise typer.BadParameter(
                f"no shards under {d}; run 'python -m dataprep.cli prepare' "
                f"or pass --dataset-slug"
            )


@app.command("convert")
def convert_command(
    output: Path = typer.Option(DEFAULT_CHECKPOINT, help="where to write the converted weights"),
    source: Optional[Path] = typer.Option(
        None, help="published safetensors file (default: bf16 repo in the HF cache)"
    ),
    force: bool = typer.Option(False, help="overwrite an existing output file"),
) -> None:
    """Remap the published Miso checkpoint onto MisoRVQDepthDecoder and cache it.

    Stored as bf16: the source weights are bf16 too, so this is lossless
    relative to them, and eval casts back to fp32 on load.
    """
    from safetensors.torch import save_file

    from convert import BF16_REPO, load_miso_decoder_state_dict

    if output.exists() and not force:
        raise typer.BadParameter(f"{output} exists; pass --force to overwrite")

    state = load_miso_decoder_state_dict(source, repo=BF16_REPO, dtype=torch.bfloat16)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output))
    nbytes = sum(t.numel() * t.element_size() for t in state.values())
    typer.echo(f"wrote {output}  ({len(state)} tensors, bfloat16, {nbytes / 1e9:.2f} GB)")


@app.command("eval")
def eval_command(
    split: str = typer.Option("val", help="wds split to score"),
    checkpoint: Optional[Path] = typer.Option(
        None, help="converted checkpoint; omit for random init"
    ),
    data_root: Path = typer.Option(DEFAULT_DATASET_ROOT, help="dataset root"),
    slug: str = typer.Option(
        DEFAULT_SHARD_SLUG, "--dataset-slug", help="shard set under the dataset root"
    ),
    batch_frames: int = typer.Option(2048, help="frames per batch"),
    device: Optional[str] = typer.Option(None, help="cpu / mps / cuda (default: auto)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress the progress bar"),
) -> None:
    """Score teacher-forced NLL over codebooks 1..31 on a WDS split."""
    if checkpoint is not None and not checkpoint.exists():
        raise typer.BadParameter(f"{checkpoint} not found; run 'convert' first")

    source = data_root / slug / split
    _check_shards(source)

    result = evaluate_nll(
        source=source,
        checkpoint=checkpoint,
        batch_frames=batch_frames,
        device=device,
        progress=not quiet,
    )

    per_codebook = result["per_codebook_nll_nats"]
    avg = result["avg_nll_per_codebook"]
    chance = math.log(2051)
    typer.echo(f"shards     : {source}")
    typer.echo(f"frames     : {result['frames']:,}  over codebooks 1..{result['num_codebooks']}")
    typer.echo(f"per-cb nll : {' '.join(f'{v:.2f}' for v in per_codebook)}")
    typer.echo(f"avg nll    : {avg:.4f} nats  (chance {chance:.2f})")
    typer.echo(f"kbit/s     : {result['kbits_per_second']:.4f}")


@app.command("train")
def train_command(
    minutes: float = typer.Option(5.0, help="wall-clock training budget"),
    preset: str = typer.Option("smoke", help="'smoke' (reduced) or 'full' (674.8M)"),
    batch_frames: int = typer.Option(1024, help="frames per step"),
    lr: float = typer.Option(1e-3, help="peak learning rate"),
    data_root: Path = typer.Option(DEFAULT_DATASET_ROOT, help="dataset root"),
    slug: str = typer.Option(
        DEFAULT_SHARD_SLUG, "--dataset-slug", help="shard set under the dataset root"
    ),
    device: Optional[str] = typer.Option(None, help="cpu / mps / cuda (default: auto)"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress the progress bar"),
) -> None:
    """Train the depth decoder for a fixed time, then score the val split."""
    from model import DepthDecoderConfig
    from trainer import run_train, smoke_config

    if preset not in ("smoke", "full"):
        raise typer.BadParameter("preset must be 'smoke' or 'full'")
    if preset == "full":
        # Measured on this repo's MPS box.
        typer.echo(
            "warning    : the full 674.8M model runs ~8.3 s/step at batch_frames=512\n"
            "             and allocated ~32 GB (enough to OOM a laptop). A 5-minute\n"
            "             budget buys ~36 steps, so the loss will not move."
        )
    cfg = DepthDecoderConfig() if preset == "full" else smoke_config()

    # Both splits up front: val is not read until training finishes, and a typo
    # there should not cost a full run.
    source, val_source = data_root / slug / "train", data_root / slug / "val"
    _check_shards(source, val_source)

    run_dir = Path("tmp") / f"train_{time.strftime('%Y%m%d_%H%M%S')}"
    result = run_train(
        source=source,
        val_source=val_source,
        out_dir=run_dir,
        config=cfg,
        minutes=minutes,
        batch_frames=batch_frames,
        lr=lr,
        device=device,
        progress=not quiet,
    )

    before = result["val_before"]["avg_nll_per_codebook"]
    after = result["val_after"]["avg_nll_per_codebook"]
    typer.echo("")
    typer.echo(
        f"steps      : {result['steps']:,} in {result['wall_seconds']:.0f}s "
        f"({result['frames_seen']:,} frames)"
    )
    typer.echo(
        f"val nll    : {before:.4f} -> {after:.4f} nats over {result['num_codebooks']} "
        f"codebooks  (chance {result['chance_nll']:.4f})"
    )
    typer.echo(f"kbit/s     : {result['kbits_per_second']:.4f}")
    typer.echo(f"wrote {run_dir}/")


if __name__ == "__main__":
    app()
