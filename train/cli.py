"""Convert the Miso depth-decoder checkpoint and score teacher-forced NLL.

Two steps.  ``convert`` reads the published Miso checkpoint, remaps the ~76
decoder tensors onto ``MisoRVQDepthDecoder`` and caches the result so later runs
skip the multi-GB source file::

    python -m train.cli convert
    python -m train.cli eval --checkpoint tmp/miso_depth_decoder.safetensors
    python -m train.cli eval                      # random init

Random init lands near ln(2051) = 7.63 nats.  The checkpoint lands near 4.26
nats / 2.38 kbit/s over codebooks 1..31, matching the 4.29 that
analysis/README.md quotes pooled over rows.  A checkpoint run stuck near 7.6
means the converter mismatched; one below ~2 means the causal mask is not
applied and later codebooks are leaking.

Numbers use the same convention as dataprep.common.nll_summary, so they are
directly comparable to data/expresso/metrics/miso/*/miso_metrics.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import torch
import typer

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_CHECKPOINT = Path("tmp/miso_depth_decoder.safetensors")
DEFAULT_FRAME_RATE = 12.5
NATS_TO_BITS = 1.0 / math.log(2.0)

app = typer.Typer(
    add_completion=False,
    help="Convert the Miso depth decoder and evaluate its teacher-forced NLL.",
)


def default_shard_urls(
    split: str, *, data_root: Path = DEFAULT_DATA_ROOT, model: str = "miso"
) -> str:
    """Brace-expanded URL covering every shard of ``split``."""
    shard_dir = data_root / "expresso" / "wds" / model / split
    shards = sorted(shard_dir.glob(f"{model}_{split}_*.tar"))
    if not shards:
        raise typer.BadParameter(f"no shards under {shard_dir}; run dataprep.export_wds first")
    if len(shards) == 1:
        return str(shards[0])
    lo = shards[0].stem.rsplit("_", 1)[1]
    hi = shards[-1].stem.rsplit("_", 1)[1]
    return str(shard_dir / f"{model}_{split}_{{{lo}..{hi}}}.tar")


def expected_frames(
    split: str, *, data_root: Path = DEFAULT_DATA_ROOT, model: str = "miso"
) -> int | None:
    """Frame count for ``split`` from dataset_info.json, for the progress bar."""
    info = data_root / "expresso" / "wds" / model / "dataset_info.json"
    if not info.exists():
        return None
    try:
        return json.loads(info.read_text()).get(f"total_{split}_frames")
    except (json.JSONDecodeError, OSError):
        return None


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

    Accepts either a converted checkpoint (as written by ``convert``) or a raw
    published one, so the cache is an optimization rather than a prerequisite.
    """
    from safetensors.torch import load_file

    from train.convert import convert_miso_decoder_state_dict, decoder_source_keys
    from train.model import MisoRVQDepthDecoder

    torch.manual_seed(seed)
    model = MisoRVQDepthDecoder()

    if checkpoint is not None:
        state = load_file(str(checkpoint))
        if any(k.startswith("decoder.layers.0.attn.") for k in state):
            # Raw published checkpoint handed in directly; convert on the fly.
            state = convert_miso_decoder_state_dict(
                {k: state[k] for k in decoder_source_keys()}
            )
        # Cached checkpoints are bf16; the model runs in fp32, and load_state_dict
        # would otherwise copy bf16 values in and change eval numerics.
        state = {k: v.float() for k, v in state.items()}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # embed_tokens is the unused vocab_size=1 dummy (we feed inputs_embeds).
        # Anything else missing means a weight was silently skipped.
        if unexpected:
            raise typer.BadParameter(f"unexpected keys in {checkpoint}: {unexpected}")
        if list(missing) != ["decoder.embed_tokens.weight"]:
            raise typer.BadParameter(f"missing keys in {checkpoint}: {missing}")

    return model.to(resolve_device(device) if device is None else device).eval()


def evaluate_nll(
    *,
    shard_urls,
    checkpoint: Path | None = None,
    batch_frames: int = 2048,
    device: str | None = None,
    max_batches: int | None = None,
    seed: int = 0,
    frame_rate: float = DEFAULT_FRAME_RATE,
    total_frames: int | None = None,
    progress: bool = True,
) -> dict:
    """Frame-weighted mean NLL over codebooks 1..K-1 of a single pass."""
    from tqdm import tqdm

    from train.dataset import FramePackingIterableDataset
    from train.model import codebook_nll

    dev = resolve_device(device)
    label = checkpoint.name if checkpoint else "random-init"
    tqdm.write(f"model      : {label}")
    tqdm.write(f"device     : {dev}")

    model = load_model(checkpoint, seed=seed, device=dev)
    params = sum(p.numel() for p in model.parameters())
    tqdm.write(f"params     : {params / 1e6:.1f}M")

    dataset = FramePackingIterableDataset(
        shard_urls,
        batch_frames=batch_frames,
        shuffle_buffer=0,
        resampled=False,
        drop_last=False,  # single pass: keep every val frame
    )

    totals = torch.zeros(model.config.num_residual_levels, dtype=torch.float64)
    frames = 0
    bar = tqdm(
        total=total_frames,
        desc="scoring",
        unit="frame",
        unit_scale=True,
        disable=not progress,
        leave=False,
    )
    with bar:
        for index, batch in enumerate(dataset):
            if max_batches is not None and index >= max_batches:
                break
            hiddens = batch["hiddens"].to(dev)
            targets = batch["targets"].to(dev)
            with torch.no_grad():
                nll = codebook_nll(model(hiddens, targets), targets)  # (n, K-1)
            # .cpu() before .double(): MPS has no float64.
            totals += nll.sum(dim=0).cpu().double()
            frames += nll.shape[0]
            bar.update(nll.shape[0])
            bar.set_postfix(nll=f"{float(totals.sum()) / (frames * len(totals)):.3f}")

    if frames == 0:
        raise typer.BadParameter("no frames evaluated; check shard urls")

    if total_frames and frames != total_frames and max_batches is None:
        tqdm.write(f"warning    : scored {frames} frames, expected {total_frames}")

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


@app.command("convert")
def convert_command(
    output: Path = typer.Option(DEFAULT_CHECKPOINT, help="where to write the converted weights"),
    source: Optional[Path] = typer.Option(
        None, help="published safetensors file (default: bf16 repo in the HF cache)"
    ),
    repo: str = typer.Option("bf16", help="'bf16' or 'fp32' when --source is omitted"),
    fp32: bool = typer.Option(
        False, "--fp32", help="store fp32 instead of bf16 (2x the disk, same eval)"
    ),
    force: bool = typer.Option(False, help="overwrite an existing output file"),
) -> None:
    """Remap the published Miso checkpoint onto MisoRVQDepthDecoder and cache it.

    Stored as bf16 by default: the source weights are themselves bf16, so this
    is lossless relative to them, and eval casts back to fp32 on load.
    """
    from safetensors.torch import save_file

    from train.convert import BF16_REPO, FP32_REPO, load_miso_decoder_state_dict

    if output.exists() and not force:
        raise typer.BadParameter(f"{output} exists; pass --force to overwrite")

    repos = {"bf16": BF16_REPO, "fp32": FP32_REPO}
    if source is None and repo not in repos:
        raise typer.BadParameter(f"repo must be one of {sorted(repos)}")

    dtype = torch.float32 if fp32 else torch.bfloat16
    state = load_miso_decoder_state_dict(source, repo=repos.get(repo, BF16_REPO), dtype=dtype)

    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(output))
    total = sum(t.numel() for t in state.values())
    nbytes = sum(t.numel() * t.element_size() for t in state.values())
    typer.echo(
        f"wrote {output}  ({len(state)} tensors, {total / 1e6:.1f}M params, "
        f"{dtype.__str__().removeprefix('torch.')}, {nbytes / 1e9:.2f} GB)"
    )


@app.command("eval")
def eval_command(
    split: str = typer.Option("val", help="wds split to score"),
    checkpoint: Optional[Path] = typer.Option(
        None, help="converted checkpoint; omit for random init"
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="dataset root"),
    shard_urls: Optional[str] = typer.Option(None, help="override the shard pattern"),
    batch_frames: int = typer.Option(2048, help="frames per batch"),
    device: Optional[str] = typer.Option(None, help="cpu / mps / cuda (default: auto)"),
    max_batches: Optional[int] = typer.Option(None, help="stop early, for smoke tests"),
    seed: int = typer.Option(0, help="seed for random init"),
    json_out: Optional[Path] = typer.Option(None, "--json", help="also write results here"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="suppress the progress bar"),
) -> None:
    """Score teacher-forced NLL over codebooks 1..31 on a WDS split."""
    if checkpoint is not None and not checkpoint.exists():
        raise typer.BadParameter(f"{checkpoint} not found; run 'convert' first")

    urls = shard_urls or default_shard_urls(split, data_root=data_root)
    typer.echo(f"split      : {split}")
    typer.echo(f"shards     : {urls}")

    result = evaluate_nll(
        shard_urls=urls,
        checkpoint=checkpoint,
        batch_frames=batch_frames,
        device=device,
        max_batches=max_batches,
        seed=seed,
        total_frames=None if max_batches else expected_frames(split, data_root=data_root),
        progress=not quiet,
    )

    per_codebook = result["per_codebook_nll_nats"]
    typer.echo(f"frames     : {result['frames']:,}")
    typer.echo(f"codebooks  : 1..{result['num_codebooks']}")
    typer.echo("")
    typer.echo("per-codebook nll (nats)")
    for start in range(0, len(per_codebook), 8):
        chunk = per_codebook[start : start + 8]
        heads = "  ".join(f"cb{start + i + 1:<6d}" for i in range(len(chunk)))
        vals = "  ".join(f"{v:<8.3f}" for v in chunk)
        typer.echo(f"  {heads}")
        typer.echo(f"  {vals}")
    typer.echo("")

    avg = result["avg_nll_per_codebook"]
    typer.echo(f"avg nll    : {avg:.4f} nats  (range {min(per_codebook):.3f}-{max(per_codebook):.3f})")
    typer.echo(f"kbit/s     : {result['kbits_per_second']:.4f}")

    # Orient the number: chance is the random-init floor, and analysis/README.md
    # quotes 4.29 nats for this checkpoint pooled over rows.
    chance = math.log(2051)
    if avg > chance - 0.1:
        typer.echo(f"note       : at chance ({chance:.2f}) — untrained or weights not loaded")
    elif avg < 2.0:
        typer.echo(f"note       : suspiciously low — check codebook-axis causal masking")
    else:
        typer.echo(f"note       : {chance - avg:.2f} nats below chance ({chance:.2f})")

    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(result, indent=2))
        typer.echo(f"wrote {json_out}")


if __name__ == "__main__":
    app()
