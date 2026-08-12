"""Turn dataset rows into training-ready WebDataset shards.

Two verbs, in the order you reach for them::

    python -m dataprep.cli inspect --model miso --rows 3
    python -m dataprep.cli prepare --model miso

``inspect`` is the debugging path. It downloads a handful of rows to
``data/DATASET/raw/`` and writes every intermediate to its own directory, so you
can open a row's ``sequences.pt`` and see what the tokenizer did. ``--stage``
stops after tokenize or featurize. Only sane for a few rows -- the per-row PT
files cost more disk than the shards they would become.

``prepare`` is the whole-dataset path::

    HF dataset -> raw example stream -> shard_prepare -> sample stream -> tars

Rows stream from the hub and go straight to shard writers; no wav, no
``sequences.pt``, nothing per-row on disk. ``shard_prepare`` drives tokenize and
featurize from inside its write loop, so the forward pass runs only for rows that
get written. Memory stays flat regardless of dataset size: rows stream one at a
time and shards roll over as they fill. Samples are written in arrival order, so
a sample's place in the set depends only on ``(row, seq_id)``; batches get mixed
by the training dataloader instead.

Train/val is decided by hashing each row id, so every sequence from a row lands
on the same side and one speaker's turns cannot leak across the split.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from tqdm import tqdm

from dataprep.expresso import (
    DATASET_NAME,
    DEFAULT_DATASET,
    DEFAULT_SPLIT,
    download_expresso,
)
from dataprep.pipeline import (
    DEFAULT_DATA_ROOT,
    DEFAULT_LOG_ROOT,
    featurize_row,
    load_tokenizer,
    prepare_row,
)

MODELS = ("miso", "qwen3", "fish")
STAGES = ("tokenize", "featurize", "all")

app = typer.Typer(
    add_completion=False,
    help="Prepare dataset rows into WebDataset shards for training.",
)


def _check_model(model: str) -> None:
    if model not in MODELS:
        raise typer.BadParameter(f"model must be one of {' / '.join(MODELS)}")


@app.command("inspect")
def inspect_command(
    model: str = typer.Option(..., help=f"tokenizer backend: {' / '.join(MODELS)}"),
    rows: int = typer.Option(3, help="number of rows to prepare, from row 0"),
    stage: str = typer.Option("all", help=f"pipeline stage: {' / '.join(STAGES)}"),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="dataset root"),
    dataset: str = typer.Option(DATASET_NAME, help="on-disk dataset slug"),
    log_root: Path = typer.Option(DEFAULT_LOG_ROOT, help="directory for failure logs"),
    model_id: Optional[str] = typer.Option(None, help="override the backend model id"),
    device: Optional[str] = typer.Option(None, help="cpu / mps / cuda (default: auto)"),
    dump_kv: bool = typer.Option(False, help="save the slow-AR layer KV cache"),
    pack_segments: bool = typer.Option(
        False, help="pack consecutive segments up to the model limit"
    ),
) -> None:
    """Prepare a few rows into per-row directories you can open and inspect.

    Writes raw/, tokenized/, and featurized/ artifacts under --data-root. No
    shards: 'prepare' is the path that writes those.
    """
    _check_model(model)
    if stage not in STAGES:
        raise typer.BadParameter(f"stage must be one of {' / '.join(STAGES)}")
    if rows < 1:
        raise typer.BadParameter("--rows requires a positive row count")

    row_ids = list(range(rows))
    raw_root = data_root / dataset / "raw"

    typer.echo(f"model      : {model}")
    typer.echo(f"stage      : {stage}")
    typer.echo(f"rows       : {rows}")
    typer.echo(f"data root  : {data_root}")

    missing = [
        row
        for row in row_ids
        if not (raw_root / str(row) / "transcript_segments.json").exists()
    ]
    if missing:
        download_expresso(missing, root=raw_root)

    tokenizer = load_tokenizer(model, model_id=model_id, device=device)

    paths: list[Path] = []
    if stage in ("tokenize", "all"):
        for row in tqdm(row_ids, desc=f"Tokenizing {model}", unit="row"):
            paths.append(
                prepare_row(
                    raw_root / str(row),
                    model=model,
                    tokenizer=tokenizer,
                    output_root=data_root,
                    dataset=dataset,
                    pack_segments=pack_segments,
                    log_root=log_root,
                )
            )
    if stage in ("featurize", "all"):
        for row in tqdm(row_ids, desc=f"Featurizing {model}", unit="row"):
            paths.append(
                featurize_row(
                    row,
                    model=model,
                    featurizer=tokenizer.featurizer,
                    data_root=data_root,
                    dataset=dataset,
                    dump_kv=dump_kv,
                    log_root=log_root,
                )
            )
    for path in paths:
        typer.echo(str(path))


@app.command("prepare")
def prepare_command(
    model: str = typer.Option(..., help=f"tokenizer backend: {' / '.join(MODELS)}"),
    rows: Optional[int] = typer.Option(
        None, help="cap at the first N streamed rows, for smoke tests"
    ),
    slug: Optional[str] = typer.Option(
        None,
        help="name of the shard set under data/sharded_wds/ "
        "(default: DATASET-rowsN, or DATASET-full)",
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="dataset root"),
    dataset: str = typer.Option(DATASET_NAME, help="source dataset slug"),
    split_ratio: float = typer.Option(0.95, help="fraction of rows routed to train"),
    samples_per_shard: int = typer.Option(250, help="sequences per tar shard"),
    shuffle_buffer: int = typer.Option(
        0,
        help="write-time sequence shuffle buffer (0 = deterministic order: recommended,"
        " makes a run resumable)",
    ),
    seed: int = typer.Option(42, help="seed for shuffling and the train/val split"),
    include_logits: bool = typer.Option(
        False,
        "--include-logits",
        help="also store the teacher's per-head logits (~16x the shard size)",
    ),
    device: Optional[str] = typer.Option(None, help="cpu / mps / cuda (default: auto)"),
    log_root: Path = typer.Option(DEFAULT_LOG_ROOT, help="directory for failure logs"),
    force: bool = typer.Option(False, help="overwrite an existing shard set"),
    bucket_frames: int = typer.Option(
        32,
        help="pad sequences to a multiple of N frames so the featurizer sees a "
        "bounded set of shapes (0 disables). avoids memory growth/fragmentation "
        "of ~+50MiB per new length upto 60GB+ on Pytorch MPS backend",
    ),
) -> None:
    """Stream the whole dataset straight into WebDataset shards.

    Nothing per-row touches disk. Failed rows are logged to failures.jsonl and
    skipped rather than aborting the run.
    """
    _check_model(model)
    if rows is not None and rows < 1:
        raise typer.BadParameter("--rows requires a positive row count")

    from dataprep.expresso import stream_expresso
    from dataprep.shards import shard_prepare, shard_root

    resolved_slug = slug or (f"{dataset}-rows{rows}" if rows else f"{dataset}-full")
    wds_root = shard_root(resolved_slug, data_root=data_root)
    # Shard sets are expensive to rebuild and easy to clobber by rerunning with
    # the same name, so an existing one is an error until asked otherwise.
    if wds_root.exists() and any(wds_root.iterdir()) and not force:
        raise typer.BadParameter(
            f"{wds_root} already exists; pass --force to overwrite or pick another --slug"
        )

    typer.echo(f"model      : {model}")
    typer.echo(f"source     : {DEFAULT_DATASET} [{DEFAULT_SPLIT}]")
    typer.echo(f"rows       : {rows if rows is not None else 'all (streaming)'}")
    typer.echo(f"slug       : {resolved_slug}")
    typer.echo(f"output     : {wds_root}")
    typer.echo(f"logits     : {'yes' if include_logits else 'no'}")
    typer.echo(
        f"shuffle buf: {f'{shuffle_buffer} sequences' if shuffle_buffer else 'off (deterministic order)'}"
    )
    typer.echo(f"buckets    : {f'multiples of {bucket_frames}' if bucket_frames else 'off'}")

    tokenizer = load_tokenizer(model, device=device, bucket_frames=bucket_frames)

    shard_prepare(
        stream_expresso(limit=rows),
        model=model,
        tokenizer=tokenizer,
        wds_root=wds_root,
        split_ratio=split_ratio,
        samples_per_shard=samples_per_shard,
        include_logits=include_logits,
        shuffle_seed=seed,
        shuffle_buffer=shuffle_buffer,
        log_root=log_root,
    )


if __name__ == "__main__":
    app()
