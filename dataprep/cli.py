"""Turn dataset utterances into training-ready WebDataset shards.

Two verbs, in the order you reach for them::

    python -m dataprep.cli inspect --model chatterbox --rows 3
    python -m dataprep.cli prepare --model chatterbox

Both consume a stream of :class:`~dataprep.common.Segment`, one per utterance,
from a loader (``--dataset emilia`` or ``expresso``). The only dataset-aware
code here is :func:`_load_segments`.

``inspect`` is the debugging path: every intermediate lands under
``--data-root``, so you can open a segment's ``sequences.pt`` and see what the
tokenizer did. ``--stage`` stops after tokenize. Only sane for a few segments.

``prepare`` is the whole-dataset path::

    HF dataset -> Segment stream -> shard_prepare -> sample stream -> tars

Nothing per-segment touches disk, and memory stays flat regardless of dataset
size. Samples are written in arrival order; batches get mixed by the training
dataloader instead.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator, Optional

import typer

from dataprep.common import Segment
from dataprep.pipeline import (
    DEFAULT_DATA_ROOT,
    DEFAULT_LOG_ROOT,
    inspect_segments,
    load_tokenizer,
)

MODELS = ("miso", "chatterbox")
DATASETS = ("emilia", "expresso")
#: Fetched to disk rather than streamed, so these are not `--dataset` loaders.
FETCH_DATASETS = ("seedtts",)
STAGES = ("tokenize", "featurize", "all")

MODEL_HELP = "tokenizer backend: miso, chatterbox"
DATASET_HELP = f"source dataset loader: {' / '.join(DATASETS)}"

app = typer.Typer(
    add_completion=False,
    help="Prepare dataset utterances into WebDataset shards for training.",
)


def _load_segments(
    dataset: str, *, limit: int | None, data_files: str | None
) -> Iterator[Segment]:
    """Open a loader's segment stream.

    Imports stay inside the branches so an Expresso run never pulls in
    Emilia's gated-auth path, and vice versa.
    """
    if dataset == "emilia":
        from dataprep.datasources.emilia import DEFAULT_DATA_FILES, stream_emilia

        return stream_emilia(
            data_files=data_files or DEFAULT_DATA_FILES, limit=limit
        )
    if dataset == "expresso":
        from dataprep.datasources.expresso import stream_expresso

        if data_files:
            raise typer.BadParameter("--data-files applies to --dataset emilia only")
        return stream_expresso(limit=limit)
    raise typer.BadParameter(f"dataset must be one of {' / '.join(DATASETS)}")


def _tokenizer_device(tokenizer: Any) -> str:
    """Device the backend actually placed its models on.

    Read off the tokenizer rather than the ``--device`` option, which is
    normally ``None`` and resolved per backend.
    """
    for attr in ("featurizer", "audio_codec", "voice_encoder"):
        device = getattr(getattr(tokenizer, attr, None), "device", None)
        if device is not None:
            return str(device)
    return "n/a"


def _check_model(model: str) -> None:
    if model not in MODELS:
        raise typer.BadParameter(f"model must be one of {' / '.join(MODELS)}")


@app.command("fetch")
def fetch_command(
    dataset: str = typer.Option(
        "seedtts", help=f"eval set to download: {' / '.join(FETCH_DATASETS)}"
    ),
    rows: Optional[int] = typer.Option(
        None,
        help="only fetch the clips the first N prompts cite (default: all). "
        "Rows reuse clips, so this is far fewer files than N",
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="dataset root"),
    ground_truth: bool = typer.Option(
        False,
        "--ground-truth",
        help="fetch the human recordings for --rows instead of the reference "
        "clips as an oracle for scoring",
    ),
) -> None:
    """Download an eval set for the decode bench, not for sharding.

    Writes prompts/ and a flat ref_audio/ under DATA_ROOT/DATASET, which is
    what `mlx_decode/bench.py --ref-audio-dir` expects.
    """
    if dataset not in FETCH_DATASETS:
        raise typer.BadParameter(
            f"fetch supports {' / '.join(FETCH_DATASETS)}; "
            f"{' / '.join(DATASETS)} are streamed by prepare instead"
        )
    if rows is not None and rows < 1:
        raise typer.BadParameter("--rows requires a positive count")

    dest = data_root / dataset
    typer.echo(f"dataset    : {dataset}")
    typer.echo(f"destination: {dest}")
    typer.echo(f"prompts    : {rows if rows is not None else 'all'}")

    if ground_truth:
        if rows is None:
            raise typer.BadParameter("--ground-truth needs --rows to size the split")
        from dataprep.datasources.seedtts import fetch_ground_truth

        gt = fetch_ground_truth(dest, rows=rows)
        typer.echo(f"ground truth: {gt['ground_truth']}")
        typer.echo(f"wavs        : {gt['wavs']} of {gt['requested']} requested")
        typer.echo(f"extracted   : {gt['extracted']} new")
        if gt["missing"]:
            typer.echo(f"WARNING: {len(gt['missing'])} missing, e.g. {gt['missing'][:3]}")
        return

    from dataprep.datasources.seedtts import fetch_seedtts

    result = fetch_seedtts(dest, limit=rows)

    typer.echo(f"prompts    : {result['prompts']}")
    typer.echo(f"ref audio  : {result['ref_audio']} ({result['wavs']} wavs)")
    typer.echo(f"extracted  : {result['extracted']} new")


@app.command("inspect")
def inspect_command(
    model: str = typer.Option(..., help=MODEL_HELP),
    rows: int = typer.Option(3, help="number of utterances to prepare"),
    stage: str = typer.Option("all", help=f"pipeline stage: {' / '.join(STAGES)}"),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="dataset root"),
    dataset: str = typer.Option("emilia", help=DATASET_HELP),
    data_files: Optional[str] = typer.Option(
        None, help="emilia only: tar path glob, e.g. 'Emilia/EN/EN-B00000*.tar'"
    ),
    log_root: Path = typer.Option(DEFAULT_LOG_ROOT, help="directory for failure logs"),
    model_id: Optional[str] = typer.Option(None, help="override the backend model id"),
    device: Optional[str] = typer.Option(None, help="cpu / mps / cuda (default: auto)"),
    dump_kv: bool = typer.Option(False, help="save the slow-AR layer KV cache"),
) -> None:
    """Prepare a few utterances into per-segment directories you can open.

    Writes tokenized/ and featurized/ artifacts under --data-root. No shards:
    'prepare' is the path that writes those.
    """
    _check_model(model)
    if stage not in STAGES:
        raise typer.BadParameter(f"stage must be one of {' / '.join(STAGES)}")
    if rows < 1:
        raise typer.BadParameter("--rows requires a positive count")

    segments = _load_segments(dataset, limit=rows, data_files=data_files)
    tokenizer = load_tokenizer(model, model_id=model_id, device=device)

    typer.echo(f"model      : {model}")
    typer.echo(f"dataset    : {dataset}")
    typer.echo(f"stage      : {stage}")
    typer.echo(f"utterances : {rows}")
    typer.echo(f"data root  : {data_root}")
    typer.echo(f"device     : {_tokenizer_device(tokenizer)}")

    paths = inspect_segments(
        segments,
        model=model,
        tokenizer=tokenizer,
        output_root=data_root,
        dataset=dataset,
        stage=stage,
        dump_kv=dump_kv,
        log_root=log_root,
    )
    for path in paths:
        typer.echo(str(path))


@app.command("prepare")
def prepare_command(
    model: str = typer.Option(..., help=MODEL_HELP),
    rows: Optional[int] = typer.Option(
        None, help="cap at the first N streamed utterances, for smoke tests"
    ),
    slug: Optional[str] = typer.Option(
        None,
        help="name of the shard set under data/sharded_wds/ "
        "(default: DATASET-rowsN, or DATASET-full)",
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="dataset root"),
    dataset: str = typer.Option("emilia", help=DATASET_HELP),
    data_files: Optional[str] = typer.Option(
        None,
        help="emilia only: tar path glob selecting language and subset "
        "(default: all EN). Size the run with --rows, not this",
    ),
    split_ratio: float = typer.Option(0.95, help="fraction of speakers routed to train"),
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

    Nothing per-utterance touches disk. Failed segments are logged to
    failures.jsonl and skipped rather than aborting the run.
    """
    _check_model(model)
    if rows is not None and rows < 1:
        raise typer.BadParameter("--rows requires a positive count")

    from dataprep.shards import shard_prepare, shard_root

    resolved_slug = slug or (f"{dataset}-rows{rows}" if rows else f"{dataset}-full")
    wds_root = shard_root(resolved_slug, data_root=data_root)
    # Shard sets are expensive to rebuild and easy to clobber by rerunning with
    # the same name, so an existing one is an error until asked otherwise.
    if wds_root.exists() and any(wds_root.iterdir()) and not force:
        raise typer.BadParameter(
            f"{wds_root} already exists; pass --force to overwrite or pick another --slug"
        )

    segments = _load_segments(dataset, limit=rows, data_files=data_files)
    tokenizer = load_tokenizer(model, device=device, bucket_frames=bucket_frames)

    typer.echo(f"model      : {model}")
    typer.echo(f"dataset    : {dataset}")
    if dataset == "emilia":
        from dataprep.datasources.emilia import DEFAULT_DATA_FILES

        typer.echo(f"data files : {data_files or DEFAULT_DATA_FILES}")
    typer.echo(f"utterances : {rows if rows is not None else 'all (streaming)'}")
    typer.echo(f"slug       : {resolved_slug}")
    typer.echo(f"output     : {wds_root}")
    typer.echo(f"device     : {_tokenizer_device(tokenizer)}")
    typer.echo(f"logits     : {'yes' if include_logits else 'no'}")
    typer.echo(
        f"shuffle buf: {f'{shuffle_buffer} sequences' if shuffle_buffer else 'off (deterministic order)'}"
    )
    typer.echo(f"buckets    : {f'multiples of {bucket_frames}' if bucket_frames else 'off'}")

    shard_prepare(
        segments,
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
