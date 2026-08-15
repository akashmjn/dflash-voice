"""Tokenize and featurize utterances: the two compute stages of dataprep.

``prepare_segment`` is the whole of it -- encode one segment's audio, lay it out
as tokens. ``stream_prepared_samples`` runs that plus the featurizer over a
stream of segments and yields shard-ready samples, touching no disk;
``dataprep.shards.shard_prepare`` drives it.

Dataset-shaped work belongs to the loaders (``dataprep.datasources.emilia``,
``dataprep.datasources.expresso``), which hand over ready :class:`Segment` objects.

Library module: the command line lives in ``dataprep.cli``.
"""

from __future__ import annotations

import inspect
import json
import traceback
from pathlib import Path
from typing import Any, Iterable, Iterator

from tqdm import tqdm

from dataprep.common import Segment, ShardSample, TokenizedSequence

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_LOG_ROOT = DEFAULT_DATA_ROOT / "dataprep_logs"


def log_failure(
    log_root: Path, *, stage: str, seq_id: str, exc: BaseException
) -> None:
    """Append a failure record to ``log_root/failures.jsonl`` and print it."""
    tqdm.write(f"{stage}: segment {seq_id} failed, skipping: {exc}")

    log_root.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "seq_id": seq_id,
        "error": str(exc),
        "traceback": traceback.format_exc(),
    }
    with (log_root / "failures.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def load_tokenizer(
    model: str,
    model_id: str | None = None,
    device: str | None = None,
    *,
    bucket_frames: int = 0,
):
    """Build a tokenizer backend.

    ``bucket_frames`` pads sequences up to a multiple, which bounds the number of
    the shapes seen by featurizer calls. Workaround to avoid unbounded memory growth
    of ~+50MiB per new length upto 60GB+ on Pytorch MPS backend.
        https://github.com/pytorch/pytorch/issues/181213
        https://github.com/pytorch/pytorch/pull/181485
    """
    if model == "miso":
        from dataprep.miso import MisoAudioCodec, MisoFeaturizer, MisoTokenizer

        return MisoTokenizer(
            audio_codec=MisoAudioCodec(device=device),
            featurizer=MisoFeaturizer(device=device),
            bucket_frames=bucket_frames,
        )

    if model == "chatterbox":
        from dataprep.chatterbox import CHATTERBOX_REPO, ChatterboxTokenizer

        return ChatterboxTokenizer(
            model_id or CHATTERBOX_REPO,
            device=device,
            bucket_frames=bucket_frames,
        )

    raise ValueError(f"Unknown model {model!r}")


def prepare_segment(segment: Segment, tokenizer) -> tuple[TokenizedSequence, Any]:
    """Encode one segment's audio and tokenize it.

    Returns ``(sequence, embedding_context)``; the context is None unless the
    backend implements ``embedding_context`` (Chatterbox's speaker embedding).
    """
    codes = tokenizer.audio_codec.encode(segment.audio, segment.sample_rate)
    sequence = tokenizer.apply_chat_template(segment, audio_codes=codes)
    build_context = getattr(tokenizer, "embedding_context", None)
    context = None if build_context is None else build_context(segment)
    return sequence, context


def _featurize(featurizer, sequence, *, include_kv: bool, context):
    """Pass ``context`` only to backends that take it -- Chatterbox needs its
    precomputed speaker embedding; the rest condition on tokens alone."""
    if "context" in inspect.signature(featurizer.featurize).parameters:
        return featurizer.featurize(sequence, include_kv=include_kv, context=context)
    return featurizer.featurize(sequence, include_kv=include_kv)


def stream_prepared_samples(
    segments: Iterable[Segment],
    *,
    model: str,
    tokenizer,
    log_root: Path | None = None,
    include_kv: bool = False,
    include_logits: bool = False,
    skip_ids: set[str] | None = None,
) -> Iterator[ShardSample]:
    """Tokenize + featurize segments, yielding WDS samples, nothing on disk.

    Lazy: a sample costs its forward pass only when pulled. Segments in
    ``skip_ids`` are dropped before tokenize, so skipping one is free.

    Failed segments are logged to ``failures.jsonl`` and skipped -- one bad
    utterance should not abort a multi-hour pass.
    """
    from dataprep.shards import build_sample

    log_root = DEFAULT_LOG_ROOT if log_root is None else log_root
    frame_rate = float(tokenizer.audio_codec.frame_rate)
    for segment in tqdm(segments, desc="prepare", unit="seg"):
        if skip_ids and segment.id in skip_ids:
            continue
        try:
            sequence, context = prepare_segment(segment, tokenizer)
            feature = _featurize(
                tokenizer.featurizer,
                sequence,
                include_kv=include_kv,
                context=context,
            )
            feature.validate(sequence_length=sequence.length)
            sample = build_sample(
                sequence,
                feature,
                segment=segment,
                model=model,
                frame_rate=frame_rate,
                include_kv=include_kv,
                include_logits=include_logits,
            )
        except Exception as exc:
            log_failure(log_root, stage="prepare", seq_id=segment.id, exc=exc)
            continue
        if sample is not None:
            yield sample


def inspect_segments(
    segments: Iterable[Segment],
    *,
    model: str,
    tokenizer,
    output_root: Path = DEFAULT_DATA_ROOT,
    dataset: str = "expresso",
    stage: str = "all",
    dump_kv: bool = False,
    log_root: Path | None = None,
) -> list[Path]:
    """Write per-segment artifacts you can open, one directory per segment id.

    The debugging counterpart to :func:`stream_prepared_samples`: same compute,
    but every intermediate lands on disk under
    ``output_root/dataset/{tokenized,featurized}/model/SEQ_ID/``. Only sane for a
    handful of segments -- the ``.pt`` files cost far more disk than the shards
    they would become.
    """
    from dataprep.common import FeaturizedSequence, SequenceEmbeddingContext

    log_root = DEFAULT_LOG_ROOT if log_root is None else log_root
    frame_rate = tokenizer.audio_codec.frame_rate
    written: list[Path] = []

    for segment in tqdm(segments, desc=f"Inspecting {model}", unit="seg"):
        try:
            sequence, context = prepare_segment(segment, tokenizer)
        except Exception as exc:
            log_failure(log_root, stage="tokenize", seq_id=segment.id, exc=exc)
            continue

        # save_all takes a list per directory; here it is always one segment.
        token_dir = Path(output_root) / dataset / "tokenized" / model / segment.id
        metadata = {"model": model, "dataset": dataset, "frame_rate": frame_rate}
        TokenizedSequence.save_all(token_dir, [sequence], metadata=metadata)
        SequenceEmbeddingContext.save_all(token_dir, [context])
        written.append(token_dir)
        if stage == "tokenize":
            continue

        try:
            feature = _featurize(
                tokenizer.featurizer, sequence, include_kv=dump_kv, context=context
            )
            feature.validate(sequence_length=sequence.length)
        except Exception as exc:
            log_failure(log_root, stage="featurize", seq_id=segment.id, exc=exc)
            continue

        feature_dir = Path(output_root) / dataset / "featurized" / model / segment.id
        FeaturizedSequence.save_all(feature_dir, [feature], metadata=metadata)
        written.append(feature_dir)

    return written


if __name__ == "__main__":  # pragma: no cover - moved to dataprep.cli
    raise SystemExit(
        "dataprep.pipeline is a library module; run 'python -m dataprep.cli --help'"
    )

