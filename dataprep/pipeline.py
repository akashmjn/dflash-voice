"""Tokenize and featurize dataset rows: the two compute stages of dataprep.

Two paths over the same stages. ``tokenize_row``/``featurize_row`` read and write
per-row directories under ``data/`` for inspection; ``stream_prepared_samples``
runs the same work over a stream of decoded rows and yields shard-ready samples,
touching no disk. ``dataprep.shards.shard_prepare`` drives the latter.

Library module: the command line lives in ``dataprep.cli``.
"""

from __future__ import annotations

import inspect
import json
import traceback
import warnings
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from tqdm import tqdm

from dataprep.common import (
    FeaturizedSequence,
    Segment,
    SequenceEmbeddingContext,
    ShardSample,
    TokenizedSequence,
    _as_torch,
)
from dataprep.expresso import (
    DATASET_NAME,
    DEFAULT_DATASET,
    DecodedExample,
    load_raw_example,
)


# Deprecated MLX-only backends; see dataprep/mlx_backends/README.md. Kept only so
# the experiments/expresso_nll_entropy/ comparison stays reproducible.
DEPRECATED_MLX_MODELS = {
    "qwen3": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "fish": "mlx-community/fish-audio-s2-pro-8bit",
}

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_LOG_ROOT = DEFAULT_DATA_ROOT / "dataprep_logs"


def log_failure(log_root: Path, *, stage: str, row: int, exc: BaseException) -> None:
    """Append a failure record to ``log_root/failures.jsonl`` and print it."""
    tqdm.write(f"{stage}: row {row} failed, skipping: {exc}")

    log_root.mkdir(parents=True, exist_ok=True)
    record = {
        "stage": stage,
        "row": row,
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

    ``miso`` and ``chatterbox`` are the maintained backends. ``qwen3`` and
    ``fish`` are deprecated MLX-only backends kept for reproducing
    ``experiments/expresso_nll_entropy/``; they warn on use -- see
    ``dataprep/mlx_backends/README.md``.

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

        if bucket_frames:
            raise ValueError(f"{model!r} does not support bucket_frames")
        return ChatterboxTokenizer(model_id or CHATTERBOX_REPO, device=device)

    if model not in DEPRECATED_MLX_MODELS:
        raise ValueError(f"Unknown model {model!r}")

    # Deprecated MLX-only path. Imports stay inside this branch so nothing here
    # loads -- or has to keep working -- during a normal pipeline run.
    from dataprep.mlx_backends import DEPRECATION_NOTE

    warnings.warn(DEPRECATION_NOTE.format(model=model), DeprecationWarning, stacklevel=2)
    if bucket_frames:
        raise ValueError(f"{model!r} does not support bucket_frames")
    if model == "qwen3":
        from dataprep.mlx_backends.qwen3 import Qwen3Tokenizer

        return Qwen3Tokenizer(model_id or DEPRECATED_MLX_MODELS["qwen3"])

    from dataprep.mlx_backends.fish import FishTokenizer

    return FishTokenizer(model_id or DEPRECATED_MLX_MODELS["fish"])


def save_codebooks(
    path: Path,
    channel_codes: Sequence[Any],
    *,
    sample_rate: int,
    frame_rate: float,
    num_codebooks: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "channels": [_as_torch(codes).long() for codes in channel_codes],
            "sample_rate": sample_rate,
            "frame_rate": frame_rate,
            "num_codebooks": num_codebooks,
        },
        path,
    )


def slice_segment_codes(
    segments: Sequence[Segment],
    channel_codes: Sequence[Any],
    *,
    frame_rate: float,
) -> dict[int, Any]:
    codes_by_segment: dict[int, Any] = {}
    for segment in segments:
        codes = channel_codes[segment.source_audio_channel_id]
        start, end = segment.frame_bounds(
            frame_rate=frame_rate, max_frames=int(codes.shape[0])
        )
        codes_by_segment[segment.segment_id] = codes[start:end]
    return codes_by_segment


def slice_segment_audio(
    segments: Sequence[Segment],
    audio: Any,
    *,
    sample_rate: int,
) -> dict[int, Any]:
    """Cut each segment's waveform out of the row audio, by sample.

    The sibling of :func:`slice_segment_codes`, for backends that need the audio
    itself -- to encode per segment (see :func:`encode_segment_codes`), or to
    derive something else from it such as a speaker embedding.
    """
    audio_by_segment: dict[int, Any] = {}
    for segment in segments:
        channel = audio[segment.source_audio_channel_id]
        total = int(channel.shape[-1])
        start = max(0, int(segment.start_sec * sample_rate))
        end = min(total, int(round(segment.end_sec * sample_rate)))
        if end <= start:
            raise ValueError(
                f"Segment {segment.segment_id} maps to an empty audio range"
            )
        audio_by_segment[segment.segment_id] = channel[start:end]
    return audio_by_segment


def encode_segment_codes(
    segments: Sequence[Segment],
    audio_slices: Mapping[int, Any],
    tokenizer,
    *,
    sample_rate: int,
) -> dict[int, Any]:
    """Encode each segment's own waveform, one codec call per segment.

    The default path encodes a whole channel once and slices it by frame, which
    is cheaper and correct for a causal or strictly local codec. It is wrong for
    an encoder that attends across the clip: S3 is bidirectional, so sliced
    frames would not match what the model sees given that segment alone.
    """
    return {
        segment.segment_id: tokenizer.audio_codec.encode(
            audio_slices[segment.segment_id], sample_rate
        )
        for segment in tqdm(segments, desc="audio encode", unit="seg", leave=False)
    }


def _pack_segments(
    segments: Sequence[Segment],
    tokenizer,
    *,
    audio_codes: Mapping[int, Any],
) -> list[TokenizedSequence]:
    """Greedily group consecutive segments up to the model sequence limit."""
    chunks: list[TokenizedSequence] = []
    pending: list[Segment] = []
    for segment in segments:
        candidate = [*pending, segment]
        try:
            tokenizer.apply_chat_template(candidate, audio_codes=audio_codes)
        except ValueError as error:
            if "exceeds" not in str(error) or not pending:
                raise
            chunks.append(
                tokenizer.apply_chat_template(pending, audio_codes=audio_codes)
            )
            pending = [segment]
            tokenizer.apply_chat_template(pending, audio_codes=audio_codes)
        else:
            pending = candidate
    if pending:
        chunks.append(tokenizer.apply_chat_template(pending, audio_codes=audio_codes))
    return chunks


def build_sequences(
    segments: Sequence[Segment],
    tokenizer,
    *,
    audio_codes: Mapping[int, Any],
    row: int,
    log_root: Path,
    pack_segments: bool = False,
    audio_slices: Mapping[int, Any] | None = None,
    sample_rate: int | None = None,
) -> tuple[list[TokenizedSequence], list[Any]]:
    """Tokenize each segment, returning ``(sequences, embedding_contexts)``.

    ``embedding_contexts`` runs parallel to ``sequences``, all-``None`` unless
    the backend implements ``embedding_context`` (see
    :class:`dataprep.common.SequenceEmbeddingContext`). Building it here rather
    than in ``apply_chat_template`` keeps the tokenizer signature uniform and
    keeps the two lists in step when a failing segment is dropped.
    """
    if pack_segments:
        chunks = _pack_segments(segments, tokenizer, audio_codes=audio_codes)
        return chunks, [None] * len(chunks)

    build_context = getattr(tokenizer, "embedding_context", None)
    sequences: list[TokenizedSequence] = []
    contexts: list[Any] = []
    for segment in tqdm(segments, desc="tokenize", unit="seg", leave=False):
        try:
            sequence = tokenizer.apply_chat_template(
                [segment], audio_codes=audio_codes
            )
            context = None
            if build_context is not None and audio_slices is not None:
                context = build_context(
                    segment, audio_slices[segment.segment_id], sample_rate
                )
        except Exception as exc:
            log_failure(log_root, stage="tokenize", row=row, exc=exc)
            continue
        sequences.append(sequence)
        contexts.append(context)
    return sequences, contexts


def tokenize_example(
    example,
    audio,
    *,
    tokenizer,
    log_root: Path = DEFAULT_LOG_ROOT,
    pack_segments: bool = False,
) -> tuple[list[TokenizedSequence], list, list[Any]]:
    """Encode a raw example's audio and build its tokenized sequences.

    Returns ``(sequences, channel_codes, embedding_contexts)``. Shared by the
    disk-backed and streaming paths so both number speakers and slice codes
    identically.

    A backend setting ``audio_codec.segmented_encode`` is encoded one segment at
    a time (see :func:`encode_segment_codes`), leaving ``channel_codes`` empty
    since no whole-channel pass runs.
    """
    row = example.row
    segmented = bool(getattr(tokenizer.audio_codec, "segmented_encode", False))
    channel_codes = []
    if not segmented:
        channel_codes = [
            tokenizer.audio_codec.encode(audio[channel], example.sample_rate)
            for channel in tqdm(
                range(example.num_channels),
                desc=f"row {row} audio encode",
                unit="ch",
                leave=False,
            )
        ]

    segments = [
        Segment.from_transcript_item(raw, source_dataset_id=row)
        for raw in example.segments
    ]
    # Number the speakers 0, 1, ... in order of first appearance in the row, so
    # the id is the small turn-taking index Miso's text prefix expects.
    speaker_ids: dict[str, int] = {}
    for segment in segments:
        segment.speaker_id = speaker_ids.setdefault(segment.speaker, len(speaker_ids))

    audio_slices = None
    if segmented:
        audio_slices = slice_segment_audio(
            segments, audio, sample_rate=example.sample_rate
        )
        audio_codes = encode_segment_codes(
            segments, audio_slices, tokenizer, sample_rate=example.sample_rate
        )
    else:
        audio_codes = slice_segment_codes(
            segments, channel_codes, frame_rate=tokenizer.audio_codec.frame_rate
        )

    sequences, contexts = build_sequences(
        segments,
        tokenizer,
        audio_codes=audio_codes,
        row=row,
        log_root=log_root,
        pack_segments=pack_segments,
        audio_slices=audio_slices,
        sample_rate=example.sample_rate,
    )
    return sequences, channel_codes, contexts


def stream_prepared_samples(
    examples: Iterable[DecodedExample],
    *,
    model: str,
    tokenizer,
    log_root: Path | None = None,
    pack_segments: bool = False,
    include_kv: bool = False,
    include_logits: bool = False,
    skip_rows: set[int] | None = None,
) -> Iterator[ShardSample]:
    """Tokenize + featurize raw examples, yielding WDS samples, nothing on disk.

    Lazy: a sample costs its forward pass only when pulled. Rows in
    ``skip_rows`` are dropped before tokenize, so skipping one is free.

    Failed rows are logged to ``failures.jsonl`` and skipped -- one bad row
    should not abort a multi-hour pass.
    """
    from dataprep.shards import build_sample

    log_root = DEFAULT_LOG_ROOT if log_root is None else log_root
    frame_rate = float(tokenizer.audio_codec.frame_rate)
    for example in examples:
        row = example.row
        if skip_rows and row in skip_rows:
            continue
        audio = example.audio
        try:
            sequences, _, contexts = tokenize_example(
                example,
                audio,
                tokenizer=tokenizer,
                log_root=log_root,
                pack_segments=pack_segments,
            )
        except Exception as exc:
            log_failure(log_root, stage="tokenize", row=row, exc=exc)
            continue

        for sequence, context in tqdm(
            list(zip(sequences, contexts)),
            desc=f"row {row} featurize",
            unit="seq",
            leave=False,
        ):
            seq_id = sequence.seq_id
            try:
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
                    row=row,
                    seq_id=seq_id,
                    model=model,
                    frame_rate=frame_rate,
                    include_kv=include_kv,
                    include_logits=include_logits,
                )
            except Exception as exc:
                log_failure(log_root, stage="featurize", row=row, exc=exc)
                continue
            if sample is not None:
                yield sample


def _featurize(featurizer, sequence, *, include_kv: bool, context):
    """Pass ``context`` only to backends that take it -- Chatterbox needs its
    precomputed speaker embedding; the rest condition on tokens alone."""
    if "context" in inspect.signature(featurizer.featurize).parameters:
        return featurizer.featurize(sequence, include_kv=include_kv, context=context)
    return featurizer.featurize(sequence, include_kv=include_kv)


def tokenize_row(
    row_dir: str | Path,
    *,
    model: str,
    tokenizer,
    output_root: str | Path = DEFAULT_DATA_ROOT,
    dataset: str = DATASET_NAME,
    pack_segments: bool = False,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> Path:
    example = load_raw_example(row_dir)
    audio = example.audio
    row = example.row
    row_dir = Path(row_dir)
    tqdm.write(
        f"Tokenizing row {row}: {example.num_channels} channel(s), "
        f"{len(example.segments)} segment(s)"
    )

    sequences, channel_codes, contexts = tokenize_example(
        example,
        audio,
        tokenizer=tokenizer,
        log_root=log_root,
        pack_segments=pack_segments,
    )
    # The per-row codec dump is an inspection artifact; the streaming path skips
    # it, and a segmented-encode backend has no whole-channel pass to dump.
    if channel_codes:
        save_codebooks(
            row_dir / f"{model}_codebooks.pt",
            channel_codes,
            sample_rate=tokenizer.audio_codec.sample_rate,
            frame_rate=tokenizer.audio_codec.frame_rate,
            num_codebooks=tokenizer.audio_codec.num_codebooks,
        )

    output_dir = Path(output_root) / dataset / "tokenized" / model / str(row)
    TokenizedSequence.save_all(
        output_dir,
        sequences,
        metadata={
            "model": model,
            "dataset": DEFAULT_DATASET,
            "row": example.row,
            "frame_rate": tokenizer.audio_codec.frame_rate,
        },
    )
    SequenceEmbeddingContext.save_all(output_dir, contexts)
    tqdm.write(f"Tokenizing row {row}: wrote {len(sequences)} sequence(s) to {output_dir}")
    return output_dir


def featurize_row(
    row: int,
    *,
    model: str,
    featurizer,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    dataset: str = DATASET_NAME,
    dump_kv: bool = False,
    log_root: Path = DEFAULT_LOG_ROOT,
) -> Path:
    input_dir = Path(data_root) / dataset / "tokenized" / model / str(row)
    sequences, source_metadata = TokenizedSequence.load_all(input_dir)
    contexts = SequenceEmbeddingContext.load_all(input_dir, count=len(sequences))
    tqdm.write(
        f"Featurizing row {row}: {len(sequences)} sequence(s)"
        + (" with KV cache" if dump_kv else "")
    )
    features = []
    for sequence, context in tqdm(
        list(zip(sequences, contexts)),
        desc=f"row {row} featurize",
        unit="seq",
        leave=False,
    ):
        try:
            feature = _featurize(
                featurizer, sequence, include_kv=dump_kv, context=context
            )
            feature.validate(sequence_length=sequence.length)
        except Exception as exc:
            log_failure(log_root, stage="featurize", row=row, exc=exc)
            continue
        features.append(feature)

    feature_dir = Path(data_root) / dataset / "featurized" / model / str(row)
    FeaturizedSequence.save_all(
        feature_dir,
        features,
        metadata={
            "model": model,
            "dataset": source_metadata["dataset"],
            "row": row,
            "frame_rate": source_metadata["frame_rate"],
        },
    )
    tqdm.write(f"Featurizing row {row}: wrote {feature_dir}")
    return feature_dir


if __name__ == "__main__":  # pragma: no cover - moved to dataprep.cli
    raise SystemExit(
        "dataprep.pipeline is a library module; run 'python -m dataprep.cli --help'"
    )
