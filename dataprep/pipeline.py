"""Tokenize and featurize dataset rows: the two compute stages of dataprep.

Two paths over the same stages. ``prepare_row``/``featurize_row`` read and write
per-row directories under ``data/`` for inspection; ``stream_prepared_samples``
runs the same work over a stream of decoded rows and yields shard-ready samples,
touching no disk. ``dataprep.shards.shard_prepare`` drives the latter.

Library module: the command line lives in ``dataprep.cli``.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from tqdm import tqdm

from dataprep.common import (
    FeaturizedSequence,
    Segment,
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


DEFAULT_MODELS = {
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
    if bucket_frames:
        raise ValueError(f"{model!r} does not support bucket_frames")
    if model == "qwen3":
        from dataprep.qwen3 import Qwen3Tokenizer

        return Qwen3Tokenizer(model_id or DEFAULT_MODELS["qwen3"])
    if model == "fish":
        from dataprep.fish import FishTokenizer

        return FishTokenizer(model_id or DEFAULT_MODELS["fish"])
    raise ValueError(f"Unknown model {model!r}")


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
) -> list[TokenizedSequence]:
    if pack_segments:
        return _pack_segments(segments, tokenizer, audio_codes=audio_codes)
    sequences: list[TokenizedSequence] = []
    for segment in tqdm(segments, desc="tokenize", unit="seg", leave=False):
        try:
            sequences.append(
                tokenizer.apply_chat_template([segment], audio_codes=audio_codes)
            )
        except Exception as exc:
            log_failure(log_root, stage="tokenize", row=row, exc=exc)
    return sequences


def tokenize_example(
    example,
    audio,
    *,
    tokenizer,
    log_root: Path = DEFAULT_LOG_ROOT,
    pack_segments: bool = False,
) -> tuple[list[TokenizedSequence], list]:
    """Encode a raw example's audio and build its tokenized sequences.

    Returns ``(sequences, channel_codes)``. Shared by the disk-backed and
    streaming paths so both number speakers and slice codes identically.
    """
    row = example.row
    channel_codes = [
        tokenizer.audio_codec.encode(audio[channel], example.sample_rate)
        for channel in tqdm(
            range(example.num_channels), desc=f"row {row} encode", unit="ch", leave=False
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

    sequences = build_sequences(
        segments,
        tokenizer,
        audio_codes=slice_segment_codes(
            segments, channel_codes, frame_rate=tokenizer.audio_codec.frame_rate
        ),
        row=row,
        log_root=log_root,
        pack_segments=pack_segments,
    )
    return sequences, channel_codes


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
            sequences, _ = tokenize_example(
                example,
                audio,
                tokenizer=tokenizer,
                log_root=log_root,
                pack_segments=pack_segments,
            )
        except Exception as exc:
            log_failure(log_root, stage="tokenize", row=row, exc=exc)
            continue

        for sequence in tqdm(
            sequences, desc=f"row {row} featurize", unit="seq", leave=False
        ):
            seq_id = sequence.seq_id
            try:
                feature = tokenizer.featurizer.featurize(sequence, include_kv=include_kv)
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


def prepare_row(
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

    sequences, channel_codes = tokenize_example(
        example,
        audio,
        tokenizer=tokenizer,
        log_root=log_root,
        pack_segments=pack_segments,
    )
    # The per-row codec dump is an inspection artifact; the streaming path skips it.
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
    tqdm.write(
        f"Featurizing row {row}: {len(sequences)} sequence(s)"
        + (" with KV cache" if dump_kv else "")
    )
    features = []
    for sequence in tqdm(sequences, desc=f"row {row} featurize", unit="seq", leave=False):
        try:
            feature = featurizer.featurize(sequence, include_kv=dump_kv)
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
