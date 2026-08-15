"""Expresso loader: multi-turn rows in, one :class:`Segment` per turn out.

A row is a multi-channel recording holding several timed speaker turns, unlike
every other source we prepare. The flattening lives here rather than in the
pipeline: cutting a turn's waveform out of its channel, numbering the row's
speakers, and synthesizing the id and speaker label other datasets supply
natively.
"""

from __future__ import annotations

import io
import json
from typing import Any, Iterator

import numpy as np

from dataprep.common import Segment

DEFAULT_DATASET = "Zackh/expresso-contextual"
DEFAULT_SPLIT = "train"

#: Slug for the on-disk artifact directory, as opposed to the hub dataset id.
DATASET_NAME = "expresso"


def _segment_id(row: int, index: int) -> str:
    """Emilia-shaped id for a turn: zero-padded, so ids sort in stream order."""
    return f"expresso_r{row:06d}_s{index:03d}"


def _speaker_label(row: int, speaker: str) -> str:
    """Row-scoped speaker label.

    Expresso's bare labels ("ex01", ...) repeat across rows, and
    ``dataprep.shards.assign_split`` hashes this string -- unprefixed, the
    dataset would collapse onto a handful of split buckets.
    """
    return f"r{row:06d}_{speaker}"


def _parse_segments(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("json", "turns", "transcript", "segments"):
        if key not in row:
            continue
        value = json.loads(row[key]) if isinstance(row[key], str) else row[key]
        if isinstance(value, dict):
            value = value.get("turns")
        if isinstance(value, list):
            result = []
            for index, item in enumerate(value):
                segment = dict(item)
                start_ms = float(
                    segment.get("start_time_ms", segment.get("start", 0) * 1000)
                )
                end_ms = float(segment.get("end_time_ms", segment.get("end", 0) * 1000))
                if end_ms <= start_ms:
                    raise ValueError(
                        f"Invalid timing for segment {index}: {start_ms}..{end_ms} ms"
                    )
                segment.update(
                    {
                        "segment_id": index,
                        "start": start_ms / 1000.0,
                        "end": end_ms / 1000.0,
                        "channel": int(segment.get("channel", 0)),
                        "speaker": str(segment.get("speaker", "speaker")),
                        "text": str(segment.get("text", "")),
                    }
                )
                result.append(segment)
            return result
    raise KeyError(f"Could not find transcript segments in row keys: {sorted(row)}")


def _decode_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    """Decode the row's audio column as channel-first ``(C, samples)``."""
    import soundfile as sf

    for value in row.values():
        if not isinstance(value, dict):
            continue
        if value.get("bytes") is not None:
            audio, sample_rate = sf.read(
                io.BytesIO(value["bytes"]), dtype="float32", always_2d=True
            )
            return audio.T, int(sample_rate)
        if value.get("path"):
            audio, sample_rate = sf.read(value["path"], dtype="float32", always_2d=True)
            return audio.T, int(sample_rate)
        if "array" in value and "sampling_rate" in value:
            audio = np.asarray(value["array"], dtype=np.float32)
            if audio.ndim == 1:
                audio = audio[None]
            elif audio.shape[0] > audio.shape[1]:
                audio = audio.T
            return audio, int(value["sampling_rate"])
    raise KeyError(f"Could not find an audio column in row keys: {sorted(row)}")


def _validate_segments(
    row_index: int, segments: list[dict[str, Any]], audio: np.ndarray, sample_rate: int
) -> None:
    """Reject segments referencing a missing channel or timings past the audio."""
    duration = audio.shape[1] / sample_rate
    for segment in segments:
        if segment["channel"] >= audio.shape[0]:
            raise ValueError(
                f"Row {row_index} segment {segment['segment_id']} references channel "
                f"{segment['channel']} but audio has {audio.shape[0]} channels"
            )
        if segment["start"] < 0 or segment["end"] > duration + 1 / sample_rate:
            raise ValueError(
                f"Row {row_index} segment timing falls outside {duration:.3f}s audio"
            )


def _cut(
    audio: np.ndarray, raw: dict[str, Any], *, sample_rate: int, seq_id: str
) -> np.ndarray:
    """Cut one turn's waveform out of its channel, by sample."""
    channel = audio[raw["channel"]]
    total = int(channel.shape[-1])
    start = max(0, int(raw["start"] * sample_rate))
    end = min(total, int(round(raw["end"] * sample_rate)))
    if end <= start:
        raise ValueError(f"Segment {seq_id} maps to an empty audio range")
    return channel[start:end]


def stream_expresso(
    *,
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
    limit: int | None = None,
) -> Iterator[Segment]:
    """Yield one :class:`Segment` per turn, straight from the hub.

    ``limit`` caps *rows* pulled from the stream, not segments, since a row is
    the unit the hub hands over -- one row yields several segments.

    Rows arrive in dataset order. ``IterableDataset.shuffle`` would also
    shuffle shard *order* and prefetch from several of the 36 audio files at
    once, stalling the first row for minutes; the sequence-level reservoir in
    ``dataprep.shards.shuffle_stream`` breaks up per-row correlation for far
    less.
    """
    from datasets import Audio, load_dataset

    stream = load_dataset(dataset, split=split, streaming=True)
    if stream.features is not None:
        for name, feature in stream.features.items():
            if isinstance(feature, Audio):
                stream = stream.cast_column(name, Audio(decode=False))

    # Tag each row with its dataset-order id; segment ids are built from it.
    stream = stream.map(lambda _row, idx: {"__row_index__": idx}, with_indices=True)
    if limit is not None:
        stream = stream.take(limit)

    for candidate in stream:
        row = dict(candidate)
        row_index = int(row.pop("__row_index__"))
        audio, sample_rate = _decode_audio(row)
        raw_segments = _parse_segments(row)
        _validate_segments(row_index, raw_segments, audio, sample_rate)

        # Number the speakers 0, 1, ... in order of first appearance in the row,
        # so the id is the small turn-taking index Miso's text prefix expects.
        speaker_ids: dict[str, int] = {}
        for index, raw in enumerate(raw_segments):
            speaker = str(raw["speaker"])
            seq_id = _segment_id(row_index, index)
            yield Segment(
                id=seq_id,
                text=str(raw["text"]),
                speaker=_speaker_label(row_index, speaker),
                audio=_cut(audio, raw, sample_rate=sample_rate, seq_id=seq_id),
                sample_rate=sample_rate,
                speaker_id=speaker_ids.setdefault(speaker, len(speaker_ids)),
            )
