from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_DATASET = "Zackh/expresso-contextual"
DEFAULT_SPLIT = "train"


@dataclass(frozen=True)
class RawExample:
    row: int
    audio_path: Path
    transcript_path: Path
    sample_rate: int
    num_channels: int
    segments: list[dict[str, Any]]


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


def _iter_selected_rows(dataset: str, split: str, rows: set[int]):
    from datasets import Audio, load_dataset

    stream = load_dataset(dataset, split=split, streaming=True)
    if stream.features is not None:
        for name, feature in stream.features.items():
            if isinstance(feature, Audio):
                stream = stream.cast_column(name, Audio(decode=False))
    for row_index, candidate in enumerate(stream):
        if row_index in rows:
            yield row_index, dict(candidate)
        if row_index >= max(rows):
            return


def download_expresso(
    rows: Iterable[int] = (0, 1, 2),
    *,
    root: str | Path = "data/expresso/raw",
    dataset: str = DEFAULT_DATASET,
    split: str = DEFAULT_SPLIT,
) -> list[RawExample]:
    import soundfile as sf

    selected = set(rows)
    if not selected or min(selected) < 0:
        raise ValueError("At least one non-negative row is required")
    root = Path(root)
    results = []
    found = set()
    for row_index, row in _iter_selected_rows(dataset, split, selected):
        audio, sample_rate = _decode_audio(row)
        segments = _parse_segments(row)
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

        row_dir = root / str(row_index)
        row_dir.mkdir(parents=True, exist_ok=True)
        audio_path = row_dir / "audio.wav"
        transcript_path = row_dir / "transcript_segments.json"
        sf.write(audio_path, audio.T, sample_rate)
        transcript_path.write_text(
            json.dumps(
                {
                    "dataset": dataset,
                    "split": split,
                    "row": row_index,
                    "sample_rate": sample_rate,
                    "num_channels": int(audio.shape[0]),
                    "duration_sec": duration,
                    "segments": segments,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        results.append(
            RawExample(
                row=row_index,
                audio_path=audio_path,
                transcript_path=transcript_path,
                sample_rate=sample_rate,
                num_channels=int(audio.shape[0]),
                segments=segments,
            )
        )
        found.add(row_index)
    missing = selected - found
    if missing:
        raise IndexError(f"Expresso rows not found: {sorted(missing)}")
    return sorted(results, key=lambda example: example.row)


def load_raw_example(row_dir: str | Path) -> tuple[RawExample, np.ndarray]:
    import soundfile as sf

    row_dir = Path(row_dir)
    transcript_path = row_dir / "transcript_segments.json"
    payload = json.loads(transcript_path.read_text(encoding="utf-8"))
    audio_path = row_dir / "audio.wav"
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.T
    return (
        RawExample(
            row=int(payload["row"]),
            audio_path=audio_path,
            transcript_path=transcript_path,
            sample_rate=int(sample_rate),
            num_channels=int(audio.shape[0]),
            segments=list(payload["segments"]),
        ),
        audio,
    )
