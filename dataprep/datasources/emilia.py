"""Emilia loader: one row is one utterance, so a row is one :class:`Segment`.

``amphion/Emilia-Dataset`` is gated -- ``HF_TOKEN`` must be set, or
``load_dataset`` fails on auth rather than on a missing file.

The hub layout is WebDataset tars, one ``KEY.json`` + ``KEY.mp3`` per utterance,
presented as the columns ``json``, ``mp3``, ``__key__``, ``__url__``. Selection
is a path glob, not stream filtering: language is a directory, so
``Emilia/EN/EN-B00000*.tar`` downloads only what it needs.

Two metadata schemas share the repo:

- ``Emilia/EN`` -- ``{id, wav, text, duration, speaker, language, dnsmos}``,
  speaker ``EN_B00000_S00000``
- ``Emilia-YODAS/DE`` -- ``{_id, ...}`` (no ``wav``), speaker
  ``DE_wSq11gYbUgU_SPEAKER_04``, diarized out of one source recording (see
  ``dataprep.shards.split_key``)
"""

from __future__ import annotations

import io
import os
from typing import Any, Iterator

import numpy as np

from dataprep.common import Segment

DEFAULT_DATASET = "amphion/Emilia-Dataset"

#: All English tars (1140 in ``Emilia/EN``, 1362 in ``Emilia-YODAS/EN``).
#:
#: The glob picks language and subset, not size -- one EN tar already holds ~68h
#: across ~25k utterances, so size a run with ``limit``. Names carry six digits
#: (``EN-B000000``), so a five-digit bracket matches nothing. See
#: ``modal_apps/dataprep/SKILL_SHARDSIZING.md``.
DEFAULT_DATA_FILES = "Emilia/EN/*.tar"

#: Slug for the on-disk artifact directory, as opposed to the hub dataset id.
DATASET_NAME = "emilia"


def _utterance_id(meta: dict[str, Any]) -> str:
    """Emilia spells the id ``id``; Emilia-YODAS spells it ``_id``."""
    for key in ("id", "_id"):
        value = meta.get(key)
        if value:
            return str(value)
    raise KeyError(f"No utterance id in metadata keys: {sorted(meta)}")


def _decode_mp3(value: Any) -> tuple[np.ndarray, int]:
    """Decode the ``mp3`` column to a mono float32 waveform.

    ``datasets`` hands this over as raw bytes or as an ``Audio`` dict depending
    on whether it inferred the feature, so both shapes are accepted.
    """
    import soundfile as sf

    if isinstance(value, dict):
        if value.get("array") is not None:
            audio = np.asarray(value["array"], dtype=np.float32)
            return _mono(audio), int(value["sampling_rate"])
        value = value.get("bytes") or value.get("path")
    if value is None:
        raise ValueError("Emilia row carries no decodable audio")

    source = io.BytesIO(value) if isinstance(value, (bytes, bytearray)) else value
    audio, sample_rate = sf.read(source, dtype="float32", always_2d=True)
    return _mono(audio.T), int(sample_rate)


def _mono(audio: np.ndarray) -> np.ndarray:
    """Collapse to a ``(samples,)`` mono waveform."""
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=0)
    raise ValueError(f"Expected mono or (C, samples) audio, got {audio.shape}")


def _undecoded_audio(stream):
    """Hand over the encoded mp3 bytes instead of a decoded array.

    ``datasets`` infers an ``Audio`` feature for ``mp3`` and runs every yielded
    row through ``Audio.encode_example``, which requires ``torchcodec``.
    ``cast_column(..., decode=False)`` does not help -- encode runs regardless
    of the decode flag. Clearing the schema does: the webdataset builder
    already yields ``{"path", "bytes"}``, which then passes through untouched.
    """
    stream._info.features = None
    return stream


def stream_emilia(
    *,
    dataset: str = DEFAULT_DATASET,
    data_files: str = DEFAULT_DATA_FILES,
    limit: int | None = None,
    token: str | None = None,
) -> Iterator[Segment]:
    """Yield one :class:`Segment` per utterance, straight from the hub.

    ``limit`` counts utterances -- unlike Expresso, a row is already one.
    """
    from datasets import load_dataset

    token = token or os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError(
            f"{dataset} is gated: set HF_TOKEN (see .env) or pass token=..."
        )

    stream = load_dataset(
        dataset,
        data_files={"train": data_files},
        split="train",
        streaming=True,
        token=token,
    )
    stream = _undecoded_audio(stream)
    if limit is not None:
        stream = stream.take(limit)

    for candidate in stream:
        row = dict(candidate)
        meta = row["json"]
        audio, sample_rate = _decode_mp3(row.get("mp3"))
        yield Segment(
            id=_utterance_id(meta),
            text=str(meta.get("text", "")),
            speaker=str(meta["speaker"]),
            audio=audio,
            sample_rate=sample_rate,
        )
