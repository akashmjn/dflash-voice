"""Shared dataprep records and on-disk serialization.

Pipeline objects are intentionally small and self-describing:

- ``Segment`` — transcript metadata only (no tensors)
- ``TokenizedSequence`` — model-ready ``(L, C+1)`` tokens/mask + spans/layout
- ``FeaturizedSequence`` — teacher-forced outputs of length ``L-1``, indexed so
  that position ``i`` is the model state after consuming ``tokens[i]``
  (predicting ``tokens[i+1]``). Use spans to select regions; no separate
  ``audio_positions`` list.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np


class SpanKind(str, Enum):
    """Region labels on a flat ``(L, C+1)`` sequence.

    Consumers should branch on these instead of model-specific layout rules.
    """

    TEXT = "text"
    AUDIO = "audio"
    SPECIAL = "special"  # bos/eos/pad/prefix/chat-control tokens


@dataclass
class Segment:
    """One speaker turn. Metadata only — codec frames live under ``raw/``."""

    source_dataset_id: int
    source_audio_channel_id: int
    segment_id: int
    text: str
    speaker: str
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_transcript_item(
        cls, item: dict[str, Any], *, source_dataset_id: int
    ) -> Segment:
        return cls(
            source_dataset_id=source_dataset_id,
            source_audio_channel_id=int(item["channel"]),
            segment_id=int(item["segment_id"]),
            text=str(item["text"]),
            speaker=str(item.get("speaker", "speaker")),
            start_sec=float(item["start"]),
            end_sec=float(item["end"]),
        )

    def frame_bounds(self, *, frame_rate: float, max_frames: int) -> tuple[int, int]:
        import math

        start = max(0, int(math.floor(self.start_sec * frame_rate)))
        end = min(max_frames, int(math.ceil(self.end_sec * frame_rate)))
        if end <= start:
            raise ValueError(
                f"Segment {self.segment_id} maps to empty codec frame range"
            )
        return start, end


@dataclass(frozen=True)
class TokenSequenceSpan:
    source_dataset_id: int
    segment_id: int
    start: int
    end: int
    kind: SpanKind

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_dataset_id": self.source_dataset_id,
            "segment_id": self.segment_id,
            "start": self.start,
            "end": self.end,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenSequenceSpan:
        return cls(
            source_dataset_id=int(payload["source_dataset_id"]),
            segment_id=int(payload["segment_id"]),
            start=int(payload["start"]),
            end=int(payload["end"]),
            kind=SpanKind(payload["kind"]),
        )


@dataclass(frozen=True)
class TokenizedSequenceLayout:
    """Enough channel geometry for a consumer to interpret tokens/mask."""

    num_codebooks: int
    text_channel: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenizedSequenceLayout:
        return cls(
            num_codebooks=int(payload["num_codebooks"]),
            text_channel=int(payload["text_channel"]),
        )


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if type(value).__module__.startswith("mlx."):
        import mlx.core as mx

        if str(value.dtype) == "mlx.core.bfloat16":
            value = value.astype(mx.float32)
        mx.eval(value)
        return np.asarray(value)
    return np.asarray(value)


def _as_torch(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if (
        type(value).__module__.startswith("mlx.")
        and str(value.dtype) == "mlx.core.bfloat16"
    ):
        import mlx.core as mx

        value = value.astype(mx.float32)
    return torch.from_numpy(np.asarray(value)).cpu()


def _as_torch_tree(value: Any):
    if isinstance(value, dict):
        return {key: _as_torch_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_torch_tree(item) for item in value]
    if value is None:
        return None
    return _as_torch(value)


@dataclass
class TokenizedSequence:
    """Model-ready tokens/mask shaped ``(L, num_codebooks + 1)``."""

    tokens: Any
    mask: Any
    spans: list[TokenSequenceSpan]
    layout: TokenizedSequenceLayout

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])

    def validate(self) -> None:
        """Check ``(L, C+1)`` layout and that mask/token channels agree."""
        expected_channels = self.layout.num_codebooks + 1
        tokens = _as_numpy(self.tokens)
        mask = _as_numpy(self.mask).astype(bool)
        if tokens.ndim != 2:
            raise ValueError(
                f"Expected sequence tokens shaped (L, C+1), got {tokens.shape}"
            )
        if mask.shape != tokens.shape:
            raise ValueError("Sequence mask must have the same shape as tokens")
        if int(tokens.shape[1]) != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} sequence channels for "
                f"{self.layout.num_codebooks} codebooks, got {tokens.shape[1]}"
            )

        text_channel = self.layout.text_channel % expected_channels
        audio_channels = [
            idx for idx in range(expected_channels) if idx != text_channel
        ]
        audio_active = mask[:, audio_channels].any(axis=1)
        if np.any(tokens[~audio_active][:, audio_channels] != 0):
            raise ValueError(
                "Audio-channel tokens must be zero where the audio mask is inactive"
            )
        if np.any(tokens[~mask[:, text_channel], text_channel] != 0):
            raise ValueError(
                "Text-channel tokens must be zero where the text mask is inactive"
            )

        for span in self.spans:
            if not 0 <= span.start < span.end <= self.length:
                raise ValueError(f"Invalid sequence span {span}")

    def spans_of(self, kind: SpanKind | str) -> list[TokenSequenceSpan]:
        kind = SpanKind(kind)
        return [span for span in self.spans if span.kind == kind]

    @staticmethod
    def save_all(
        directory: str | Path,
        sequences: Sequence[TokenizedSequence],
        *,
        metadata: dict[str, Any],
    ) -> Path:
        """Write ``sequences.pt`` (tokens+masks) and ``metadata.json`` (spans/layout)."""
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            [
                {
                    "tokens": _as_torch(item.tokens).long(),
                    "mask": _as_torch(item.mask).bool(),
                }
                for item in sequences
            ],
            directory / "sequences.pt",
        )
        # Drop legacy split artifacts from earlier dataprep layouts.
        for stale in ("masks.pt", "codebooks.pt"):
            (directory / stale).unlink(missing_ok=True)
        payload = {
            **metadata,
            "sequences": [
                {
                    "sequence_id": index,
                    "shape": list(_as_numpy(item.tokens).shape),
                    "layout": item.layout.to_dict(),
                    "spans": [span.to_dict() for span in item.spans],
                }
                for index, item in enumerate(sequences)
            ],
        }
        (directory / "metadata.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return directory

    @staticmethod
    def load_all(
        directory: str | Path,
    ) -> tuple[list[TokenizedSequence], dict[str, Any]]:
        import torch

        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        rows = torch.load(
            directory / "sequences.pt", map_location="cpu", weights_only=True
        )
        if len(rows) != len(metadata["sequences"]):
            raise ValueError(f"Inconsistent tokenized artifacts in {directory}")

        sequences = []
        for row, sequence_meta in zip(rows, metadata["sequences"]):
            layout = TokenizedSequenceLayout.from_dict(sequence_meta["layout"])
            spans = [
                TokenSequenceSpan.from_dict(payload)
                for payload in sequence_meta["spans"]
            ]
            sequence = TokenizedSequence(
                tokens=row["tokens"],
                mask=row["mask"],
                spans=spans,
                layout=layout,
            )
            sequence.validate()
            sequences.append(sequence)
        return sequences, metadata


@dataclass
class FeaturizedSequence:
    """Teacher-forced outputs aligned to a ``TokenizedSequence`` of length ``L``.

    ``hiddens`` and each ``logits[k]`` have length ``L - 1``. Index ``i`` is the
    model output after ``tokens[i]``, i.e. the distribution / state used to
    predict ``tokens[i + 1]``. Select regions with the same spans as the source
    sequence (for an audio span ``[s, e)``, predictions live at ``[s - 1, e - 1)``).
    """

    logits: dict[int, Any]
    hiddens: Any
    spans: list[TokenSequenceSpan]
    layout: TokenizedSequenceLayout
    kv_cache: Any | None = None

    @property
    def length(self) -> int:
        """Feature length ``L - 1``."""
        return int(self.hiddens.shape[0])

    def feature_slice_for_targets(self, start: int, end: int) -> slice:
        """Slice of features that predict ``tokens[start:end]``."""
        if start < 1:
            raise ValueError(
                "No teacher-forced prediction exists for tokens[0]; start must be >= 1"
            )
        if end <= start:
            raise ValueError(f"Empty target range [{start}, {end})")
        return slice(start - 1, end - 1)

    def validate(self, *, sequence_length: int | None = None) -> None:
        feature_len = self.length
        if sequence_length is not None and feature_len != sequence_length - 1:
            raise ValueError(
                f"Expected features of length {sequence_length - 1}, got {feature_len}"
            )
        for index, logits in self.logits.items():
            if int(logits.shape[0]) != feature_len:
                raise ValueError(
                    f"logits[{index}] length {logits.shape[0]} != hiddens length {feature_len}"
                )
        for span in self.spans:
            if not 0 <= span.start < span.end <= feature_len + 1:
                raise ValueError(
                    f"Span {span} incompatible with feature length {feature_len}"
                )

    @staticmethod
    def save_all(
        directory: str | Path,
        sequences: Sequence[FeaturizedSequence],
        *,
        metadata: dict[str, Any],
    ) -> Path:
        import torch

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            [
                {
                    "logits": {
                        index: _as_torch(logits)
                        for index, logits in item.logits.items()
                    },
                    "hiddens": _as_torch(item.hiddens),
                }
                for item in sequences
            ],
            directory / "features.pt",
        )
        for stale in ("logits.pt", "hiddens.pt"):
            (directory / stale).unlink(missing_ok=True)
        kv_path = directory / "kv_context.pt"
        if any(item.kv_cache is not None for item in sequences):
            torch.save(
                [_as_torch_tree(item.kv_cache) for item in sequences],
                kv_path,
            )
        else:
            kv_path.unlink(missing_ok=True)

        payload = {
            **metadata,
            "sequences": [
                {
                    "sequence_id": index,
                    "feature_length": item.length,
                    "layout": item.layout.to_dict(),
                    "spans": [span.to_dict() for span in item.spans],
                    "hidden_shape": list(_as_numpy(item.hiddens).shape),
                    "logit_shapes": {
                        str(codebook): list(_as_numpy(logits).shape)
                        for codebook, logits in item.logits.items()
                    },
                }
                for index, item in enumerate(sequences)
            ],
        }
        (directory / "metadata.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return directory

    @staticmethod
    def load_all(
        directory: str | Path,
    ) -> tuple[list[FeaturizedSequence], dict[str, Any]]:
        import torch

        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        rows = torch.load(
            directory / "features.pt", map_location="cpu", weights_only=True
        )
        kv_path = directory / "kv_context.pt"
        kv_rows = (
            torch.load(kv_path, map_location="cpu", weights_only=True)
            if kv_path.exists()
            else [None] * len(rows)
        )
        if len(rows) != len(metadata["sequences"]):
            raise ValueError(f"Inconsistent featurized artifacts in {directory}")

        sequences = []
        for row, kv_cache, sequence_meta in zip(rows, kv_rows, metadata["sequences"]):
            item = FeaturizedSequence(
                logits=row["logits"],
                hiddens=row["hiddens"],
                spans=[
                    TokenSequenceSpan.from_dict(payload)
                    for payload in sequence_meta["spans"]
                ],
                layout=TokenizedSequenceLayout.from_dict(sequence_meta["layout"]),
                kv_cache=kv_cache,
            )
            item.validate()
            sequences.append(item)
        return sequences, metadata
