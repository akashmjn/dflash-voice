from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class Segment:
    """One speaker turn and, when available, its encoded audio."""

    text: str
    speaker: int | str = 0
    audio_codes: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SequenceSpan:
    segment_index: int
    start: int
    end: int
    kind: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenizedSequence:
    """Model-ready tokens and validity mask shaped (seq_len, num_codebooks + 1)."""

    tokens: Any
    mask: Any
    spans: list[SequenceSpan]

    @property
    def length(self) -> int:
        return int(self.tokens.shape[0])


@runtime_checkable
class AudioCodec(Protocol):
    sample_rate: int
    frame_rate: float
    num_codebooks: int

    def encode(self, audio: Any, sample_rate: int) -> Any:
        """Return integer codec indexes shaped (frames, num_codebooks)."""

    def decode(self, codes: Any) -> Any:
        """Decode (frames, num_codebooks) indexes to a mono waveform."""


@runtime_checkable
class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, token_ids: Sequence[int]) -> str: ...


class Tokenizer(Protocol):
    audio_codec: AudioCodec
    text_tokenizer: TextTokenizer
    max_seq_length: int

    def apply_chat_template(self, segments: Sequence[Segment]) -> TokenizedSequence: ...


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def validate_sequence(
    sequence: TokenizedSequence,
    num_codebooks: int,
    *,
    text_channel: int = -1,
) -> None:
    """Validate (L, C+1) layout and that mask/token channels agree."""
    expected_channels = num_codebooks + 1
    tokens = _as_numpy(sequence.tokens)
    mask = _as_numpy(sequence.mask).astype(bool)
    if tokens.ndim != 2:
        raise ValueError(
            f"Expected sequence tokens shaped (L, C+1), got {tokens.shape}"
        )
    if mask.shape != tokens.shape:
        raise ValueError("Sequence mask must have the same shape as tokens")
    if int(tokens.shape[1]) != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} sequence channels for {num_codebooks} codebooks, "
            f"got {tokens.shape[1]}"
        )

    text_channel = text_channel % expected_channels
    audio_channels = [idx for idx in range(expected_channels) if idx != text_channel]
    audio_active = mask[:, audio_channels].any(axis=1)
    if np.any(tokens[~audio_active][:, audio_channels] != 0):
        raise ValueError(
            "Audio-channel tokens must be zero where the audio mask is inactive"
        )
    if np.any(tokens[~mask[:, text_channel], text_channel] != 0):
        raise ValueError(
            "Text-channel tokens must be zero where the text mask is inactive"
        )

    for span in sequence.spans:
        if not 0 <= span.start < span.end <= sequence.length:
            raise ValueError(f"Invalid sequence span {span}")
