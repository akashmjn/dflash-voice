from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence, runtime_checkable


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
    """Model-ready, channel-first tokens and their validity mask."""

    tokens: Any
    mask: Any
    spans: list[SequenceSpan]

    @property
    def length(self) -> int:
        return int(self.tokens.shape[-1])


@runtime_checkable
class AudioCodec(Protocol):
    sample_rate: int
    frame_rate: float
    num_codebooks: int

    def encode(self, audio: Any, sample_rate: int) -> Any:
        """Return integer codec indexes shaped (num_codebooks, frames)."""

    def decode(self, codes: Any) -> Any:
        """Decode (num_codebooks, frames) indexes to a mono waveform."""


@runtime_checkable
class TextTokenizer(Protocol):
    def encode(self, text: str) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        ...


class Tokenizer(Protocol):
    audio_codec: AudioCodec
    text_tokenizer: TextTokenizer
    max_seq_length: int

    def apply_chat_template(self, segments: Sequence[Segment]) -> TokenizedSequence:
        ...


def validate_sequence(sequence: TokenizedSequence, num_codebooks: int) -> None:
    expected_rows = num_codebooks + 1
    if sequence.tokens.ndim != 2:
        raise ValueError(f"Expected sequence tokens shaped (C+1, L), got {sequence.tokens.shape}")
    if tuple(sequence.mask.shape) != tuple(sequence.tokens.shape):
        raise ValueError("Sequence mask must have the same shape as tokens")
    if int(sequence.tokens.shape[0]) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} sequence rows for {num_codebooks} codebooks, "
            f"got {sequence.tokens.shape[0]}"
        )
    for span in sequence.spans:
        if not 0 <= span.start < span.end <= sequence.length:
            raise ValueError(f"Invalid sequence span {span}")
