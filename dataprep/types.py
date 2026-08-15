"""The records passed between dataprep stages.

One ``Segment`` becomes one ``TokenizedSequence`` becomes one ``ShardSample``;
identity is the loader-supplied ``Segment.id``, carried through unchanged.

- ``Segment`` -- one utterance: transcript, speaker, and its own waveform
- ``TokenizedSequence`` -- model-ready ``(L, C+1)`` tokens + spans/layout
- ``FeaturizedSequence`` -- teacher-forced outputs of length ``L-1``, indexed so
  that position ``i`` is the model state after consuming ``tokens[i]``
  (predicting ``tokens[i+1]``). Use spans to select regions; no separate
  ``audio_positions`` list.
- ``SequenceEmbeddingContext`` -- precomputed *continuous* conditioning that has
  no integer token to live in, kept beside the sequences rather than inside them.
- ``ShardSample`` -- one sequence serialized for a WebDataset shard.

Scoring these records lives in ``dataprep.utils``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from dataprep.utils import _as_numpy, _as_torch, _as_torch_tree, _wds_key, bucket_length


class TokenSpanKind(str, Enum):
    """Region labels on a flat ``(L, C+1)`` sequence.

    Consumers should branch on these instead of model-specific layout rules.

    Boundary markers get their own kinds rather than a shared ``SPECIAL``
    because token values cannot identify them: id 0 is a real EOS for some
    models and the grid's "no token here" fill everywhere else.

    ``PREFIX`` and ``PADDING`` are both unsupervised but not interchangeable: a
    prefix frame is fed to the model as a real embedding, padding is dead space.
    """

    TEXT = "text"
    AUDIO = "audio"
    BOS_TEXT = "bos_text"
    EOS_TEXT = "eos_text"
    BOS_AUDIO = "bos_audio"
    EOS_AUDIO = "eos_audio"
    PREFIX = "prefix"  # reserved, never supervised
    PADDING = "padding"  # trailing bucket fill, never supervised
    SPECIAL = "special"  # anything not worth naming; also the pre-split label

TokenSpanIdRange = dict[TokenSpanKind, tuple[int, int]]

@dataclass
class Segment:
    """One utterance: transcript, speaker, and its own waveform.

    The unit the whole pipeline runs on -- one Segment becomes one
    :class:`TokenizedSequence` becomes one :class:`ShardSample`. Loaders own
    both halves of that: cutting a multi-turn source row into per-utterance
    segments (see :func:`dataprep.datasources.expresso.stream_expresso`) and assigning
    ``id``. Nothing downstream infers identity from spans or slices by timing.
    """

    #: Globally unique, loader-supplied. Emilia uses its native metadata id
    #: (``EN_B00000_S00000_W000000``); Expresso synthesizes an equivalent.
    id: str
    text: str
    #: Native speaker label. The split hashes a prefix of this, so it must
    #: identify the source recording -- see ``dataprep.shards.split_key``.
    speaker: str
    #: ``(samples,)`` mono waveform for this utterance alone.
    audio: np.ndarray
    sample_rate: int
    #: Per-sequence speaker index, consumed by Miso's ``[{speaker_id}]`` text
    #: prefix. 0 for one-utterance datasets; the Expresso loader numbers a
    #: row's turns 0, 1, ... in order of first appearance.
    speaker_id: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Metadata only. Not ``asdict``: that would deep-copy the waveform
        into something json cannot serialize anyway."""
        return {
            "id": self.id,
            "text": self.text,
            "speaker": self.speaker,
            "sample_rate": self.sample_rate,
            "speaker_id": self.speaker_id,
        }

@dataclass(frozen=True)
class TokenSequenceSpan:
    """Pointer to a [start, end) contiguous region of a ``(L, C+1)`` sequence.

    Carries no identity: one sequence is one utterance, so the sequence's
    ``seq_id`` already covers every span in it.
    """

    start: int
    end: int
    kind: TokenSpanKind

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "kind": self.kind.value,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenSequenceSpan:
        # Extra keys are ignored, so artifacts written before spans dropped
        # their identity fields still load.
        return cls(
            start=int(payload["start"]),
            end=int(payload["end"]),
            kind=TokenSpanKind(payload["kind"]),
        )

@dataclass(frozen=True)
class TokenizedSequenceLayout:
    """Channel geometry that makes a serialized ``(L, C+1)`` sequence self-describing.

    A frame has ``num_codebooks + 1`` channels. Exactly one is the ``text_channel``
    (text tokens, and for a semantic-LM model like Fish the semantic token too);
    the rest are ``audio_channels`` — the codec codes, in codebook order, that the
    waveform decoder consumes.

    ``head_targets`` maps which saved token column targets are teacher-forced
    against. Usually code head ``k`` predicts ``audio_channels[k]``. The exception is 
    Fish, which predicts the semantic codes in ``text_channel`` rather than the 
    audio code. Consumers can score/read logits without re-deriving this per model.

    ``hidden_dim`` and ``logit_dims`` describe the featurized side: the model's
    hidden width and each head's vocabulary size. They are constant for a model,
    so they live here rather than being repeated per sequence, and are unset
    (``None`` / empty) on a purely tokenized sequence.
    """

    num_codebooks: int
    text_channel: int
    head_targets: tuple[int, ...] = ()
    hidden_dim: int | None = None
    logit_dims: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        # Default: each head predicts its own audio channel (no semantic LM head).
        if not self.head_targets:
            object.__setattr__(self, "head_targets", tuple(self.audio_channels))
        if len(self.head_targets) != self.num_codebooks:
            raise ValueError(
                f"head_targets must have {self.num_codebooks} entries, "
                f"got {len(self.head_targets)}"
            )

    @property
    def num_channels(self) -> int:
        return self.num_codebooks + 1

    @property
    def text_column(self) -> int:
        return self.text_channel % self.num_channels

    @property
    def audio_channels(self) -> tuple[int, ...]:
        """Columns holding codec audio codes, in codebook order."""
        return tuple(c for c in range(self.num_channels) if c != self.text_column)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "num_codebooks": self.num_codebooks,
            "text_channel": self.text_channel,
            "head_targets": list(self.head_targets),
        }
        if self.hidden_dim is not None:
            payload["hidden_dim"] = self.hidden_dim
        if self.logit_dims:
            payload["logit_dims"] = list(self.logit_dims)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> TokenizedSequenceLayout:
        hidden_dim = payload.get("hidden_dim")
        return cls(
            num_codebooks=int(payload["num_codebooks"]),
            text_channel=int(payload["text_channel"]),
            head_targets=tuple(payload.get("head_targets", ())),
            hidden_dim=None if hidden_dim is None else int(hidden_dim),
            logit_dims=tuple(payload.get("logit_dims", ())),
        )

@dataclass
class TokenizedSequence:
    """Model-ready stack of ``tokens`` (integer array) shaped ``(L, num_codebooks + 1)``.

    Contains text tokens (+1), and semantic + audio codec tokens in a model-specific
    arrangement described by ``layout``. ``spans`` demarcates contiguous regions of the
    token sequence (e.g. text, audio, ...) for interpretation by the consumer.

    Bucket padding is trailing fill appended to reach a bucket size, marked by a
    ``PADDING`` span: ``length`` is the padded height, ``unpadded_length`` the real
    content. No other kind covers it, so consumers slicing by span never see it.

    Spans are the only record of which columns of a frame carry a real token --
    a zero is equally a genuine id or the grid's fill. Backends needing a
    per-channel mask derive one from spans and ``layout``; none is stored.
    """

    tokens: Any
    spans: list[TokenSequenceSpan]
    layout: TokenizedSequenceLayout
    #: The source :class:`Segment`'s id. Loader-supplied, so it cannot be
    #: reconstructed from the sequence's own geometry.
    seq_id: str = ""

    @property
    def length(self) -> int:
        """Padded height of ``tokens`` (== ``unpadded_length`` when unpadded)."""
        return int(self.tokens.shape[0])

    @property
    def padding(self) -> int:
        """Trailing bucket-fill frames."""
        return sum(span.end - span.start for span in self.spans_of(TokenSpanKind.PADDING))

    @property
    def unpadded_length(self) -> int:
        """Real length, ignoring bucket padding."""
        return self.length - self.padding

    def validate(self, token_ranges: "TokenSpanIdRange | None" = None) -> None:
        """Check the ``(L, C+1)`` grid, treating spans as the source of truth.

        Structural by default: shapes, padding, and spans ordered,
        non-overlapping, and clear of padding. With ``token_ranges``, also
        checks token values against what each span kind claims to hold --
        the only way to catch a bad boundary frame, since id 0 is both a
        real EOS and the grid's "no token here" fill.
        """
        expected_channels = self.layout.num_codebooks + 1
        tokens = _as_numpy(self.tokens)
        if tokens.ndim != 2:
            raise ValueError(
                f"Expected sequence tokens shaped (L, C+1), got {tokens.shape}"
            )
        if int(tokens.shape[1]) != expected_channels:
            raise ValueError(
                f"Expected {expected_channels} sequence channels for "
                f"{self.layout.num_codebooks} codebooks, got {tokens.shape[1]}"
            )

        padding_spans = self.spans_of(TokenSpanKind.PADDING)
        if len(padding_spans) > 1:
            raise ValueError(
                f"Expected at most one PADDING span, got {len(padding_spans)}"
            )
        if padding_spans:
            pad = padding_spans[0]
            if (pad.start, pad.end) != (self.unpadded_length, self.length):
                raise ValueError(
                    f"PADDING span [{pad.start}, {pad.end}) must be the trailing "
                    f"[{self.unpadded_length}, {self.length}) frames"
                )
            if np.any(tokens[pad.start :] != 0):
                raise ValueError("Padded frames must have zero tokens")

        previous_end = 0
        for span in sorted(self.spans, key=lambda item: item.start):
            if not 0 <= span.start < span.end <= self.length:
                raise ValueError(f"Invalid sequence span {span}")
            # Every kind but PADDING covers real frames only.
            if span.kind is not TokenSpanKind.PADDING and span.end > self.unpadded_length:
                raise ValueError(
                    f"Span {span.kind.value} [{span.start}, {span.end}) runs into "
                    f"bucket padding, which starts at {self.unpadded_length}"
                )
            if span.start < previous_end:
                raise ValueError(
                    f"Span {span.kind.value} [{span.start}, {span.end}) overlaps "
                    f"the previous span, which ends at {previous_end}"
                )
            previous_end = span.end

            if token_ranges is None:
                continue
            bounds = token_ranges.get(span.kind)
            if bounds is None:
                continue
            low, high = bounds
            # Check only the channels the kind names; the other one is
            # zero-filled and would drag every lower bound down to 0.
            # PREFIX and SPECIAL name neither, so they cover all channels.
            if span.kind.value.endswith("text"):
                channels = [self.layout.text_column]
            elif span.kind.value.endswith("audio"):
                channels = list(self.layout.audio_channels)
            else:
                channels = list(range(expected_channels))
            block = tokens[span.start : span.end][:, channels]
            if block.size and (block.min() < low or block.max() >= high):
                raise ValueError(
                    f"Span {span.kind.value} [{span.start}, {span.end}) holds "
                    f"tokens in [{int(block.min())}, {int(block.max())}] on "
                    f"channels {channels}, outside the declared range "
                    f"[{low}, {high})"
                )

    def spans_of(self, kind: TokenSpanKind | str) -> list[TokenSequenceSpan]:
        kind = TokenSpanKind(kind)
        return [span for span in self.spans if span.kind == kind]

    @staticmethod
    def save_all(
        directory: str | Path,
        sequences: Sequence[TokenizedSequence],
        *,
        metadata: dict[str, Any],
    ) -> Path:
        """Write ``sequences.pt`` (tokens) and ``metadata.json`` (spans/layout).

        ``layout`` is written once at the top level; it is constant for a model.
        """
        if not sequences:
            raise ValueError("Cannot save an empty sequence list")
        layout = sequences[0].layout
        if any(item.layout != layout for item in sequences):
            raise ValueError("All sequences in a row must share one layout")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            [{"tokens": _as_torch(item.tokens).long()} for item in sequences],
            directory / "sequences.pt",
        )
        # Drop legacy split artifacts from earlier dataprep layouts.
        for stale in ("masks.pt", "codebooks.pt"):
            (directory / stale).unlink(missing_ok=True)
        payload = {
            **metadata,
            "layout": layout.to_dict(),
            "sequences": [
                {
                    "sequence_id": index,
                    "seq_id": item.seq_id,
                    "sequence_length": item.length,
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
        directory = Path(directory)
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        rows = torch.load(
            directory / "sequences.pt", map_location="cpu", weights_only=True
        )
        if len(rows) != len(metadata["sequences"]):
            raise ValueError(f"Inconsistent tokenized artifacts in {directory}")

        layout = TokenizedSequenceLayout.from_dict(metadata["layout"])
        sequences = []
        for row, sequence_meta in zip(rows, metadata["sequences"]):
            spans = [
                TokenSequenceSpan.from_dict(payload)
                for payload in sequence_meta["spans"]
            ]
            sequence = TokenizedSequence(
                tokens=row["tokens"],
                spans=spans,
                layout=layout,
                # Absent in artifacts written before ids became explicit.
                seq_id=str(sequence_meta.get("seq_id", "")),
            )
            sequence.validate()
            sequences.append(sequence)
        return sequences, metadata

@dataclass
class SequenceEmbeddingContext:
    """Precomputed continuous conditioning for one sequence.

    Some models condition on vectors rather than tokens, and those have no
    integer id to occupy a column of the ``(L, C+1)`` token grid. Chatterbox is the
    motivating case: a 256-d GE2E speaker embedding. Keeping them out of
    :class:`TokenizedSequence` leaves it an integer token array for every
    backend, rather than widening its dtype or adding a float channel that
    ``validate`` cannot check.

    ``values`` is free-form so a backend can add a field without touching this
    class.
    """

    values: dict[str, Any]  # chatterbox saves {'speaker_emb': (256,) tensor}

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def __contains__(self, key: str) -> bool:
        return key in self.values

    @staticmethod
    def save_all(
        directory: str | Path,
        contexts: Sequence["SequenceEmbeddingContext | None"],
    ) -> Path | None:
        """Write ``embedding_context.pt``, or remove it when nothing is stored.

        Indexed positionally against the sequences saved alongside. A backend
        with no continuous conditioning writes no file, so this stays invisible
        to models that do not need it.
        """
        directory = Path(directory)
        path = directory / "embedding_context.pt"
        if not any(item is not None for item in contexts):
            path.unlink(missing_ok=True)
            return None
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(
            [
                None
                if item is None
                else {key: _as_torch(value) for key, value in item.values.items()}
                for item in contexts
            ],
            path,
        )
        return path

    @staticmethod
    def load_all(
        directory: str | Path, *, count: int
    ) -> list["SequenceEmbeddingContext | None"]:
        """Read ``embedding_context.pt``; all-``None`` when the file is absent."""
        path = Path(directory) / "embedding_context.pt"
        if not path.exists():
            return [None] * count
        rows = torch.load(path, map_location="cpu", weights_only=True)
        if len(rows) != count:
            raise ValueError(
                f"{path} holds {len(rows)} contexts for {count} sequences"
            )
        return [
            None if row is None else SequenceEmbeddingContext(values=dict(row))
            for row in rows
        ]

@dataclass
class FeaturizedSequence:
    """Teacher-forced outputs aligned to a ``TokenizedSequence`` of length ``L``.

    ``hiddens`` and each ``logits[k]`` have length ``L - 1``. Index ``i`` holds
    model outputs when processing ``tokens[i]``, i.e. the distribution / state used to
    predict ``tokens[i + 1]``.

    For example, given a span of audio tokens ``[s, e)`` in the tokenized sequence,
    corresponding features predicting it are ``[s - 1, e - 1)`` in featurized sequence.

    When the source sequence is bucket-padded, input/output alignment remains the same.
    """

    logits: dict[int, Any]
    hiddens: Any
    spans: list[TokenSequenceSpan]
    layout: TokenizedSequenceLayout
    kv_cache: Any | None = None

    @property
    def length(self) -> int:
        """Feature length ``L - 1``, including any bucket padding."""
        return int(self.hiddens.shape[0])

    @property
    def padding(self) -> int:
        """Trailing bucket-fill frames.

        Spans are carried over unshifted, so this counts padded *tokens*; the
        ``L - 1`` truncation drops a real frame, not a padded one.
        """
        return sum(span.end - span.start for span in self.spans_of(TokenSpanKind.PADDING))

    @property
    def unpadded_length(self) -> int:
        """Feature length excluding bucket padding."""
        return self.length - self.padding

    def feature_layout(self) -> TokenizedSequenceLayout:
        """Source layout with the model's hidden/logit widths filled in."""
        return replace(
            self.layout,
            hidden_dim=int(self.hiddens.shape[-1]),
            logit_dims=tuple(
                int(self.logits[index].shape[-1])
                for index in range(self.layout.num_codebooks)
            ),
        )

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

    def spans_of(self, kind: TokenSpanKind | str) -> list[TokenSequenceSpan]:
        kind = TokenSpanKind(kind)
        return [span for span in self.spans if span.kind == kind]

    @staticmethod
    def save_all(
        directory: str | Path,
        sequences: Sequence[FeaturizedSequence],
        *,
        metadata: dict[str, Any],
    ) -> Path:
        """Write ``features.pt`` and ``metadata.json`` (top-level layout, per-sequence spans).

        The saved layout carries the model's ``hidden_dim`` / ``logit_dims``, read
        off the first sequence since both are constant across a row.
        """
        if not sequences:
            raise ValueError("Cannot save an empty sequence list")
        layout = sequences[0].feature_layout()
        if any(item.feature_layout() != layout for item in sequences):
            raise ValueError("All sequences in a row must share one layout")

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
            "layout": layout.to_dict(),
            "sequences": [
                {
                    "sequence_id": index,
                    "sequence_length": item.length,
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
    ) -> tuple[list[FeaturizedSequence], dict[str, Any]]:
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

        layout = TokenizedSequenceLayout.from_dict(metadata["layout"])
        sequences = []
        for row, kv_cache, sequence_meta in zip(rows, kv_rows, metadata["sequences"]):
            item = FeaturizedSequence(
                logits=row["logits"],
                hiddens=row["hiddens"],
                spans=[
                    TokenSequenceSpan.from_dict(payload)
                    for payload in sequence_meta["spans"]
                ],
                layout=layout,
                kv_cache=kv_cache,
            )
            item.validate()
            sequences.append(item)
        return sequences, metadata

@dataclass
class ShardSample:
    """One sequence as it is stored in a WebDataset shard.

    The arrays are kept as encoded ``.npy`` bytes rather than live tensors: the
    exporter holds a shuffle buffer of these, and fp16 bytes are far smaller than
    the fp32 tensors they came from. ``to_wds`` renders the wire format
    ``webdataset.TarWriter`` expects -- a flat dict of ``{key}.{ext}`` -- so the
    field-per-part shape stays in Python and only the boundary deals in dicts.
    """

    #: The source :class:`Segment`'s id -- the sole identity, and what a
    #: resumed run skips on.
    seq_id: str
    #: Native speaker label, so a consumer can group or audit the train/val
    #: split without re-deriving it.
    speaker: str
    audio_frames: int
    frame_rate: float
    model: str
    hidden_dim: int
    num_codebooks: int
    hiddens: bytes
    targets: bytes
    #: Teacher per-head logits; present only when the run asked for them.
    logits: bytes | None = None
    vocab_size: int | None = None
    #: Which token column each head is scored against. A consumer needs it to
    #: reproduce :func:`audio_frame_metrics`.
    head_targets: list[int] | None = None
    #: Per-layer KV slices over the audio span, when the model exposed a cache.
    kv: bytes | None = None

    @property
    def key(self) -> str:
        """WDS sample key: the segment id, sanitized for a tar member name."""
        return _wds_key(self.seq_id)

    def meta(self) -> dict:
        """Provenance written as ``{key}.meta.json``."""
        meta = {
            "seq_id": self.seq_id,
            "speaker": self.speaker,
            "audio_frames": self.audio_frames,
            "frame_rate": self.frame_rate,
            "model": self.model,
            "hidden_dim": self.hidden_dim,
            "num_codebooks": self.num_codebooks,
            "has_kv": self.kv is not None,
            "has_logits": self.logits is not None,
        }
        if self.logits is not None:
            meta["vocab_size"] = self.vocab_size
            meta["head_targets"] = list(self.head_targets or [])
        return meta

    def to_wds(self) -> dict:
        """Flatten to the ``{key}.{ext}`` dict ``webdataset.TarWriter`` writes."""
        sample = {
            "__key__": self.key,
            "hiddens.npy": self.hiddens,
            "targets.npy": self.targets,
            "meta.json": json.dumps(self.meta()).encode(),
        }
        if self.logits is not None:
            sample["logits.npy"] = self.logits
        if self.kv is not None:
            sample["kv.npy"] = self.kv
        return sample
