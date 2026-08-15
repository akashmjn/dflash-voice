"""Backwards-compatible re-exports.

The records moved to :mod:`dataprep.types` and their helpers to
:mod:`dataprep.utils`. Importing from here still works; prefer the specific
module in new code.
"""

from __future__ import annotations

from dataprep.types import (
    FeaturizedSequence,
    Segment,
    SequenceEmbeddingContext,
    ShardSample,
    TokenizedSequence,
    TokenizedSequenceLayout,
    TokenSequenceSpan,
    TokenSpanIdRange,
    TokenSpanKind,
)
from dataprep.utils import (
    NATS_TO_BITS,
    _as_numpy,
    _as_torch,
    _as_torch_tree,
    _wds_key,
    audio_frame_metrics,
    bucket_length,
    nll_summary,
)

__all__ = [
    "FeaturizedSequence",
    "NATS_TO_BITS",
    "Segment",
    "SequenceEmbeddingContext",
    "ShardSample",
    "TokenSequenceSpan",
    "TokenSpanIdRange",
    "TokenSpanKind",
    "TokenizedSequence",
    "TokenizedSequenceLayout",
    "audio_frame_metrics",
    "bucket_length",
    "nll_summary",
]
