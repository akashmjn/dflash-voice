from dataprep.prepare import (
    _build_sequences,
    _pack_segments,
    resolve_prepare_rows,
    segment_frame_bounds,
)
from dataprep.common import Segment
import pytest


def test_segment_frame_bounds_use_floor_and_ceil():
    assert segment_frame_bounds(
        {"segment_id": 7, "start": 0.081, "end": 0.159},
        frame_rate=12.5,
        max_frames=100,
    ) == (1, 2)


def test_default_builds_one_sequence_per_segment():
    class CountingTokenizer:
        def apply_chat_template(self, segments):
            return tuple(segment.text for segment in segments)

    sequences = _build_sequences(
        [Segment("a"), Segment("b"), Segment("c")],
        CountingTokenizer(),
    )
    assert sequences == [("a",), ("b",), ("c",)]


def test_pack_segments_respects_model_limit():
    class TwoSegmentTokenizer:
        def apply_chat_template(self, segments):
            if len(segments) > 2:
                raise ValueError("Sequence exceeds model limit")
            return tuple(segment.text for segment in segments)

    chunks = _pack_segments(
        [Segment("a"), Segment("b"), Segment("c")],
        TwoSegmentTokenizer(),
    )
    assert chunks == [("a", "b"), ("c",)]
    assert (
        _build_sequences(
            [Segment("a"), Segment("b"), Segment("c")],
            TwoSegmentTokenizer(),
            pack_segments=True,
        )
        == chunks
    )


def test_debug_mode_selects_first_rows_and_full_run_is_deferred():
    assert resolve_prepare_rows(debug=3) == [0, 1, 2]
    with pytest.raises(NotImplementedError, match="parquet"):
        resolve_prepare_rows(debug=None)
    with pytest.raises(ValueError, match="positive"):
        resolve_prepare_rows(debug=0)
