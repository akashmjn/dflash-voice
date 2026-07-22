from dataprep.expresso import segment_frame_bounds
from dataprep.prepare import _chunk_segments
from dataprep.tokenizer import Segment


def test_segment_frame_bounds_use_floor_and_ceil():
    assert segment_frame_bounds(
        {"segment_id": 7, "start": 0.081, "end": 0.159},
        frame_rate=12.5,
        max_frames=100,
    ) == (1, 2)


def test_segment_chunking_respects_model_limit():
    class TwoSegmentTokenizer:
        def apply_chat_template(self, segments):
            if len(segments) > 2:
                raise ValueError("Sequence exceeds model limit")
            return tuple(segment.text for segment in segments)

    chunks = _chunk_segments(
        [Segment("a"), Segment("b"), Segment("c")],
        TwoSegmentTokenizer(),
    )
    assert chunks == [("a", "b"), ("c",)]
