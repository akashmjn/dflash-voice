import pytest

from dataprep.common import SpanKind, TokenizedSequenceLayout
from dataprep.prepare import load_tokenizer
from dataprep.tests.conftest import tokenize_segment


@pytest.mark.parametrize("model", ["miso", "qwen3", "fish"])
def test_tokenize_segment0(segment0, model, expected_tokenized):
    expected = expected_tokenized[model]
    sequence = tokenize_segment(segment0, load_tokenizer(model))
    sequence.validate()

    assert sequence.length == expected["length"]
    assert sequence.layout == TokenizedSequenceLayout.from_dict(expected["layout"])
    assert [(span.kind.value, span.start, span.end) for span in sequence.spans] == [
        tuple(span) for span in expected["spans"]
    ]
    assert list(sequence.tokens.shape) == expected["tokens_shape"]
    assert list(sequence.mask.shape) == expected["mask_shape"]

    audio_span = sequence.spans_of(SpanKind.AUDIO)[0]
    assert audio_span.end - audio_span.start == expected["audio_frames"]
