import pytest

from dataprep.common import TokenSpanKind, TokenizedSequenceLayout
from dataprep.pipeline import load_tokenizer
from dataprep.tests.conftest import tokenize_segment


@pytest.mark.parametrize(
    "model",
    [
        # xfail on Apple silicon: Mimi's 30 s encode chunk is past what MPS
        # conv1d accepts. Not a tokenize bug, and it passes on CPU.
        pytest.param(
            "miso",
            marks=[pytest.mark.expensive, pytest.mark.xfail(
                raises=NotImplementedError, reason="MPS conv1d output cap"
            )],
        ),
        # Deprecated MLX backends: skipped by default, run with -m deprecated.
        pytest.param("qwen3", marks=pytest.mark.deprecated),
        pytest.param("fish", marks=pytest.mark.deprecated),
    ],
)
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

    audio_span = sequence.spans_of(TokenSpanKind.AUDIO)[0]
    assert audio_span.end - audio_span.start == expected["audio_frames"]


@pytest.mark.expensive
def test_tokenize_segment0_chatterbox(segment0, expected_tokenized):
    """Ids match the gold dumped from upstream chatterbox-flash.

    ``fixtures/segment0/expected/chatterbox.json`` comes from the upstream
    tokenizer rather than ``dataprep.chatterbox``, so it pins the ids
    themselves, not just the layout we built around them.
    """
    pytest.importorskip("chatterbox", reason="needs the dataprep-chatterbox extra")
    expected = expected_tokenized["chatterbox"]
    tokenizer = load_tokenizer("chatterbox")
    sequence = tokenize_segment(segment0, tokenizer)
    sequence.validate(tokenizer.span_token_ranges)

    assert sequence.length == expected["length"]
    assert sequence.layout == TokenizedSequenceLayout.from_dict(expected["layout"])
    assert [(span.kind.value, span.start, span.end) for span in sequence.spans] == [
        tuple(span) for span in expected["spans"]
    ]
    assert list(sequence.tokens.shape) == expected["tokens_shape"]
    assert list(sequence.mask.shape) == expected["mask_shape"]

    text_span = sequence.spans_of(TokenSpanKind.TEXT)[0]
    audio_span = sequence.spans_of(TokenSpanKind.AUDIO)[0]
    assert audio_span.end - audio_span.start == expected["audio_frames"]
    assert (
        sequence.tokens[text_span.start : text_span.end, -1].tolist()
        == expected["text_tokens"]
    )
    assert (
        sequence.tokens[audio_span.start : audio_span.end, 0].tolist()
        == expected["speech_tokens"]
    )
