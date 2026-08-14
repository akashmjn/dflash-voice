import pytest
import torch

from dataprep.common import (
    TokenSequenceSpan,
    TokenSpanKind,
    TokenizedSequence,
    TokenizedSequenceLayout,
)
from dataprep.miso import _channel_mask
from dataprep.pipeline import load_tokenizer
from dataprep.tests.conftest import tokenize_segment


def test_miso_channel_mask_follows_spans():
    """EOS_AUDIO is the case worth pinning: all zeros, so only its span kind
    distinguishes it from the bucket padding that must stay dead.
    """
    layout = TokenizedSequenceLayout(num_codebooks=2, text_channel=-1)
    spans = [
        (0, 3, TokenSpanKind.TEXT),
        (3, 5, TokenSpanKind.AUDIO),
        (5, 6, TokenSpanKind.EOS_AUDIO),
        (6, 8, TokenSpanKind.PADDING),
    ]
    sequence = TokenizedSequence(
        tokens=torch.zeros(8, 3, dtype=torch.long),
        spans=[
            TokenSequenceSpan(
                source_dataset_id=0, segment_id=0, start=start, end=end, kind=kind
            )
            for start, end, kind in spans
        ],
        layout=layout,
    )
    sequence.validate()
    assert sequence.padding == 2
    assert sequence.unpadded_length == 6

    expected = torch.zeros(8, 3, dtype=torch.bool)
    expected[0:3, -1] = True  # text frames: text column only
    expected[3:6, :-1] = True  # audio + EOS_AUDIO: codebooks only
    assert torch.equal(_channel_mask(sequence), expected)


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
        pytest.param("chatterbox", marks=pytest.mark.expensive),
        # Deprecated MLX backends: skipped by default, run with -m deprecated.
        pytest.param("qwen3", marks=pytest.mark.deprecated),
        pytest.param("fish", marks=pytest.mark.deprecated),
    ],
)
def test_tokenize_segment0(segment0, model, expected_tokenized):
    expected = expected_tokenized[model]
    tokenizer = load_tokenizer(model)
    sequence = tokenize_segment(segment0, tokenizer)
    sequence.validate(getattr(tokenizer, "span_token_ranges", None))

    assert sequence.length == expected["length"]
    assert sequence.layout == TokenizedSequenceLayout.from_dict(expected["layout"])
    assert [(span.kind.value, span.start, span.end) for span in sequence.spans] == [
        tuple(span) for span in expected["spans"]
    ]
    assert list(sequence.tokens.shape) == expected["tokens_shape"]

    audio_span = sequence.spans_of(TokenSpanKind.AUDIO)[0]
    assert audio_span.end - audio_span.start == expected["audio_frames"]

    # Only chatterbox.json carries ids: dumped from the upstream tokenizer, so
    # it pins the ids themselves and not just the layout built around them.
    if "text_tokens" in expected:
        text_span = sequence.spans_of(TokenSpanKind.TEXT)[0]
        assert (
            sequence.tokens[text_span.start : text_span.end, -1].tolist()
            == expected["text_tokens"]
        )
        assert (
            sequence.tokens[audio_span.start : audio_span.end, 0].tolist()
            == expected["speech_tokens"]
        )
