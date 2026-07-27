import pytest

from dataprep.common import SpanKind
from dataprep.prepare import load_tokenizer
from dataprep.tests.conftest import tokenize_segment


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("miso", marks=pytest.mark.expensive),
        "qwen3",
        "fish",
    ],
)
def test_featurize_segment0(segment0, model, expected_featurized):
    tokenizer = load_tokenizer(model)
    expected = expected_featurized[model]
    sequence = tokenize_segment(segment0, tokenizer)
    features = tokenizer.featurizer.featurize(sequence)
    features.validate(sequence_length=sequence.length)

    assert features.length == expected["feature_length"]
    assert features.length == sequence.length - 1

    audio_span = sequence.spans_of(SpanKind.AUDIO)[0]
    pred = features.feature_slice_for_targets(audio_span.start, audio_span.end)
    assert pred.stop - pred.start == expected["pred_slice_len"]

    assert list(features.hiddens.shape) == expected["hiddens_shape"]
    assert list(features.logits[0].shape) == expected["logits0_shape"]
    assert len(features.logits) == expected["num_logits"]
