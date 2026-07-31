import pytest
import torch

from dataprep.common import SpanKind, audio_frame_metrics, nll_summary
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

    assert features.length == expected["sequence_length"]
    assert features.length == sequence.length - 1

    audio_span = sequence.spans_of(SpanKind.AUDIO)[0]
    pred = features.feature_slice_for_targets(audio_span.start, audio_span.end)
    assert pred.stop - pred.start == expected["pred_slice_len"]

    # The serialized layout carries the model's constant hidden/logit widths.
    layout = features.feature_layout()
    assert layout.hidden_dim == expected["hidden_dim"]
    assert layout.logit_dims[0] == expected["logit_dim0"]
    assert len(layout.logit_dims) == expected["num_logits"]
    assert len(features.logits) == expected["num_logits"]

    # Score the logits against the ground-truth codes. This pins the numerics of
    # the forward pass, not just its shapes, and because the target column comes
    # from layout.head_targets it also guards against incorrect mapping of columns
    metrics = audio_frame_metrics(features, sequence.tokens, layout.num_codebooks)
    nll = metrics["nll"].numpy()
    assert nll.shape == (expected["nll_frames"], expected["num_logits"])

    torch.testing.assert_close(
        torch.tensor(nll.mean(axis=0), dtype=torch.float32),
        torch.tensor(expected["codebook_nll_nats"], dtype=torch.float32),
        atol=3e-3,
        rtol=3e-3,
    )

    summary = nll_summary(nll, tokenizer.audio_codec.frame_rate)
    for group, values in expected["nll_summary"].items():
        for unit, value in values.items():
            assert summary[group][unit] == pytest.approx(value, abs=3e-3), (
                f"{group}.{unit}"
            )
