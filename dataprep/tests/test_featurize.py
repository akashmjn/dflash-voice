import math

import pytest
import torch

from dataprep.common import TokenSpanKind, audio_frame_metrics, nll_summary
from dataprep.pipeline import load_tokenizer
from dataprep.tests.conftest import featurize_segment, tokenize_segment


@pytest.mark.parametrize(
    "model",
    [
        pytest.param("miso", marks=pytest.mark.expensive),
        pytest.param("chatterbox", marks=pytest.mark.expensive),
    ],
)
def test_featurize_segment0(segment0, model, expected_featurized):
    tokenizer = load_tokenizer(model)
    expected = expected_featurized[model]
    sequence = tokenize_segment(segment0, tokenizer)
    features = featurize_segment(segment0, tokenizer, sequence)
    features.validate(sequence_length=sequence.length)

    assert features.length == expected["sequence_length"]
    assert features.length == sequence.length - 1

    audio_span = sequence.spans_of(TokenSpanKind.AUDIO)[0]
    pred = features.feature_slice_for_targets(audio_span.start, audio_span.end)
    assert pred.stop - pred.start == expected["pred_slice_len"]

    # The serialized layout carries the model's constant hidden/logit widths.
    layout = features.feature_layout()
    assert layout.hidden_dim == expected["hidden_dim"]
    assert layout.logit_dims[0] == expected["logit_dim0"]
    assert len(layout.logit_dims) == expected["num_logits"]
    assert len(features.logits) == expected["num_logits"]

    # Score logits against the ground-truth codes to sanity check correctness
    metrics = audio_frame_metrics(features, sequence.tokens, layout.num_codebooks)
    nll = metrics["nll"].numpy()
    entropy = metrics["entropy"].numpy()
    assert nll.shape == (expected["nll_frames"], expected["num_logits"])

    # Sanity check: NLL must beat chance on every codebook. Chance is over the
    # full head, so it is loose where a head has unreachable ids (chatterbox
    # spans 8194 but only 0..6562 are valid targets).
    codebook_nll = nll.mean(axis=0)
    codebook_entropy = entropy.mean(axis=0)
    for index, (value, chance) in enumerate(
        zip(codebook_nll, (math.log(dim) for dim in layout.logit_dims))
    ):
        assert value < 0.9 * chance, (
            f"codebook {index} NLL {value:.3f} nats is not meaningfully better than "
            f"chance ({chance:.3f})"
        )
    # Sanity check: NLL must stay near predictive entropy. Above it means the
    # logits are scored against the wrong targets; below it means future context
    # leaks in.
    excess = codebook_nll - codebook_entropy
    worst = int(abs(excess).argmax())
    assert abs(excess[worst]) < 1.0, (
        f"codebook {worst} is confidently wrong: NLL {codebook_nll[worst]:.3f} "
        f"differs from entropy {codebook_entropy[worst]:.3f} by "
        f"{excess[worst]:+.3f} nats"
    )

    torch.testing.assert_close(
        torch.tensor(nll.mean(axis=0), dtype=torch.float32),
        torch.tensor(expected["codebook_nll_nats"], dtype=torch.float32),
        atol=3e-3,
        rtol=3e-3,
    )

    # nll_summary splits codebooks positionally ([:1] semantic, [1:] audio), so
    # at C=1 the audio group is empty and semantic == total. The chatterbox
    # fixture records that degenerate output rather than asserting on it.
    if layout.num_codebooks > 1:
        summary = nll_summary(nll, tokenizer.audio_codec.frame_rate)
        for group, values in expected["nll_summary"].items():
            for unit, value in values.items():
                assert summary[group][unit] == pytest.approx(value, rel=3e-3), (
                    f"{group}.{unit}"
                )
