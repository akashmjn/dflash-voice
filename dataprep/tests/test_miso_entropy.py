import pytest
import torch

from dataprep.common import SpanKind
from dataprep.tests.conftest import tokenize_segment


def _entropy(logits):
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@pytest.mark.expensive
def test_miso_entropy_matches_fixture(segment0, miso_entropy_reference):
    from dataprep.prepare import load_tokenizer

    expected = miso_entropy_reference["segment"]
    tokenizer = load_tokenizer("miso")
    sequence = tokenize_segment(segment0, tokenizer)
    features = tokenizer.featurizer.featurize(sequence)

    audio_span = sequence.spans_of(SpanKind.AUDIO)[0]
    pred = features.feature_slice_for_targets(audio_span.start, audio_span.end)
    actual = torch.stack(
        [_entropy(features.logits[index][pred]) for index in range(32)],
        dim=1,
    )
    saved = torch.tensor(expected["codebook_entropy_per_frame"], dtype=torch.float32)

    assert actual.shape == saved.shape == (expected["num_frames"], 32)
    torch.testing.assert_close(actual, saved, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        actual.mean(dim=0),
        torch.tensor(expected["codebook_entropy"], dtype=torch.float32),
        atol=3e-3,
        rtol=3e-3,
    )
