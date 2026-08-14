"""Guards in ``ChatterboxFeaturizer.featurize`` that the golden cannot reach.

``test_featurize_segment0[chatterbox]`` covers alignment by scoring it -- an
off-by-one moves the NLL. These two cases raise before any of that, so they need
a fake model rather than the real one.
"""

from __future__ import annotations

import pytest
import torch

from dataprep.common import (
    SequenceEmbeddingContext,
    TokenSpanKind,
    TokenizedSequence,
    TokenizedSequenceLayout,
    TokenSequenceSpan,
)

from dataprep.tests.conftest import BACKEND_REQUIREMENTS

# Must stop collection before the chatterbox import below, so importorskip
# rather than the conftest hook (which runs after the module is imported).
pytest.importorskip(
    BACKEND_REQUIREMENTS["chatterbox"][0],
    reason=f"needs the {BACKEND_REQUIREMENTS['chatterbox'][1]} extra",
)

from chatterbox.models.t3.modules.t3_config import T3Config  # noqa: E402

from dataprep.chatterbox import COND_PREFIX_LEN, ChatterboxFeaturizer  # noqa: E402

HIDDEN_DIM = 8
TEXT_LEN = 5
AUDIO_FRAMES = 7


class _FakeT3:
    """Zero embeddings of the right width; enough to reach the guards."""

    def __init__(self, cond_len: int = COND_PREFIX_LEN):
        self.hp = T3Config.english_only()
        self._cond_len = cond_len

    def prepare_conditioning(self, cond):
        return torch.zeros(1, self._cond_len, HIDDEN_DIM)

    def _zeros(self, tokens):
        return torch.zeros(1, tokens.shape[1], HIDDEN_DIM)

    text_emb = text_pos_emb = speech_emb = speech_pos_emb = _zeros

    def tfmr(self, *, inputs_embeds, use_cache, return_dict):
        hidden = torch.zeros(1, inputs_embeds.shape[1], HIDDEN_DIM)
        return type("Out", (), {"last_hidden_state": hidden})()

    def speech_head(self, hidden):
        return hidden.new_zeros(hidden.shape[0], self.hp.speech_tokens_dict_size)


@pytest.fixture
def sequence():
    """A minimal ``[cond | SOT text EOT | SOS y EOS]`` grid."""
    hp = T3Config.english_only()
    length = COND_PREFIX_LEN + (TEXT_LEN + 2) + (AUDIO_FRAMES + 2)
    tokens = torch.zeros(length, 2, dtype=torch.long)
    spans = []
    position = COND_PREFIX_LEN

    def add(start, end, kind):
        spans.append(
            TokenSequenceSpan(
                source_dataset_id=0, segment_id=0, start=start, end=end, kind=kind
            )
        )

    add(0, position, TokenSpanKind.PREFIX)
    tokens[position, -1] = hp.start_text_token
    add(position, position + 1, TokenSpanKind.BOS_TEXT)
    position += 1
    tokens[position : position + TEXT_LEN, -1] = torch.arange(1, TEXT_LEN + 1)
    add(position, position + TEXT_LEN, TokenSpanKind.TEXT)
    position += TEXT_LEN
    add(position, position + 1, TokenSpanKind.EOS_TEXT)  # EOT is id 0
    position += 1
    tokens[position, 0] = hp.start_speech_token
    add(position, position + 1, TokenSpanKind.BOS_AUDIO)
    position += 1
    tokens[position : position + AUDIO_FRAMES, 0] = torch.arange(10, 10 + AUDIO_FRAMES)
    add(position, position + AUDIO_FRAMES, TokenSpanKind.AUDIO)
    position += AUDIO_FRAMES
    tokens[position, 0] = hp.stop_speech_token
    add(position, position + 1, TokenSpanKind.EOS_AUDIO)

    result = TokenizedSequence(
        tokens=tokens,
        spans=spans,
        layout=TokenizedSequenceLayout(num_codebooks=1, text_channel=-1),
    )
    result.validate()
    return result


def test_wrong_conditioning_width_is_rejected(sequence):
    """A drifted COND_PREFIX_LEN must fail loudly, not silently misalign.

    Unreachable with the real model, whose prefix is always 34 wide -- which is
    the point: the assertion fires only once that stops being true.
    """
    featurizer = ChatterboxFeaturizer(model=_FakeT3(cond_len=COND_PREFIX_LEN - 1))
    context = SequenceEmbeddingContext(values={"speaker_emb": torch.zeros(256)})
    with pytest.raises(ValueError, match="COND_PREFIX_LEN"):
        featurizer.featurize(sequence, context=context)


def test_missing_speaker_embedding_is_rejected(sequence):
    featurizer = ChatterboxFeaturizer(model=_FakeT3())
    with pytest.raises(ValueError, match="speaker_emb"):
        featurizer.featurize(sequence, context=None)
    with pytest.raises(ValueError, match="speaker_emb"):
        featurizer.featurize(sequence, context=SequenceEmbeddingContext(values={}))
