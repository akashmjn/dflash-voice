"""Guards in ``ChatterboxFeaturizer.featurize`` that the golden cannot reach.

``test_featurize_segment0[chatterbox]`` covers alignment by scoring it -- an
off-by-one moves the NLL. These cases inspect or raise before any of that, so
they need a fake model rather than the real one.
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

from dataprep.chatterbox import cond_prefix_len, ChatterboxFeaturizer  # noqa: E402

HIDDEN_DIM = 8
TEXT_LEN = 5
AUDIO_FRAMES = 7
COND_PREFIX_LEN = cond_prefix_len(T3Config.english_only(), AUDIO_FRAMES)

class _FakeT3:
    """Zero embeddings of the right width; enough to reach the guards."""

    def __init__(self, cond_len: int = COND_PREFIX_LEN):
        self.hp = T3Config.english_only()
        self._cond_len = cond_len
        self.cond_prompt = None

    def prepare_conditioning(self, cond):
        self.cond_prompt = cond.cond_prompt_speech_tokens
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
                start=start, end=end, kind=kind
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
    with pytest.raises(ValueError, match="embeddings length"):
        featurizer.featurize(sequence, context=context)


def test_short_utterance_prompt_stops_at_the_audio_span(sequence):
    """An utterance shorter than ``speech_cond_prompt_len`` must not condition on
    what follows its audio, which under bucketing is padding zeros.
    """
    hp = T3Config.english_only()
    assert AUDIO_FRAMES < hp.speech_cond_prompt_len, "fixture must be under the cap"

    padded = TokenizedSequence(
        tokens=torch.cat([sequence.tokens, torch.zeros(9, 2, dtype=torch.long)]),
        spans=[
            *sequence.spans,
            TokenSequenceSpan(
                start=sequence.length,
                end=sequence.length + 9,
                kind=TokenSpanKind.PADDING,
            ),
        ],
        layout=sequence.layout,
    )
    padded.validate()

    audio = padded.spans_of(TokenSpanKind.AUDIO)[0]
    expected = padded.tokens[audio.start : audio.end, 0]

    model = _FakeT3()
    featurizer = ChatterboxFeaturizer(model=model)
    context = SequenceEmbeddingContext(values={"speaker_emb": torch.zeros(256)})
    featurizer.featurize(padded, context=context)

    assert torch.equal(model.cond_prompt[0].cpu(), expected)


def test_missing_speaker_embedding_is_rejected(sequence):
    featurizer = ChatterboxFeaturizer(model=_FakeT3())
    with pytest.raises(ValueError, match="speaker_emb"):
        featurizer.featurize(sequence, context=None)
    with pytest.raises(ValueError, match="speaker_emb"):
        featurizer.featurize(sequence, context=SequenceEmbeddingContext(values={}))
