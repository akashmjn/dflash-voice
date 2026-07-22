import numpy as np
import pytest

from dataprep.tokenizer import SequenceSpan, TokenizedSequence, validate_sequence


def test_seq_channel_layout_and_mask_consistency():
    tokens = np.zeros((5, 3), dtype=np.int64)
    mask = np.zeros((5, 3), dtype=bool)
    tokens[:2, -1] = [11, 12]
    mask[:2, -1] = True
    tokens[2:, :-1] = [[1, 2], [3, 4], [5, 6]]
    mask[2:, :-1] = True

    sequence = TokenizedSequence(
        tokens=tokens,
        mask=mask,
        spans=[SequenceSpan(0, 0, 2, "text"), SequenceSpan(0, 2, 5, "audio")],
    )
    validate_sequence(sequence, num_codebooks=2, text_channel=-1)
    assert sequence.length == 5


def test_mask_rejects_audio_tokens_outside_audio_mask():
    tokens = np.zeros((3, 3), dtype=np.int64)
    mask = np.zeros((3, 3), dtype=bool)
    tokens[:, -1] = [1, 2, 3]
    mask[:, -1] = True
    tokens[1, 0] = 9
    with pytest.raises(ValueError, match="Audio-channel tokens must be zero"):
        validate_sequence(
            TokenizedSequence(tokens=tokens, mask=mask, spans=[SequenceSpan(0, 0, 3, "text")]),
            num_codebooks=2,
            text_channel=-1,
        )
