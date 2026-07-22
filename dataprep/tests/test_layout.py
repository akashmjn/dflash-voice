import numpy as np

from dataprep.tokenizer import SequenceSpan, TokenizedSequence, validate_sequence


def test_channel_first_sequence_layout():
    sequence = TokenizedSequence(
        tokens=np.zeros((3, 5), dtype=np.int64),
        mask=np.ones((3, 5), dtype=bool),
        spans=[SequenceSpan(0, 0, 2, "text"), SequenceSpan(0, 2, 5, "audio")],
    )

    validate_sequence(sequence, num_codebooks=2)
    assert sequence.length == 5
