import pytest

from dataprep.qwen3 import Qwen3AudioCodec


def test_qwen_requires_speech_encoder():
    with pytest.raises(ValueError, match="no speech encoder"):
        Qwen3AudioCodec(object())
