import pytest

from dataprep.fish import FishTextTokenizer
from dataprep.qwen3 import Qwen3AudioCodec, Qwen3TextTokenizer


class FakeTextTokenizer:
    def encode(self, text):
        return [len(text), 9]

    def decode(self, token_ids):
        return ",".join(map(str, token_ids))


def test_text_adapters_delegate_and_qwen_requires_encoder():
    for adapter in (Qwen3TextTokenizer(FakeTextTokenizer()), FishTextTokenizer(FakeTextTokenizer())):
        assert adapter.encode("abc") == [3, 9]
        assert adapter.decode([3, 9]) == "3,9"

    with pytest.raises(ValueError, match="no speech encoder"):
        Qwen3AudioCodec(object())
