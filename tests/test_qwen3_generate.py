import os

import numpy as np
import pytest

MODEL_ID = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit"
TEST_TEXT = "Hello, this is a quick Qwen3 TTS test on Apple Silicon."
TEST_VOICE = "Ryan"
TEST_LANGUAGE = "english"


def _skip_integration():
    if os.environ.get("DFLASH_VOICE_SKIP_INTEGRATION") == "1":
        pytest.skip("DFLASH_VOICE_SKIP_INTEGRATION=1")


@pytest.fixture(scope="module")
def gen_kwargs():
    return dict(
        text=TEST_TEXT,
        voice=TEST_VOICE,
        language=TEST_LANGUAGE,
        temperature=0,
        top_k=0,
    )


@pytest.fixture(scope="module")
def ref_model():
    _skip_integration()
    from mlx_audio.tts.utils import load_model as load_ref

    return load_ref(MODEL_ID)


@pytest.fixture(scope="module")
def our_model():
    _skip_integration()
    from tts_mlx.qwen3 import load_model

    return load_model(MODEL_ID)


def _concat_stream(results):
    import mlx.core as mx

    if not results:
        return mx.zeros((0,), dtype=mx.float32)
    return mx.concatenate([r.audio for r in results], axis=0)


@pytest.mark.integration
def test_non_stream_matches_mlx_audio(ref_model, our_model, gen_kwargs):
    ref = list(ref_model.generate(**gen_kwargs, lang_code=TEST_LANGUAGE, stream=False))
    ours = list(our_model.generate(**gen_kwargs, stream=False))

    assert len(ref) == 1
    assert len(ours) == 1

    ref_result = ref[0]
    our_result = ours[0]

    ref_audio = np.array(ref_result.audio)
    our_audio = np.array(our_result.audio)

    assert ref_audio.shape == our_audio.shape
    np.testing.assert_allclose(our_audio, ref_audio, atol=1e-4)

    assert our_result.sample_rate == 24000
    assert our_result.samples > 0
    assert our_result.token_count > 0
    assert our_result.audio.shape[0] == our_result.samples


@pytest.mark.integration
def test_stream_matches_mlx_audio(ref_model, our_model, gen_kwargs):
    stream_kwargs = dict(
        gen_kwargs,
        stream=True,
        streaming_interval=0.32,
    )

    ref = list(ref_model.generate(**stream_kwargs, lang_code=TEST_LANGUAGE))
    ours = list(our_model.generate(**stream_kwargs))

    ref_audio = np.array(_concat_stream(ref))
    our_audio = np.array(_concat_stream(ours))

    assert ref_audio.shape == our_audio.shape
    assert ref_audio.size > 0
    np.testing.assert_allclose(our_audio, ref_audio, atol=1e-4)
