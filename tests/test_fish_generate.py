import os

import numpy as np
import pytest

MODEL_ID = "mlx-community/fish-audio-s2-pro-8bit"
TEST_TEXT = "[excited] Hello, this is a quick Fish Audio test on Apple Silicon."


def _skip_integration():
    if os.environ.get("DFLASH_VOICE_SKIP_INTEGRATION") == "1":
        pytest.skip("DFLASH_VOICE_SKIP_INTEGRATION=1")


@pytest.fixture(scope="module")
def gen_kwargs():
    return dict(
        text=TEST_TEXT,
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
    from dflash_voice.tts_mlx.fish import load_model

    return load_model(MODEL_ID)


@pytest.mark.integration
def test_non_stream_matches_mlx_audio(ref_model, our_model, gen_kwargs):
    import mlx.core as mx

    mx.random.seed(42)
    ref = list(ref_model.generate(**gen_kwargs, stream=False))
    mx.random.seed(42)
    ours = list(our_model.generate(**gen_kwargs, stream=False))

    assert len(ref) == 1
    assert len(ours) == 1

    ref_result = ref[0]
    our_result = ours[0]

    ref_audio = np.array(ref_result.audio)
    our_audio = np.array(our_result.audio)

    assert ref_audio.shape == our_audio.shape
    np.testing.assert_allclose(our_audio, ref_audio, atol=1e-4)

    assert our_result.sample_rate == 44100
    assert our_result.samples > 0
    assert our_result.token_count > 0
    assert our_result.audio.shape[0] == our_result.samples
