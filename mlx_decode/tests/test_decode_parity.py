"""Parity tests: mlx_decode backends vs the mlx-audio implementations they port.

Parity covers both the generated audio and the wall-clock time to produce it: the
point of these ports is to be no slower than mlx-audio while staying bit-comparable.
"""

import time
from dataclasses import dataclass, field

import numpy as np
import pytest

pytestmark = pytest.mark.integration

# Our port may be faster, but should never be meaningfully slower than the reference.
MAX_SLOWDOWN = 1.10
# MLX's allocator leaves whichever model ran most recently at an advantage, so
# both sides get a quiet gap to settle before they are timed.
SETTLE_SECONDS = 2.0
WARMUP_TEXT = "Hello."


@dataclass(frozen=True)
class Backend:
    model_id: str
    loader: str
    sample_rate: int
    gen_kwargs: dict
    # mlx-audio's Qwen3 entrypoint takes the language as `lang_code`.
    ref_kwargs: dict = field(default_factory=dict)
    supports_streaming: bool = True
    # Fish falls back to a nonzero RAS temperature (see mlx_decode/fish.py), so its
    # decoding stays stochastic even at temperature=0 and needs a fixed seed.
    needs_seed: bool = False


BACKENDS = {
    "qwen3": Backend(
        model_id="mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        loader="mlx_decode.qwen3",
        sample_rate=24000,
        gen_kwargs=dict(
            text="Hello, this is a quick Qwen3 TTS test on Apple Silicon.",
            voice="Ryan",
            language="english",
            temperature=0,
            top_k=0,
        ),
        ref_kwargs=dict(lang_code="english"),
    ),
    "voxtral": Backend(
        model_id="mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
        loader="mlx_decode.voxtral",
        sample_rate=24000,
        gen_kwargs=dict(
            text="Hello, this is a quick Voxtral TTS test on Apple Silicon.",
            voice="casual_male",
        ),
        # The flow-matching head samples fresh noise per frame, so there is no
        # temperature to zero out -- pin the seed instead.
        needs_seed=True,
        supports_streaming=False,
    ),
    "fish": Backend(
        model_id="mlx-community/fish-audio-s2-pro-8bit",
        loader="mlx_decode.fish",
        sample_rate=44100,
        gen_kwargs=dict(
            text="[excited] Hello, this is a quick Fish Audio test on Apple Silicon.",
            temperature=0,
            top_k=0,
        ),
        supports_streaming=False,
        needs_seed=True,
    ),
}


@pytest.fixture(scope="module", params=list(BACKENDS), ids=list(BACKENDS))
def backend(request) -> Backend:
    return BACKENDS[request.param]


@pytest.fixture(scope="module")
def ref_model(backend):
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model

    model = load_model(backend.model_id)
    _warmup(model, backend, **backend.ref_kwargs)
    yield model
    # Backends are timed on wall-clock, so one backend's weights must not still
    # be resident while the next is measured.
    del model
    mx.clear_cache()


@pytest.fixture(scope="module")
def our_model(backend):
    import importlib

    import mlx.core as mx

    model = importlib.import_module(backend.loader).load_model(backend.model_id)
    _warmup(model, backend)
    yield model
    del model
    mx.clear_cache()


def _warmup(model, backend, **extra):
    """Compile MLX kernels so the first timed run doesn't also measure compilation."""
    kwargs = dict(backend.gen_kwargs, text=WARMUP_TEXT, stream=False, **extra)
    _run(model, backend, **kwargs)


def _run(model, backend, **kwargs):
    """Generate under a fixed seed, returning results, audio and wall-clock seconds."""
    import mlx.core as mx

    if backend.needs_seed:
        mx.random.seed(42)

    start = time.perf_counter()
    results = list(model.generate(**kwargs))
    # MLX is lazy, so the audio must be forced before the clock stops.
    mx.eval([r.audio for r in results])
    elapsed = time.perf_counter() - start

    audio = np.concatenate([np.array(r.audio) for r in results]) if results else None
    return results, audio, elapsed


def _timed_run(model, backend, **kwargs):
    """Let the machine settle, then time one generation."""
    import mlx.core as mx

    mx.clear_cache()
    time.sleep(SETTLE_SECONDS)
    return _run(model, backend, **kwargs)


def _time_parity(our_model, ref_model, backend, our_kwargs, ref_kwargs):
    """Time both implementations, each after its own settle gap."""
    return _timed_run(our_model, backend, **our_kwargs), _timed_run(
        ref_model, backend, **ref_kwargs
    )


def _assert_time_parity(our_seconds, ref_seconds):
    assert our_seconds <= ref_seconds * MAX_SLOWDOWN, (
        f"ours took {our_seconds:.2f}s vs mlx-audio {ref_seconds:.2f}s "
        f"({our_seconds / ref_seconds:.2f}x, limit {MAX_SLOWDOWN:.2f}x)"
    )


def test_non_stream_parity(backend, ref_model, our_model):
    kwargs = dict(backend.gen_kwargs, stream=False)
    ref_kwargs = dict(kwargs, **backend.ref_kwargs)
    (ours, our_audio, our_seconds), (ref, ref_audio, ref_seconds) = _time_parity(
        our_model, ref_model, backend, kwargs, ref_kwargs
    )

    assert len(ref) == 1
    assert len(ours) == 1
    assert our_audio.shape == ref_audio.shape
    np.testing.assert_allclose(our_audio, ref_audio, atol=1e-4)
    _assert_time_parity(our_seconds, ref_seconds)

    result = ours[0]
    assert result.sample_rate == backend.sample_rate
    assert result.samples > 0
    assert result.token_count > 0
    assert result.audio.shape[0] == result.samples


def test_stream_parity(backend, ref_model, our_model):
    if not backend.supports_streaming:
        pytest.skip(f"{backend.loader} does not implement streaming")

    kwargs = dict(backend.gen_kwargs, stream=True, streaming_interval=0.32)
    ref_kwargs = dict(kwargs, **backend.ref_kwargs)
    (_, our_audio, our_seconds), (_, ref_audio, ref_seconds) = _time_parity(
        our_model, ref_model, backend, kwargs, ref_kwargs
    )

    assert ref_audio.size > 0
    assert our_audio.shape == ref_audio.shape
    np.testing.assert_allclose(our_audio, ref_audio, atol=1e-4)
    _assert_time_parity(our_seconds, ref_seconds)
