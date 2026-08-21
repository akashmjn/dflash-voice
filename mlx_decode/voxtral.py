"""Readable Voxtral-TTS inference (preset voice embeddings).

Voxtral's depth stage is a flow-matching acoustic head, not an autoregressive
RVQ chain: one frame is a semantic code (argmax off the backbone hidden state)
plus 36 acoustic codes emitted together by Euler-integrating a 3-layer
transformer. So ``depth_audio_s`` covers denoising steps over all codebooks at
once rather than per-codebook decodes. The head itself
(``FlowMatchingAudioTransformer.decode_one_frame``) is called as-is; model
weights and nn.Module architecture are loaded via mlx-audio, and this file owns
prompt construction, the frame autoregression loop, and codec decode.

Prompt encoding needs ``mistral-common[audio]`` on top of the ``mlx_decode`` extra.

Ported from mlx-audio 0.4.4
(https://github.com/Blaizzy/mlx-audio, PyPI: mlx-audio==0.4.4),
all from ``mlx_audio/tts/models/voxtral_tts/voxtral_tts.py``:

- ``Model.generate`` prefill → ``_prepare_prompt``
- ``Model.generate`` frame loop → ``_generate_frames``, ``VoxtralTTS.generate``
- ``Model.generate`` final ``audio_tokenizer.decode`` → ``_decode_frames``

Reference model: ``mlx-community/Voxtral-4B-TTS-2603-mlx-6bit``
"""

from __future__ import annotations

import time
from typing import Generator, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from mlx_decode._common import (
    GenerationProfile,
    GenerationResult,
    StepTiming,
    _make_result,
)

MLX_AUDIO_VERSION = "0.4.4"

DEFAULT_VOICE = "casual_male"

# Semantic codes 0 (empty_audio) and 1 (end_audio) terminate generation.
END_AUDIO_MAX_CODE = 1


# ---------------------------------------------------------------------------
# Prompt construction (from mlx_audio Model.generate prefill)
# ---------------------------------------------------------------------------


def _prepare_prompt(model, text: str, voice: str, cache) -> mx.array:
    """Prefill the backbone, returning the hidden state for the first frame.

    Two passes: the prompt, then a lone AUDIO token whose hidden state
    conditions the first codec frame.
    """
    backbone = model.language_model.model.model

    input_ids = mx.array(model._encode_text(text, voice))[None, :]
    input_embeddings = model._build_input_embeddings(input_ids, voice)
    backbone(input_ids, cache=cache, input_embeddings=input_embeddings)

    audio_token = mx.array([[model.config.audio_token_id]])
    audio_embed = model.language_model.embed_tokens(audio_token)
    return backbone(audio_token, cache=cache, input_embeddings=audio_embed)


def _next_frame_embedding(model, codes: mx.array) -> mx.array:
    """Sum a frame's 37 codes into one backbone input vector.

    Each codebook owns its own slice of the shared embedding table, so codes are
    shifted to global indices first.
    """
    global_codes = model._codes_to_global_indices(codes)
    code_embeddings = model.audio_codebook_embeddings["embeddings"](global_codes)
    return code_embeddings.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Frame generation loop (from mlx_audio Model.generate)
# ---------------------------------------------------------------------------


def _generate_frames(
    model,
    hidden: mx.array,
    cache,
    *,
    max_tokens: int = 4096,
    profile: Optional[GenerationProfile] = None,
) -> List[mx.array]:
    """Autoregressively generate 37-codebook frames at 12.5 Hz.

    No sampling settings: upstream decides the semantic code by argmax and the
    acoustic codes by flow matching, so there is no temperature/top-k here.
    """
    backbone = model.language_model.model.model
    audio_token = mx.array([[model.config.audio_token_id]])
    frames: List[mx.array] = []

    # Upstream runs the backbone for frame N+1 at the tail of step N; timing it
    # there would charge each step with its successor's forward pass. Deferring
    # the embedding moves that pass to the top of the step it belongs to.
    pending_embedding: Optional[mx.array] = None

    for step in range(max_tokens):
        if profile is not None:
            t_step = time.perf_counter()

        if pending_embedding is not None:
            hidden = backbone(
                audio_token, cache=cache, input_embeddings=pending_embedding
            )
        h_t = hidden[:, -1, :]

        if profile is not None:
            mx.eval(h_t)
            backbone_semantic_s = time.perf_counter() - t_step
            t_depth = time.perf_counter()

        # Semantic argmax + 36 acoustic codes from the flow-matching head.
        codes = model.acoustic_transformer.decode_one_frame(h_t)

        if profile is not None:
            mx.eval(codes)
            depth_audio_s = time.perf_counter() - t_depth

        if int(codes[0, 0]) <= END_AUDIO_MAX_CODE:
            break

        frames.append(codes[:, None, :])
        pending_embedding = _next_frame_embedding(model, codes)

        if profile is not None:
            mx.eval(pending_embedding)
            profile.step_timings.append(
                StepTiming(
                    step_idx=len(frames) - 1,
                    backbone_semantic_s=backbone_semantic_s,
                    depth_audio_s=depth_audio_s,
                    total_s=time.perf_counter() - t_step,
                )
            )

        if step % 50 == 0:
            mx.clear_cache()

    return frames


def _decode_frames(model, frames: List[mx.array]) -> mx.array:
    """Decode stacked codec frames to a 24 kHz waveform."""
    if not frames:
        return mx.zeros((0,), dtype=mx.float32)

    audio = model.audio_tokenizer.decode(mx.concatenate(frames, axis=1)).squeeze(0)
    mx.eval(audio)
    return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class VoxtralTTS:
    """Thin wrapper around mlx-audio's Voxtral Model with a readable generate()."""

    def __init__(self, mlx_model):
        self._model = mlx_model

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def generate(
        self,
        text: str,
        voice: str = DEFAULT_VOICE,
        max_tokens: int = 4096,
        profile: Optional[GenerationProfile] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text using a preset voice embedding."""
        del kwargs

        model = self._model
        start_time = time.perf_counter()

        cache = make_prompt_cache(model.language_model.model)
        hidden = _prepare_prompt(model, text, voice, cache)

        frames = _generate_frames(
            model, hidden, cache, max_tokens=max_tokens, profile=profile
        )
        if not frames:
            return

        if profile is not None:
            t_decode = time.perf_counter()
            audio = _decode_frames(model, frames)
            profile.codec_decode_s = time.perf_counter() - t_decode
            profile.num_steps = len(frames)
        else:
            audio = _decode_frames(model, frames)

        yield _make_result(
            model,
            audio,
            segment_idx=0,
            token_count=len(frames),
            start_time=start_time,
            profile=profile,
        )
        mx.clear_cache()


def load_model(model_id: str) -> VoxtralTTS:
    """Load a Voxtral-TTS model via mlx-audio and wrap it for readable inference."""
    from mlx_audio.tts.utils import load_model as _load

    return VoxtralTTS(_load(model_id))
