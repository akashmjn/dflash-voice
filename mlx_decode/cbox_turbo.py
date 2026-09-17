"""Readable Chatterbox Turbo (T3) inference, timed per speech token.

Like Chatterbox AR, Turbo has no depth stage: T3 emits one speech token per step
at 25 Hz and S3Gen renders the finished sequence in one pass, so
``depth_audio_s`` is always zero and the S3Gen pass is charged to
``codec_decode_s``. It differs from AR in the backbone and the prompt:

- the backbone is GPT-2, not Llama. ``GPT2Model`` takes ``inputs_embeds``,
  returns ``(hidden, cache)``, and allocates its own ``KVCache`` list when
  passed ``cache=None``, so the cache is threaded from the prefill's return
  value rather than built with ``make_prompt_cache``.
- positions come from GPT-2's ``wpe``, keyed on ``cache[0].offset``. There is no
  ``speech_pos_emb`` to add per step.
- text tokens are raw GPT-2 BPE with no start/stop bookends.
- CFG is accepted and ignored, matching upstream: Turbo runs a batch of 1.

Ported from mlx-audio 0.4.4
(https://github.com/Blaizzy/mlx-audio, PyPI: mlx-audio==0.4.4),
from ``mlx_audio/tts/models/chatterbox_turbo/``:

- ``models/t3/t3.py::T3.inference_turbo`` prefill → ``_prepare_prompt``
- ``models/t3/t3.py::T3.inference_turbo`` token loop → ``_generate_tokens``
- ``chatterbox_turbo.py::ChatterboxTurboTTS.generate`` cleanup + s3gen
  → ``_decode_tokens``

Reference model: ``mlx-community/Chatterbox-Turbo-TTS-8bit``
"""

from __future__ import annotations

import time
from typing import Generator, List, Optional, Tuple

import mlx.core as mx

from mlx_decode._common import (
    GenerationProfile,
    GenerationResult,
    StepTiming,
    _make_result,
)

MLX_AUDIO_VERSION = "0.4.4"

# T3 emits speech tokens at 25 Hz; S3Gen renders them at 24 kHz.
TOKEN_RATE_HZ = 25.0

# FSQ codebook size. Tokens at or above this are the BOS/EOS bookends, which
# S3Gen cannot embed.
SPEECH_VOCAB_SIZE = 6561

# S3Gen is conditioned on a trailing silence, as upstream does.
N_TRAILING_SILENCE = 3

# Meanflow S3Gen converges in 2 CFM steps.
N_CFM_TIMESTEPS = 2


# ---------------------------------------------------------------------------
# Prompt construction (from T3.inference_turbo prefill)
# ---------------------------------------------------------------------------


def _prepare_prompt(model, text: str) -> Tuple[mx.array, list]:
    """Tokenize, prefill the GPT-2 backbone, and return (hidden, cache).

    The prompt is ``[cond | text | bos_speech]``. Upstream splits long text at
    sentence boundaries and runs this once per chunk; we keep a single chunk so
    that one prompt maps to one decode loop and the per-step timings stay
    comparable with the other models.
    """
    from mlx_audio.tts.models.chatterbox_turbo.chatterbox_turbo import punc_norm

    t3 = model.t3
    encoded = model.tokenizer(
        punc_norm(text), return_tensors="np", padding=True, truncation=True
    )
    text_tokens = mx.array(encoded.input_ids)

    bos = mx.full((text_tokens.shape[0], 1), t3.hp.start_speech_token, dtype=mx.int32)
    # Caches cond_prompt_speech_emb onto the shared T3Cond on first use, so the
    # 375-token conditioning prefix is embedded once per process, not per prompt.
    embeddings, _ = t3.prepare_input_embeds(
        t3_cond=model._conds.t3, text_tokens=text_tokens, speech_tokens=bos
    )
    return t3.tfmr(inputs_embeds=embeddings, cache=None)


# ---------------------------------------------------------------------------
# Token generation loop (from T3.inference_turbo)
# ---------------------------------------------------------------------------


def _generate_tokens(
    model,
    hidden: mx.array,
    cache: list,
    *,
    max_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
    profile: Optional[GenerationProfile] = None,
) -> List[int]:
    """Autoregressively sample speech tokens at 25 Hz.

    Every step is backbone work, so ``depth_audio_s`` stays zero -- sampling is
    counted as backbone time, matching how the other models charge their
    semantic step.
    """
    t3 = model.t3
    generated: List[int] = []

    # Upstream runs the backbone for token N+1 at the tail of step N; timing it
    # there would charge each step with its successor's forward pass. Deferring
    # the embedding moves that pass to the top of the step it belongs to. Step 0
    # is the exception: its logits come from the prefill, which is prompt cost
    # rather than step cost.
    pending_embedding: Optional[mx.array] = None

    for step in range(max_tokens):
        if profile is not None:
            t_step = time.perf_counter()

        if pending_embedding is not None:
            hidden, cache = t3.tfmr(inputs_embeds=pending_embedding, cache=cache)

        logits = t3.speech_head(hidden[:, -1:, :])

        # Upstream penalises against every token emitted so far, so the context
        # window grows with the sequence.
        history = mx.array([generated], dtype=mx.int32) if generated else None
        next_token = t3._sample_token(
            logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generated_tokens=history,
            repetition_penalty=repetition_penalty,
        )
        mx.eval(next_token)
        token_id = int(next_token[0, 0])
        generated.append(token_id)

        if profile is not None:
            backbone_semantic_s = time.perf_counter() - t_step

        if token_id == t3.hp.stop_speech_token:
            break

        pending_embedding = t3.speech_emb(next_token)

        if profile is not None:
            mx.eval(pending_embedding)
            profile.step_timings.append(
                StepTiming(
                    step_idx=step,
                    backbone_semantic_s=backbone_semantic_s,
                    depth_audio_s=0.0,
                    total_s=time.perf_counter() - t_step,
                )
            )

        if step % 50 == 0:
            mx.clear_cache()

    return generated


def _decode_tokens(model, token_ids: List[int]) -> mx.array:
    """Render speech tokens to a 24 kHz waveform with S3Gen."""
    from mlx_audio.tts.models.chatterbox_turbo.models.s3gen import S3GEN_SIL

    ids = [t for t in token_ids if t < SPEECH_VOCAB_SIZE]
    if not ids:
        return mx.zeros((0,), dtype=mx.float32)

    ids = ids + [S3GEN_SIL] * N_TRAILING_SILENCE

    mx.clear_cache()
    wav, _ = model.s3gen.inference(
        speech_tokens=mx.array([ids], dtype=mx.int32),
        ref_dict=model._conds.gen,
        n_cfm_timesteps=N_CFM_TIMESTEPS,
    )
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    mx.eval(wav)
    return wav


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ChatterboxTurbo:
    """Thin wrapper around mlx-audio's Chatterbox Turbo Model with a readable generate()."""

    def __init__(self, mlx_model):
        self._model = mlx_model

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def set_reference(self, path: str) -> float:
        """Encode a reference clip into voice conditionals; returns seconds taken.

        Overrides the conds.safetensors the checkpoint ships with.
        """
        t_start = time.perf_counter()
        self._model.prepare_conditionals(path)
        # MLX is lazy: without this the returned time excludes the actual encode.
        mx.eval(self._model._conds.t3.speaker_emb, self._model._conds.gen["prompt_feat"])
        return time.perf_counter() - t_start

    def generate(
        self,
        text: str,
        temperature: float = 0.8,
        top_k: int = 1000,
        top_p: float = 0.95,
        repetition_penalty: float = 1.2,
        max_tokens: int = 1000,
        profile: Optional[GenerationProfile] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text using the checkpoint's built-in voice.

        ``cfg_weight``, ``min_p`` and ``exaggeration`` are accepted and ignored,
        as upstream does -- Turbo supports none of them.
        """
        del kwargs

        model = self._model
        if getattr(model, "_conds", None) is None:
            raise ValueError(
                "No voice conditionals; the checkpoint should ship conds.safetensors."
            )

        start_time = time.perf_counter()
        mx.clear_cache()

        hidden, cache = _prepare_prompt(model, text)
        token_ids = _generate_tokens(
            model,
            hidden,
            cache,
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            profile=profile,
        )

        if profile is not None:
            t_decode = time.perf_counter()
            audio = _decode_tokens(model, token_ids)
            profile.codec_decode_s = time.perf_counter() - t_decode
            profile.num_steps = len(profile.step_timings)
        else:
            audio = _decode_tokens(model, token_ids)

        if audio.shape[0] == 0:
            return

        yield _make_result(
            model,
            audio,
            segment_idx=0,
            token_count=len(token_ids),
            start_time=start_time,
            profile=profile,
        )
        mx.clear_cache()


def load_model(model_id: str) -> ChatterboxTurbo:
    """Load a Chatterbox Turbo model via mlx-audio and wrap it for readable inference."""
    from mlx_audio.tts.utils import load_model as _load

    return ChatterboxTurbo(_load(model_id))
