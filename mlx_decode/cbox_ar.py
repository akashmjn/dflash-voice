"""Readable Chatterbox AR (T3) inference, timed per speech token.

Chatterbox has no depth stage: T3 is a single-codebook autoregressive Llama that
emits one speech token per step, and S3Gen converts the whole finished sequence
to a waveform in one pass. So ``depth_audio_s`` is always zero and the S3Gen
pass is charged to ``codec_decode_s`` -- unlike the RVQ and flow-matching models,
whose per-frame audio decoders run inside the loop. That makes this the control
case for the decode breakdown: all per-step time is backbone time.

Generation uses classifier-free guidance, so each step runs the backbone on a
batch of 2 (conditional + unconditional).

The mlx-community checkpoint ships no ``conds.safetensors``, so a reference clip
is required to condition generation. ``prepare_conditionals`` runs once at load
(see ``set_reference``) rather than per prompt, keeping voice encoding out of the
timed region.

Ported from mlx-audio 0.4.4
(https://github.com/Blaizzy/mlx-audio, PyPI: mlx-audio==0.4.4),
from ``mlx_audio/tts/models/chatterbox/``:

- ``t3/t3.py::T3.inference`` prefill → ``_prepare_prompt``
- ``t3/t3.py::T3.inference`` token loop → ``_generate_tokens``
- ``chatterbox.py::Model.generate`` token cleanup + s3gen → ``_decode_tokens``

Reference model: ``mlx-community/Chatterbox-TTS-8bit``
"""

from __future__ import annotations

import time
from typing import Generator, List, Optional

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler

from mlx_decode._common import (
    GenerationProfile,
    GenerationResult,
    StepTiming,
    _make_result,
)

MLX_AUDIO_VERSION = "0.4.4"

# T3 emits speech tokens at 25 Hz; S3Gen renders them at 24 kHz.
TOKEN_RATE_HZ = 25.0


# ---------------------------------------------------------------------------
# Prompt construction (from T3.inference prefill)
# ---------------------------------------------------------------------------


def _build_input_embeddings(t3, t3_cond, text_tokens: mx.array, cfg_weight: float) -> mx.array:
    """Assemble the [cond | text | bos] prefill batch.

    With CFG the second batch row is the unconditional branch: same conditioning
    and BOS, but zeroed text.
    """
    bos_token = mx.array([[t3.hp.start_speech_token]], dtype=mx.int32)
    bos_embed = t3.speech_emb(bos_token) + t3.speech_pos_emb.get_fixed_embedding(0)

    cond_emb = t3.prepare_conditioning(t3_cond)
    text_emb = t3.text_emb(text_tokens)

    if cfg_weight > 0.0:
        text_emb = mx.concatenate([text_emb[:1], mx.zeros_like(text_emb[:1])], axis=0)
        bos_embed = mx.concatenate([bos_embed, bos_embed], axis=0)

    if t3.hp.input_pos_emb == "learned":
        text_emb = text_emb + t3.text_pos_emb(text_tokens)

    if cond_emb.shape[0] != text_emb.shape[0]:
        cond_emb = mx.broadcast_to(cond_emb, (text_emb.shape[0],) + cond_emb.shape[1:])

    return mx.concatenate([cond_emb, text_emb, bos_embed], axis=1)


def _prepare_prompt(model, text: str, cfg_weight: float, cache) -> mx.array:
    """Tokenize, prefill the T3 backbone, and return the hidden state for step 0."""
    from mlx_audio.tts.models.chatterbox.chatterbox import punc_norm

    t3 = model.t3
    text_tokens = model.tokenizer.text_to_tokens(punc_norm(text))
    if cfg_weight > 0.0:
        text_tokens = mx.concatenate([text_tokens, text_tokens], axis=0)

    sot = mx.full((text_tokens.shape[0], 1), t3.hp.start_text_token, dtype=mx.int32)
    eot = mx.full((text_tokens.shape[0], 1), t3.hp.stop_text_token, dtype=mx.int32)
    text_tokens = mx.concatenate([sot, text_tokens, eot], axis=1)

    embeddings = _build_input_embeddings(t3, model._conds.t3, text_tokens, cfg_weight)
    return t3.tfmr.model(inputs=None, input_embeddings=embeddings, cache=cache)


# ---------------------------------------------------------------------------
# Token generation loop (from T3.inference)
# ---------------------------------------------------------------------------


def _generate_tokens(
    model,
    hidden: mx.array,
    cache,
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float,
    repetition_penalty: float,
    cfg_weight: float,
    profile: Optional[GenerationProfile] = None,
) -> List[int]:
    """Autoregressively sample speech tokens at 25 Hz.

    Every step is backbone work, so ``depth_audio_s`` stays zero -- sampling and
    the CFG mix are counted as backbone time, matching how the other models
    charge their semantic step.
    """
    t3 = model.t3
    sampler = make_sampler(temp=temperature, top_p=top_p, min_p=min_p)
    processors = make_logits_processors(
        logit_bias=None,
        repetition_penalty=repetition_penalty,
        repetition_context_size=max_tokens,
    )

    generated: List[int] = [t3.hp.start_speech_token]

    # Upstream runs the backbone for token N+1 at the tail of step N; timing it
    # there would charge each step with its successor's forward pass. Deferring
    # the embedding moves that pass to the top of the step it belongs to.
    pending_embedding: Optional[mx.array] = None

    for step in range(max_tokens):
        if profile is not None:
            t_step = time.perf_counter()

        if pending_embedding is not None:
            hidden = t3.tfmr.model(
                inputs=None, input_embeddings=pending_embedding, cache=cache
            )

        logits = t3.speech_head(hidden[:, -1:, :]).squeeze(1)
        if cfg_weight > 0.0 and logits.shape[0] > 1:
            cond, uncond = logits[0:1, :], logits[1:2, :]
            logits = cond + cfg_weight * (cond - uncond)
        else:
            logits = logits[0:1, :]

        for processor in processors:
            logits = processor(mx.array([generated], dtype=mx.int32), logits)

        next_token = sampler(logits)
        mx.eval(next_token)
        token_id = int(next_token[0])
        generated.append(token_id)

        if profile is not None:
            backbone_semantic_s = time.perf_counter() - t_step

        if token_id == t3.hp.stop_speech_token:
            break

        embed = t3.speech_emb(mx.array([[token_id]]))
        embed = embed + t3.speech_pos_emb.get_fixed_embedding(step + 1)
        pending_embedding = (
            mx.concatenate([embed, embed], axis=0) if cfg_weight > 0.0 else embed
        )

        if profile is not None:
            mx.eval(pending_embedding)
            profile.step_timings.append(
                StepTiming(
                    step_idx=len(generated) - 2,
                    backbone_semantic_s=backbone_semantic_s,
                    depth_audio_s=0.0,
                    total_s=time.perf_counter() - t_step,
                )
            )

        if step % 50 == 0:
            mx.clear_cache()

    return generated


def _decode_tokens(model, token_ids: List[int]) -> mx.array:
    """Render speech tokens to a 24 kHz waveform with S3Gen.

    Drops the BOS/EOS bookends and any token outside the S3 vocabulary, which
    S3Gen cannot embed.
    """
    from mlx_audio.tts.models.chatterbox.chatterbox import SPEECH_VOCAB_SIZE

    t3 = model.t3
    ids = [
        t
        for t in token_ids
        if t not in (t3.hp.start_speech_token, t3.hp.stop_speech_token)
        and t < SPEECH_VOCAB_SIZE
    ]
    if not ids:
        return mx.zeros((0,), dtype=mx.float32)

    mx.clear_cache()
    wav = model.s3gen(
        speech_tokens=mx.array([ids], dtype=mx.int32),
        ref_dict=model._conds.gen,
        finalize=True,
    )
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    mx.eval(wav)
    return wav


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class ChatterboxAR:
    """Thin wrapper around mlx-audio's Chatterbox Model with a readable generate()."""

    def __init__(self, mlx_model):
        self._model = mlx_model

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def set_reference(self, path: str, exaggeration: float = 0.1) -> None:
        """Compute the voice conditionals once, outside any timed region."""
        import soundfile as sf

        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        self._model._conds = self._model.prepare_conditionals(
            mx.array(audio), sr, exaggeration
        )

    def generate(
        self,
        text: str,
        temperature: float = 0.8,
        top_p: float = 1.0,
        min_p: float = 0.05,
        repetition_penalty: float = 1.2,
        cfg_weight: float = 0.5,
        max_tokens: int = 1000,
        profile: Optional[GenerationProfile] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text using the checkpoint's built-in voice."""
        del kwargs

        model = self._model
        if model._conds is None:
            raise ValueError(
                "No voice conditionals; the mlx-community checkpoint ships none, "
                "so call set_reference(path) before generate()."
            )

        start_time = time.perf_counter()
        mx.clear_cache()

        cache = make_prompt_cache(model.t3.tfmr)
        hidden = _prepare_prompt(model, text, cfg_weight, cache)

        token_ids = _generate_tokens(
            model,
            hidden,
            cache,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            cfg_weight=cfg_weight,
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


def load_model(model_id: str, ref_audio: Optional[str] = None) -> ChatterboxAR:
    """Load a Chatterbox model via mlx-audio and wrap it for readable inference.

    ``ref_audio`` is required unless the checkpoint carries ``conds.safetensors``.
    """
    from mlx_audio.tts.utils import load_model as _load

    wrapper = ChatterboxAR(_load(model_id))
    if ref_audio is not None:
        wrapper.set_reference(ref_audio)
    return wrapper
