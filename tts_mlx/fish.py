"""Readable Fish Audio S2 Pro inference (text-to-speech + optional voice clone).

This module reimplements the mlx-audio Fish Speech **generate** path in a single,
annotated file. Model weights and nn.Module architecture are loaded via
mlx-audio; this file owns prompt construction, the DualAR autoregression loop,
and DAC decode.

Ported from mlx-audio 0.4.4
(https://github.com/Blaizzy/mlx-audio, PyPI: mlx-audio==0.4.4):

- ``Model._build_conversation`` / ``_prepare_reference_prompt`` → prompt helpers
  (``mlx_audio/tts/models/fish_qwen3_omni/fish_speech.py``)
- ``Model._sample_semantic`` / ``_sample_logits`` → ``_sample_semantic``,
  ``_sample_logits``
- ``Model._generate_codes_for_batch`` → ``_generate_codes``
- ``Model._decode_codes`` → ``_decode_codes``
- ``Model.generate`` → ``FishAudioTTS.generate``

Reference model: ``mlx-community/fish-audio-s2-pro-8bit``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import partial
from typing import Generator, List, Optional

import mlx.core as mx

from tts_mlx._common import GenerationResult, _make_result
from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
    Conversation,
    Message,
    TextPart,
    VQPart,
    group_turns_into_batches,
    split_text_by_speaker,
)
from mlx_audio.tts.models.fish_qwen3_omni.tokenizer import IM_END_TOKEN

MLX_AUDIO_VERSION = "0.4.4"

RAS_WIN_SIZE = 10
RAS_HIGH_TEMP = 1.0
RAS_HIGH_TOP_P = 0.9

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class TokenTiming:
    token_idx: int
    backbone_semantic_s: float
    depth_audio_s: float
    total_s: float


@dataclass
class GenerationProfile:
    token_timings: List[TokenTiming] = field(default_factory=list)
    codec_decode_s: float = 0.0
    num_tokens: int = 0


# ---------------------------------------------------------------------------
# Sampling (from mlx_audio Model._sample_logits / _sample_semantic)
# ---------------------------------------------------------------------------


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _sample_logits(
    logits: mx.array, temperature: float, top_p: float, top_k: int
) -> mx.array:
    if temperature <= 0:
        return mx.argmax(logits, axis=-1).astype(mx.int32)

    vocab_size = logits.shape[-1]
    if top_k <= 0 or top_k > vocab_size:
        top_k = vocab_size

    sorted_indices = mx.argsort(-logits, axis=-1)
    sorted_logits = mx.take_along_axis(logits, sorted_indices, axis=-1)
    cum_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)

    rank_indices = mx.arange(vocab_size, dtype=sorted_indices.dtype)
    if sorted_logits.ndim > 1:
        rank_indices = mx.broadcast_to(rank_indices, sorted_logits.shape)
    tokens_to_remove = (cum_probs > top_p) | (rank_indices >= top_k)
    tokens_to_remove[..., 0] = False

    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(vocab_size, dtype=sorted_indices.dtype),
        axis=-1,
    )
    tokens_to_remove = mx.take_along_axis(tokens_to_remove, inverse_indices, axis=-1)
    filtered_logits = mx.where(tokens_to_remove, -mx.inf, logits).astype(mx.float32)
    probs = mx.softmax(filtered_logits * (1.0 / max(temperature, 1e-5)), axis=-1)
    noise = -mx.log(mx.random.uniform(shape=probs.shape, low=1e-6, high=1.0))
    return mx.argmax(probs / noise, axis=-1).astype(mx.int32)


def _sample_semantic(
    model,
    logits: mx.array,
    previous_semantic_tokens: list[int],
    *,
    top_p: float,
    top_k: int,
    temperature: float,
) -> mx.array:
    """Sample a semantic token with RAS (Repetition-Aware Sampling)."""
    if model.semantic_logit_bias is None:
        raise ValueError("Semantic logits bias is not initialized.")

    biased_logits = logits + model.semantic_logit_bias.astype(logits.dtype)
    normal = _sample_logits(
        biased_logits, temperature=temperature, top_p=top_p, top_k=top_k
    )

    token_value = int(normal[0].item())
    should_use_high = (
        token_value in previous_semantic_tokens
        and model.config.semantic_start_token_id
        <= token_value
        <= model.config.semantic_end_token_id
    )
    if not should_use_high:
        return normal

    return _sample_logits(
        biased_logits,
        temperature=RAS_HIGH_TEMP,
        top_p=RAS_HIGH_TOP_P,
        top_k=top_k,
    )


# ---------------------------------------------------------------------------
# Prompt construction (from mlx_audio Model._build_conversation)
# ---------------------------------------------------------------------------


def _build_conversation(
    prompt_texts: list[str],
    prompt_tokens: list[mx.array],
    instruct: Optional[str] = None,
) -> Conversation:
    style_instruction = instruct.strip() if instruct else ""
    conversation = Conversation()
    if prompt_texts and prompt_tokens:
        tagged_prompt_texts = []
        for idx, text in enumerate(prompt_texts):
            if "<|speaker:" in text:
                tagged_prompt_texts.append(text)
            else:
                tagged_prompt_texts.append(f"<|speaker:{idx}|>{text}")
        system_prompt = (
            "convert the provided text to speech reference to the following:\n\n"
        )
        if style_instruction:
            system_prompt += f"Style instruction: {style_instruction}\n\n"
        system_prompt += "Text:\n"
        system_parts = [
            TextPart(system_prompt),
            TextPart("\n".join(tagged_prompt_texts)),
            TextPart("\n\nSpeech:\n"),
            VQPart(mx.concatenate(prompt_tokens, axis=1)),
        ]
    else:
        system_prompt = "convert the provided text to speech"
        if style_instruction:
            system_prompt += f"\n\nStyle instruction: {style_instruction}"
        system_parts = [TextPart(system_prompt)]

    conversation.append(
        Message(
            role="system",
            parts=system_parts,
            add_im_start=True,
            add_im_end=True,
        )
    )
    return conversation


def _prepare_reference_prompt(
    model,
    ref_audio: Optional[mx.array],
    ref_text: Optional[str],
) -> tuple[list[str], list[mx.array]]:
    prompt_tokens: list[mx.array] = []
    prompt_texts: list[str] = []
    if ref_audio is not None:
        if model.codec is None:
            raise ValueError("Codec not loaded.")

        audio = ref_audio
        if audio.ndim == 1:
            audio = audio[None, None, :]
        elif audio.ndim == 2:
            audio = audio[None, :, :]
        if audio.shape[1] != 1:
            audio = mx.mean(audio, axis=1, keepdims=True)
        indices, feature_lengths = model.codec.encode(audio)
        prompt_length = int(feature_lengths[0].item())
        prompt_tokens.append(indices[0, :, :prompt_length])
        prompt_texts.append(ref_text or "")

    return prompt_texts, prompt_tokens


def _split_generation_text(text: str, chunk_length: int) -> list[str]:
    turns = split_text_by_speaker(text)
    return (
        group_turns_into_batches(turns, max_speakers=5, max_bytes=chunk_length)
        if turns
        else [text]
    )


# ---------------------------------------------------------------------------
# Codec generation loop (from mlx_audio Model._generate_codes_for_batch)
# ---------------------------------------------------------------------------


def _generate_codes(
    model,
    conversation: Conversation,
    batch_text: str,
    *,
    max_new_tokens: int = 1024,
    top_p: float = 0.7,
    top_k: int = 30,
    temperature: float = 0.7,
    profile: Optional[GenerationProfile] = None,
) -> mx.array:
    """Autoregressively generate semantic + residual codebook tokens."""
    if model.tokenizer is None:
        raise ValueError("Tokenizer not loaded.")

    prompt_conversation = Conversation(list(conversation.messages))
    prompt_conversation.append(
        Message(
            role="assistant",
            parts=[],
            modality="voice",
            add_im_start=True,
            add_im_end=False,
        )
    )
    prompt = prompt_conversation.encode_for_inference(
        model.tokenizer, num_codebooks=model.model.num_codebooks
    )
    prompt = prompt[None, :, :]

    cache = model.model.make_cache()
    result = model.model(prompt, cache=cache)
    logits = result.logits[:, -1]
    hidden_state = result.hidden_states[:, -1]

    previous_semantic_tokens: list[int] = []
    generated_steps: list[mx.array] = []
    im_end_id = model.tokenizer.get_token_id(IM_END_TOKEN)
    text_token_count = len(model.tokenizer.encode(batch_text))
    semantic_token_budget = min(
        max_new_tokens,
        max(32, text_token_count * 12),
    )

    for step in range(semantic_token_budget):
        if profile is not None:
            t_step = time.perf_counter()
            t0 = time.perf_counter()

        semantic_token = _sample_semantic(
            model,
            logits=logits,
            previous_semantic_tokens=previous_semantic_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
        )
        semantic_token_id = int(semantic_token[0].item())
        if semantic_token_id == im_end_id:
            break

        previous_semantic_tokens.append(semantic_token_id)
        previous_semantic_tokens = previous_semantic_tokens[-RAS_WIN_SIZE:]

        semantic_code = (semantic_token - model.config.semantic_start_token_id).astype(
            mx.int32
        )
        semantic_code = mx.clip(
            semantic_code, 0, model.config.audio_decoder_config.vocab_size - 1
        )
        previous_codebooks = semantic_code[:, None]
        fast_cache = model.model.make_fast_cache()
        fast_prefill = model.model.fast_forward_cached(hidden_state, fast_cache)
        mx.async_eval(fast_prefill)
        fast_hidden = model.model.fast_embeddings(semantic_code)

        if profile is not None:
            mx.eval(semantic_token, previous_codebooks)
            backbone_semantic_s = time.perf_counter() - t0
            t1 = time.perf_counter()

        for _ in range(model.model.num_codebooks - 1):
            residual_logits = model.model.fast_forward_cached(fast_hidden, fast_cache)
            residual_token = _sample_logits(
                residual_logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
            )
            previous_codebooks = mx.concatenate(
                [previous_codebooks, residual_token[:, None]], axis=1
            )
            fast_hidden = model.model.fast_embeddings(residual_token)

        if profile is not None:
            mx.eval(previous_codebooks)
            depth_audio_s = time.perf_counter() - t1
            total_s = time.perf_counter() - t_step
            profile.token_timings.append(
                TokenTiming(
                    token_idx=len(generated_steps),
                    backbone_semantic_s=backbone_semantic_s,
                    depth_audio_s=depth_audio_s,
                    total_s=total_s,
                )
            )

        generated_steps.append(previous_codebooks[0])

        next_input = mx.concatenate(
            [semantic_token[:, None].astype(mx.int32), previous_codebooks], axis=1
        )
        next_result = model.model(next_input[:, :, None], cache=cache)
        logits = next_result.logits[:, -1]
        hidden_state = next_result.hidden_states[:, -1]

        if step > 0 and step % 50 == 0:
            mx.clear_cache()

    if not generated_steps:
        raise RuntimeError(
            f"No audio tokens were generated for batch text: {batch_text!r}"
        )

    return mx.stack(generated_steps, axis=1).astype(mx.int32)


# ---------------------------------------------------------------------------
# Codec decode (codes → waveform)
# ---------------------------------------------------------------------------


def _decode_codes(model, codes: mx.array) -> mx.array:
    if model.codec is None:
        raise ValueError("Codec not loaded.")
    feature_lengths = mx.array([codes.shape[1]], dtype=mx.int32)
    audio, audio_lengths = model.codec.decode(codes[None, :, :], feature_lengths)
    length = int(audio_lengths[0].item())
    return audio[0, 0, :length]


def _adjust_speed(audio: mx.array, speed: float) -> mx.array:
    if abs(speed - 1.0) < 1e-6:
        return audio
    old_length = int(audio.shape[0])
    new_length = max(1, int(old_length / speed))
    positions = mx.linspace(0, old_length - 1, new_length)
    left = mx.floor(positions).astype(mx.int32)
    right = mx.minimum(left + 1, old_length - 1)
    right_weight = positions - left
    left_weight = 1.0 - right_weight
    return left_weight * audio[left] + right_weight * audio[right]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FishAudioTTS:
    """Thin wrapper around mlx-audio's Fish Speech Model with a readable generate()."""

    def __init__(self, mlx_model):
        self._model = mlx_model

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        ref_audio: Optional[mx.array] = None,
        ref_text: Optional[str] = None,
        instruct: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.7,
        top_k: int = 30,
        repetition_penalty: float = 1.2,
        stream: bool = False,
        speed: float = 1.0,
        chunk_length: int = 300,
        profile: Optional[GenerationProfile] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text, optionally cloning a reference voice."""
        del voice, repetition_penalty, kwargs

        if stream:
            raise NotImplementedError("Fish Audio streaming is not implemented yet.")
        if self._model.tokenizer is None:
            raise ValueError("Tokenizer not loaded.")
        if self._model.codec is None:
            raise ValueError("Codec not loaded.")

        prompt_texts, prompt_tokens = _prepare_reference_prompt(
            self._model, ref_audio, ref_text
        )
        base_conversation = _build_conversation(
            prompt_texts, prompt_tokens, instruct=instruct
        )
        batches = _split_generation_text(text, chunk_length)

        conversation = Conversation(list(base_conversation.messages))
        for segment_idx, batch_text in enumerate(batches):
            start_time = time.time()

            conversation.append(
                Message(
                    role="user",
                    parts=[TextPart(batch_text)],
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            segment_profile = profile

            codes = _generate_codes(
                self._model,
                conversation,
                batch_text,
                max_new_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                profile=segment_profile,
            )

            if segment_profile is not None:
                t_decode = time.perf_counter()
                audio = _decode_codes(self._model, codes)
                segment_profile.codec_decode_s = time.perf_counter() - t_decode
                segment_profile.num_tokens = int(codes.shape[1])
            else:
                audio = _decode_codes(self._model, codes)

            if abs(speed - 1.0) > 1e-6:
                audio = _adjust_speed(audio, speed)
            mx.eval(audio)

            conversation.append(
                Message(
                    role="assistant",
                    parts=[VQPart(codes)],
                    modality="voice",
                    add_im_start=True,
                    add_im_end=True,
                )
            )

            yield _make_result(
                self._model,
                audio,
                segment_idx=segment_idx,
                token_count=int(codes.shape[1]),
                start_time=start_time,
                profile=segment_profile,
            )
            mx.clear_cache()


def load_model(model_id: str) -> FishAudioTTS:
    """Load a Fish Audio model via mlx-audio and wrap it for readable inference."""
    from mlx_audio.tts.utils import load_model as _load

    return FishAudioTTS(_load(model_id))
