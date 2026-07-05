"""Readable Qwen3-TTS Base inference (preset voice + streaming).

This module reimplements the mlx-audio Qwen3-TTS **Base preset-voice** generate
path in a single, annotated file. Model weights and nn.Module architecture are
loaded via mlx-audio; this file owns prompt construction, the codec
autoregression loop, and vocoder decode.

Ported from mlx-audio 0.4.4
(https://github.com/Blaizzy/mlx-audio, PyPI: mlx-audio==0.4.4):

- ``Model._prepare_generation_inputs`` → ``_prepare_prompt``
  (``mlx_audio/tts/models/qwen3_tts/qwen3_tts.py``)
- ``Model._sample_token`` / ``_apply_probability_filters`` → ``_sample_codebook_token``
- Base ``Model.generate`` codec loop (talker + code predictor + streaming decode)
  → ``_generate_codec_frames``, ``Qwen3TTS.generate``
- ``Model._decode_*`` / ``speech_tokenizer.decode`` → ``_codec_decode_frames``,
  ``_codec_decode_stream_chunk``

Reference model: ``mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit``
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.sample_utils import (
    apply_min_p,
    apply_top_k,
    apply_top_p,
    categorical_sampling,
)

MLX_AUDIO_VERSION = "0.4.4"

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


@dataclass
class FrameTiming:
    frame_idx: int
    backbone_s: float
    depth_s: float
    total_s: float


@dataclass
class GenerationProfile:
    frame_timings: List[FrameTiming] = field(default_factory=list)
    codec_decode_s: float = 0.0
    num_frames: int = 0


@dataclass
class GenerationResult:
    audio: mx.array
    samples: int
    sample_rate: int
    segment_idx: int
    token_count: int
    audio_duration: str
    real_time_factor: float
    prompt: dict
    audio_samples: dict
    processing_time_seconds: float
    peak_memory_usage: float
    is_streaming_chunk: bool = False
    is_final_chunk: bool = False
    profile: Optional[GenerationProfile] = None


# ---------------------------------------------------------------------------
# Sampling (from mlx_audio Model._sample_token)
# ---------------------------------------------------------------------------


def _apply_probability_filters(
    logits: mx.array,
    top_p: float,
    min_p: float,
) -> mx.array:
    if not (0.0 < top_p < 1.0 or min_p > 0.0):
        return logits

    logprobs = nn.log_softmax(logits, axis=-1)
    if 0.0 < top_p < 1.0:
        logprobs = apply_top_p(logprobs, top_p)
    if min_p > 0.0:
        logprobs = apply_min_p(logprobs, min_p)

    return mx.where(logprobs == -mx.inf, -float("inf"), logits)


def _sample_codebook_token(
    logits: mx.array,
    *,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    generated_tokens: Optional[List[int]] = None,
    suppress_tokens: Optional[List[int]] = None,
    eos_token_id: Optional[int] = None,
    min_p: float = 0.0,
) -> mx.array:
    """Sample s_t (codebook 0) from backbone or depth-decoder logits [1, seq, vocab]."""
    logits = logits[:, -1, :]

    if suppress_tokens:
        suppress_idx = mx.array(suppress_tokens, dtype=mx.int32)
        logits = mx.put_along_axis(
            logits,
            suppress_idx[None, :],
            mx.array(float("-inf"), logits.dtype),
            axis=-1,
        )

    if generated_tokens and repetition_penalty != 1.0:
        unique_tokens = list(set(generated_tokens))
        valid_tokens = [t for t in unique_tokens if t < logits.shape[-1]]
        if valid_tokens:
            token_ids = mx.array(valid_tokens, dtype=mx.int32)
            selected_logits = mx.take(logits, token_ids, axis=-1)
            penalized = mx.where(
                selected_logits < 0,
                selected_logits * repetition_penalty,
                selected_logits / repetition_penalty,
            )
            logits = mx.put_along_axis(logits, token_ids[None, :], penalized, axis=-1)

    if temperature <= 0:
        return mx.argmax(logits, axis=-1, keepdims=True)

    if temperature != 1.0:
        logits = logits / temperature

    eos_logit = None
    if eos_token_id is not None and eos_token_id < logits.shape[-1]:
        eos_logit = logits[:, eos_token_id : eos_token_id + 1]

    if top_k > 0 and top_k < logits.shape[-1]:
        logits = apply_top_k(logits, top_k)

    logits = _apply_probability_filters(logits, top_p, min_p)

    if eos_logit is not None:
        eos_idx = mx.array([[eos_token_id]], dtype=mx.int32)
        logits = mx.put_along_axis(logits, eos_idx, eos_logit, axis=-1)

    token = categorical_sampling(logits, 1.0)
    return token[:, None]


# ---------------------------------------------------------------------------
# Prompt construction (from mlx_audio Model._prepare_generation_inputs)
# ---------------------------------------------------------------------------


def _prepare_prompt(
    model,
    text: str,
    *,
    voice: Optional[str] = None,
    language: str = "auto",
) -> Tuple[mx.array, mx.array, mx.array]:
    """Build talker prefill embeddings for Base preset-voice generation.

    Returns:
        input_embeds: prefill sequence for the talker KV cache
        text_stream: one text embedding per codec frame
        text_pad_embed: padding embed after text is exhausted
    """
    if model.tokenizer is None:
        raise ValueError("Tokenizer not loaded")

    config = model.config.talker_config

    chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    input_ids = mx.array(model.tokenizer.encode(chat_text))[None, :]

    text_embed = model.talker.text_projection(
        model.talker.get_text_embeddings()(input_ids)
    )

    tts_tokens = mx.array(
        [
            [
                model.config.tts_bos_token_id,
                model.config.tts_eos_token_id,
                model.config.tts_pad_token_id,
            ]
        ]
    )
    tts_embeds = model.talker.text_projection(
        model.talker.get_text_embeddings()(tts_tokens)
    )
    tts_bos_embed = tts_embeds[:, 0:1, :]
    tts_eos_embed = tts_embeds[:, 1:2, :]
    tts_pad_embed = tts_embeds[:, 2:3, :]

    speaker_embed = None
    if voice and voice.lower() in (config.spk_id or {}):
        spk_ids = mx.array([[config.spk_id[voice.lower()]]])
        speaker_embed = model.talker.get_input_embeddings()(spk_ids)

    language_id = None
    if language.lower() != "auto" and config.codec_language_id:
        if language.lower() in config.codec_language_id:
            language_id = config.codec_language_id[language.lower()]

    if (
        language.lower() in ["chinese", "auto"]
        and voice
        and voice.lower() in (config.spk_is_dialect or {})
        and config.spk_is_dialect[voice.lower()]
    ):
        dialect = config.spk_is_dialect[voice.lower()]
        if dialect in config.codec_language_id:
            language_id = config.codec_language_id[dialect]

    if language_id is None:
        codec_prefill = [
            config.codec_nothink_id,
            config.codec_think_bos_id,
            config.codec_think_eos_id,
        ]
    else:
        codec_prefill = [
            config.codec_think_id,
            config.codec_think_bos_id,
            language_id,
            config.codec_think_eos_id,
        ]

    codec_embed = model.talker.get_input_embeddings()(mx.array([codec_prefill]))
    codec_embed_suffix = model.talker.get_input_embeddings()(
        mx.array([[config.codec_pad_id, config.codec_bos_id]])
    )

    if speaker_embed is not None:
        codec_embed = mx.concatenate(
            [codec_embed, speaker_embed.reshape(1, 1, -1), codec_embed_suffix],
            axis=1,
        )
    else:
        codec_embed = mx.concatenate([codec_embed, codec_embed_suffix], axis=1)

    role_embed = text_embed[:, :3, :]

    pad_count = codec_embed.shape[1] - 2
    pad_embeds = mx.broadcast_to(tts_pad_embed, (1, pad_count, tts_pad_embed.shape[-1]))
    combined_embed = mx.concatenate([pad_embeds, tts_bos_embed], axis=1)
    combined_embed = combined_embed + codec_embed[:, :-1, :]

    input_embeds = mx.concatenate([role_embed, combined_embed], axis=1)
    first_text_embed = text_embed[:, 3:4, :] + codec_embed[:, -1:, :]
    input_embeds = mx.concatenate([input_embeds, first_text_embed], axis=1)

    text_stream = mx.concatenate(
        [text_embed[:, 4:-5, :], tts_eos_embed],
        axis=1,
    )

    return input_embeds, text_stream, tts_pad_embed


# ---------------------------------------------------------------------------
# Codec generation loop (from mlx_audio Model.generate Base path)
# ---------------------------------------------------------------------------


def _reset_depth_cache(depth_cache) -> None:
    for cache in depth_cache:
        cache.keys = None
        cache.values = None
        cache.offset = 0


def _depth_decode_audio_tokens(
    model,
    s_t: mx.array,
    h_t: mx.array,
    depth_cache,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> List[mx.array]:
    """Predict a_t (codebooks 1..N-1) given semantic token s_t and backbone hidden h_t."""
    config = model.config.talker_config
    a_t: List[mx.array] = []
    depth_hidden = h_t[:, -1:, :]

    _reset_depth_cache(depth_cache)

    for code_idx in range(config.num_code_groups - 1):
        if code_idx == 0:
            s_embed = model.talker.get_input_embeddings()(s_t)
            depth_input = mx.concatenate([depth_hidden, s_embed], axis=1)
        else:
            depth_input = model.talker.code_predictor.codec_embedding[code_idx - 1](
                a_t[-1]
            )

        depth_logits, depth_cache, _ = model.talker.code_predictor(
            depth_input,
            cache=depth_cache,
            generation_step=code_idx,
        )
        a_t.append(
            _sample_codebook_token(
                depth_logits,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        )

    return a_t


def _build_frame_embedding(
    model,
    s_t: mx.array,
    a_t: List[mx.array],
    text_stream: mx.array,
    text_pad_embed: mx.array,
    text_idx: int,
) -> Tuple[mx.array, int]:
    """Build e_t = text_embed + embed(s_t) + sum(embed(a_i)) for the next backbone step."""
    if text_idx < text_stream.shape[1]:
        text_embed = text_stream[:, text_idx : text_idx + 1, :]
        text_idx += 1
    else:
        text_embed = text_pad_embed

    codec_embed = model.talker.get_input_embeddings()(s_t)
    for i, a_i in enumerate(a_t):
        codec_embed = codec_embed + model.talker.code_predictor.codec_embedding[i](a_i)

    return text_embed + codec_embed, text_idx


def _generate_codec_frames(
    model,
    prefill_embeds: mx.array,
    text_stream: mx.array,
    text_pad_embed: mx.array,
    *,
    max_tokens: int = 4096,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    repetition_penalty: float = 1.05,
    stream: bool = False,
    streaming_interval: float = 2.0,
    profile: Optional[GenerationProfile] = None,
) -> Generator[Tuple[str, object], None, None]:
    """Autoregressively generate 16-codebook frames at ~12.5 Hz.

    Yields:
        ("stream", (audio_chunk, new_token_count, is_final)) for streaming decode chunks
        ("done", generated_frames) when batch decode will follow
    """
    if model.speech_tokenizer is None:
        raise ValueError("Speech tokenizer not loaded")

    config = model.config.talker_config
    eos_token_id = config.codec_eos_token_id
    suppress_tokens = [
        i
        for i in range(config.vocab_size - 1024, config.vocab_size)
        if i != eos_token_id
    ]

    backbone_cache = model.talker.make_cache()
    depth_cache = model.talker.code_predictor.make_cache()
    generated_token_ids: List[int] = []
    generated_frames: List[mx.array] = []
    text_idx = 0
    e_t = prefill_embeds

    streaming_chunk_size = max(1, int(streaming_interval * 12.5))
    decoded_frames = 0

    if stream:
        model.speech_tokenizer.decoder.reset_streaming_state()

    def _maybe_emit_stream(is_final: bool = False):
        nonlocal decoded_frames
        if not stream:
            return
        pending = len(generated_frames) - decoded_frames
        if pending <= 0:
            return
        if not is_final and pending < streaming_chunk_size:
            return

        new_tokens = pending
        audio_chunk = _codec_decode_stream_chunk(
            model, generated_frames[decoded_frames:]
        )
        decoded_frames = len(generated_frames)
        yield ("stream", (audio_chunk, new_tokens, is_final))

    for step in range(max_tokens):
        if profile is not None:
            t_step = time.perf_counter()
            t0 = time.perf_counter()

        logits, h_t = model.talker(e_t, cache=backbone_cache)
        s_t = _sample_codebook_token(
            logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            generated_tokens=generated_token_ids or None,
            suppress_tokens=suppress_tokens,
            eos_token_id=eos_token_id,
        )

        if profile is not None:
            mx.eval(s_t, h_t)
            backbone_s = time.perf_counter() - t0
            t1 = time.perf_counter()

        is_eos = s_t[0, 0] == eos_token_id

        a_t = _depth_decode_audio_tokens(
            model,
            s_t,
            h_t,
            depth_cache,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        if profile is not None:
            if a_t:
                mx.eval(a_t[-1])
            depth_s = time.perf_counter() - t1

        e_t, text_idx = _build_frame_embedding(
            model,
            s_t,
            a_t,
            text_stream,
            text_pad_embed,
            text_idx,
        )

        mx.eval(e_t, is_eos)

        if profile is not None:
            total_s = time.perf_counter() - t_step

        if is_eos.item():
            break

        generated_token_ids.append(int(s_t[0, 0]))
        frame_t = mx.concatenate([s_t, *a_t], axis=1)
        generated_frames.append(frame_t)

        if profile is not None:
            profile.frame_timings.append(
                FrameTiming(
                    frame_idx=len(generated_frames) - 1,
                    backbone_s=backbone_s,
                    depth_s=depth_s,
                    total_s=total_s,
                )
            )

        if stream:
            if len(generated_frames) - decoded_frames >= streaming_chunk_size:
                yield from _maybe_emit_stream(is_final=False)
                mx.clear_cache()
        elif step > 0 and step % 50 == 0:
            mx.clear_cache()

    if stream:
        yield from _maybe_emit_stream(is_final=True)
        model.speech_tokenizer.decoder.reset_streaming_state()
    else:
        yield ("done", generated_frames)


# ---------------------------------------------------------------------------
# Codec decode (frames → waveform)
# ---------------------------------------------------------------------------


def _codec_decode_frames(model, generated_frames: List[mx.array]) -> mx.array:
    """Decode stacked codec frames to a 24 kHz waveform."""
    if not generated_frames:
        return mx.zeros((0,), dtype=mx.float32)

    codes = mx.stack(generated_frames, axis=1)
    audio, audio_lengths = model.speech_tokenizer.decode(codes)
    audio = audio[0]

    valid_len = int(audio_lengths[0])
    if valid_len > 0 and valid_len < audio.shape[0]:
        audio = audio[:valid_len]

    mx.eval(audio)
    return audio


def _codec_decode_stream_chunk(model, frames: List[mx.array]) -> mx.array:
    codes_chunk = mx.stack(frames, axis=1)
    codes_for_decoder = mx.transpose(codes_chunk, (0, 2, 1))
    mx.eval(codes_for_decoder)

    wav = model.speech_tokenizer.decoder.streaming_step(codes_for_decoder)
    audio_chunk = wav.squeeze(1)[0]
    mx.eval(audio_chunk)
    return audio_chunk


def _make_result(
    model,
    audio: mx.array,
    *,
    segment_idx: int,
    token_count: int,
    start_time: float,
    is_streaming_chunk: bool = False,
    is_final_chunk: bool = False,
    profile: Optional[GenerationProfile] = None,
) -> GenerationResult:
    elapsed = time.time() - start_time
    samples = int(audio.shape[0])
    duration_seconds = samples / model.sample_rate
    rtf = duration_seconds / elapsed if elapsed > 0 else 0.0

    return GenerationResult(
        audio=audio,
        samples=samples,
        sample_rate=model.sample_rate,
        segment_idx=segment_idx,
        token_count=token_count,
        audio_duration=_format_duration(duration_seconds),
        real_time_factor=rtf,
        prompt={
            "tokens": token_count,
            "tokens-per-sec": token_count / elapsed if elapsed > 0 else 0.0,
        },
        audio_samples={
            "samples": samples,
            "samples-per-sec": samples / elapsed if elapsed > 0 else 0.0,
        },
        processing_time_seconds=elapsed,
        peak_memory_usage=mx.get_peak_memory() / 1e9,
        is_streaming_chunk=is_streaming_chunk,
        is_final_chunk=is_final_chunk,
        profile=profile,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class Qwen3TTS:
    """Thin wrapper around mlx-audio's Qwen3 Model with a readable generate()."""

    def __init__(self, mlx_model):
        self._model = mlx_model

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def generate(
        self,
        text: str,
        voice: Optional[str] = None,
        language: str = "auto",
        lang_code: Optional[str] = None,
        split_pattern: str = "\n",
        max_tokens: int = 4096,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        repetition_penalty: float = 1.05,
        stream: bool = False,
        streaming_interval: float = 2.0,
        profile: Optional[GenerationProfile] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text using a Base preset voice."""
        del kwargs

        if getattr(self._model.config, "tts_model_type", "base") != "base":
            raise ValueError("Qwen3TTS wrapper supports Base preset-voice models only")

        if self._model.speech_tokenizer is None:
            raise ValueError("Speech tokenizer not loaded")

        lang = lang_code if lang_code is not None else language

        if split_pattern:
            segments = [s.strip() for s in text.split(split_pattern) if s.strip()]
        else:
            segments = [text]

        for segment_idx, segment_text in enumerate(segments):
            start_time = time.time()

            prefill_embeds, text_stream, text_pad_embed = _prepare_prompt(
                self._model,
                segment_text,
                voice=voice,
                language=lang,
            )

            generated_frames: List[mx.array] = []
            chunk_start = start_time

            segment_profile = profile if profile is not None and not stream else None

            for event, payload in _generate_codec_frames(
                self._model,
                prefill_embeds,
                text_stream,
                text_pad_embed,
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                stream=stream,
                streaming_interval=streaming_interval,
                profile=segment_profile,
            ):
                if event == "stream":
                    audio_chunk, new_tokens, is_final = payload
                    yield _make_result(
                        self._model,
                        audio_chunk,
                        segment_idx=segment_idx,
                        token_count=new_tokens,
                        start_time=chunk_start,
                        is_streaming_chunk=True,
                        is_final_chunk=is_final,
                    )
                    chunk_start = time.time()
                    if is_final:
                        mx.clear_cache()
                elif event == "done":
                    generated_frames = payload

            if stream:
                continue

            if not generated_frames:
                continue

            if segment_profile is not None:
                t_decode = time.perf_counter()
                audio = _codec_decode_frames(self._model, generated_frames)
                segment_profile.codec_decode_s = time.perf_counter() - t_decode
                segment_profile.num_frames = len(generated_frames)
            else:
                audio = _codec_decode_frames(self._model, generated_frames)

            yield _make_result(
                self._model,
                audio,
                segment_idx=segment_idx,
                token_count=len(generated_frames),
                start_time=start_time,
                profile=segment_profile,
            )
            mx.clear_cache()


def load_model(model_id: str) -> Qwen3TTS:
    """Load a Qwen3-TTS model via mlx-audio and wrap it for readable inference."""
    from mlx_audio.tts.utils import load_model as _load

    return Qwen3TTS(_load(model_id))
