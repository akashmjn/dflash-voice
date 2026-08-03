"""Readable MisoTTS inference (text prompt, no voice cloning).

MisoTTS is architecturally Sesame CSM with an 8B backbone, so mlx-audio's
``sesame`` model already covers it. Model weights and nn.Module architecture are
loaded via mlx-audio; this file owns prompt construction, the frame
autoregression loop, and Mimi decode, with the per-frame timing split the
benchmark reports.

Ported from mlx-audio 0.4.4
(https://github.com/Blaizzy/mlx-audio, PyPI: mlx-audio==0.4.4),
all from ``mlx_audio/tts/models/sesame/sesame.py``:

- ``SesameModel.generate_frame`` → ``_generate_frame``
- ``Model.generate`` frame loop → ``_generate_frames``, ``MisoTTS.generate``
- ``Model.generate_result`` Mimi decode chunking → ``_decode_frames``

Reference model: ``mlx-community/MisoLabs-MisoTTS-8bit``
"""

from __future__ import annotations

import time
from typing import Callable, Generator, List, Optional, Sequence, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from benchmark_mlx._common import (
    GenerationProfile,
    GenerationResult,
    StepTiming,
    _make_result,
)

MLX_AUDIO_VERSION = "0.4.4"

FRAME_RATE = 12.5  # Mimi frames per second
MAX_SEQ_LEN = 2048  # Miso backbone context

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _make_sampler(temperature: float, top_k: int, top_p: float):
    """Build the frame sampler, normalizing logits only when top-p needs it.

    ``mlx_lm.apply_top_p`` exponentiates its input and compares a cumulative sum
    against ``1 - top_p``, so it requires normalized log-probabilities. Sesame
    hands the sampler raw logits, whose exponentials sum to millions -- top-p
    then keeps nearly the whole vocabulary and silently does nothing.

    Normalizing is mathematically a no-op for top-k (monotonic) and temperature
    (softmax ignores constant shifts), but the subtraction rounds differently in
    the model's working precision, which is enough to flip an occasional token
    and diverge the sequence. So it is applied only when top-p is actually on,
    leaving the default path bit-identical to upstream ``Model.generate``.
    """
    sampler = make_sampler(temp=temperature, top_k=top_k, top_p=top_p)
    if not 0.0 < top_p < 1.0:
        return sampler
    return lambda logits: sampler(nn.log_softmax(logits, axis=-1))


# ---------------------------------------------------------------------------
# Frame generation (from mlx_audio SesameModel.generate_frame)
# ---------------------------------------------------------------------------


def _generate_frame(
    model,
    tokens: mx.array,
    tokens_mask: mx.array,
    sampler: Callable[[mx.array], mx.array],
    profile: Optional[GenerationProfile] = None,
    step_idx: int = 0,
) -> mx.array:
    """One frame: backbone predicts codebook 0, the depth decoder the other 31.

    Inlined rather than calling ``SesameModel.generate_frame`` so the benchmark
    can separate backbone from depth-decoder time. Upstream's ``input_pos`` is
    dropped: ``LlamaModel`` derives positions from the KV cache offset, and
    upstream threads the value through without ever reading it.
    """
    sm = model.model
    t_step = time.perf_counter()

    embeds = sm._embed_tokens(tokens)
    h = mx.sum(embeds * mx.expand_dims(tokens_mask, -1), axis=2)
    h = sm.backbone(h, cache=sm.backbone_cache)

    last_h = h[:, -1, :]
    c0_sample = mx.expand_dims(sampler(sm.codebook0_head(last_h)), axis=-1)

    backbone_semantic_s = 0.0
    if profile is not None:
        mx.eval(c0_sample, last_h)
        backbone_semantic_s = time.perf_counter() - t_step
    t_depth = time.perf_counter()

    # The depth decoder runs over the codebook axis: [h_t, emb(c0), .. emb(c30)].
    curr_h = mx.concat([mx.expand_dims(last_h, 1), sm._embed_audio(0, c0_sample)], axis=1)
    curr_sample = c0_sample
    sm.decoder_cache = make_prompt_cache(sm.decoder)

    for i in range(1, sm.args.audio_num_codebooks):
        decoder_h = sm.decoder(sm.projection(curr_h), cache=sm.decoder_cache)
        ci_sample = mx.expand_dims(
            sampler(mx.matmul(decoder_h[:, -1, :], sm.audio_head[i - 1])), axis=-1
        )
        curr_h = sm._embed_audio(i, ci_sample)
        curr_sample = mx.concat([curr_sample, ci_sample], axis=1)

    if profile is not None:
        mx.eval(curr_sample)
        profile.step_timings.append(
            StepTiming(
                step_idx=step_idx,
                backbone_semantic_s=backbone_semantic_s,
                depth_audio_s=time.perf_counter() - t_depth,
                total_s=time.perf_counter() - t_step,
            )
        )

    return curr_sample


def _generate_frames(
    model,
    prompt_tokens: mx.array,
    prompt_mask: mx.array,
    *,
    max_frames: int,
    sampler: Callable[[mx.array], mx.array],
    profile: Optional[GenerationProfile] = None,
) -> List[mx.array]:
    """Autoregress frames until the model emits an all-zero (EOS) frame."""
    model.model.reset_caches()

    curr_tokens = mx.expand_dims(prompt_tokens, axis=0)
    curr_tokens_mask = mx.expand_dims(prompt_mask, axis=0)

    if curr_tokens.shape[1] >= MAX_SEQ_LEN - max_frames:
        raise ValueError(
            f"Prompt of {curr_tokens.shape[1]} tokens leaves no room for "
            f"{max_frames} frames within the {MAX_SEQ_LEN}-token context"
        )

    frames: List[mx.array] = []
    for _ in range(max_frames):
        sample = _generate_frame(
            model,
            curr_tokens,
            curr_tokens_mask,
            sampler,
            profile=profile,
            step_idx=len(frames),
        )
        if mx.all(sample == 0):
            break

        frames.append(sample)

        # Next input is the frame just generated, padded with an unused text column.
        curr_tokens = mx.expand_dims(
            mx.concat([sample, mx.zeros((1, 1)).astype(mx.int32)], axis=1), axis=1
        )
        curr_tokens_mask = mx.expand_dims(
            mx.concat(
                [
                    mx.ones_like(sample).astype(mx.bool_),
                    mx.zeros((1, 1)).astype(mx.bool_),
                ],
                axis=1,
            ),
            axis=1,
        )

    return frames


# ---------------------------------------------------------------------------
# Codec decode (frames → waveform)
# ---------------------------------------------------------------------------


def _decode_frames(model, frames: List[mx.array]) -> mx.array:
    """Decode stacked Mimi frames to a 24 kHz waveform."""
    # MimiStreamingDecoder carries convolution and KV state across calls, and
    # upstream only resets it on the streaming path -- without this, prompt N+1
    # decodes on top of prompt N's state.
    model._streaming_decoder.reset()

    codes = mx.transpose(mx.stack(frames), axes=[1, 2, 0])

    chunk = min(len(frames), int(FRAME_RATE * 5))  # ~5s per chunk, bounds memory
    chunks = []
    for start in range(0, codes.shape[2], chunk):
        audio = model._streaming_decoder.decode_frames(codes[:, :, start : start + chunk])
        chunks.append(audio.squeeze(0).squeeze(0))

    audio = mx.concat(chunks, axis=0)
    mx.eval(audio)
    return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class MisoTTS:
    """Thin wrapper around mlx-audio's sesame Model with a readable generate()."""

    def __init__(self, mlx_model):
        self._model = mlx_model

    @property
    def sample_rate(self) -> int:
        return self._model.sample_rate

    def generate(
        self,
        text: str,
        speaker: int = 0,
        context: Optional[Sequence[Tuple[int, str, str]]] = None,
        max_tokens: int = 1024,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
        stream: bool = False,
        profile: Optional[GenerationProfile] = None,
        **kwargs,
    ) -> Generator[GenerationResult, None, None]:
        """Generate speech from text. ``max_tokens`` counts frames, not tokens.

        Each frame is 32 codebook tokens at 12.5 Hz, so the default is ~82s of
        audio. The parameter keeps its name because the benchmark harness passes
        ``max_tokens=`` to every backend.

        ``context`` is a list of ``(speaker, text, audio_path)`` turns prepended
        as conversation history. Without one the model must invent a voice while
        articulating the first frames, which destabilizes the start of the
        utterance. Encoding it costs a Mimi forward pass per turn, inside the
        timed window.
        """
        del kwargs

        if stream:
            raise NotImplementedError("MisoTTS wrapper does not support streaming")

        start_time = time.perf_counter()

        parts = [
            self._model._tokenize_segment(
                self._model.prepare_prompt(
                    ctx_text, ctx_speaker, str(ctx_audio), self.sample_rate
                )
            )
            for ctx_speaker, ctx_text, ctx_audio in context or ()
        ]
        # Produces "[speaker] text", matching how dataprep/miso.py tokenizes.
        parts.append(self._model._tokenize_text_segment(text, speaker))

        prompt_tokens = mx.concat([tokens for tokens, _ in parts], axis=0)
        prompt_mask = mx.concat([mask for _, mask in parts], axis=0)

        frames = _generate_frames(
            self._model,
            prompt_tokens.astype(mx.int32),
            prompt_mask.astype(mx.bool_),
            max_frames=max_tokens,
            sampler=_make_sampler(temperature, top_k, top_p),
            profile=profile,
        )
        if not frames:
            return

        if profile is not None:
            t_decode = time.perf_counter()
            audio = _decode_frames(self._model, frames)
            profile.codec_decode_s = time.perf_counter() - t_decode
            profile.num_steps = len(frames)
        else:
            audio = _decode_frames(self._model, frames)

        yield _make_result(
            self._model,
            audio,
            segment_idx=0,
            token_count=len(frames),
            start_time=start_time,
            profile=profile,
        )

        mx.clear_cache()


def load_model(model_id: str) -> MisoTTS:
    from mlx_audio.tts.utils import load_model as _load_mlx_audio_model

    return MisoTTS(_load_mlx_audio_model(model_id))
