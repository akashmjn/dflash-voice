"""Shared helpers for readable MLX TTS inference modules."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx


def _format_duration(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


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
    profile: Optional[Any] = None


def _make_result(
    model,
    audio: mx.array,
    *,
    segment_idx: int,
    token_count: int,
    start_time: float,
    is_streaming_chunk: bool = False,
    is_final_chunk: bool = False,
    profile: Optional[Any] = None,
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
