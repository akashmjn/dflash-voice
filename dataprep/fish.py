from __future__ import annotations

from math import gcd
from typing import Any, Sequence

import numpy as np

from dataprep.tokenizer import Segment, SequenceSpan, TokenizedSequence, validate_sequence


def _mx():
    import mlx.core as mx

    return mx


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(np.float32)


class FishTextTokenizer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids)))


class FishAudioCodec:
    sample_rate = 44_100
    frame_rate = 21.0
    num_codebooks = 10

    def __init__(self, codec: Any, *, chunk_duration_sec: float = 15.0):
        if codec is None:
            raise ValueError("Fish codec is not loaded")
        self._codec = codec
        self.chunk_duration_sec = chunk_duration_sec

    def encode(self, audio: Any, sample_rate: int):
        mx = _mx()
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono audio, got {waveform.shape}")
        waveform = _resample(waveform, sample_rate, self.sample_rate)
        chunk_samples = int(self.sample_rate * self.chunk_duration_sec)
        chunks = []
        for start in range(0, waveform.shape[0], chunk_samples):
            indices, feature_lengths = self._codec.encode(
                mx.array(waveform[start : start + chunk_samples])[None, None, :]
            )
            length = int(feature_lengths[0].item())
            if length:
                chunks.append(indices[0, :, :length])
        if not chunks:
            raise ValueError("Fish codec produced no frames")
        return mx.concatenate(chunks, axis=1)

    def decode(self, codes: Any):
        mx = _mx()
        codes = mx.array(codes)
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks:
            raise ValueError(f"Expected codes shaped ({self.num_codebooks}, F), got {codes.shape}")
        chunk_frames = max(1, int(self.frame_rate * self.chunk_duration_sec))
        chunks = []
        for start in range(0, codes.shape[1], chunk_frames):
            chunk = codes[:, start : start + chunk_frames]
            lengths = mx.array([chunk.shape[1]], dtype=mx.int32)
            audio, audio_lengths = self._codec.decode(chunk[None], lengths)
            chunks.append(audio[0, 0, : int(audio_lengths[0].item())])
        return mx.concatenate(chunks)


class FishTokenizer:
    def __init__(
        self,
        model_id: str = "mlx-community/fish-audio-s2-pro-8bit",
        *,
        model: Any | None = None,
    ):
        if model is None:
            from tts_mlx.fish import load_model

            model = load_model(model_id)._model
        self._model = model
        self.audio_codec = FishAudioCodec(model.codec)
        self.text_tokenizer = FishTextTokenizer(model.tokenizer)
        self.max_seq_length = int(
            getattr(model.config, "max_seq_len", getattr(model.config, "max_length", 32_768))
        )

    def _encode_segment(self, segment: Segment):
        if segment.audio_codes is None:
            raise ValueError("Fish supervised segments require audio_codes")
        mx = _mx()
        from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
            Conversation,
            Message,
            TextPart,
            VQPart,
        )

        codes = mx.array(segment.audio_codes, dtype=mx.int32)
        if codes.ndim != 2 or codes.shape[0] != self.audio_codec.num_codebooks:
            raise ValueError(
                f"Expected ({self.audio_codec.num_codebooks}, F) Fish codes, got {codes.shape}"
            )
        speaker_text = (
            segment.text
            if "<|speaker:" in segment.text
            else f"<|speaker:{segment.speaker}|>{segment.text}"
        )
        conversation = Conversation()
        conversation.append(
            Message(
                role="user",
                parts=[TextPart(speaker_text)],
                add_im_start=True,
                add_im_end=True,
                modality="text",
            )
        )
        conversation.append(
            Message(
                role="assistant",
                parts=[VQPart(codes)],
                add_im_start=True,
                add_im_end=True,
                modality="voice",
            )
        )
        return conversation.encode_for_inference(
            self._model.tokenizer,
            num_codebooks=self._model.model.num_codebooks,
        )

    def apply_chat_template(self, segments: Sequence[Segment]) -> TokenizedSequence:
        mx = _mx()
        encoded = []
        masks = []
        spans: list[SequenceSpan] = []
        position = 0
        tokenizer = self._model.tokenizer

        for segment_index, segment in enumerate(segments):
            tokens = self._encode_segment(segment)
            audio_positions = (tokens[0] >= tokenizer.semantic_begin_id) & (
                tokens[0] <= tokenizer.semantic_end_id
            )
            mask = mx.concatenate(
                [
                    mx.ones((1, tokens.shape[1]), dtype=mx.bool_),
                    mx.broadcast_to(audio_positions[None], (self.audio_codec.num_codebooks, tokens.shape[1])),
                ],
                axis=0,
            )
            encoded.append(tokens)
            masks.append(mask)
            spans.append(
                SequenceSpan(segment_index, position, position + tokens.shape[1], "segment", segment.metadata)
            )
            position += tokens.shape[1]

        if not encoded:
            raise ValueError("At least one segment is required")
        result = TokenizedSequence(
            tokens=mx.concatenate(encoded, axis=1),
            mask=mx.concatenate(masks, axis=1),
            spans=spans,
        )
        validate_sequence(result, self.audio_codec.num_codebooks)
        if result.length > self.max_seq_length:
            raise ValueError(f"Sequence length {result.length} exceeds Fish limit {self.max_seq_length}")
        return result
