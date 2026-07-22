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


class Qwen3TextTokenizer:
    def __init__(self, tokenizer: Any):
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids)))


class Qwen3AudioCodec:
    sample_rate = 24_000
    frame_rate = 12.5
    num_codebooks = 16

    def __init__(self, speech_tokenizer: Any):
        if speech_tokenizer is None:
            raise ValueError("Qwen3 speech tokenizer is not loaded")
        if not getattr(speech_tokenizer, "has_encoder", False):
            raise ValueError(
                "This Qwen3 checkpoint has no speech encoder; use a checkpoint containing "
                "speech_tokenizer/encoder_config.json for dataprep"
            )
        self._codec = speech_tokenizer
        self.sample_rate = int(getattr(speech_tokenizer, "input_sample_rate", self.sample_rate))
        encoder_config = getattr(getattr(speech_tokenizer, "encoder_model", None), "config", None)
        self.frame_rate = float(getattr(encoder_config, "frame_rate", self.frame_rate))

    def encode(self, audio: Any, sample_rate: int):
        mx = _mx()
        waveform = np.asarray(audio, dtype=np.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(axis=0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono audio, got {waveform.shape}")
        waveform = _resample(waveform, sample_rate, self.sample_rate)
        codes = self._codec.encode(mx.array(waveform)[None, None, :])
        return codes[0, : self.num_codebooks]

    def decode(self, codes: Any):
        mx = _mx()
        codes = mx.array(codes)
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks:
            raise ValueError(f"Expected codes shaped ({self.num_codebooks}, F), got {codes.shape}")
        audio, audio_lengths = self._codec.decode(mx.transpose(codes, (1, 0))[None])
        return audio[0, : int(audio_lengths[0].item())]


class Qwen3Tokenizer:
    def __init__(
        self,
        model_id: str = "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        *,
        model: Any | None = None,
        voice: str = "Ryan",
        language: str = "english",
    ):
        if model is None:
            from tts_mlx.qwen3 import load_model

            model = load_model(model_id)._model
        self._model = model
        self.voice = voice
        self.language = language
        self.audio_codec = Qwen3AudioCodec(model.speech_tokenizer)
        self.text_tokenizer = Qwen3TextTokenizer(model.tokenizer)
        talker_config = model.config.talker_config
        self.max_seq_length = int(getattr(talker_config, "max_position_embeddings", 2048))

    def _codec_prefix(self) -> list[int]:
        config = self._model.config.talker_config
        language_id = getattr(config, "codec_language_id", {}).get(self.language.lower())
        if language_id is None:
            prefix = [config.codec_nothink_id, config.codec_think_bos_id, config.codec_think_eos_id]
        else:
            prefix = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]
        speaker_id = getattr(config, "spk_id", {}).get(self.voice.lower())
        if speaker_id is not None:
            prefix.append(speaker_id)
        return [*prefix, config.codec_pad_id, config.codec_bos_id]

    def apply_chat_template(self, segments: Sequence[Segment]) -> TokenizedSequence:
        mx = _mx()
        blocks = []
        masks = []
        spans: list[SequenceSpan] = []
        position = 0
        rows = self.audio_codec.num_codebooks + 1
        config = self._model.config.talker_config

        for segment_index, segment in enumerate(segments):
            chat = f"<|im_start|>assistant\n{segment.text}<|im_end|>\n<|im_start|>assistant\n"
            text_ids = self.text_tokenizer.encode(chat)
            text = mx.zeros((rows, len(text_ids)), dtype=mx.int32)
            text[-1] = mx.array(text_ids, dtype=mx.int32)
            text_mask = mx.zeros(text.shape, dtype=mx.bool_)
            text_mask[-1] = True
            blocks.append(text)
            masks.append(text_mask)
            spans.append(
                SequenceSpan(segment_index, position, position + len(text_ids), "text", segment.metadata)
            )
            position += len(text_ids)

            if segment.audio_codes is not None:
                prefix_ids = self._codec_prefix()
                prefix = mx.zeros((rows, len(prefix_ids)), dtype=mx.int32)
                prefix[0] = mx.array(prefix_ids, dtype=mx.int32)
                prefix_mask = mx.zeros(prefix.shape, dtype=mx.bool_)
                prefix_mask[0] = True
                blocks.append(prefix)
                masks.append(prefix_mask)
                position += len(prefix_ids)

                codes = mx.array(segment.audio_codes, dtype=mx.int32)
                if codes.ndim != 2 or codes.shape[0] != self.audio_codec.num_codebooks:
                    raise ValueError(
                        f"Expected ({self.audio_codec.num_codebooks}, F) Qwen3 codes, got {codes.shape}"
                    )
                eos = mx.zeros((self.audio_codec.num_codebooks, 1), dtype=mx.int32)
                eos[0, 0] = config.codec_eos_token_id
                codes = mx.concatenate([codes, eos], axis=1)
                audio = mx.zeros((rows, codes.shape[1]), dtype=mx.int32)
                audio[:-1] = codes
                audio_mask = mx.zeros(audio.shape, dtype=mx.bool_)
                audio_mask[:-1] = True
                blocks.append(audio)
                masks.append(audio_mask)
                spans.append(
                    SequenceSpan(segment_index, position, position + codes.shape[1], "audio", segment.metadata)
                )
                position += codes.shape[1]

        if not blocks:
            raise ValueError("At least one segment is required")
        result = TokenizedSequence(
            tokens=mx.concatenate(blocks, axis=1),
            mask=mx.concatenate(masks, axis=1),
            spans=spans,
        )
        validate_sequence(result, self.audio_codec.num_codebooks)
        if result.length > self.max_seq_length:
            raise ValueError(f"Sequence length {result.length} exceeds Qwen3 limit {self.max_seq_length}")
        return result
