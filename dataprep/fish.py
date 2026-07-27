from __future__ import annotations

from math import gcd
from typing import Any, Sequence

import numpy as np

from dataprep.common import (
    ForwardFeatures,
    Segment,
    SequenceSpan,
    TokenizedSequence,
    validate_sequence,
)


def _mx():
    import mlx.core as mx

    return mx


def _resample(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    divisor = gcd(source_rate, target_rate)
    return resample_poly(audio, target_rate // divisor, source_rate // divisor).astype(
        np.float32
    )


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
                chunks.append(mx.transpose(indices[0, :, :length], (1, 0)))
        if not chunks:
            raise ValueError("Fish codec produced no frames")
        return mx.concatenate(chunks, axis=0)

    def decode(self, codes: Any):
        mx = _mx()
        codes = mx.array(codes)
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"Expected codes shaped (F, {self.num_codebooks}), got {codes.shape}"
            )
        chunk_frames = max(1, int(self.frame_rate * self.chunk_duration_sec))
        chunks = []
        for start in range(0, codes.shape[0], chunk_frames):
            chunk = mx.transpose(codes[start : start + chunk_frames], (1, 0))
            lengths = mx.array([chunk.shape[1]], dtype=mx.int32)
            audio, audio_lengths = self._codec.decode(chunk[None], lengths)
            chunks.append(audio[0, 0, : int(audio_lengths[0].item())])
        return mx.concatenate(chunks)


class FishFeaturizer:
    """Teacher-force Fish's slow and fast autoregressive transformers."""

    num_codebooks = 10

    def __init__(self, model: Any):
        self._model = model

    def featurize(
        self, sequence: TokenizedSequence, *, include_kv: bool = False
    ) -> ForwardFeatures:
        mx = _mx()
        tokens = mx.array(sequence.tokens, dtype=mx.int32)
        validate_sequence(sequence, self.num_codebooks, text_channel=0)

        audio_positions = mx.array(
            np.flatnonzero(np.asarray(sequence.mask)[:, 1:].any(axis=1)),
            dtype=mx.int32,
        )
        if int(audio_positions.shape[0]) == 0:
            raise ValueError("Fish sequence contains no audio frames")
        if bool(mx.any(audio_positions == 0).item()):
            raise ValueError("Fish audio targets require a preceding context token")

        # Fish's native model layout is (batch, channels, sequence). Position t
        # predicts the semantic token and residual codebooks at position t + 1.
        model_input = mx.transpose(tokens, (1, 0))[None, :, :]
        cache = self._model.model.make_cache() if include_kv else None
        result = self._model.model(model_input, cache=cache)
        source_positions = audio_positions - 1
        slow_logits = result.logits[0, source_positions]
        slow_hiddens = result.hidden_states[0, source_positions]
        targets = tokens[audio_positions, 1:]

        if self._model.semantic_logit_bias is None:
            raise ValueError("Fish semantic logit bias is not initialized")
        semantic_logits = slow_logits + self._model.semantic_logit_bias.astype(
            slow_logits.dtype
        )
        logits = {0: semantic_logits}
        for codebook in range(1, self.num_codebooks):
            logits[codebook] = self._model.model.fast_forward(
                slow_hiddens, targets[:, :codebook]
            )

        mx.eval(*logits.values(), slow_hiddens, targets, audio_positions)
        kv_cache = None
        if cache is not None:
            kv_cache = [(layer.keys, layer.values) for layer in cache]
            mx.eval(
                *[
                    value
                    for layer in kv_cache
                    for value in layer
                    if value is not None
                ]
            )

        semantic_targets = tokens[audio_positions, 0]
        config = self._model.config
        valid_semantic = (
            semantic_targets >= config.semantic_start_token_id
        ) & (semantic_targets <= config.semantic_end_token_id)
        if not bool(mx.all(valid_semantic).item()):
            raise ValueError("Fish audio mask includes a non-semantic target token")

        return ForwardFeatures(
            logits=logits,
            hiddens=slow_hiddens,
            audio_positions=audio_positions,
            targets=targets,
            kv_cache=kv_cache,
        )


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
        self.text_tokenizer = model.tokenizer
        self.featurizer = FishFeaturizer(model)
        self.max_seq_length = int(
            getattr(
                model.config, "max_seq_len", getattr(model.config, "max_length", 32_768)
            )
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
        if codes.ndim != 2 or codes.shape[1] != self.audio_codec.num_codebooks:
            raise ValueError(
                f"Expected (F, {self.audio_codec.num_codebooks}) Fish codes, got {codes.shape}"
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
                parts=[VQPart(mx.transpose(codes, (1, 0)))],
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
            tokens_cf = self._encode_segment(segment)
            tokens = mx.transpose(tokens_cf, (1, 0))
            audio_positions = (tokens[:, 0] >= tokenizer.semantic_begin_id) & (
                tokens[:, 0] <= tokenizer.semantic_end_id
            )
            mask = mx.concatenate(
                [
                    mx.ones((tokens.shape[0], 1), dtype=mx.bool_),
                    mx.broadcast_to(
                        audio_positions[:, None],
                        (tokens.shape[0], self.audio_codec.num_codebooks),
                    ),
                ],
                axis=1,
            )
            encoded.append(tokens)
            masks.append(mask)
            spans.append(
                SequenceSpan(
                    segment_index,
                    position,
                    position + tokens.shape[0],
                    "segment",
                    segment.metadata,
                )
            )
            position += tokens.shape[0]

        if not encoded:
            raise ValueError("At least one segment is required")
        result = TokenizedSequence(
            tokens=mx.concatenate(encoded, axis=0),
            mask=mx.concatenate(masks, axis=0),
            spans=spans,
        )
        validate_sequence(result, self.audio_codec.num_codebooks, text_channel=0)
        if result.length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {result.length} exceeds Fish limit {self.max_seq_length}"
            )
        return result
