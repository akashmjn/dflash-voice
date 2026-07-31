from __future__ import annotations

from math import gcd
from typing import Any, Mapping, Sequence

import numpy as np

from dataprep.common import (
    FeaturizedSequence,
    Segment,
    TokenizedSequenceLayout,
    TokenSequenceSpan,
    SpanKind,
    TokenizedSequence,
    _as_numpy,
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


def _contiguous_spans(
    flags: np.ndarray,
    *,
    kind: SpanKind,
    source_dataset_id: int,
    segment_id: int,
    offset: int,
) -> list[TokenSequenceSpan]:
    spans: list[TokenSequenceSpan] = []
    start = None
    for index, flag in enumerate(flags):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            spans.append(
                TokenSequenceSpan(
                    source_dataset_id=source_dataset_id,
                    segment_id=segment_id,
                    start=offset + start,
                    end=offset + index,
                    kind=kind,
                )
            )
            start = None
    if start is not None:
        spans.append(
            TokenSequenceSpan(
                source_dataset_id=source_dataset_id,
                segment_id=segment_id,
                start=offset + start,
                end=offset + len(flags),
                kind=kind,
            )
        )
    return spans


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
    ) -> FeaturizedSequence:
        mx = _mx()
        tokens = mx.array(sequence.tokens, dtype=mx.int32)
        sequence.validate()

        audio_mask = np.asarray(sequence.mask)[:, 1:].any(axis=1)
        audio_positions = np.flatnonzero(audio_mask)
        if audio_positions.size == 0:
            raise ValueError("Fish sequence contains no audio frames")
        if audio_positions[0] == 0:
            raise ValueError("Fish audio targets require a preceding context token")

        # Fish's native model layout is (batch, channels, sequence). Position t
        # predicts the semantic token and residual codebooks at position t + 1.
        model_input = mx.transpose(tokens, (1, 0))[None, :, :]
        cache = self._model.model.make_cache() if include_kv else None
        result = self._model.model(model_input, cache=cache)
        slow_logits = result.logits[0, :-1]
        slow_hiddens = result.hidden_states[0, :-1]
        feature_len = sequence.length - 1
        source_positions = mx.array(audio_positions - 1, dtype=mx.int32)
        targets = tokens[mx.array(audio_positions, dtype=mx.int32), 1:]

        if self._model.semantic_logit_bias is None:
            raise ValueError("Fish semantic logit bias is not initialized")
        semantic_logits = slow_logits + self._model.semantic_logit_bias.astype(
            slow_logits.dtype
        )

        logits: dict[int, Any] = {0: semantic_logits}
        for codebook in range(1, self.num_codebooks):
            residual = self._model.model.fast_forward(
                slow_hiddens[source_positions], targets[:, :codebook]
            )
            aligned_np = np.zeros(
                (feature_len, int(residual.shape[-1])), dtype=np.float32
            )
            aligned_np[audio_positions - 1] = _as_numpy(residual).astype(np.float32)
            logits[codebook] = mx.array(aligned_np)

        mx.eval(*logits.values(), slow_hiddens)
        kv_cache = None
        if cache is not None:
            kv_cache = [(layer.keys, layer.values) for layer in cache]
            mx.eval(
                *[value for layer in kv_cache for value in layer if value is not None]
            )

        semantic_targets = tokens[mx.array(audio_positions, dtype=mx.int32), 0]
        config = self._model.config
        valid_semantic = (semantic_targets >= config.semantic_start_token_id) & (
            semantic_targets <= config.semantic_end_token_id
        )
        if not bool(mx.all(valid_semantic).item()):
            raise ValueError("Fish audio mask includes a non-semantic target token")

        return FeaturizedSequence(
            logits=logits,
            hiddens=slow_hiddens,
            spans=list(sequence.spans),
            layout=sequence.layout,
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

    def _encode_segment(self, segment: Segment, codes: Any):
        mx = _mx()
        from mlx_audio.tts.models.fish_qwen3_omni.prompt import (
            Conversation,
            Message,
            TextPart,
            VQPart,
        )

        codes = mx.array(codes, dtype=mx.int32)
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

    def apply_chat_template(
        self,
        segments: Sequence[Segment],
        *,
        audio_codes: Mapping[int, Any] | None = None,
    ) -> TokenizedSequence:
        mx = _mx()
        audio_codes = audio_codes or {}
        encoded = []
        masks = []
        spans: list[TokenSequenceSpan] = []
        position = 0
        tokenizer = self._model.tokenizer
        # Head 0 is a semantic LM head predicting the channel-0 token; heads 1..N-1
        # predict audio channels 2..N (channel 1 holds the redundant semantic code).
        num_codebooks = self.audio_codec.num_codebooks
        layout = TokenizedSequenceLayout(
            num_codebooks=num_codebooks,
            text_channel=0,
            head_targets=(0, *range(2, num_codebooks + 1)),
        )

        for segment in segments:
            codes = audio_codes.get(segment.segment_id)
            if codes is None:
                raise ValueError("Fish supervised segments require audio_codes")
            tokens_cf = self._encode_segment(segment, codes)
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

            audio_flags = np.asarray(audio_positions, dtype=bool)
            # Non-audio frames are chat/control framing around the VQ block.
            spans.extend(
                _contiguous_spans(
                    ~audio_flags,
                    kind=SpanKind.SPECIAL,
                    source_dataset_id=segment.source_dataset_id,
                    segment_id=segment.segment_id,
                    offset=position,
                )
            )
            spans.extend(
                _contiguous_spans(
                    audio_flags,
                    kind=SpanKind.AUDIO,
                    source_dataset_id=segment.source_dataset_id,
                    segment_id=segment.segment_id,
                    offset=position,
                )
            )
            # Speaker/text content is mixed into the leading special region by
            # Fish's conversation encoder; keep it under SPECIAL so consumers
            # only treat AUDIO spans as codec targets.
            position += int(tokens.shape[0])

        if not encoded:
            raise ValueError("At least one segment is required")
        spans.sort(key=lambda span: span.start)
        result = TokenizedSequence(
            tokens=mx.concatenate(encoded, axis=0),
            mask=mx.concatenate(masks, axis=0),
            spans=spans,
            layout=layout,
        )
        result.validate()
        if result.length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {result.length} exceeds Fish limit {self.max_seq_length}"
            )
        return result
