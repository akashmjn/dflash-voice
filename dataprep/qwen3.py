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
        self.sample_rate = int(
            getattr(speech_tokenizer, "input_sample_rate", self.sample_rate)
        )
        encoder_config = getattr(
            getattr(speech_tokenizer, "encoder_model", None), "config", None
        )
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
        return mx.transpose(codes[0, : self.num_codebooks], (1, 0))

    def decode(self, codes: Any):
        mx = _mx()
        codes = mx.array(codes)
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"Expected codes shaped (F, {self.num_codebooks}), got {codes.shape}"
            )
        audio, audio_lengths = self._codec.decode(codes[None])
        return audio[0, : int(audio_lengths[0].item())]


class Qwen3Featurizer:
    """Replay Qwen3-TTS generation with ground-truth codec frames."""

    num_codebooks = 16

    def __init__(self, model: Any):
        self._model = model

    def _prepare_segment(
        self,
        tokens,
        *,
        text_span: SequenceSpan,
        audio_span: SequenceSpan,
    ):
        mx = _mx()
        model = self._model
        text_ids = tokens[text_span.start : text_span.end, -1][None, :]
        text_embed = model.talker.text_projection(
            model.talker.get_text_embeddings()(text_ids)
        )

        tts_tokens = mx.array(
            [
                [
                    model.config.tts_bos_token_id,
                    model.config.tts_eos_token_id,
                    model.config.tts_pad_token_id,
                ]
            ],
            dtype=mx.int32,
        )
        tts_embeds = model.talker.text_projection(
            model.talker.get_text_embeddings()(tts_tokens)
        )
        tts_bos, tts_eos, tts_pad = (
            tts_embeds[:, 0:1],
            tts_embeds[:, 1:2],
            tts_embeds[:, 2:3],
        )

        prefix_ids = tokens[text_span.end : audio_span.start, 0][None, :]
        codec_embed = model.talker.get_input_embeddings()(prefix_ids)
        if int(codec_embed.shape[1]) < 2:
            raise ValueError("Qwen3 codec prefix is missing pad/BOS tokens")
        pad_count = int(codec_embed.shape[1]) - 2
        combined = mx.concatenate(
            [
                mx.broadcast_to(
                    tts_pad, (1, pad_count, int(tts_pad.shape[-1]))
                ),
                tts_bos,
            ],
            axis=1,
        )
        input_embeds = mx.concatenate(
            [
                text_embed[:, :3],
                combined + codec_embed[:, :-1],
                text_embed[:, 3:4] + codec_embed[:, -1:],
            ],
            axis=1,
        )
        text_stream = mx.concatenate([text_embed[:, 4:-5], tts_eos], axis=1)
        return input_embeds, text_stream, tts_pad

    def featurize(
        self, sequence: TokenizedSequence, *, include_kv: bool = False
    ) -> ForwardFeatures:
        mx = _mx()
        validate_sequence(sequence, self.num_codebooks, text_channel=-1)
        tokens = mx.array(sequence.tokens, dtype=mx.int32)
        config = self._model.config.talker_config

        spans_by_segment: dict[int, dict[str, SequenceSpan]] = {}
        for span in sequence.spans:
            spans_by_segment.setdefault(span.segment_index, {})[span.kind] = span

        all_positions = []
        all_targets = []
        all_hiddens = []
        all_logits: dict[int, list[Any]] = {
            index: [] for index in range(self.num_codebooks)
        }
        saved_caches = []

        for segment_index in sorted(spans_by_segment):
            spans = spans_by_segment[segment_index]
            if "text" not in spans or "audio" not in spans:
                continue
            text_span, audio_span = spans["text"], spans["audio"]
            e_t, text_stream, text_pad = self._prepare_segment(
                tokens, text_span=text_span, audio_span=audio_span
            )
            target_values = np.asarray(sequence.tokens)[
                audio_span.start : audio_span.end, :-1
            ]
            keep = target_values[:, 0] != config.codec_eos_token_id
            positions = mx.array(
                np.arange(audio_span.start, audio_span.end)[keep],
                dtype=mx.int32,
            )
            targets = mx.array(target_values[keep], dtype=mx.int32)
            if int(targets.shape[0]) == 0:
                continue

            talker_cache = self._model.talker.make_cache()
            segment_hiddens = []
            segment_logits: dict[int, list[Any]] = {
                index: [] for index in range(self.num_codebooks)
            }
            for frame_index in range(int(targets.shape[0])):
                semantic_logits, hidden = self._model.talker(
                    e_t, cache=talker_cache
                )
                semantic = targets[frame_index : frame_index + 1, 0:1]
                residuals = targets[frame_index : frame_index + 1, 1:]
                segment_logits[0].append(semantic_logits[:, -1])
                segment_hiddens.append(hidden[:, -1])

                depth_cache = self._model.talker.code_predictor.make_cache()
                depth_input = mx.concatenate(
                    [
                        hidden[:, -1:],
                        self._model.talker.get_input_embeddings()(semantic),
                    ],
                    axis=1,
                )
                for codebook in range(1, self.num_codebooks):
                    depth_logits, depth_cache, _ = (
                        self._model.talker.code_predictor(
                            depth_input,
                            cache=depth_cache,
                            generation_step=codebook - 1,
                        )
                    )
                    segment_logits[codebook].append(depth_logits[:, -1])
                    if codebook < self.num_codebooks - 1:
                        depth_input = (
                            self._model.talker.code_predictor.codec_embedding[
                                codebook - 1
                            ](residuals[:, codebook - 1 : codebook])
                        )

                if frame_index < int(text_stream.shape[1]):
                    text_embed = text_stream[:, frame_index : frame_index + 1]
                else:
                    text_embed = text_pad
                codec_embed = self._model.talker.get_input_embeddings()(semantic)
                for codebook in range(1, self.num_codebooks):
                    codec_embed = codec_embed + (
                        self._model.talker.code_predictor.codec_embedding[
                            codebook - 1
                        ](residuals[:, codebook - 1 : codebook])
                    )
                e_t = text_embed + codec_embed

            all_positions.append(positions)
            all_targets.append(targets)
            all_hiddens.append(mx.concatenate(segment_hiddens, axis=0))
            for codebook in range(self.num_codebooks):
                all_logits[codebook].append(
                    mx.concatenate(segment_logits[codebook], axis=0)
                )
            if include_kv:
                saved_caches.append(
                    [(layer.keys, layer.values) for layer in talker_cache]
                )

        if not all_targets:
            raise ValueError("Qwen3 sequence contains no non-EOS audio frames")
        positions = mx.concatenate(all_positions, axis=0)
        targets = mx.concatenate(all_targets, axis=0)
        hiddens = mx.concatenate(all_hiddens, axis=0)
        logits = {
            codebook: mx.concatenate(chunks, axis=0)
            for codebook, chunks in all_logits.items()
        }
        mx.eval(positions, targets, hiddens, *logits.values())
        if include_kv:
            mx.eval(
                *[
                    value
                    for segment_cache in saved_caches
                    for layer in segment_cache
                    for value in layer
                    if value is not None
                ]
            )
        return ForwardFeatures(
            logits=logits,
            hiddens=hiddens,
            audio_positions=positions,
            targets=targets,
            kv_cache=saved_caches if include_kv else None,
        )


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
        self.text_tokenizer = model.tokenizer
        self.featurizer = Qwen3Featurizer(model)
        talker_config = model.config.talker_config
        self.max_seq_length = int(
            getattr(talker_config, "max_position_embeddings", 2048)
        )

    def _codec_prefix(self) -> list[int]:
        config = self._model.config.talker_config
        language_id = getattr(config, "codec_language_id", {}).get(
            self.language.lower()
        )
        if language_id is None:
            prefix = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
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
        channels = self.audio_codec.num_codebooks + 1
        config = self._model.config.talker_config

        for segment_index, segment in enumerate(segments):
            chat = f"<|im_start|>assistant\n{segment.text}<|im_end|>\n<|im_start|>assistant\n"
            text_ids = list(self.text_tokenizer.encode(chat))
            text = mx.zeros((len(text_ids), channels), dtype=mx.int32)
            text[:, -1] = mx.array(text_ids, dtype=mx.int32)
            text_mask = mx.zeros(text.shape, dtype=mx.bool_)
            text_mask[:, -1] = True
            blocks.append(text)
            masks.append(text_mask)
            spans.append(
                SequenceSpan(
                    segment_index,
                    position,
                    position + len(text_ids),
                    "text",
                    segment.metadata,
                )
            )
            position += len(text_ids)

            if segment.audio_codes is not None:
                prefix_ids = self._codec_prefix()
                prefix = mx.zeros((len(prefix_ids), channels), dtype=mx.int32)
                prefix[:, 0] = mx.array(prefix_ids, dtype=mx.int32)
                prefix_mask = mx.zeros(prefix.shape, dtype=mx.bool_)
                prefix_mask[:, 0] = True
                blocks.append(prefix)
                masks.append(prefix_mask)
                position += len(prefix_ids)

                codes = mx.array(segment.audio_codes, dtype=mx.int32)
                if codes.ndim != 2 or codes.shape[1] != self.audio_codec.num_codebooks:
                    raise ValueError(
                        f"Expected (F, {self.audio_codec.num_codebooks}) Qwen3 codes, got {codes.shape}"
                    )
                eos = mx.zeros((1, self.audio_codec.num_codebooks), dtype=mx.int32)
                eos[0, 0] = config.codec_eos_token_id
                codes = mx.concatenate([codes, eos], axis=0)
                audio = mx.zeros((codes.shape[0], channels), dtype=mx.int32)
                audio[:, :-1] = codes
                audio_mask = mx.zeros(audio.shape, dtype=mx.bool_)
                audio_mask[:, :-1] = True
                blocks.append(audio)
                masks.append(audio_mask)
                spans.append(
                    SequenceSpan(
                        segment_index,
                        position,
                        position + codes.shape[0],
                        "audio",
                        segment.metadata,
                    )
                )
                position += codes.shape[0]

        if not blocks:
            raise ValueError("At least one segment is required")
        result = TokenizedSequence(
            tokens=mx.concatenate(blocks, axis=0),
            mask=mx.concatenate(masks, axis=0),
            spans=spans,
        )
        validate_sequence(result, self.audio_codec.num_codebooks, text_channel=-1)
        if result.length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {result.length} exceeds Qwen3 limit {self.max_seq_length}"
            )
        return result
