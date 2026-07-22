from __future__ import annotations

import os
from typing import Any, Sequence

from dataprep.tokenizer import Segment, SequenceSpan, TokenizedSequence, validate_sequence


def _torch():
    import torch

    return torch


class MisoTextTokenizer:
    def __init__(self, tokenizer: Any | None = None):
        if tokenizer is None:
            from generator import load_llama3_tokenizer

            tokenizer = load_llama3_tokenizer()
        self._tokenizer = tokenizer

    def encode(self, text: str) -> list[int]:
        return list(self._tokenizer.encode(text))

    def decode(self, token_ids: Sequence[int]) -> str:
        return str(self._tokenizer.decode(list(token_ids)))


class MisoAudioCodec:
    sample_rate = 24_000
    frame_rate = 12.5
    num_codebooks = 32

    def __init__(
        self,
        codec: Any | None = None,
        *,
        device: str | None = None,
        chunk_duration_sec: float = 30.0,
    ):
        torch = _torch()
        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        if self.device == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        self.chunk_duration_sec = chunk_duration_sec
        if codec is None:
            from huggingface_hub import hf_hub_download
            from moshi.models import loaders

            try:
                from moshi_compat import patch_bitsandbytes_import_for_unquantized_layers

                patch_bitsandbytes_import_for_unquantized_layers()
            except ImportError:
                pass
            weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            codec = loaders.get_mimi(weight, device=self.device, num_codebooks=self.num_codebooks)
            codec.set_num_codebooks(self.num_codebooks)
        self._codec = codec
        self.sample_rate = int(codec.sample_rate)
        self.frame_rate = float(codec.frame_rate)

    def encode(self, audio: Any, sample_rate: int):
        torch = _torch()
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono audio, got {tuple(waveform.shape)}")
        if sample_rate != self.sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        frame_samples = int(round(self.sample_rate / self.frame_rate))
        chunk_samples = int(round(self.sample_rate * self.chunk_duration_sec))
        chunk_samples = max(frame_samples, chunk_samples // frame_samples * frame_samples)
        chunks = []
        with torch.inference_mode():
            for start in range(0, waveform.numel(), chunk_samples):
                chunk = waveform[start : start + chunk_samples].to(self.device)
                codes = self._codec.encode(chunk[None, None])
                if codes.shape[-1]:
                    chunks.append(codes[0].cpu().long())
        if not chunks:
            raise ValueError("Mimi produced no codec frames")
        return torch.cat(chunks, dim=-1)

    def decode(self, codes: Any):
        torch = _torch()
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[0] != self.num_codebooks:
            raise ValueError(f"Expected codes shaped ({self.num_codebooks}, F), got {tuple(codes.shape)}")
        frames = []
        with torch.inference_mode(), self._codec.streaming(1):
            for frame_idx in range(codes.shape[1]):
                frame = self._codec.decode(codes[:, frame_idx : frame_idx + 1][None].to(self.device))
                frames.append(frame.cpu())
        return torch.cat(frames, dim=-1).squeeze(0).squeeze(0)


class MisoTokenizer:
    max_seq_length = 2048

    def __init__(
        self,
        audio_codec: MisoAudioCodec | None = None,
        text_tokenizer: MisoTextTokenizer | None = None,
    ):
        self.audio_codec = audio_codec or MisoAudioCodec()
        self.text_tokenizer = text_tokenizer or MisoTextTokenizer()

    def apply_chat_template(self, segments: Sequence[Segment]) -> TokenizedSequence:
        torch = _torch()
        blocks = []
        masks = []
        spans: list[SequenceSpan] = []
        position = 0

        for segment_index, segment in enumerate(segments):
            text_ids = self.text_tokenizer.encode(f"[{segment.speaker}] {segment.text.lstrip()}")
            text = torch.zeros(self.audio_codec.num_codebooks + 1, len(text_ids), dtype=torch.long)
            text[-1] = torch.tensor(text_ids, dtype=torch.long)
            text_mask = torch.zeros_like(text, dtype=torch.bool)
            text_mask[-1] = True
            blocks.append(text)
            masks.append(text_mask)
            spans.append(
                SequenceSpan(segment_index, position, position + len(text_ids), "text", segment.metadata)
            )
            position += len(text_ids)

            if segment.audio_codes is not None:
                codes = torch.as_tensor(segment.audio_codes, dtype=torch.long)
                if codes.ndim != 2 or codes.shape[0] != self.audio_codec.num_codebooks:
                    raise ValueError(
                        f"Expected ({self.audio_codec.num_codebooks}, F) Miso codes, "
                        f"got {tuple(codes.shape)}"
                    )
                codes = torch.cat([codes, torch.zeros(codes.shape[0], 1, dtype=torch.long)], dim=1)
                audio = torch.zeros(self.audio_codec.num_codebooks + 1, codes.shape[1], dtype=torch.long)
                audio[:-1] = codes
                audio_mask = torch.zeros_like(audio, dtype=torch.bool)
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
            tokens=torch.cat(blocks, dim=1),
            mask=torch.cat(masks, dim=1),
            spans=spans,
        )
        validate_sequence(result, self.audio_codec.num_codebooks)
        if result.length > self.max_seq_length:
            raise ValueError(f"Sequence length {result.length} exceeds Miso limit {self.max_seq_length}")
        return result


def build_teacher_forcing_batch(sequence: TokenizedSequence):
    """Convert one prepared Miso segment to the reference model's shifted batch."""
    torch = _torch()
    text_spans = [span for span in sequence.spans if span.kind == "text"]
    audio_spans = [span for span in sequence.spans if span.kind == "audio"]
    if len(text_spans) != 1 or len(audio_spans) != 1:
        raise ValueError("Teacher-forcing verification requires exactly one text/audio segment")
    text_span, audio_span = text_spans[0], audio_spans[0]
    if text_span.segment_index != audio_span.segment_index:
        raise ValueError("Text and audio spans must belong to the same segment")

    prepared = torch.as_tensor(sequence.tokens, dtype=torch.long)
    prepared_mask = torch.as_tensor(sequence.mask, dtype=torch.bool)
    text_tokens = prepared[:, text_span.start : text_span.end].transpose(0, 1)
    text_mask = prepared_mask[:, text_span.start : text_span.end].transpose(0, 1)
    # The prepared audio span has a final all-zero EOS frame; the reference
    # teacher-forcing path predicts real codec frames and does not target EOS.
    audio_codes = prepared[:-1, audio_span.start : audio_span.end - 1]
    num_codebooks, num_frames = audio_codes.shape
    text_len = text_tokens.shape[0]
    seq_len = text_len + num_frames - 1

    tokens = torch.zeros(seq_len, num_codebooks + 1, dtype=torch.long)
    tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
    targets = torch.zeros(seq_len, num_codebooks, dtype=torch.long)
    targets_mask = torch.zeros_like(targets, dtype=torch.bool)
    tokens[:text_len] = text_tokens
    tokens_mask[:text_len] = text_mask
    if num_frames > 1:
        tokens[text_len:, :num_codebooks] = audio_codes[:, :-1].transpose(0, 1)
        tokens_mask[text_len:, :num_codebooks] = True
    target_positions = torch.arange(text_len - 1, seq_len, dtype=torch.long)
    targets[target_positions] = audio_codes.transpose(0, 1)
    targets_mask[target_positions] = True
    return (
        tokens.unsqueeze(0),
        tokens_mask.unsqueeze(0),
        targets.unsqueeze(0),
        targets_mask.unsqueeze(0),
        target_positions.unsqueeze(0),
    )
