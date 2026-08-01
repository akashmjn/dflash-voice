from __future__ import annotations

import os
from typing import Any, Mapping, Sequence

# Moshi decode on Apple MPS needs this before Torch initializes MPS kernels, so
# it must stay above the torch import below (directly, and via dataprep.common).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from dataprep.common import (
    FeaturizedSequence,
    Segment,
    TokenizedSequenceLayout,
    TokenSequenceSpan,
    SpanKind,
    TokenizedSequence,
)


def _load_text_tokenizer(tokenizer: Any | None = None):
    if tokenizer is None:
        from generator import load_llama3_tokenizer

        tokenizer = load_llama3_tokenizer()
    return tokenizer


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
                from moshi_compat import (
                    patch_bitsandbytes_import_for_unquantized_layers,
                )

                patch_bitsandbytes_import_for_unquantized_layers()
            except ImportError:
                pass
            weight = hf_hub_download(loaders.DEFAULT_REPO, loaders.MIMI_NAME)
            codec = loaders.get_mimi(
                weight, device=self.device, num_codebooks=self.num_codebooks
            )
            codec.set_num_codebooks(self.num_codebooks)
        self._codec = codec
        self.sample_rate = int(codec.sample_rate)
        self.frame_rate = float(codec.frame_rate)

    def encode(self, audio: Any, sample_rate: int):
        waveform = torch.as_tensor(audio, dtype=torch.float32)
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0)
        if waveform.ndim != 1:
            raise ValueError(f"Expected mono audio, got {tuple(waveform.shape)}")
        if sample_rate != self.sample_rate:
            import torchaudio

            waveform = torchaudio.functional.resample(
                waveform, sample_rate, self.sample_rate
            )

        frame_samples = int(round(self.sample_rate / self.frame_rate))
        chunk_samples = int(round(self.sample_rate * self.chunk_duration_sec))
        chunk_samples = max(
            frame_samples, chunk_samples // frame_samples * frame_samples
        )
        chunks = []
        with torch.inference_mode():
            for start in range(0, waveform.numel(), chunk_samples):
                chunk = waveform[start : start + chunk_samples].to(self.device)
                codes = self._codec.encode(chunk[None, None])
                if codes.shape[-1]:
                    chunks.append(codes[0].cpu().long())
        if not chunks:
            raise ValueError("Mimi produced no codec frames")
        return torch.cat(chunks, dim=-1).transpose(0, 1).contiguous()

    def decode(self, codes: Any):
        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] != self.num_codebooks:
            raise ValueError(
                f"Expected codes shaped (F, {self.num_codebooks}), got {tuple(codes.shape)}"
            )
        codes_cf = codes.transpose(0, 1).contiguous()
        frames = []
        with torch.inference_mode(), self._codec.streaming(1):
            for frame_idx in range(codes_cf.shape[1]):
                frame = self._codec.decode(
                    codes_cf[:, frame_idx : frame_idx + 1][None].to(self.device)
                )
                frames.append(frame.cpu())
        return torch.cat(frames, dim=-1).squeeze(0).squeeze(0)


class MisoFeaturizer:
    """Teacher-force MisoTTS without enabling its inference KV caches."""

    num_codebooks = 32

    def __init__(self, generator: Any | None = None, *, device: str | None = None):
        self._generator = generator
        self.device = device

    def _load_generator(self):
        if self._generator is None:
            from generator import load_miso_8b, resolve_inference_config

            config = resolve_inference_config(device=self.device)
            self._generator = load_miso_8b(
                device=config.model_device, dtype=config.dtype
            )
        return self._generator

    def featurize(
        self, sequence: TokenizedSequence, *, include_kv: bool = False
    ) -> FeaturizedSequence:
        """Teacher-forced forward pass: feature ``t`` predicts ``tokens[t+1]``."""
        if include_kv:
            raise NotImplementedError(
                "Miso teacher-forced KV export is not available with disabled caches"
            )
        from torchtune.modules.common_utils import disable_kv_cache

        generator = self._load_generator()
        model = generator._model
        device = generator.model_device
        num_codebooks = self.num_codebooks

        tokens = torch.as_tensor(sequence.tokens, dtype=torch.long)
        mask = torch.as_tensor(sequence.mask, dtype=torch.bool)
        audio_channels = torch.tensor(sequence.layout.audio_channels)

        # tokens[L-1] is never an input, so features cover tokens[0..L-2].
        feature_len = sequence.length - 1
        input_tokens = tokens[:feature_len].unsqueeze(0).to(device)
        input_mask = mask[:feature_len].unsqueeze(0).to(device)
        # The decoder predicts codebook k from c0..c_{k-1} of the *same* frame, so
        # it is conditioned on the frame h_t predicts: tokens[t+1].
        predicted_codes = (
            tokens[1 : feature_len + 1][:, audio_channels].unsqueeze(0).to(device)
        )

        # Causal masks must be passed explicitly. torchtune only falls back to
        # implicit causal attention when `kv_cache is None and mask is None`
        # (MultiHeadAttention.forward), and disable_kv_cache leaves the cache
        # object in place -- so without these the attention is bidirectional.
        # That leaks later codebooks into earlier ones along the decoder axis.
        backbone_mask = torch.tril(
            torch.ones(feature_len, feature_len, dtype=torch.bool, device=device)
        ).unsqueeze(0)
        decoder_mask = (
            torch.tril(
                torch.ones(num_codebooks, num_codebooks, dtype=torch.bool, device=device)
            )
            .unsqueeze(0)
            .expand(feature_len, num_codebooks, num_codebooks)
        )

        with (
            torch.inference_mode(),
            disable_kv_cache(model.backbone),
            disable_kv_cache(model.decoder),
        ):
            dtype = next(model.parameters()).dtype

            embeds = model._embed_tokens(input_tokens)
            hiddens = model.backbone(
                (embeds * input_mask.unsqueeze(-1)).sum(dim=2), mask=backbone_mask
            ).to(dtype=dtype)
            semantic_logits = model.codebook0_head(hiddens)

            # Decoder runs over the codebook axis: [h_t, emb(c0) .. emb(c30)].
            # c31 is predicted but never fed back, and codebook k sits at offset
            # k * audio_vocab_size in the shared embedding table.
            context_codes = predicted_codes[..., :-1]
            offsets = model.config.audio_vocab_size * torch.arange(
                num_codebooks - 1, device=device
            )
            decoder_input = torch.cat(
                [hiddens.unsqueeze(2), model.audio_embeddings(context_codes + offsets)],
                dim=2,
            )
            decoder_h = model.decoder(
                model.projection(decoder_input)
                .view(feature_len, num_codebooks, -1)
                .to(dtype=dtype),
                mask=decoder_mask,
            ).view(1, feature_len, num_codebooks, -1)
            # audio_head[j] reads slot j+1 (context h_t, c0..cj) to predict c_{j+1}.
            residual_logits = torch.einsum(
                "bsid,idv->bsiv", decoder_h[:, :, 1:, :], model.audio_head
            )

        logits = {0: semantic_logits[0].cpu().float()}
        for codebook in range(1, num_codebooks):
            logits[codebook] = residual_logits[0, :, codebook - 1].cpu().float()

        return FeaturizedSequence(
            logits=logits,
            hiddens=hiddens[0].cpu().float(),
            spans=list(sequence.spans),
            layout=sequence.layout,
        )


class MisoTokenizer:
    max_seq_length = 2048

    def __init__(
        self,
        audio_codec: MisoAudioCodec | None = None,
        text_tokenizer: Any | None = None,
        featurizer: MisoFeaturizer | None = None,
    ):
        self.audio_codec = audio_codec or MisoAudioCodec()
        self.text_tokenizer = _load_text_tokenizer(text_tokenizer)
        self.featurizer = featurizer or MisoFeaturizer()

    def apply_chat_template(
        self,
        segments: Sequence[Segment],
        *,
        audio_codes: Mapping[int, Any] | None = None,
    ) -> TokenizedSequence:
        audio_codes = audio_codes or {}
        blocks = []
        masks = []
        spans: list[TokenSequenceSpan] = []
        position = 0
        channels = self.audio_codec.num_codebooks + 1
        layout = TokenizedSequenceLayout(
            num_codebooks=self.audio_codec.num_codebooks, text_channel=-1
        )

        for segment in segments:
            text_ids = list(
                self.text_tokenizer.encode(
                    f"[{segment.speaker_id}] {segment.text.lstrip()}"
                )
            )
            text = torch.zeros(len(text_ids), channels, dtype=torch.long)
            text[:, -1] = torch.tensor(text_ids, dtype=torch.long)
            text_mask = torch.zeros_like(text, dtype=torch.bool)
            text_mask[:, -1] = True
            blocks.append(text)
            masks.append(text_mask)
            spans.append(
                TokenSequenceSpan(
                    source_dataset_id=segment.source_dataset_id,
                    segment_id=segment.segment_id,
                    start=position,
                    end=position + len(text_ids),
                    kind=SpanKind.TEXT,
                )
            )
            position += len(text_ids)

            codes = audio_codes.get(segment.segment_id)
            if codes is not None:
                codes = torch.as_tensor(codes, dtype=torch.long)
                if codes.ndim != 2 or codes.shape[1] != self.audio_codec.num_codebooks:
                    raise ValueError(
                        f"Expected (F, {self.audio_codec.num_codebooks}) Miso codes, "
                        f"got {tuple(codes.shape)}"
                    )
                audio = torch.zeros(codes.shape[0], channels, dtype=torch.long)
                audio[:, :-1] = codes
                audio_mask = torch.zeros_like(audio, dtype=torch.bool)
                audio_mask[:, :-1] = True
                blocks.append(audio)
                masks.append(audio_mask)
                spans.append(
                    TokenSequenceSpan(
                        source_dataset_id=segment.source_dataset_id,
                        segment_id=segment.segment_id,
                        start=position,
                        end=position + codes.shape[0],
                        kind=SpanKind.AUDIO,
                    )
                )
                position += codes.shape[0]

                eos = torch.zeros(1, channels, dtype=torch.long)
                eos_mask = torch.zeros_like(eos, dtype=torch.bool)
                eos_mask[:, :-1] = True
                blocks.append(eos)
                masks.append(eos_mask)
                spans.append(
                    TokenSequenceSpan(
                        source_dataset_id=segment.source_dataset_id,
                        segment_id=segment.segment_id,
                        start=position,
                        end=position + 1,
                        kind=SpanKind.SPECIAL,
                    )
                )
                position += 1

        if not blocks:
            raise ValueError("At least one segment is required")
        result = TokenizedSequence(
            tokens=torch.cat(blocks, dim=0),
            mask=torch.cat(masks, dim=0),
            spans=spans,
            layout=layout,
        )
        result.validate()
        if result.length > self.max_seq_length:
            raise ValueError(
                f"Sequence length {result.length} exceeds Miso limit {self.max_seq_length}"
            )
        return result
