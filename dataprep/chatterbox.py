"""Chatterbox tokenize backend, shared by Chatterbox AR and Chatterbox Flash.

``ChatterboxFlashT3`` subclasses the upstream ``T3`` and differs only by an extra
``[MASK]`` row in ``speech_emb`` at id 8194 -- input-only, since ``speech_head``
keeps its 8194 outputs. Ids, tokenizer and conditioning are identical, so one
backend prepares data for both.

Unlike the RVQ backends, text and speech use separate vocabularies of the backbone
read by separate heads, concatenated along one sequence rather than
stacked as codebook channels. It maps onto the ``(L, C+1)`` grid as a
single-codebook model whose two regions are disjoint::

    [ cond | SOT text EOT | SOS y EOS ]
      col0  0             0    speech codes
      col1  0  text ids   0

``text_channel=-1`` is load-bearing: ``shards.build_sample`` slices
``tokens[s:e, :num_codebooks]``, so speech must be column 0 or the shard would
export the empty text column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from chatterbox.models.s3tokenizer import S3_SR, S3_TOKEN_RATE, S3Tokenizer
from chatterbox.models.t3.modules.t3_config import T3Config
from chatterbox.models.tokenizers import EnTokenizer
from chatterbox.models.voice_encoder import VoiceEncoder

from dataprep.common import (
    Segment,
    SequenceEmbeddingContext,
    TokenSpanKind,
    TokenSpanIdRange,
    TokenizedSequence,
    TokenizedSequenceLayout,
    TokenSequenceSpan,
)

CHATTERBOX_REPO = "ResembleAI/chatterbox"

S3GEN_FILENAME = "s3gen.safetensors"
VE_FILENAME = "ve.safetensors"
TOKENIZER_FILENAME = "tokenizer.json"

#: T3's conditioning prefix width, containing audio prompt. The same length 
#: for every utterance, containing: 1 speaker + 0 clap (unimplemented) 
#: + 32 perceiver queries + 1 emotion. The perceiver has a fixed
#: bank of 32 queries, so any audio prompt length maps to 32 frames. See
#: ``chatterbox.models.t3.modules.cond_enc.T3CondEnc.forward``.
COND_PREFIX_LEN = 34

#: Drop a segment with more than this many 25 Hz frames per text token: the
#: transcript does not cover the audio, and training on it teaches the model to
#: run past the end of its text. Inference budgets ``n_text_tokens * 6`` speech
#: tokens, but the threshold is loose because slowly-spelled letters ("A C E G I")
#: legitimately reach ~20. The floor spares short clips, whose ratio is noisy.
MAX_FRAMES_PER_TEXT_TOKEN = 60.0
MIN_FRAMES_FOR_RATIO_CHECK = 300


def _span_token_ranges(hp: T3Config) -> TokenSpanIdRange:
    """Token ids each span kind may hold, as ``{kind: (low, high)}``.

    Boundary kinds pin a single id, which is what makes EOS_TEXT checkable --
    it is id 0, indistinguishable by value from an empty frame.
    """
    return {
        TokenSpanKind.PREFIX: (0, 1),  # reserved: all zero, every channel
        TokenSpanKind.BOS_TEXT: _only(hp.start_text_token),
        TokenSpanKind.TEXT: (0, hp.text_tokens_dict_size),
        TokenSpanKind.EOS_TEXT: _only(hp.stop_text_token),
        TokenSpanKind.BOS_AUDIO: _only(hp.start_speech_token),
        TokenSpanKind.AUDIO: (0, hp.start_speech_token),
        TokenSpanKind.EOS_AUDIO: _only(hp.stop_speech_token),
    }


def _only(token_id: int) -> tuple[int, int]:
    """Range accepting exactly ``token_id``."""
    return (token_id, token_id + 1)


def _default_device(device: str | None = None) -> str:
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _fetch(repo_id: str, filename: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(repo_id=repo_id, filename=filename))


def _mono(audio: Any) -> torch.Tensor:
    waveform = torch.as_tensor(audio, dtype=torch.float32)
    if waveform.ndim == 2:
        waveform = waveform.mean(dim=0)
    if waveform.ndim != 1:
        raise ValueError(f"Expected mono audio, got {tuple(waveform.shape)}")
    return waveform


def _resample(waveform: torch.Tensor, source_rate: int, target_rate: int):
    """torchaudio, kept for speed, slight difference from librosa 
    
    Referenced in official inference repo Chatterbox Flash ``prepare_conditionals``.
    """
    if source_rate == target_rate:
        return waveform
    import torchaudio

    return torchaudio.functional.resample(waveform, source_rate, target_rate)


class ChatterboxAudioCodec:
    """S3 speech tokenizer: 16 kHz in, 25 Hz single-codebook ids out.

    Not RVQ -- S3 is an FSQ quantizer (8 scalar dims at 3 levels, base-3 packed,
    so ``3 ** 8 == 6561`` ids) with no residual structure to stack.
    """

    sample_rate = S3_SR  # 16_000
    frame_rate = float(S3_TOKEN_RATE)  # 25.0
    num_codebooks = 1

    #: Encode per segment rather than slicing one whole-channel pass. S3's
    #: encoder is bidirectional, so surrounding audio changes a segment's codes.
    segmented_encode = True

    def __init__(
        self,
        tokenizer: Any | None = None,
        *,
        device: str | None = None,
        repo_id: str = CHATTERBOX_REPO,
    ):
        self.device = _default_device(device)
        if tokenizer is None:
            tokenizer = self._load(repo_id, self.device)
        self._codec = tokenizer

    @staticmethod
    def _load(repo_id: str, device: str) -> S3Tokenizer:
        """Load just the S3 tokenizer out of the S3Gen checkpoint.

        ``s3gen.safetensors`` holds the whole vocoder, but the tokenizer's ~100
        tensors sit under a ``tokenizer.`` prefix and lift out directly, so
        there is no need to construct the full ``S3Gen``.
        """
        from safetensors.torch import load_file

        state = load_file(str(_fetch(repo_id, S3GEN_FILENAME)))
        prefix = "tokenizer."
        weights = {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if not weights:
            raise ValueError(f"No {prefix!r} weights found in {S3GEN_FILENAME}")
        codec = S3Tokenizer()
        # strict=False only for the buffers the constructor already computed;
        # everything else must match, so both lists are checked below.
        missing, unexpected = codec.load_state_dict(weights, strict=False)
        unexpected = [key for key in unexpected if key not in S3Tokenizer.ignore_state_dict_missing]
        if unexpected:
            raise ValueError(f"Unexpected S3Tokenizer weights: {sorted(unexpected)[:5]}")
        missing = [key for key in missing if key not in S3Tokenizer.ignore_state_dict_missing]
        if missing:
            raise ValueError(f"Missing S3Tokenizer weights: {sorted(missing)[:5]}")
        return codec.to(device).eval()

    def encode(self, audio: Any, sample_rate: int) -> torch.Tensor:
        """Tokenize one waveform into ``(F, 1)`` speech ids in ``[0, 6561)``."""
        waveform = _resample(_mono(audio), sample_rate, self.sample_rate)
        with torch.inference_mode():
            codes, lengths = self._codec.forward([waveform.to(self.device)])
        length = int(lengths[0].item())
        return codes[0, :length].cpu().long().unsqueeze(-1)


class ChatterboxVoiceEncoder:
    """GE2E speaker embedding: 16 kHz waveform to a 256-d vector.

    Frozen, so it is precomputed -- the only part of Chatterbox's conditioning
    worth storing. Prompt speech ids are sliced from the sequence's own audio at
    collate time, and the perceiver output is not cached because the module
    producing it is trained.
    """

    sample_rate = S3_SR
    embed_size = 256

    def __init__(
        self,
        encoder: Any | None = None,
        *,
        device: str | None = None,
        repo_id: str = CHATTERBOX_REPO,
    ):
        self.device = _default_device(device)
        if encoder is None:
            from safetensors.torch import load_file

            encoder = VoiceEncoder()
            encoder.load_state_dict(load_file(str(_fetch(repo_id, VE_FILENAME))))
            encoder = encoder.to(self.device).eval()
        self._encoder = encoder

    def embed(self, audio: Any, sample_rate: int) -> torch.Tensor:
        """Mean-pooled speaker embedding, shaped ``(256,)`` float32."""
        waveform = _resample(_mono(audio), sample_rate, self.sample_rate)
        with torch.inference_mode():
            embeds = self._encoder.embeds_from_wavs(
                [waveform.cpu().numpy()], sample_rate=self.sample_rate
            )
        embedding = torch.as_tensor(embeds, dtype=torch.float32).mean(dim=0)
        if embedding.shape != (self.embed_size,):
            raise ValueError(
                f"Expected a ({self.embed_size},) speaker embedding, "
                f"got {tuple(embedding.shape)}"
            )
        return embedding


class ChatterboxFeaturizer:
    """Placeholder: this backend prepares tokens only."""

    num_codebooks = 1

    def featurize(self, sequence: TokenizedSequence, *, include_kv: bool = False):
        raise NotImplementedError(
            "chatterbox dataprep is tokenize-only; 'inspect --stage featurize' "
            "and 'prepare' are not supported yet"
        )


class ChatterboxTokenizer:
    """Build one Chatterbox training sequence per utterance."""

    def __init__(
        self,
        model_id: str = CHATTERBOX_REPO,
        *,
        device: str | None = None,
        audio_codec: ChatterboxAudioCodec | None = None,
        voice_encoder: ChatterboxVoiceEncoder | None = None,
        text_tokenizer: Any | None = None,
        featurizer: ChatterboxFeaturizer | None = None,
        normalize_text: str = "en_us_cleaner",
        max_frames_per_text_token: float = MAX_FRAMES_PER_TEXT_TOKEN,
    ):
        self.hp = T3Config.english_only()
        self.span_token_ranges = _span_token_ranges(self.hp)
        self.max_frames_per_text_token = max_frames_per_text_token
        self.audio_codec = audio_codec or ChatterboxAudioCodec(
            device=device, repo_id=model_id
        )
        self.voice_encoder = voice_encoder or ChatterboxVoiceEncoder(
            device=device, repo_id=model_id
        )
        self.text_tokenizer = text_tokenizer or EnTokenizer(
            str(_fetch(model_id, TOKENIZER_FILENAME))
        )
        self.featurizer = featurizer or ChatterboxFeaturizer()
        self._normalize_text = self._resolve_normalizer(normalize_text)
        # Coarse bound only; the two real limits are enforced separately below.
        self.max_seq_length = self.hp.max_text_tokens + self.hp.max_speech_tokens

    @staticmethod
    def _resolve_normalizer(name: str):
        """Pick the text cleaner. ``en_us_cleaner`` is what the model saw.

        ``punc_norm``, the one chatterbox-tts ships, only tidies punctuation and
        capitalization. ``en_us_cleaner`` also expands numbers, times and phone
        numbers, so the two are not interchangeable: "2013" survives the first
        and becomes "twenty thirteen" under the second.
        """
        if name == "en_us_cleaner":
            from dataprep.chatterbox_text_norm import en_us_cleaner

            return en_us_cleaner
        if name == "punc_norm":
            from chatterbox.tts import punc_norm

            return punc_norm
        if name == "none":
            return lambda text: text
        raise ValueError(
            f"Unknown text normalizer {name!r}; "
            "expected 'en_us_cleaner', 'punc_norm' or 'none'"
        )

    def encode_text(self, text: str) -> list[int]:
        """Normalize then BPE-encode, without the SOT/EOT wrapper."""
        return list(self.text_tokenizer.encode(self._normalize_text(text)))

    def apply_chat_template(
        self,
        segments: Sequence[Segment],
        *,
        audio_codes: Mapping[int, Any] | None = None,
        speaker_embeddings: Mapping[int, Any] | None = None,
    ) -> TokenizedSequence:
        """Lay one segment out as ``[cond | SOT text EOT | SOS y EOS]``.

        One utterance per sequence: Chatterbox conditions on a single speaker
        embedding and prompt, so packed turns would describe something the model
        cannot consume, and ``TokenizedSequence.seq_id`` needs one segment id.
        """
        if len(segments) != 1:
            raise ValueError(
                "Chatterbox builds one sequence per segment; "
                f"got {len(segments)} segments (packing is not supported)"
            )
        segment = segments[0]
        audio_codes = audio_codes or {}
        codes = audio_codes.get(segment.segment_id)
        if codes is None:
            raise ValueError("Chatterbox supervised segments require audio_codes")

        codes = torch.as_tensor(codes, dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] != self.audio_codec.num_codebooks:
            raise ValueError(
                f"Expected (F, {self.audio_codec.num_codebooks}) Chatterbox codes, "
                f"got {tuple(codes.shape)}"
            )
        speech_ids = codes[:, 0]
        if speech_ids.numel() == 0:
            raise ValueError(f"Segment {segment.segment_id} produced no speech tokens")
        if int(speech_ids.max()) >= self.hp.start_speech_token:
            raise ValueError(
                f"Speech id {int(speech_ids.max())} is outside the S3 codec range "
                f"[0, {self.hp.start_speech_token})"
            )

        text_ids = self.encode_text(segment.text)
        if not text_ids:
            raise ValueError(f"Segment {segment.segment_id} normalized to empty text")

        # Limits are per stream. The messages deliberately avoid the word
        # "exceeds", which pipeline._pack_segments treats as a signal to split
        # and retry -- never valid for this backend.
        n_text = len(text_ids) + 2  # + SOT/EOT
        n_speech = int(speech_ids.numel()) + 2  # + SOS/EOS
        if n_text > self.hp.max_text_tokens:
            raise ValueError(
                f"Segment {segment.segment_id}: {n_text} text tokens is over the "
                f"Chatterbox limit of {self.hp.max_text_tokens}"
            )
        if n_speech > self.hp.max_speech_tokens:
            raise ValueError(
                f"Segment {segment.segment_id}: {n_speech} speech tokens is over the "
                f"Chatterbox limit of {self.hp.max_speech_tokens}"
            )

        frames = int(speech_ids.numel())
        ratio = frames / len(text_ids)
        if (
            frames >= MIN_FRAMES_FOR_RATIO_CHECK
            and ratio > self.max_frames_per_text_token
        ):
            raise ValueError(
                f"Segment {segment.segment_id}: {frames} speech frames for "
                f"{len(text_ids)} text tokens (ratio {ratio:.0f}, limit "
                f"{self.max_frames_per_text_token:.0f}) -- the transcript likely "
                f"does not cover the audio: {segment.text[:60]!r}"
            )

        channels = self.audio_codec.num_codebooks + 1  # speech, text
        layout = TokenizedSequenceLayout(
            num_codebooks=self.audio_codec.num_codebooks, text_channel=-1
        )
        length = COND_PREFIX_LEN + n_text + n_speech
        tokens = torch.zeros(length, channels, dtype=torch.long)
        mask = torch.zeros(length, channels, dtype=torch.bool)
        spans: list[TokenSequenceSpan] = []

        def add_span(start: int, end: int, kind: TokenSpanKind) -> None:
            spans.append(
                TokenSequenceSpan(
                    source_dataset_id=segment.source_dataset_id,
                    segment_id=segment.segment_id,
                    start=start,
                    end=end,
                    kind=kind,
                )
            )

        # [0, 34) conditioning prefix: reserved and never supervised, since these
        # frames are projections of a vector and a scalar with no token id.
        # Holding the space keeps grid positions equal to model positions, so
        # featurize can slice with the usual [start-1, end-1) offset.
        position = COND_PREFIX_LEN
        add_span(0, position, TokenSpanKind.PREFIX)

        tokens[position, -1] = self.hp.start_text_token
        mask[position, -1] = True
        add_span(position, position + 1, TokenSpanKind.BOS_TEXT)
        position += 1

        tokens[position : position + len(text_ids), -1] = torch.tensor(
            text_ids, dtype=torch.long
        )
        mask[position : position + len(text_ids), -1] = True
        add_span(position, position + len(text_ids), TokenSpanKind.TEXT)
        position += len(text_ids)

        # EOT is id 0, the same value the grid uses for "nothing here". It is a
        # real supervised target, so the mask is True and only the mask tells the
        # two apart -- leave the zero token alone. EOS_TEXT is what makes it
        # findable without going by value.
        mask[position, -1] = True
        add_span(position, position + 1, TokenSpanKind.EOS_TEXT)
        position += 1

        tokens[position, 0] = self.hp.start_speech_token
        mask[position, 0] = True
        add_span(position, position + 1, TokenSpanKind.BOS_AUDIO)
        position += 1

        tokens[position : position + frames, 0] = speech_ids
        mask[position : position + frames, 0] = True
        add_span(position, position + frames, TokenSpanKind.AUDIO)
        position += frames

        tokens[position, 0] = self.hp.stop_speech_token
        mask[position, 0] = True
        add_span(position, position + 1, TokenSpanKind.EOS_AUDIO)
        position += 1

        if position != length:
            raise AssertionError(f"Built {position} frames, expected {length}")

        result = TokenizedSequence(
            tokens=tokens, mask=mask, spans=spans, layout=layout
        )
        result.validate(self.span_token_ranges)
        return result

    def embedding_context(
        self, segment: Segment, audio: Any, sample_rate: int
    ) -> SequenceEmbeddingContext:
        """Precompute the speaker embedding for one segment's audio."""
        return SequenceEmbeddingContext(
            values={"speaker_emb": self.voice_encoder.embed(audio, sample_rate)}
        )
