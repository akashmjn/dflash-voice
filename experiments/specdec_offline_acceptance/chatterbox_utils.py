"""Chatterbox draft models: loading, tokenization, and the teacher-forced forward.

The draft checkpoints in :data:`DRAFTS` all share AR's speech codec, so their audio
spans line up frame-for-frame with dataprep's AR features. They differ in how they
are driven:

* **Flash** shares AR's text tokenizer, ids and conditioning, differing only by an
  input-only ``[MASK]`` row in ``speech_emb`` (8195 vs 8194). It reads the AR
  tokenized dump directly and is scored by :func:`featurize_block_masked`.
* **Turbo / Nano** are causal decoders sharing only the speech codec. Their text
  side is a 50276-id GPT-2 BPE, so sequences are re-tokenized by
  :func:`tokenize_for_causal_draft` and scored by ``ChatterboxFeaturizer``.

Writing the dumps to disk is ``dump_logits.py``'s job.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Running a sibling by path puts this directory first on sys.path; drop it so
# nothing here can shadow the installed `chatterbox` package.
_HERE = str(Path(__file__).resolve().parent)
if _HERE in sys.path:
    sys.path.remove(_HERE)

from dataprep.chatterbox import (  # noqa: E402
    DEFAULT_EMOTION_ADV,
    AudioPrompt,
    _fetch,
    cond_prefix_len,
    num_audio_prompt_frames,
)
from dataprep.types import (  # noqa: E402
    FeaturizedSequence,
    SequenceEmbeddingContext,
    TokenizedSequence,
    TokenizedSequenceLayout,
    TokenSequenceSpan,
    TokenSpanKind,
)

#: Input-only [MASK] row past the AR vocabulary; speech_head cannot emit it.
MASK_TOKEN = 8194
NUM_EXTRA_TOKENS = 1


@dataclass(frozen=True)
class Draft:
    """One draft checkpoint and how it departs from Chatterbox AR.

    ``llama_config`` is ``None`` only for Flash, which keeps AR's backbone; it is
    what splits the two dump paths.
    """

    repo: str
    filename: str
    llama_config: str | None = None
    #: Speech-prompt frames, causal drafts only -- Flash inherits AR's perceiver,
    #: which collapses any prompt to 32 queries. The releases use 375 (15s); the
    #: prompt comes from the utterance being scored, so it overlaps the scored
    #: frames -- see the README's conditioning caveat.
    prompt_len: int | None = None

    @property
    def is_flash(self) -> bool:
        return self.llama_config is None


DRAFTS = {
    "flash": Draft("ResembleAI/chatterbox-flash", "t3_flash.safetensors"),
    "turbo": Draft(
        "ResembleAI/chatterbox-turbo", "t3_turbo_v1.safetensors", "GPT2_medium", 150
    ),
    "nano": Draft(
        "ResembleAI/chatterbox-nano", "t3_nano_v1.safetensors", "GPT2_small", 150
    ),
}

def featurize_target_ar(
    featurizer: "ChatterboxFeaturizer",
    sequence: TokenizedSequence,
    audio_prompt: AudioPrompt,
) -> FeaturizedSequence:
    """The AR target teacher-forced on ``audio_prompt``, not the utterance's own audio.

    The perceiver collapses any prompt to 32 queries, so the 34-frame conditioning
    prefix keeps its width and the tokenized sequence carries over untouched.
    """
    return featurizer.featurize(sequence, audio_prompt=audio_prompt)


def load_flash_t3(spec: Draft, device: str, dtype: torch.dtype = torch.float32) -> "T3":
    """Upstream ``T3`` with ``speech_emb`` widened for Flash's ``[MASK]`` row."""
    import torch.nn as nn
    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.t3 import T3
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    hp = T3Config.english_only()
    model = T3(hp)
    model.speech_emb = nn.Embedding(hp.speech_tokens_dict_size + NUM_EXTRA_TOKENS, model.dim)

    state = load_file(str(hf_hub_download(repo_id=spec.repo, filename=spec.filename)))
    if "model" in state:
        state = state["model"][0]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Flash T3 weights did not match: {len(missing)} missing "
            f"{sorted(missing)[:3]}, {len(unexpected)} unexpected {sorted(unexpected)[:3]}"
        )
    return model.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------------------
# Turbo / Nano: causal drafts on a GPT-2 backbone
# ---------------------------------------------------------------------------


def causal_draft_config(spec: Draft) -> "T3Config":
    """``T3Config`` for Turbo or Nano, matching ``ChatterboxTurboTTS.from_local``."""
    from chatterbox.models.t3.modules.t3_config import T3Config

    hp = T3Config(text_tokens_dict_size=50276)
    hp.llama_config_name = spec.llama_config
    hp.speech_tokens_dict_size = 6563
    hp.input_pos_emb = None
    hp.speech_cond_prompt_len = spec.prompt_len
    hp.use_perceiver_resampler = False
    hp.emotion_adv = False
    return hp


def load_causal_draft(spec: Draft, device: str, dtype: torch.dtype = torch.float32) -> "T3":
    """Upstream ``T3`` under a Turbo/Nano config, with its released weights."""
    from chatterbox.models.t3.t3 import T3
    from safetensors.torch import load_file

    model = T3(causal_draft_config(spec))
    state = load_file(str(_fetch(spec.repo, spec.filename)))
    if "model" in state:
        state = state["model"][0]
    # T3 embeds text via text_emb; upstream deletes tfmr.wte for the same reason.
    missing, unexpected = model.load_state_dict(state, strict=False)
    unexpected = [key for key in unexpected if key != "tfmr.wte.weight"]
    if missing or unexpected:
        raise ValueError(
            f"{spec.repo} T3 weights did not match: {len(missing)} missing "
            f"{sorted(missing)[:3]}, {len(unexpected)} unexpected {sorted(unexpected)[:3]}"
        )
    return model.to(device=device, dtype=dtype).eval()


def load_causal_tokenizer(spec: Draft) -> "PreTrainedTokenizerBase":
    """The draft's own GPT-2 BPE, 50276 ids including the 19 [tag] tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(spec.repo)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


# ---------------------------------------------------------------------------
# Per-speaker reference prompts
# ---------------------------------------------------------------------------

#: One clip per Expresso speaker, cut from rows outside the scored range so a
#: sequence is never prompted with its own audio. Each covers the 150-frame prompt.
SPEAKER_PROMPT_DIR = Path(__file__).resolve().parent / "expresso_speaker_prompts"


def speaker_manifest(prompt_dir: Path = SPEAKER_PROMPT_DIR) -> dict[str, dict]:
    """The reference clips on disk, keyed by speaker."""
    return json.loads((prompt_dir / "manifest.json").read_text())


def load_speaker_references(
    prompt_dir: Path = SPEAKER_PROMPT_DIR,
) -> dict[str, AudioPrompt]:
    """Encode each speaker's reference clip to speech tokens + speaker embedding."""
    import torchaudio

    from dataprep.chatterbox import ChatterboxAudioCodec, ChatterboxVoiceEncoder

    codec = ChatterboxAudioCodec()
    encoder = ChatterboxVoiceEncoder()

    prompts = {}
    for speaker, entry in speaker_manifest(prompt_dir).items():
        audio, sample_rate = torchaudio.load(str(prompt_dir / entry["file"]))
        prompts[speaker] = AudioPrompt(
            speech_tokens=codec.encode(audio, sample_rate)[:, 0].to(torch.long),
            speaker_emb=torch.as_tensor(encoder.embed(audio, sample_rate)),
        )
    return prompts


def reference_sources(prompt_dir: Path = SPEAKER_PROMPT_DIR) -> dict[str, str]:
    """Where each speaker's clip came from, for dump metadata."""
    return {
        speaker: f"{e['source']['dataset']} {e['source']['row']}/{e['source']['segment_id']}"
        for speaker, e in speaker_manifest(prompt_dir).items()
    }


def sequence_segments(row_dir: Path, raw_dir: Path) -> list[dict]:
    """Transcript segment per tokenized sequence, matched on ``segment_id``.

    Matched on id rather than position: dataprep drops segments it cannot tokenize,
    and zipping would shift every later transcript onto the wrong audio.
    """
    _, metadata = TokenizedSequence.load_all(row_dir)
    segments = json.loads((raw_dir / "transcript_segments.json").read_text())["segments"]
    by_id = {segment["segment_id"]: segment for segment in segments}
    return [
        by_id[entry["spans"][0]["segment_id"]] for entry in metadata["sequences"]
    ]


def tokenize_for_causal_draft(
    row_dir: Path,
    raw_dir: Path,
    hp: "T3Config",
    tokenizer: "PreTrainedTokenizerBase",
    references: dict[str, AudioPrompt],
) -> tuple[list[TokenizedSequence], list[AudioPrompt]]:
    """Re-tokenize the AR row for the draft, under its own GPT-2 text vocabulary.

    Speech ids carry over from the AR dump unchanged, so only the text span is
    re-tokenized, against the transcript in ``raw_dir``. Each sequence is conditioned
    on its speaker's reference, so the prefix is sized from that rather than from the
    utterance's own audio; returns the sequences alongside those references.
    """
    from chatterbox.tts import punc_norm

    ar_sequences, _ = TokenizedSequence.load_all(row_dir)
    segments = sequence_segments(row_dir, raw_dir)

    sequences = []
    prompts = []
    for ar, segment in zip(ar_sequences, segments):
        reference = references[segment["speaker"]]
        audio_span = ar.spans_of(TokenSpanKind.AUDIO)[0]
        speech_ids = torch.as_tensor(ar.tokens)[audio_span.start : audio_span.end, 0]
        frames = int(speech_ids.numel())

        # punc_norm is what Turbo/Nano saw; dataprep's en_us_cleaner expands
        # numbers, so the two sides render the same transcript differently.
        text_ids = tokenizer(punc_norm(segment["text"]))["input_ids"]
        if not text_ids:
            raise ValueError(f"{raw_dir}: segment normalized to empty text")

        prefix_len = cond_prefix_len(hp, int(reference.prompt_tokens(hp).numel()))
        # No SOT/EOT: inference_turbo skips _ensure_BOT_EOT and the tokenizer
        # adds no bos/eos, so text is the bare BPE ids.
        length = prefix_len + len(text_ids) + (frames + 2)
        tokens = torch.zeros(length, 2, dtype=torch.long)
        spans: list[TokenSequenceSpan] = []

        position = prefix_len
        spans.append(TokenSequenceSpan(start=0, end=position, kind=TokenSpanKind.PREFIX))

        tokens[position : position + len(text_ids), -1] = torch.tensor(text_ids)
        spans.append(
            TokenSequenceSpan(
                start=position, end=position + len(text_ids), kind=TokenSpanKind.TEXT
            )
        )
        position += len(text_ids)

        tokens[position, 0] = hp.start_speech_token
        spans.append(
            TokenSequenceSpan(start=position, end=position + 1, kind=TokenSpanKind.BOS_AUDIO)
        )
        position += 1

        tokens[position : position + frames, 0] = speech_ids
        spans.append(
            TokenSequenceSpan(start=position, end=position + frames, kind=TokenSpanKind.AUDIO)
        )
        position += frames

        tokens[position, 0] = hp.stop_speech_token
        spans.append(
            TokenSequenceSpan(start=position, end=position + 1, kind=TokenSpanKind.EOS_AUDIO)
        )
        position += 1
        if position != length:
            raise AssertionError(f"laid out {position} frames, expected {length}")

        sequences.append(
            TokenizedSequence(
                tokens=tokens,
                spans=spans,
                layout=TokenizedSequenceLayout(num_codebooks=1, text_channel=-1),
                seq_id=ar.seq_id,
            )
        )
        prompts.append(reference)
    return sequences, prompts


def featurize_causal_draft(
    model: "T3",
    sequence: TokenizedSequence,
    reference: AudioPrompt,
    *,
    device: str,
    dtype: torch.dtype,
) -> FeaturizedSequence:
    """Teacher-force a causal draft on ``reference`` rather than the utterance itself.

    Mirrors ``dataprep.chatterbox.ChatterboxFeaturizer.featurize``; the drafts need
    their own copy because they have no SOT/EOT and position via GPT-2's wpe.
    """
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    hp = model.hp
    tokens = torch.as_tensor(sequence.tokens, dtype=torch.long)
    spans = {span.kind: span for span in sequence.spans}
    text = spans[TokenSpanKind.TEXT]
    speech_start = spans[TokenSpanKind.BOS_AUDIO].start
    speech_end = spans[TokenSpanKind.EOS_AUDIO].end

    text_tokens = tokens[text.start : text.end, -1].unsqueeze(0).to(device)
    speech_tokens = tokens[speech_start:speech_end, 0].unsqueeze(0).to(device)
    cond = T3Cond(
        speaker_emb=reference.speaker_emb.to(device=device, dtype=dtype).view(1, -1),
        cond_prompt_speech_tokens=reference.prompt_tokens(hp).unsqueeze(0).to(device),
        emotion_adv=None if not hp.emotion_adv else
        DEFAULT_EMOTION_ADV * torch.ones(1, 1, 1, dtype=dtype, device=device),
    )

    with torch.inference_mode():
        cond_emb = model.prepare_conditioning(cond)
        text_emb = model.text_emb(text_tokens)
        speech_emb = model.speech_emb(speech_tokens)
        # Turbo/Nano position internally via GPT-2 wpe; nothing to add here.
        if hp.input_pos_emb == "learned":
            text_emb = text_emb + model.text_pos_emb(text_tokens)
            speech_emb = speech_emb + model.speech_pos_emb(speech_tokens)
        embeds = torch.cat([cond_emb, text_emb.to(dtype), speech_emb.to(dtype)], dim=1)
        if embeds.shape[1] != sequence.unpadded_length:
            raise ValueError(
                f"Built {embeds.shape[1]} input frames for a "
                f"{sequence.unpadded_length}-frame sequence; conditioning is "
                f"{cond_emb.shape[1]} frames"
            )
        hidden = model.tfmr(
            inputs_embeds=embeds, use_cache=False, return_dict=True
        ).last_hidden_state[0]
        # tokens[L-1] is never an input, so features cover tokens[0..L-2].
        hidden = hidden[:-1].clone()
        logits = model.speech_head(hidden)

    return FeaturizedSequence(
        logits={0: logits.cpu().float()},
        hiddens=hidden.cpu().float(),
        spans=list(sequence.spans),
        layout=sequence.layout,
    )


# ---------------------------------------------------------------------------
# Forward pass simulating autoregressive blockwise masks
# ---------------------------------------------------------------------------


def inference_block_mask(context_len: int, speech_len: int, block_size: int) -> torch.Tensor:
    """``(T + S, T + S)`` bool mask for one ``[embeds | mask_embeds]`` row.

    ``T = C + S``: ``C`` frames of ``[cond | text]`` context, ``S`` of speech. In
    block form::

        TT  0
        ST  SS

    TT is causal. ST gives block ``j``'s mask queries the context plus the speech
    committed before it, ``embeds[:C + j*B]``; SS is its own block, bidirectionally.
    So each block scores against the true prefix, and no block sees another.
    """
    C, S = context_len, speech_len
    B = block_size
    T = C + S
    mask = torch.zeros(T + S, T + S, dtype=torch.bool)
    mask[:T, :T] = torch.tril(torch.ones(T, T, dtype=torch.bool))
    for lo in range(0, S, B):
        hi = min(lo + B, S)
        mask[T + lo : T + hi, : C + lo] = True          # previous [context | speech]
        mask[T + lo : T + hi, T + lo : T + hi] = True   # intra-block
    return mask


def block_masked_logits(
    model: "T3", embeds: torch.Tensor, speech_start: int, block_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Flash's masked block-diffusion logits, on the AR feature axis, in one pass.

    Appends a ``[MASK]`` embedding per speech position to ``embeds`` and runs the
    whole ``T + S`` row under :func:`inference_block_mask`, scoring every block
    against its committed prefix in one forward.

    Readout follows Fast-dLLM v2's token-shift denoising loss (arXiv 2605.30748):
    ``tokens[i]`` is predicted from the hidden state at ``i - 1``, so feature ``t``
    predicts ``tokens[t + 1]`` and the dump stays aligned with the AR one.
    """
    total_len = embeds.shape[1]
    speech_len = total_len - speech_start
    device, dtype = embeds.device, embeds.dtype

    speech_pos = torch.arange(speech_len, device=device)
    mask_ids = torch.full((1, speech_len), MASK_TOKEN, dtype=torch.long, device=device)
    mask_emb = model.speech_emb(mask_ids) + model.speech_pos_emb.emb(speech_pos)[None]
    packed = torch.cat([embeds, mask_emb.to(dtype)], dim=1)

    position_ids = torch.cat(
        [torch.arange(total_len, device=device), speech_start + speech_pos]
    )[None]
    hidden = model.tfmr(
        inputs_embeds=packed,
        attention_mask=inference_block_mask(speech_start, speech_len, block_size)[None, None].to(device),
        position_ids=position_ids,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state[0]

    # A block's first target reads off the real token before it, as a block
    # decoder would off its prefix cache; the rest off the mask slot at t.
    feature = torch.arange(speech_start - 1, total_len - 1, device=device)
    first_in_block = (feature + 1 - speech_start) % block_size == 0
    source = torch.where(first_in_block, feature, total_len + feature - speech_start)

    read_hidden = hidden[source]
    read_logits = model.speech_head(read_hidden)
    logits = torch.zeros(total_len - 1, read_logits.shape[-1], dtype=read_logits.dtype, device=device)
    hiddens = torch.zeros(total_len - 1, read_hidden.shape[-1], dtype=read_hidden.dtype, device=device)
    logits[feature] = read_logits
    hiddens[feature] = read_hidden
    return logits, hiddens


def featurize_block_masked(
    model: "T3",
    sequence: TokenizedSequence,
    audio_prompt: AudioPrompt,
    *,
    device: str,
    dtype: torch.dtype,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Build the Chatterbox input embeddings, then score them under the block mask.

    The embedding half mirrors ``dataprep.chatterbox.ChatterboxFeaturizer``, prompted
    on ``audio_prompt``; Flash shares AR's perceiver, so the prefix stays 34 frames.
    """
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    hp = model.hp
    tokens = torch.as_tensor(sequence.tokens, dtype=torch.long)
    spans = {span.kind: span for span in sequence.spans}
    text_start = spans[TokenSpanKind.BOS_TEXT].start
    text_end = spans[TokenSpanKind.EOS_TEXT].end
    speech_start = spans[TokenSpanKind.BOS_AUDIO].start
    speech_end = spans[TokenSpanKind.EOS_AUDIO].end

    text_tokens = tokens[text_start:text_end, -1].unsqueeze(0).to(device)
    speech_tokens = tokens[speech_start:speech_end, 0].unsqueeze(0).to(device)
    speaker_prompt = audio_prompt.prompt_tokens(hp)
    cond = T3Cond(
        speaker_emb=torch.as_tensor(audio_prompt.speaker_emb, dtype=dtype)
        .view(1, -1)
        .to(device),
        cond_prompt_speech_tokens=speaker_prompt.unsqueeze(0).to(device),
        emotion_adv=DEFAULT_EMOTION_ADV * torch.ones(1, 1, 1, dtype=dtype, device=device),
    )

    with torch.inference_mode():
        cond_emb = model.prepare_conditioning(cond)
        text_emb = model.text_emb(text_tokens) + model.text_pos_emb(text_tokens)
        speech_emb = model.speech_emb(speech_tokens) + model.speech_pos_emb(speech_tokens)
        embeds = torch.cat([cond_emb, text_emb.to(dtype), speech_emb.to(dtype)], dim=1)
        if embeds.shape[1] != sequence.unpadded_length:
            raise ValueError(
                f"Built {embeds.shape[1]} input frames for a "
                f"{sequence.unpadded_length}-frame sequence"
            )
        # Bucket padding is dropped: the blockwise mask is explicit.
        offset = cond_emb.shape[1] + text_emb.shape[1]
        logits, hiddens = block_masked_logits(model, embeds, offset, block_size)
    return logits.cpu().float(), hiddens.cpu().float()
