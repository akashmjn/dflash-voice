"""Teacher-force Chatterbox Flash over the tokenized dumps, for draft logits.

Dataprep featurizes the AR checkpoint only, so the Flash half of the draft/target
pair is produced here, into its own `chatterbox-flash` artifact -- `features_bN.pt`
per block size -- reading the tokenized dump both models share and staying aligned
frame-for-frame with the AR features. Flash is a block-diffusion decoder, so its logits come from
the masked forward in `block_masked_logits`.

Flash shares AR's tokenizer, ids and conditioning, differing only by an input-only
`[MASK]` row in `speech_emb` (8195 vs 8194); `speech_head` keeps 8194 outputs, so the
two logit vectors stay comparable.

```bash
python experiments/specdec_offline_acceptance/chatterbox_dump_logits.py --rows 10
python experiments/specdec_offline_acceptance/chatterbox_dump_logits.py --rows 3 --block-sizes 1,2,4,8
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import torch
import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Running this file by path puts its own directory first on sys.path, where the
# sibling chatterbox.py would shadow the installed `chatterbox` package.
_HERE = str(Path(__file__).resolve().parent)
if _HERE in sys.path:
    sys.path.remove(_HERE)

from dataprep.chatterbox import DEFAULT_EMOTION_ADV, _default_device  # noqa: E402
from dataprep.types import (  # noqa: E402
    SequenceEmbeddingContext,
    TokenizedSequence,
    TokenSpanKind,
)

FLASH_REPO = "ResembleAI/chatterbox-flash"
FLASH_T3_FILENAME = "t3_flash.safetensors"

#: Input-only [MASK] row past the AR vocabulary; speech_head cannot emit it.
MASK_TOKEN = 8194
NUM_EXTRA_TOKENS = 1

#: Tokens are shared with AR, so the input artifact differs from the output one.
TOKENIZED_ARTIFACT = "chatterbox"
FLASH_ARTIFACT = "chatterbox-flash"

app = typer.Typer(
    add_completion=False,
    help="Teacher-force Chatterbox Flash over tokenized dumps, for draft logits.",
)


def load_flash_t3(device: str, dtype: torch.dtype = torch.float32):
    """Upstream ``T3`` with ``speech_emb`` widened for Flash's ``[MASK]`` row."""
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.t3 import T3

    hp = T3Config.english_only()
    model = T3(hp)
    model.speech_emb = nn.Embedding(hp.speech_tokens_dict_size + NUM_EXTRA_TOKENS, model.dim)

    state = load_file(str(hf_hub_download(repo_id=FLASH_REPO, filename=FLASH_T3_FILENAME)))
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


def block_masked_logits(model, embeds: torch.Tensor, speech_start: int, block_size: int):
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


def featurize_block_masked(model, sequence, context, *, device, dtype, block_size):
    """Build the Chatterbox input embeddings, then score them under the block mask.

    The embedding half mirrors ``dataprep.chatterbox.ChatterboxFeaturizer``.
    """
    from chatterbox.models.t3.modules.cond_enc import T3Cond

    hp = model.hp
    tokens = torch.as_tensor(sequence.tokens, dtype=torch.long)
    spans = {span.kind: span for span in sequence.spans}
    text_start = spans[TokenSpanKind.BOS_TEXT].start
    text_end = spans[TokenSpanKind.EOS_TEXT].end
    speech_start = spans[TokenSpanKind.BOS_AUDIO].start
    audio = spans[TokenSpanKind.AUDIO]
    speech_end = spans[TokenSpanKind.EOS_AUDIO].end

    text_tokens = tokens[text_start:text_end, -1].unsqueeze(0).to(device)
    speech_tokens = tokens[speech_start:speech_end, 0].unsqueeze(0).to(device)
    speaker_prompt = tokens[
        audio.start : min(audio.start + hp.speech_cond_prompt_len, audio.end), 0
    ]
    cond = T3Cond(
        speaker_emb=torch.as_tensor(context["speaker_emb"], dtype=dtype).view(1, -1).to(device),
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


def dump_suffix(block_size: int) -> str:
    """Filename suffix for a dump; ``simulate_acceptance.py --block-size`` reads it back."""
    return f"_b{block_size}"


def dump_row(row_dir: Path, out_dir: Path, model, *, device, dtype, block_size) -> int:
    sequences, metadata = TokenizedSequence.load_all(row_dir)
    contexts = SequenceEmbeddingContext.load_all(row_dir, count=len(sequences))

    featurized = [
        featurize_block_masked(
            model, sequence, context, device=device, dtype=dtype, block_size=block_size
        )
        for sequence, context in zip(sequences, contexts)
    ]

    suffix = dump_suffix(block_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        [{"logits": {0: logits}, "hiddens": hiddens} for logits, hiddens in featurized],
        out_dir / f"features{suffix}.pt",
    )
    payload = dict(metadata)
    payload["model"] = "chatterbox-flash"
    payload["decode"] = {"mode": "blockwise", "block_size": block_size}
    (out_dir / f"metadata{suffix}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return len(featurized)


@app.command()
def main(
    rows: int = typer.Option(10, help="Dump rows 0..N-1."),
    row_start: int = typer.Option(0, help="Skip rows below this, to resume a run."),
    block_sizes: str = typer.Option(
        "1,2,4", help="Comma-separated block sizes to dump."
    ),
    artifact: str = typer.Option(
        TOKENIZED_ARTIFACT, help="Tokenized subdir to read (shared with AR)."
    ),
    flash_artifact: str = typer.Option(
        FLASH_ARTIFACT, help="Featurized subdir to write."
    ),
    dataset: str = typer.Option("expresso", help="Dataset slug for artifact paths."),
    device: Optional[str] = typer.Option(None, help="Defaults to mps/cuda/cpu."),
) -> None:
    """Dump Flash logits for each block size, into the featurized row dirs."""
    try:
        sizes = [int(part) for part in block_sizes.split(",") if part.strip()]
    except ValueError:
        raise typer.BadParameter(f"wants integers, got {block_sizes!r}")
    if not sizes or any(size < 1 for size in sizes):
        raise typer.BadParameter("wants at least one positive integer")

    resolved = _default_device(device)
    print(f"loading Chatterbox Flash on {resolved}")
    model = load_flash_t3(resolved)

    data = REPO_ROOT / "data" / dataset
    for block_size in sizes:
        print(f"block size {block_size}:")
        for row in range(row_start, rows):
            row_dir = data / "tokenized" / artifact / str(row)
            if not row_dir.exists():
                raise SystemExit(f"no tokenized dump at {row_dir}")
            out_dir = data / "featurized" / flash_artifact / str(row)
            count = dump_row(
                row_dir, out_dir, model,
                device=resolved, dtype=torch.float32, block_size=block_size,
            )
            name = f"features{dump_suffix(block_size)}.pt"
            print(f"  row {row}: {count} sequence(s) -> {out_dir}/{name}")


if __name__ == "__main__":
    app()
