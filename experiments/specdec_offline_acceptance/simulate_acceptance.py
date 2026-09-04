"""Offline speculative-decoding acceptance ratio (alpha) estimation.

Simulates a single specdec step from dumped logits: sample from the draft
distribution, verify against the target under the speculative sampling criterion,
and count acceptances over every frame in parallel. alpha = accepted / frames,
extrapolated to tau(gamma) under an i.i.d. assumption.

Measures Chatterbox Flash, Turbo or Nano drafting Chatterbox AR. All single-codebook,
so there is one axis and no depth decoder. Target and drafts are both written by
``dump_logits.py``, which must run first; all share one tokenized dump. See README.md
for what this does and does not measure.

```bash
python experiments/specdec_offline_acceptance/simulate_acceptance.py --rows 10 --block-size 4
python experiments/specdec_offline_acceptance/simulate_acceptance.py --rows 10 --draft turbo
```

``qwen3.py`` reuses the sampling core here for a multi-codebook model.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence

import torch
import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataprep.types import (  # noqa: E402
    FeaturizedSequence,
    TokenizedSequence,
    TokenSpanKind,
)

#: Draft block sizes reported for tau(gamma).
GAMMAS = (1, 2, 4, 8)

#: Trained S3 speech codes, ids 0..6560. EOS (6562) stays sampleable; the untrained
#: padding ids above it are masked.
LIVE_VOCAB_SIZE = 6561

#: Chatterbox AR's speech head width; ids past the codes are BOS/EOS then padding.
HEAD_VOCAB_SIZE = 8194
STOP_SPEECH_TOKEN = 6562

#: Target and drafts alike come from dump_logits.py, all conditioned on the same
#: per-speaker reference clips. Tokens are shared, so all read the ``chatterbox``
#: tokenized dump.
TOKENIZED_ARTIFACT = "chatterbox"
TARGET_ARTIFACT = "chatterbox-ar"

#: This experiment keeps its dumps out of dataprep's ``featurized/`` tree, where
#: model_metrics.py would discover them. ``qwen3.py`` reads dataprep's tree instead.
SPECDEC_STAGE = "specdec_offline"
FEATURIZED_STAGE = "featurized"

#: The Flash release decodes 16 tokens a block, too wide to accept here (alpha
#: falls off with block size); 4 is the middle of the 1-8 range this sweeps.
DEFAULT_BLOCK_SIZE = 4

#: HF repo per draft, for the results JSON.
DRAFT_MODELS = {
    "flash": "ResembleAI/chatterbox-flash",
    "turbo": "ResembleAI/chatterbox-turbo",
    "nano": "ResembleAI/chatterbox-nano",
}

app = typer.Typer(
    add_completion=False,
    help="Estimate specdec acceptance for Chatterbox Flash drafting Chatterbox AR.",
)


# ---------------------------------------------------------------------------
# Speculative sampling
# ---------------------------------------------------------------------------


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Row-wise softmax with off-mask ids driven to zero probability.

    Masking rather than truncating keeps ids at their original indices, so a sampled
    id needs no remapping. ``None`` means every id is live.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def sample_tokens(
    probs: torch.Tensor, rng: torch.Generator, method: str = "inverse-cdf"
) -> torch.Tensor:
    """One categorical draw per frame: ``(frames,)`` token ids.

    ``torch.multinomial`` gives identical samples but is ~17x slower on a
    (25k, 8194) batch, so inverse-CDF is the default.
    """
    if method == "multinomial":
        return torch.multinomial(probs, 1, generator=rng).squeeze(1)

    cdf = torch.cumsum(probs, dim=-1)
    cdf[:, -1] = 1.0  # guard against float drift leaving the last edge < 1
    draws = torch.rand(probs.shape[0], 1, generator=rng, dtype=probs.dtype)
    return (draws < cdf).to(torch.uint8).argmax(dim=-1)


def speculative_step(
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    rng: torch.Generator,
    method: str = "inverse-cdf",
) -> tuple[torch.Tensor, torch.Tensor]:
    """One speculative decoding step over a batch of frames.

    Takes ``(frames, vocab)`` distributions, returns ``(token, accepted)`` per frame:
    draw ``x`` from the draft, accept with probability ``min(1, target(x)/draft(x))``.
    Real specdec resamples from the residual ``[target - draft]_+`` on rejection;
    only acceptances are counted here, so that draw is skipped.
    """
    token = sample_tokens(draft_probs, rng, method)
    draft_mass = draft_probs.gather(1, token[:, None]).squeeze(1)
    target_mass = target_probs.gather(1, token[:, None]).squeeze(1)

    ratio = torch.where(
        draft_mass <= 0.0,
        torch.ones_like(draft_mass),
        torch.clamp(target_mass / draft_mass.clamp_min(1e-30), max=1.0),
    )
    accepted = (
        torch.rand(draft_probs.shape[0], generator=rng, dtype=draft_probs.dtype) < ratio
    )
    return token, accepted


def mean_nll(logits: torch.Tensor, targets: torch.Tensor,
             mask: torch.Tensor | None = None) -> float:
    """Mean teacher-forced NLL in nats/frame over ``(frames, vocab)`` logits."""
    log_probs = torch.log_softmax(
        logits.masked_fill(~mask, float("-inf")) if mask is not None else logits, dim=-1
    )
    return float(-log_probs.gather(1, targets[:, None]).squeeze(1).mean())


def tau(alpha: float, gamma: int) -> float:
    """Expected accepted tokens per verify round at draft block size ``gamma``.

    ``(1 - alpha^(gamma+1)) / (1 - alpha)``, assuming acceptance is i.i.d. across the
    gamma drafted positions -- measured acceptance is mildly correlated.
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def count_accepted(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    rng: torch.Generator,
    mask: torch.Tensor | None = None,
    method: str = "inverse-cdf",
) -> tuple[int, int]:
    """Accepted count and frame count for one aligned ``(frames, vocab)`` sequence.

    Trims to the shorter of the two so a ragged tail cannot misalign the pair.
    """
    num_frames = min(draft_logits.shape[0], target_logits.shape[0])
    if num_frames == 0:
        return 0, 0
    draft = masked_softmax(draft_logits[:num_frames], mask)
    target = masked_softmax(target_logits[:num_frames], mask)
    return int(speculative_step(draft, target, rng, method)[1].sum()), num_frames


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------


def resolve_row(
    artifact: str, row: int, dataset: str = "expresso", stage: str = SPECDEC_STAGE
) -> Path:
    """Dump directory for one dataset row, or a usage error."""
    path = REPO_ROOT / "data" / dataset / stage / artifact / str(row)
    if not path.exists():
        raise SystemExit(
            f"no {stage} dump at {path}\n"
            "  Needs teacher-forced logit dumps for both model variants."
        )
    return path


def audio_logits(features: FeaturizedSequence, head: int = 0) -> list[torch.Tensor]:
    """``head``'s logits over each audio span, one ``(frames, vocab)`` tensor per span.

    ``feature_slice_for_targets`` handles the predict-next offset.
    """
    chunks = []
    for span in features.spans_of(TokenSpanKind.AUDIO):
        window = features.feature_slice_for_targets(span.start, span.end)
        chunks.append(torch.as_tensor(features.logits[head][window], dtype=torch.float32))
    return chunks


def audio_targets(sequence, head: int = 0) -> list[torch.Tensor]:
    """Target token ids over each audio span, aligned with :func:`audio_logits`."""
    tokens = torch.as_tensor(sequence.tokens, dtype=torch.long)
    return [
        tokens[span.start : span.end, head]
        for span in sequence.spans_of(TokenSpanKind.AUDIO)
    ]


def load_row_logits(
    row_dir: Path, head: int = 0
) -> tuple[list[list[torch.Tensor]], int]:
    """Per-sequence audio logits from a featurized dump, plus the sequence count.

    Sequences with no audio span come back empty, keeping the result index-aligned
    with a second dump over the same tokenized sequences.
    """
    sequences, _ = FeaturizedSequence.load_all(row_dir)
    per_sequence = [audio_logits(f, head) for f in sequences]
    return per_sequence, sum(1 for chunks in per_sequence if chunks)


def flatten(per_sequence: Iterable[Sequence[torch.Tensor]]) -> torch.Tensor:
    """Concatenate per-sequence span logits into one ``(frames, vocab)`` tensor."""
    logits = [c for seq in per_sequence for c in seq]
    if not logits:
        raise ValueError("no audio spans found")
    return torch.cat(logits, dim=0)


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def axis_result(
    axis: str, hits: int, frames: int, sequences: int, **extra
) -> dict:
    """One axis's alpha and tau(gamma) table, as written to the results JSON."""
    alpha = float(hits / frames)
    return {
        "axis": axis,
        "frames": frames,
        "sequences": sequences,
        "alpha": round(alpha, 4),
        "tau": {str(g): round(tau(alpha, g), 3) for g in GAMMAS},
        **extra,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_summary(results: dict, labels: dict[str, str], width: int = 32) -> None:
    """The alpha / tau(gamma) table, one row per measured axis."""
    print(f"\n  {'axis':{width}} {'alpha':>6} " + " ".join(f"g={g:<4}" for g in GAMMAS))
    for key, label in labels.items():
        entry = results.get(key)
        if entry is None:
            continue
        cells = " ".join(f"{entry['tau'][str(g)]:6.2f}" for g in GAMMAS)
        print(f"  {label:{width}} {entry['alpha']:6.3f} {cells}")


def write_results(results: dict, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nresults -> {out}")


# ---------------------------------------------------------------------------
# Chatterbox: dump layout, live vocabulary, and the CLI
# ---------------------------------------------------------------------------


def live_token_mask(vocab_size: int = HEAD_VOCAB_SIZE) -> torch.Tensor:
    """Ids the model may emit: the trained speech codes plus EOS."""
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[:LIVE_VOCAB_SIZE] = True
    mask[STOP_SPEECH_TOKEN] = True
    return mask


def pad_to_head(logits: torch.Tensor, width: int = HEAD_VOCAB_SIZE) -> torch.Tensor:
    """Right-pad a narrower head to ``width`` with ``-inf``.

    Turbo and Nano emit 6563 logits against AR's 8194. The extra ids are dead on
    both sides, but the tensors must agree in width to be compared.
    """
    if logits.shape[-1] >= width:
        return logits
    pad = logits.new_full((*logits.shape[:-1], width - logits.shape[-1]), float("-inf"))
    return torch.cat([logits, pad], dim=-1)


def load_draft_logits(draft_dir: Path) -> list[list[torch.Tensor]]:
    """Per-sequence draft logits from a dump that carries its own spans.

    Turbo and Nano condition on a variable-length speech prompt, so their sequences
    do not share AR's geometry and spans must come from the draft's own metadata.
    """
    sequences, _ = FeaturizedSequence.load_all(draft_dir)
    return [[pad_to_head(c) for c in audio_logits(f)] for f in sequences]


def load_flash_logits(
    flash_dir: Path, target_dir: Path, name: str
) -> list[list[torch.Tensor]]:
    """Per-sequence Flash logits for one row.

    Raw tensors rather than a FeaturizedSequence, so spans come off the AR metadata
    in ``target_dir`` -- the two artifacts share tokenized sequences.
    """
    path = flash_dir / name
    if not path.exists():
        raise SystemExit(
            f"no Flash logits at {path}\n"
            "  Run: python experiments/specdec_offline_acceptance/dump_logits.py"
        )
    sequences, _ = FeaturizedSequence.load_all(target_dir)
    raw = torch.load(path, weights_only=False)
    if len(raw) != len(sequences):
        raise ValueError(f"{path}: {len(raw)} Flash sequences vs {len(sequences)} AR")

    per_sequence = []
    for entry, reference in zip(raw, sequences):
        stand_in = FeaturizedSequence(
            logits={0: entry["logits"][0]},
            hiddens=entry.get("hiddens"),
            spans=reference.spans,
            layout=reference.layout,
        )
        per_sequence.append(audio_logits(stand_in))
    return per_sequence


def measure(
    rows: int,
    target_artifact: str,
    draft_artifact: str,
    dataset: str,
    flash_features: str | None,
    rng: torch.Generator,
) -> dict:
    """Acceptance over the backbone axis, plus both sides' NLL on the same frames."""
    mask = live_token_mask()
    hits = 0
    total_frames = 0
    total_sequences = 0
    draft_nll = 0.0
    target_nll = 0.0

    for row in range(rows):
        row_dir = resolve_row(target_artifact, row, dataset)
        flash_dir = resolve_row(draft_artifact, row, dataset)
        target_rows, used = load_row_logits(row_dir)
        # Flash dumps bare tensors per block size; the others dump full sequences.
        draft_rows = (
            load_flash_logits(flash_dir, row_dir, flash_features)
            if flash_features
            else load_draft_logits(flash_dir)
        )
        sequences, _ = TokenizedSequence.load_all(
            REPO_ROOT / "data" / dataset / "tokenized" / TOKENIZED_ARTIFACT / str(row)
        )

        draft = flatten(draft_rows)
        target = flatten(target_rows)
        targets = torch.cat([t for s in sequences for t in audio_targets(s)])
        row_hits, row_frames = count_accepted(draft, target, rng, mask)

        draft_nll += mean_nll(draft[:row_frames], targets[:row_frames], mask) * row_frames
        target_nll += mean_nll(target[:row_frames], targets[:row_frames], mask) * row_frames
        hits += row_hits
        total_frames += row_frames
        total_sequences += used
        print(f"  row {row}: {used} sequence(s), {row_frames} frames")

    return axis_result(
        "cb0", hits, total_frames, total_sequences,
        live_vocab=int(mask.sum()),
        draft_nll=round(draft_nll / total_frames, 4),
        target_nll=round(target_nll / total_frames, 4),
    )


@app.command()
def main(
    rows: int = typer.Option(10, help="Use rows 0..N-1."),
    draft: str = typer.Option("flash", help="Draft model: flash, turbo or nano."),
    block_size: int = typer.Option(
        DEFAULT_BLOCK_SIZE, help="Flash only: read the dump for this block size."
    ),
    target_artifact: str = typer.Option(
        TARGET_ARTIFACT, help="AR target subdir under specdec_offline/."
    ),
    draft_artifact: Optional[str] = typer.Option(
        None, help="Draft subdir under specdec_offline/ (default: chatterbox-DRAFT)."
    ),
    dataset: str = typer.Option("expresso", help="Dataset slug for artifact paths."),
    seed: int = typer.Option(0, help="Sampling RNG seed."),
    out: Optional[Path] = typer.Option(
        None, help="Results JSON (default: results/chatterbox_{bN,DRAFT}.json)."
    ),
) -> None:
    """Score one draft model and write its results JSON."""
    if rows < 1:
        raise typer.BadParameter("--rows wants a positive row count")
    if draft not in DRAFT_MODELS:
        raise typer.BadParameter(f"--draft wants one of {', '.join(DRAFT_MODELS)}")
    if block_size < 1:
        raise typer.BadParameter("--block-size wants a positive integer")

    # Only Flash dumps per block size.
    flash_features = f"features_b{block_size}.pt" if draft == "flash" else None
    tag = f"b{block_size}" if draft == "flash" else draft
    draft_artifact = draft_artifact or f"chatterbox-{draft}"
    # Default the output per draft, so a sweep cannot overwrite its own runs.
    out = out or Path(__file__).with_name("results") / f"chatterbox_{tag}.json"

    print(f"Chatterbox {draft.capitalize()} drafting Chatterbox AR:")
    results = {
        "draft_model": DRAFT_MODELS[draft],
        "target_model": "ResembleAI/chatterbox",
        "target_artifact": target_artifact,
        "draft_artifact": draft_artifact,
        "rows": rows,
        "seed": seed,
        "gammas": list(GAMMAS),
        **({"block_size": block_size} if draft == "flash" else {}),
        "cb0": measure(
            rows, target_artifact, draft_artifact, dataset, flash_features,
            torch.Generator().manual_seed(seed),
        ),
    }

    print_summary(results, {"cb0": "backbone (single codebook)"}, width=26)
    write_results(results, out)


if __name__ == "__main__":
    app()
