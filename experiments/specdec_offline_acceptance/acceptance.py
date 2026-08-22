"""Shared offline speculative-decoding acceptance measurement.

Model-independent half of the experiment: the speculative sampling rule, the
acceptance accumulator, dump loading, and result formatting. Per-model scripts
(``qwen3.py``, ``chatterbox.py``) supply the loading
parameters -- which artifact dirs hold `p` and `q`, how many axes the model has,
and which vocabulary ids are live -- and call in here to do the measuring.

Everything is offline: `p` and `q` are read from pre-dumped teacher-forced logits,
so at every frame both models share a ground-truth prefix.

Runs on CPU torch tensors throughout: the dumps are already torch, and its softmax
and cumsum are multithreaded where numpy's are not -- worth ~6x on these shapes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataprep.types import FeaturizedSequence, TokenSpanKind  # noqa: E402

#: Draft block sizes reported for tau(gamma).
GAMMAS = (1, 2, 3, 4, 5, 7)


# ---------------------------------------------------------------------------
# Speculative sampling
# ---------------------------------------------------------------------------


def masked_softmax(logits: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    """Row-wise softmax, with the off-mask ids driven to zero probability.

    Masking rather than truncating leaves surviving ids at their original indices,
    so a sampled id needs no remapping. ``None`` means every id is live.
    """
    if mask is not None:
        logits = logits.masked_fill(~mask, float("-inf"))
    return torch.softmax(logits, dim=-1)


def _sample_rows(probs: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
    """One categorical draw per row: ``(frames,)`` token ids.

    Inverse-CDF rather than ``torch.multinomial``, which is markedly slower at
    these shapes and caps the vocabulary it will sample from.
    """
    cdf = torch.cumsum(probs, dim=-1)
    cdf[:, -1] = 1.0  # guard against float drift leaving the last edge < 1
    draws = torch.rand(probs.shape[0], 1, generator=rng, dtype=probs.dtype)
    return (draws < cdf).to(torch.uint8).argmax(dim=-1)


def vanilla_step(
    p_probs: torch.Tensor, q_probs: torch.Tensor, rng: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    """Standard speculative sampling over a batch of frames.

    Takes ``(frames, vocab)`` distributions and returns ``(draft_token, accepted)``
    per frame: draft ``x ~ p``, accepted with probability ``min(1, q(x)/p(x))``.

    On rejection real speculative sampling emits a resample from the residual
    ``[q - p]_+``; we only ever count acceptances here, so that draw is skipped.
    """
    draft = _sample_rows(p_probs, rng)
    p_mass = p_probs.gather(1, draft[:, None]).squeeze(1)
    q_mass = q_probs.gather(1, draft[:, None]).squeeze(1)

    ratio = torch.where(
        p_mass <= 0.0,
        torch.ones_like(p_mass),
        torch.clamp(q_mass / p_mass.clamp_min(1e-30), max=1.0),
    )
    accepted = torch.rand(p_probs.shape[0], generator=rng, dtype=p_probs.dtype) < ratio
    return draft, accepted


def tau(alpha: float, gamma: int) -> float:
    """Expected accepted tokens per verify round at draft block size ``gamma``.

    ``(1 - alpha^(gamma+1)) / (1 - alpha)``, which assumes acceptance is i.i.d.
    across the gamma drafted positions -- measured acceptance is mildly correlated.
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


def count_accepted(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
    rng: torch.Generator,
    mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    """Accepted count and frame count for one aligned ``(frames, vocab)`` block.

    Trims to the shorter of the two so a ragged tail can never misalign `p` and `q`.
    """
    num_frames = min(draft_logits.shape[0], target_logits.shape[0])
    if num_frames == 0:
        return 0, 0
    p = masked_softmax(draft_logits[:num_frames], mask)
    q = masked_softmax(target_logits[:num_frames], mask)
    return int(vanilla_step(p, q, rng)[1].sum()), num_frames


def count_accepted_blocks(
    blocks: Iterable[tuple[torch.Tensor, torch.Tensor]],
    rng: torch.Generator,
    mask: torch.Tensor | None = None,
) -> tuple[int, int]:
    """``count_accepted`` summed over several blocks.

    Blocking is the caller's choice -- per row, per sequence, or per span -- and
    changes nothing statistically: acceptance is drawn per frame either way.
    """
    hits = 0
    frames = 0
    for draft, target in blocks:
        block_hits, block_frames = count_accepted(draft, target, rng, mask)
        hits += block_hits
        frames += block_frames
    return hits, frames


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------


def resolve_row(artifact: str, row: int, dataset: str = "expresso") -> Path:
    """Featurized dump directory for one dataset row, or a usage error."""
    path = REPO_ROOT / "data" / dataset / "featurized" / artifact / str(row)
    if not path.exists():
        raise SystemExit(
            f"no featurized dump at {path}\n"
            "  Needs teacher-forced logit dumps for both model variants."
        )
    return path


def audio_logits(features: FeaturizedSequence, head: int = 0) -> list[torch.Tensor]:
    """``head``'s logits over each audio span, teacher-forced onto its targets.

    One ``(frames, vocab)`` tensor per span; ``feature_slice_for_targets`` handles
    the predict-next offset rather than reimplementing it. The dumps are already
    torch tensors, so this is a view, not a conversion.
    """
    chunks = []
    for span in features.spans_of(TokenSpanKind.AUDIO):
        window = features.feature_slice_for_targets(span.start, span.end)
        chunks.append(torch.as_tensor(features.logits[head][window], dtype=torch.float32))
    return chunks


def load_row_logits(
    row_dir: Path, head: int = 0
) -> tuple[list[list[torch.Tensor]], int]:
    """Per-sequence audio logits from a featurized dump, plus the sequence count.

    Sequences with no audio span come back as an empty chunk list, so the result
    stays index-aligned with a second dump over the same tokenized sequences.
    """
    sequences, _ = FeaturizedSequence.load_all(row_dir)
    per_sequence = [audio_logits(f, head) for f in sequences]
    return per_sequence, sum(1 for chunks in per_sequence if chunks)


def flatten(per_sequence: Iterable[Sequence[torch.Tensor]]) -> torch.Tensor:
    """Concatenate per-sequence span chunks into one ``(frames, vocab)`` block."""
    chunks = [c for seq in per_sequence for c in seq]
    if not chunks:
        raise ValueError("no audio spans found")
    return torch.cat(chunks, dim=0)


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
