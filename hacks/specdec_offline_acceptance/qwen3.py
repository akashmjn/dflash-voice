"""Speculative-decoding acceptance for Qwen3-TTS, 0.6B drafting 1.7B.

Qwen3-specific half: the two-artifact dump layout (one dir per model size) and the
depth axis. The sampling rule, accumulator and reporting are imported from
``experiments/specdec_offline_acceptance/simulate_acceptance.py``.

Two axes, because the model splits generation in two: `cb0`, the semantic backbone
stepping over time, and `depth`, the audio RVQ decoder stepping over codebooks 1..15.
The whole speech vocabulary is trained, so no live-token mask is needed.

Unmaintained: the specdec workstream moved to Chatterbox, so this is kept for the
RVQ-depth numbers in README.md rather than actively run.

Usage:
    python hacks/specdec_offline_acceptance/qwen3.py --axis both --rows 10
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import torch
import typer

# The shared core stayed in experiments/ when this moved to hacks/.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "experiments" / "specdec_offline_acceptance"),
)

from simulate_acceptance import (  # noqa: E402
    FEATURIZED_STAGE,
    GAMMAS,
    FeaturizedSequence,
    TokenSpanKind,
    audio_logits,
    axis_result,
    count_accepted,
    flatten,
    print_summary,
    resolve_row,
    write_results,
)

#: code_predictor covers codebooks 1..15; codebook 0 comes from the talker.
NUM_DEPTH_LEVELS = 15

app = typer.Typer(
    add_completion=False,
    help="Estimate specdec acceptance for Qwen3-TTS 0.6B drafting 1.7B.",
)


def load_heads(row_dir: Path, heads: Sequence[int]) -> tuple[list[torch.Tensor], int]:
    """Each head's logits concatenated over frames, plus the sequence count.

    ``FeaturizedSequence.load_all`` pulls the whole dump off disk, so all the
    requested heads are sliced out of one load rather than reloading per head.
    """
    sequences, _ = FeaturizedSequence.load_all(row_dir)
    used = sum(1 for f in sequences if f.spans_of(TokenSpanKind.AUDIO))
    per_head = [flatten([audio_logits(f, head) for f in sequences]) for head in heads]
    return per_head, used


def measure_cb0(
    draft_artifact: str, target_artifact: str, dataset: str, rows: int,
    rng: torch.Generator,
) -> dict:
    """Acceptance on the semantic backbone, pooled over rows."""
    hits = 0
    total_frames = 0
    total_sequences = 0

    for row in range(rows):
        (draft,), _ = load_heads(resolve_row(draft_artifact, row, dataset, FEATURIZED_STAGE), [0])
        (target,), used = load_heads(resolve_row(target_artifact, row, dataset, FEATURIZED_STAGE), [0])

        row_hits, row_frames = count_accepted(draft, target, rng)
        hits += row_hits
        total_frames += row_frames
        total_sequences += used
        print(f"  row {row}: {used} sequence(s), {row_frames} frames")

    return axis_result("cb0", hits, total_frames, total_sequences)


def measure_depth(
    draft_artifact: str, target_artifact: str, dataset: str, rows: int,
    rng: torch.Generator,
) -> dict:
    """Acceptance on the RVQ decoder, accumulated per depth level.

    Levels are kept separate so the per-level breakdown survives; the headline
    alpha pools them, which is what tau(gamma) along the depth axis needs.
    """
    heads = list(range(1, NUM_DEPTH_LEVELS + 1))  # head 0 is the talker's
    hits = [0] * NUM_DEPTH_LEVELS
    frames = [0] * NUM_DEPTH_LEVELS
    total_sequences = 0

    for row in range(rows):
        draft_levels, _ = load_heads(resolve_row(draft_artifact, row, dataset, FEATURIZED_STAGE), heads)
        target_levels, used = load_heads(resolve_row(target_artifact, row, dataset, FEATURIZED_STAGE), heads)

        for level, (draft, target) in enumerate(zip(draft_levels, target_levels)):
            level_hits, level_frames = count_accepted(draft, target, rng)
            hits[level] += level_hits
            frames[level] += level_frames

        total_sequences += used
        print(f"  row {row}: {used} sequence(s), {frames[0]} frames")  # equal across levels

    return axis_result(
        "depth",
        sum(hits),
        sum(frames),
        total_sequences,
        frames_per_level=frames[0],
        alpha_per_level=[round(h / f, 4) for h, f in zip(hits, frames)],
    )


@app.command()
def main(
    draft_artifact: str = typer.Option("qwen3-0.6b", help="featurized dir for p"),
    target_artifact: str = typer.Option("qwen3-1.7b", help="featurized dir for q"),
    dataset: str = typer.Option("expresso", help="dataset slug for artifact paths"),
    rows: int = typer.Option(10, help="use rows 0..N-1"),
    axis: str = typer.Option("both", help="which axis to score: depth / cb0 / both"),
    seed: int = typer.Option(0, help="sampling RNG seed"),
    out: Path = typer.Option(
        Path(__file__).with_name("results") / "qwen3.json", help="results file"
    ),
) -> None:
    if axis not in ("depth", "cb0", "both"):
        raise typer.BadParameter("axis must be one of depth / cb0 / both")

    results = {
        "draft_artifact": draft_artifact,
        "target_artifact": target_artifact,
        "draft_model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "target_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "rows": rows,
        "seed": seed,
        "gammas": list(GAMMAS),
    }

    if axis in ("cb0", "both"):
        print("codebook 0 (semantic backbone):")
        results["cb0"] = measure_cb0(
            draft_artifact, target_artifact, dataset, rows,
            torch.Generator().manual_seed(seed),
        )
    if axis in ("depth", "both"):
        print("depth (audio RVQ decoder, cb1-15):")
        results["depth"] = measure_depth(
            draft_artifact, target_artifact, dataset, rows,
            torch.Generator().manual_seed(seed),
        )

    print_summary(
        results,
        {"cb0": "semantic backbone (cb0)", "depth": "audio depth decoder (cb1-15)"},
    )

    if "depth" in results:
        print("\n  per-level alpha (cb1..cb15):")
        for level, rate in enumerate(results["depth"]["alpha_per_level"], start=1):
            print(f"    cb{level:<3} {rate:.3f}")

    write_results(results, out)


if __name__ == "__main__":
    app()
