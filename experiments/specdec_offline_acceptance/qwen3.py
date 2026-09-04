"""Speculative-decoding acceptance for Qwen3-TTS, 0.6B drafting 1.7B.

Qwen3-specific half: the two-artifact dump layout (one dir per model size) and the
depth axis. The sampling rule, accumulator and reporting live in ``acceptance``.

Two axes, because the model splits generation in two: `cb0`, the semantic backbone
stepping over time, and `depth`, the audio RVQ decoder stepping over codebooks 1..15.
The whole speech vocabulary is trained, so no live-token mask is needed.

Usage:
    python experiments/specdec_offline_acceptance/qwen3.py --axis both --rows 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import torch

sys.path.insert(0, str(Path(__file__).parent))

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


def load_heads(row_dir: Path, heads: Sequence[int]) -> tuple[list[torch.Tensor], int]:
    """Each head's logits concatenated over frames, plus the sequence count.

    ``FeaturizedSequence.load_all`` pulls the whole dump off disk, so all the
    requested heads are sliced out of one load rather than reloading per head.
    """
    sequences, _ = FeaturizedSequence.load_all(row_dir)
    used = sum(1 for f in sequences if f.spans_of(TokenSpanKind.AUDIO))
    per_head = [flatten([audio_logits(f, head) for f in sequences]) for head in heads]
    return per_head, used


def measure_cb0(args: argparse.Namespace, rng: torch.Generator) -> dict:
    """Acceptance on the semantic backbone, pooled over rows."""
    hits = 0
    total_frames = 0
    total_sequences = 0

    for row in range(args.rows):
        (draft,), _ = load_heads(resolve_row(args.draft_artifact, row, args.dataset, FEATURIZED_STAGE), [0])
        (target,), used = load_heads(resolve_row(args.target_artifact, row, args.dataset, FEATURIZED_STAGE), [0])

        row_hits, row_frames = count_accepted(draft, target, rng)
        hits += row_hits
        total_frames += row_frames
        total_sequences += used
        print(f"  row {row}: {used} sequence(s), {row_frames} frames")

    return axis_result("cb0", hits, total_frames, total_sequences)


def measure_depth(args: argparse.Namespace, rng: torch.Generator) -> dict:
    """Acceptance on the RVQ decoder, accumulated per depth level.

    Levels are kept separate so the per-level breakdown survives; the headline
    alpha pools them, which is what tau(gamma) along the depth axis needs.
    """
    heads = list(range(1, NUM_DEPTH_LEVELS + 1))  # head 0 is the talker's
    hits = [0] * NUM_DEPTH_LEVELS
    frames = [0] * NUM_DEPTH_LEVELS
    total_sequences = 0

    for row in range(args.rows):
        draft_levels, _ = load_heads(resolve_row(args.draft_artifact, row, args.dataset, FEATURIZED_STAGE), heads)
        target_levels, used = load_heads(resolve_row(args.target_artifact, row, args.dataset, FEATURIZED_STAGE), heads)

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-artifact", default="qwen3-0.6b", help="featurized dir for p")
    parser.add_argument("--target-artifact", default="qwen3-1.7b", help="featurized dir for q")
    parser.add_argument("--dataset", default="expresso")
    parser.add_argument("--rows", type=int, default=10, help="use rows 0..N-1")
    parser.add_argument("--axis", choices=("depth", "cb0", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).with_name("results") / "qwen3.json",
    )
    args = parser.parse_args()

    results = {
        "draft_artifact": args.draft_artifact,
        "target_artifact": args.target_artifact,
        "draft_model": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
        "target_model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "rows": args.rows,
        "seed": args.seed,
        "gammas": list(GAMMAS),
    }

    if args.axis in ("cb0", "both"):
        print("codebook 0 (semantic backbone):")
        results["cb0"] = measure_cb0(args, torch.Generator().manual_seed(args.seed))
    if args.axis in ("depth", "both"):
        print("depth (audio RVQ decoder, cb1-15):")
        results["depth"] = measure_depth(args, torch.Generator().manual_seed(args.seed))

    print_summary(
        results,
        {"cb0": "semantic backbone (cb0)", "depth": "audio depth decoder (cb1-15)"},
    )

    if "depth" in results:
        print("\n  per-level alpha (cb1..cb15):")
        for level, rate in enumerate(results["depth"]["alpha_per_level"], start=1):
            print(f"    cb{level:<3} {rate:.3f}")

    write_results(results, args.out)


if __name__ == "__main__":
    main()
