"""Baseline speculative-decoding acceptance for Qwen3-TTS, 0.6B drafting 1.7B.

Measures the acceptance ratio alpha under standard speculative sampling, and the
expected accepted tokens per verify round tau(gamma), on two axes: `cb0` (the
semantic backbone stepping over time) and `depth` (the audio RVQ decoder stepping
over codebooks 1..15).

Offline: reads pre-dumped teacher-forced logits from
`data/expresso/featurized/ARTIFACT/ROW/` rather than running a live draft/verify
loop, so `p` and `q` at each frame share a ground-truth prefix.

Usage:
    python experiments/specdec_offline_acceptance/acceptance_qwen3.py --axis both --rows 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataprep.types import FeaturizedSequence, TokenSpanKind  # noqa: E402

#: code_predictor covers codebooks 1..15; codebook 0 comes from the talker.
NUM_DEPTH_LEVELS = 15

GAMMAS = (1, 2, 3, 4, 5, 7)


# ---------------------------------------------------------------------------
# Acceptance
# ---------------------------------------------------------------------------


def softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last axis of a ``(frames, vocab)`` block."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted, out=shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _sample_rows(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One categorical draw per row, by inverse CDF: ``(frames,)`` token ids."""
    cdf = np.cumsum(probs, axis=-1)
    cdf[:, -1] = 1.0  # guard against float drift leaving the last edge < 1
    draws = rng.random(probs.shape[0]).astype(probs.dtype)[:, None]
    return (draws < cdf).argmax(axis=-1)


def vanilla_step(
    p_probs: np.ndarray, q_probs: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Standard speculative sampling over a batch of frames.

    Takes ``(frames, vocab)`` distributions and returns ``(draft_token, accepted)``
    per frame: draft ``x ~ p``, accepted with probability ``min(1, q(x)/p(x))``.

    On rejection real speculative sampling emits a resample from the residual
    ``[q - p]_+``; we only ever count acceptances here, so that draw is skipped.
    """
    rows = np.arange(p_probs.shape[0])
    draft = _sample_rows(p_probs, rng)

    p_mass = p_probs[rows, draft]
    q_mass = q_probs[rows, draft]
    ratio = np.where(p_mass <= 0.0, 1.0, np.minimum(1.0, q_mass / np.maximum(p_mass, 1e-30)))
    return draft, rng.random(p_probs.shape[0]) < ratio


def tau(alpha: float, gamma: int) -> float:
    """Expected accepted tokens per verify round at draft block size ``gamma``.

    ``(1 - alpha^(gamma+1)) / (1 - alpha)``, which assumes acceptance is i.i.d.
    across the gamma drafted positions -- measured acceptance is mildly correlated.
    """
    if alpha >= 1.0:
        return float(gamma + 1)
    return float((1.0 - alpha ** (gamma + 1)) / (1.0 - alpha))


# ---------------------------------------------------------------------------
# Dump loading
# ---------------------------------------------------------------------------


def resolve_row(artifact: str, row: int) -> Path:
    path = REPO_ROOT / "data" / "expresso" / "featurized" / artifact / str(row)
    if not path.exists():
        raise SystemExit(
            f"no featurized dump at {path}\n"
            "  Needs teacher-forced logit dumps for both model variants."
        )
    return path


def load_depth_logits(row_dir: Path) -> tuple[list[np.ndarray], int]:
    """Per depth level, the (frames, 2048) logits, plus the sequence count."""
    sequences, _ = FeaturizedSequence.load_all(row_dir)

    used = 0
    per_level: list[list[np.ndarray]] = [[] for _ in range(NUM_DEPTH_LEVELS)]
    for features in sequences:
        spans = features.spans_of(TokenSpanKind.AUDIO)
        if not spans:
            continue
        used += 1
        for level in range(1, NUM_DEPTH_LEVELS + 1):  # head 0 is the talker's
            for span in spans:
                window = features.feature_slice_for_targets(span.start, span.end)
                per_level[level - 1].append(
                    np.asarray(features.logits[level][window], dtype=np.float32)
                )
    if not per_level[0]:
        raise ValueError(f"no audio spans in {row_dir}")
    return [np.concatenate(chunks, axis=0) for chunks in per_level], used


def load_cb0_logits(row_dir: Path) -> tuple[np.ndarray, int]:
    """Head-0 logits concatenated over frames, plus the sequence count."""
    sequences, _ = FeaturizedSequence.load_all(row_dir)
    used = 0
    chunks = []
    for features in sequences:
        spans = features.spans_of(TokenSpanKind.AUDIO)
        if spans:
            used += 1
        for span in spans:
            window = features.feature_slice_for_targets(span.start, span.end)
            chunks.append(np.asarray(features.logits[0][window], dtype=np.float32))
    if not chunks:
        raise ValueError(f"no audio spans in {row_dir}")
    return np.concatenate(chunks, axis=0), used


# ---------------------------------------------------------------------------
# Drivers
# ---------------------------------------------------------------------------


def measure_depth(args: argparse.Namespace, rng: np.random.Generator) -> dict:
    hits = np.zeros(NUM_DEPTH_LEVELS, dtype=np.int64)
    total_frames = 0
    total_sequences = 0

    for row in range(args.rows):
        draft, _ = load_depth_logits(resolve_row(args.draft_artifact, row))
        target, used = load_depth_logits(resolve_row(args.target_artifact, row))
        num_frames = min(draft[0].shape[0], target[0].shape[0])

        for level in range(NUM_DEPTH_LEVELS):
            p = softmax(draft[level][:num_frames])
            q = softmax(target[level][:num_frames])
            hits[level] += int(vanilla_step(p, q, rng)[1].sum())
        total_frames += num_frames
        total_sequences += used
        print(f"  row {row}: {used} sequence(s), {num_frames} frames")

    per_level = (hits / total_frames).round(4).tolist()
    alpha = float(hits.sum() / (total_frames * NUM_DEPTH_LEVELS))
    return {
        "axis": "depth",
        "frames_per_level": total_frames,
        "sequences": total_sequences,
        "alpha_per_level": per_level,
        "alpha": round(alpha, 4),
        "tau": {str(g): round(tau(alpha, g), 3) for g in GAMMAS},
    }


def measure_cb0(args: argparse.Namespace, rng: np.random.Generator) -> dict:
    hits = 0
    total_frames = 0
    total_sequences = 0

    for row in range(args.rows):
        draft, _ = load_cb0_logits(resolve_row(args.draft_artifact, row))
        target, used = load_cb0_logits(resolve_row(args.target_artifact, row))
        num_frames = min(draft.shape[0], target.shape[0])

        p = softmax(draft[:num_frames])
        q = softmax(target[:num_frames])
        hits += int(vanilla_step(p, q, rng)[1].sum())
        total_frames += num_frames
        total_sequences += used
        print(f"  row {row}: {used} sequence(s), {num_frames} frames")

    alpha = float(hits / total_frames)
    return {
        "axis": "cb0",
        "frames": total_frames,
        "sequences": total_sequences,
        "alpha": round(alpha, 4),
        "tau": {str(g): round(tau(alpha, g), 3) for g in GAMMAS},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-artifact", default="qwen3-0.6b", help="featurized dir for p")
    parser.add_argument("--target-artifact", default="qwen3-1.7b", help="featurized dir for q")
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
        results["cb0"] = measure_cb0(args, np.random.default_rng(args.seed))
    if args.axis in ("depth", "both"):
        print("depth (audio RVQ decoder, cb1-15):")
        results["depth"] = measure_depth(args, np.random.default_rng(args.seed))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")

    print(f"\n  {'axis':32} {'alpha':>6} " + " ".join(f"g={g:<4}" for g in GAMMAS))
    labels = {"cb0": "semantic backbone (cb0)", "depth": "audio depth decoder (cb1-15)"}
    for key in ("cb0", "depth"):
        if key not in results:
            continue
        entry = results[key]
        cells = " ".join(f"{entry['tau'][str(g)]:6.2f}" for g in GAMMAS)
        print(f"  {labels[key]:32} {entry['alpha']:6.3f} {cells}")

    if "depth" in results:
        print("\n  per-level alpha (cb1..cb15):")
        for level, rate in enumerate(results["depth"]["alpha_per_level"], start=1):
            print(f"    cb{level:<3} {rate:.3f}")

    print(f"\nresults -> {args.out}")


if __name__ == "__main__":
    main()
