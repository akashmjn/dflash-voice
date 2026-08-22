"""Speculative-decoding acceptance for Chatterbox, Flash drafting AR.

Chatterbox-specific half: the AR/Flash dump layout and the live-vocabulary mask.
The sampling rule, accumulator and reporting live in ``acceptance``.

Single-codebook, so there is one axis rather than Qwen3's two: the 0.5B backbone
stepping over time, with no depth decoder underneath it. Both models write into
the same ``chatterbox`` artifact dir -- AR from the dataprep pipeline, Flash from
``chatterbox_dump_flash.py``, which must be run first.

Usage:
    python experiments/specdec_offline_acceptance/chatterbox.py --rows 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from acceptance import (  # noqa: E402
    GAMMAS,
    FeaturizedSequence,
    audio_logits,
    axis_result,
    count_accepted_blocks,
    load_row_logits,
    print_summary,
    resolve_row,
    write_results,
)

#: Trained S3 speech codes, ids 0..6560. EOS (6562) stays sampleable -- it is a real
#: decision the model makes -- while the untrained padding ids above it are masked.
LIVE_VOCAB_SIZE = 6561

#: Chatterbox AR's speech head width; ids past the codes are BOS/EOS then padding.
HEAD_VOCAB_SIZE = 8194
STOP_SPEECH_TOKEN = 6562

#: Sidecar written by chatterbox_dump_flash.py, alongside the AR dump.
FLASH_FEATURES = "features_flash.pt"


def live_token_mask(vocab_size: int = HEAD_VOCAB_SIZE) -> torch.Tensor:
    """Ids the model may emit: the trained speech codes plus EOS.

    Everything above is untrained padding whose rows never saw a gradient.
    """
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    mask[:LIVE_VOCAB_SIZE] = True
    mask[STOP_SPEECH_TOKEN] = True
    return mask


def load_flash_logits(row_dir: Path) -> list[list[torch.Tensor]]:
    """Per-sequence Flash logits, from the sidecar dump.

    Stored as raw tensors rather than a FeaturizedSequence dump, so spans are read
    off the AR metadata that sits alongside -- the two share tokenized sequences.
    """
    path = row_dir / FLASH_FEATURES
    if not path.exists():
        raise SystemExit(
            f"no Flash logits at {path}\n"
            "  Run: python experiments/specdec_offline_acceptance/chatterbox_dump_flash.py"
        )
    sequences, _ = FeaturizedSequence.load_all(row_dir)
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


def measure(args: argparse.Namespace, rng: torch.Generator) -> dict:
    """Acceptance over the backbone axis, one block per audio span."""
    mask = live_token_mask()
    hits = 0
    total_frames = 0
    total_sequences = 0

    for row in range(args.rows):
        row_dir = resolve_row(args.artifact, row, args.dataset)
        target_rows, used = load_row_logits(row_dir)
        draft_rows = load_flash_logits(row_dir)

        blocks = (
            (draft, target)
            for draft_chunks, target_chunks in zip(draft_rows, target_rows)
            for draft, target in zip(draft_chunks, target_chunks)
        )
        row_hits, row_frames = count_accepted_blocks(blocks, rng, mask)

        hits += row_hits
        total_frames += row_frames
        total_sequences += used
        print(f"  row {row}: {used} sequence(s), {row_frames} frames")

    return axis_result(
        "cb0", hits, total_frames, total_sequences, live_vocab=int(mask.sum())
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10, help="use rows 0..N-1")
    parser.add_argument("--artifact", default="chatterbox")
    parser.add_argument("--dataset", default="expresso")
    parser.add_argument("--seed", type=int, default=0, help="sampling RNG seed")
    parser.add_argument(
        "--out", type=Path,
        default=Path(__file__).with_name("results") / "chatterbox.json",
    )
    args = parser.parse_args()

    print("Chatterbox Flash drafting Chatterbox AR:")
    results = {
        "draft_model": "ResembleAI/chatterbox-flash",
        "target_model": "ResembleAI/chatterbox",
        "artifact": args.artifact,
        "rows": args.rows,
        "seed": args.seed,
        "gammas": list(GAMMAS),
        "cb0": measure(args, torch.Generator().manual_seed(args.seed)),
    }

    print_summary(results, {"cb0": "backbone (single codebook)"}, width=26)
    write_results(results, args.out)


if __name__ == "__main__":
    main()
