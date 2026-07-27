"""Summarize teacher-forced predictive entropy from featurized Expresso rows.

Reads `data/expresso/MODEL_featurized/ROW/{logits.pt,metadata.json}` written by
`dataprep.prepare --stage featurize` and writes the inputs for
`notebooks/entropy_explore.py`:

```text
data/expresso/MODEL_entropy/ROW/
  entropy.npz   # per-frame entropy, audio positions, per-sequence offsets
  entropy.json  # row metadata, per-sequence means, spans
```

Per-frame entropy lives in the `.npz` because a full row is millions of floats;
only the small per-codebook means and span metadata stay in JSON.

```bash
python notebooks/entropy_compute.py --model fish
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch


ARRAYS_NAME = "entropy.npz"
SUMMARY_NAME = "entropy.json"


def predictive_entropy(logits: Any) -> torch.Tensor:
    """Compute categorical predictive entropy in nats over the last axis."""
    logits = torch.as_tensor(np.asarray(logits)).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def sequence_entropy(logits: dict[int, Any], num_codebooks: int) -> torch.Tensor:
    """Stack per-codebook entropy into a (frames, num_codebooks) tensor."""
    return torch.stack(
        [predictive_entropy(logits[index]) for index in range(num_codebooks)],
        dim=1,
    )


def summarize_row(row: int, *, model: str, data_root: str | Path = "data") -> Path:
    feature_dir = Path(data_root) / "expresso" / f"{model}_featurized" / str(row)
    metadata_path = feature_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Missing featurized row {feature_dir}; run "
            f"`python -m dataprep.prepare --model {model} --stage featurize` first"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    logits = torch.load(
        feature_dir / "logits.pt", map_location="cpu", weights_only=True
    )
    if len(logits) != len(metadata["sequences"]):
        raise ValueError(f"Inconsistent featurized artifacts in {feature_dir}")
    if not logits:
        raise ValueError(f"No featurized sequences in {feature_dir}")
    num_codebooks = int(metadata["num_codebooks"])

    per_frame: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    sequences: list[dict[str, Any]] = []
    offset = 0
    for sequence_logits, sequence_meta in zip(logits, metadata["sequences"]):
        entropy = sequence_entropy(sequence_logits, num_codebooks)
        num_frames = int(sequence_meta["num_audio_frames"])
        if tuple(entropy.shape) != (num_frames, num_codebooks):
            raise ValueError(
                f"Entropy shape {tuple(entropy.shape)} does not match "
                f"{num_frames} audio frames and {num_codebooks} codebooks"
            )
        per_frame.append(entropy.numpy().astype(np.float32))
        positions.append(
            np.asarray(sequence_meta["audio_positions"], dtype=np.int32)
        )
        sequences.append(
            {
                "sequence_id": sequence_meta["sequence_id"],
                "num_frames": num_frames,
                "frame_offset": offset,
                "codebook_entropy": entropy.mean(dim=0).tolist(),
                "spans": sequence_meta["spans"],
            }
        )
        offset += num_frames

    entropy_dir = Path(data_root) / "expresso" / f"{model}_entropy" / str(row)
    entropy_dir.mkdir(parents=True, exist_ok=True)
    frame_offsets = np.cumsum(
        [0, *(len(item) for item in per_frame)], dtype=np.int64
    )
    np.savez_compressed(
        entropy_dir / ARRAYS_NAME,
        entropy=np.concatenate(per_frame),
        audio_positions=np.concatenate(positions),
        frame_offsets=frame_offsets,
    )
    summary = {
        "model": metadata["model"],
        "dataset": metadata["dataset"],
        "row": int(metadata["row"]),
        "entropy_unit": "nats",
        "frame_rate": metadata["frame_rate"],
        "num_codebooks": num_codebooks,
        "semantic_logits_masked": metadata["semantic_logits_masked"],
        "arrays": ARRAYS_NAME,
        "total_frames": int(frame_offsets[-1]),
        "sequences": sequences,
    }
    (entropy_dir / SUMMARY_NAME).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    _print_summary(summary, np.concatenate(per_frame), entropy_dir)
    return entropy_dir


def _print_summary(
    summary: dict[str, Any], per_frame: np.ndarray, entropy_dir: Path
) -> None:
    """Print enough of the row to eyeball against the marimo charts."""
    print(
        f"{summary['model']} row {summary['row']}: "
        f"{len(summary['sequences'])} sequence(s), {summary['total_frames']} frames, "
        f"{summary['num_codebooks']} codebooks -> {entropy_dir}"
    )
    row_means = per_frame.mean(axis=0)
    print(
        "  row codebook mean (nats): "
        + " ".join(f"{index:02d}={value:.3f}" for index, value in enumerate(row_means))
    )
    for item in summary["sequences"]:
        means = item["codebook_entropy"]
        print(
            f"  seq {item['sequence_id']:>3} frames={item['num_frames']:>5} "
            f"mean={sum(means) / len(means):.3f} "
            f"cb00={means[0]:.3f} cb{len(means) - 1:02d}={means[-1]:.3f} nats"
        )


def discover_rows(model: str, data_root: str | Path) -> list[int]:
    root = Path(data_root) / "expresso" / f"{model}_featurized"
    rows = sorted(
        int(path.name)
        for path in root.glob("*")
        if path.is_dir() and path.name.isdigit()
    )
    if not rows:
        raise FileNotFoundError(f"No featurized rows found under {root}")
    return rows


def summarize_rows(
    rows: Sequence[int], *, model: str, data_root: str | Path = "data"
) -> list[Path]:
    return [summarize_row(row, model=model, data_root=data_root) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute predictive entropy summaries from featurized rows."
    )
    parser.add_argument("--model", choices=("miso", "qwen3", "fish"), required=True)
    parser.add_argument(
        "--rows",
        nargs="+",
        type=int,
        default=None,
        help="Expresso rows to summarize (default: every featurized row).",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    args = parser.parse_args()
    rows = args.rows or discover_rows(args.model, args.data_root)
    for path in summarize_rows(rows, model=args.model, data_root=args.data_root):
        print(path)


if __name__ == "__main__":
    main()
