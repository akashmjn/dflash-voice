"""Entropy computation and summarization for featurized Expresso rows.

Reads `DATA_ROOT/MODEL_featurized/ROW/{features.pt,metadata.json}` and writes
`DATA_ROOT/entropy/ROW/MODEL_entropy.{npz,json}` for `entropy_explore.py`.

```bash
python notebooks/entropy_utils.py
python notebooks/entropy_utils.py --model fish
```

`DEFAULT_DATA_ROOT` is the repo-relative Expresso artifact root (`data/expresso`).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataprep.common import FeaturizedSequence, SpanKind

DEFAULT_DATA_ROOT = Path("data/expresso")
NATS_TO_BITS = 1.0 / math.log(2.0)


def group_entropy_bits(entropy_bits: np.ndarray, mid: int) -> dict[str, np.ndarray]:
    """Per-frame entropy (bits) for each codebook group: ``{label: (frames,)}``.

    ``semantic`` is codebook 0; the remaining codebooks split into two halves.
    """
    return {
        "semantic": entropy_bits[:, 0],
        "audio_half_1": entropy_bits[:, 1 : mid + 1].mean(axis=1),
        "audio_half_2": entropy_bits[:, mid + 1 :].mean(axis=1),
    }


def codebook_mean_records(entropy_bits: np.ndarray) -> list[dict[str, Any]]:
    """Bar-chart data: mean entropy (bits) per codebook over the given frames."""
    return [
        {"codebook": f"{index:02d}", "entropy_bits": float(value)}
        for index, value in enumerate(entropy_bits.mean(axis=0))
    ]


def group_frame_records(
    entropy_bits: np.ndarray,
    mid: int,
    x_field: str,
    frame_rate: float | None = None,
) -> list[dict[str, Any]]:
    """Long-form raw per-frame records for the group time-series charts.

    One row per (frame, group). ``x_field="second"`` maps frame index to time
    via ``frame_rate``; any other name emits the raw frame index. Smoothing is
    left to the chart (Altair loess), so no pooling happens here.
    """
    scale = 1.0 / frame_rate if x_field == "second" and frame_rate else 1.0
    return [
        {x_field: index * scale, "group": label, "entropy_bits": float(value)}
        for label, series in group_entropy_bits(entropy_bits, mid).items()
        for index, value in enumerate(series)
    ]


def predictive_entropy(logits: Any) -> torch.Tensor:
    logits = torch.as_tensor(np.asarray(logits)).float()
    log_probs = torch.log_softmax(logits, dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def audio_entropy_frames(
    features: FeaturizedSequence, num_codebooks: int
) -> tuple[torch.Tensor, torch.Tensor]:
    per_frame_parts: list[torch.Tensor] = []
    position_parts: list[torch.Tensor] = []
    for span in features.spans_of(SpanKind.AUDIO):
        pred = features.feature_slice_for_targets(span.start, span.end)
        entropy = torch.stack(
            [
                predictive_entropy(features.logits[index][pred])
                for index in range(num_codebooks)
            ],
            dim=1,
        )
        per_frame_parts.append(entropy)
        position_parts.append(torch.arange(span.start, span.end, dtype=torch.int32))
    if not per_frame_parts:
        raise ValueError("Sequence has no audio spans")
    return torch.cat(per_frame_parts, dim=0), torch.cat(position_parts, dim=0)


def load_entropy(entropy_dir: Path, model: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    entropy_dir = Path(entropy_dir)
    payload = json.loads(
        (entropy_dir / f"{model}_entropy.json").read_text(encoding="utf-8")
    )
    arrays = dict(np.load(entropy_dir / f"{model}_entropy.npz"))
    if "sequence_positions" not in arrays:
        arrays["sequence_positions"] = arrays["audio_positions"]
    return payload, arrays


def compute_entropy_row(
    row: int, *, model: str, data_root: str | Path = DEFAULT_DATA_ROOT
) -> Path:
    feature_dir = Path(data_root) / f"{model}_featurized" / str(row)
    if not (feature_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"Missing featurized row {feature_dir}; run "
            f"`python -m dataprep.prepare --model {model} --stage featurize` first"
        )
    sequences, metadata = FeaturizedSequence.load_all(feature_dir)
    if not sequences:
        raise ValueError(f"No featurized sequences in {feature_dir}")
    num_codebooks = int(metadata["num_codebooks"])

    per_frame: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    sequence_summaries: list[dict[str, Any]] = []
    for features, sequence_meta in zip(sequences, metadata["sequences"]):
        entropy, sequence_positions = audio_entropy_frames(features, num_codebooks)
        per_frame.append(entropy.numpy().astype(np.float32))
        positions.append(sequence_positions.numpy())
        sequence_summaries.append(
            {
                "sequence_id": sequence_meta["sequence_id"],
                "spans": [
                    {
                        "segment_id": span.segment_id,
                        "start": span.start,
                        "end": span.end,
                    }
                    for span in features.spans_of(SpanKind.AUDIO)
                ],
            }
        )

    entropy_dir = Path(data_root) / "entropy" / str(row)
    entropy_dir.mkdir(parents=True, exist_ok=True)
    per_frame_concat = np.concatenate(per_frame)
    sequence_positions = np.concatenate(positions)
    frame_offsets = np.cumsum([0, *(len(item) for item in per_frame)], dtype=np.int64)

    np.savez_compressed(
        entropy_dir / f"{model}_entropy.npz",
        entropy=per_frame_concat,
        sequence_positions=sequence_positions,
        frame_offsets=frame_offsets,
    )
    (entropy_dir / f"{model}_entropy.json").write_text(
        json.dumps(
            {
                "model": metadata["model"],
                "row": int(metadata["row"]),
                "frame_rate": metadata["frame_rate"],
                "num_codebooks": num_codebooks,
                "codebook_entropy": per_frame_concat.mean(axis=0).tolist(),
                "sequences": sequence_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    mean_bits = float(per_frame_concat.mean() * NATS_TO_BITS)
    print(
        f"{model} row {row}: {len(per_frame_concat):,} frames, "
        f"{mean_bits:.3f} bits mean -> {entropy_dir}"
    )
    return entropy_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute predictive entropy summaries from featurized rows. "
            f"Defaults: all MODEL_featurized under {DEFAULT_DATA_ROOT}, "
            "every featurized row."
        ),
    )
    parser.add_argument("--model", choices=("miso", "qwen3", "fish"), default=None)
    parser.add_argument("--rows", nargs="+", type=int, default=None)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Expresso artifact root (default: {DEFAULT_DATA_ROOT}).",
    )
    args = parser.parse_args()
    root = args.data_root

    models = (
        [args.model]
        if args.model
        else sorted(
            path.name.removesuffix("_featurized")
            for path in root.glob("*_featurized")
            if path.is_dir()
        )
    )
    if not models:
        raise FileNotFoundError(f"No MODEL_featurized directories under {root}")

    for model in models:
        rows = args.rows
        if rows is None:
            featurized = root / f"{model}_featurized"
            rows = sorted(
                int(path.name)
                for path in featurized.glob("*")
                if path.is_dir() and path.name.isdigit()
            )
            if not rows:
                raise FileNotFoundError(f"No featurized rows found under {featurized}")
        for row in rows:
            compute_entropy_row(row, model=model, data_root=root)


if __name__ == "__main__":
    main()
