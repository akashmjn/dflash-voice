"""Per-frame metrics for featurized Expresso rows.

Reads `DATA_ROOT/{MODEL_featurized,MODEL_tokenized}/ROW/` and writes
`DATA_ROOT/metrics/ROW/MODEL_metrics.{npz,json}` for `metrics_explore.py`.

Per audio frame and codebook we record predictive entropy and the teacher-forced
ground-truth NLL (the CE loss), then summarize NLL for the semantic / audio /
total code groups in two units: nats per frame, and the equivalent bitrate in
kbit/s. Scoring itself lives in `dataprep.common` (`audio_frame_metrics` and
`nll_summary`) so the dataprep tests can assert against it; this module is the
batch driver plus the record shaping that `metrics_explore.py` charts.

```bash
python notebooks/frame_metrics.py
python notebooks/frame_metrics.py --model fish
```

`DEFAULT_DATA_ROOT` is the repo-relative Expresso artifact root (`data/expresso`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dataprep.common import (
    NATS_TO_BITS,
    FeaturizedSequence,
    SpanKind,
    TokenizedSequence,
    _as_numpy,
    audio_frame_metrics,
    nll_summary,
)

DEFAULT_DATA_ROOT = Path("data/expresso")


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


def nll_breakdown_records(
    nll: np.ndarray, frame_rate: float
) -> list[dict[str, Any]]:
    """Table rows: NLL per group and per codebook, in nats/frame and kbit/s.

    ``nll`` is a ``(frames, num_codebooks)`` slice in nats — pass the whole row or
    one sequence's frames. Emits the ``semantic`` / ``audio`` / ``total`` groups
    from :func:`nll_summary` followed by one row per audio codebook, each with the
    share of total NLL it accounts for.
    """
    summary = nll_summary(nll, frame_rate)
    total_nats = summary["total"]["nats_per_frame"]
    records = [
        {
            "scope": name,
            "codebook": "—",
            "nats_per_frame": values["nats_per_frame"],
            "kbits_per_second": values["kbits_per_second"],
            "share_pct": 100.0 * values["nats_per_frame"] / total_nats
            if total_nats
            else 0.0,
        }
        for name, values in summary.items()
    ]
    per_codebook = nll.mean(axis=0)
    records.extend(
        {
            "scope": "semantic" if index == 0 else "audio",
            "codebook": f"{index:02d}",
            "nats_per_frame": float(nats),
            "kbits_per_second": float(nats) * NATS_TO_BITS * frame_rate / 1000.0,
            "share_pct": 100.0 * float(nats) / total_nats if total_nats else 0.0,
        }
        for index, nats in enumerate(per_codebook)
    )
    return records


def load_metrics(
    metrics_dir: Path, model: str
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Load one row's ``{summary payload, per-frame arrays}`` written by this module."""
    metrics_dir = Path(metrics_dir)
    payload = json.loads(
        (metrics_dir / f"{model}_metrics.json").read_text(encoding="utf-8")
    )
    return payload, dict(np.load(metrics_dir / f"{model}_metrics.npz"))


def compute_row_metrics(
    row: int, *, model: str, data_root: str | Path = DEFAULT_DATA_ROOT
) -> Path:
    """Score one featurized row and write its ``MODEL_metrics.{npz,json}``."""
    feature_dir = Path(data_root) / f"{model}_featurized" / str(row)
    if not (feature_dir / "metadata.json").exists():
        raise FileNotFoundError(
            f"Missing featurized row {feature_dir}; run "
            f"`python -m dataprep.prepare --model {model} --stage featurize` first"
        )
    sequences, metadata = FeaturizedSequence.load_all(feature_dir)
    if not sequences:
        raise ValueError(f"No featurized sequences in {feature_dir}")
    num_codebooks = sequences[0].layout.num_codebooks

    token_sequences, _ = TokenizedSequence.load_all(
        Path(data_root) / f"{model}_tokenized" / str(row)
    )

    entropy_parts: list[np.ndarray] = []
    nll_parts: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    sequence_summaries: list[dict[str, Any]] = []
    for features, tokenized, sequence_meta in zip(
        sequences, token_sequences, metadata["sequences"]
    ):
        metrics = audio_frame_metrics(features, tokenized.tokens, num_codebooks)
        entropy_parts.append(metrics["entropy"].numpy().astype(np.float32))
        nll_parts.append(metrics["nll"].numpy().astype(np.float32))
        positions.append(metrics["positions"].numpy())
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

    metrics_dir = Path(data_root) / "metrics" / str(row)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    entropy = np.concatenate(entropy_parts)
    nll = np.concatenate(nll_parts)
    sequence_positions = np.concatenate(positions)
    frame_offsets = np.cumsum([0, *(len(item) for item in entropy_parts)], dtype=np.int64)

    np.savez_compressed(
        metrics_dir / f"{model}_metrics.npz",
        entropy=entropy,
        nll=nll,
        sequence_positions=sequence_positions,
        frame_offsets=frame_offsets,
    )
    frame_rate = float(metadata["frame_rate"])
    summary = nll_summary(nll, frame_rate)
    (metrics_dir / f"{model}_metrics.json").write_text(
        json.dumps(
            {
                "model": metadata["model"],
                "row": int(metadata["row"]),
                "frame_rate": metadata["frame_rate"],
                "num_codebooks": num_codebooks,
                "codebook_entropy": (entropy.mean(axis=0) * NATS_TO_BITS).tolist(),
                "codebook_nll_bits": (nll.mean(axis=0) * NATS_TO_BITS).tolist(),
                "nll_summary": summary,
                "sequences": sequence_summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"{model} row {row}: {len(entropy):,} frames | "
        + " | ".join(
            f"{name} {values['nats_per_frame']:.2f} nats/frame "
            f"({values['kbits_per_second']:.2f} kbit/s)"
            for name, values in summary.items()
        )
        + f" -> {metrics_dir}"
    )
    return metrics_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-frame entropy and NLL summaries from featurized rows. "
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
            compute_row_metrics(row, model=model, data_root=root)


if __name__ == "__main__":
    main()
