"""Per-frame metrics for featurized Expresso rows.

Reads `DATA_ROOT/DATASET/{featurized,tokenized}/MODEL/ROW/` and writes
`DATA_ROOT/DATASET/metrics/MODEL/ROW/MODEL_metrics.{npz,json}` for `metrics_explore.py`.

Per audio frame and codebook we record predictive entropy and the teacher-forced
ground-truth NLL (the CE loss), then summarize NLL for the semantic / audio /
total code groups in two units: the average NLL per codebook, and the bitrate
that implies (`avg_nll × log2(e) × frame_rate × num_codebooks / 1000`), which is
what makes models with different frame rates and codebook counts comparable.
Scoring itself lives in `dataprep.common` (`audio_frame_metrics` and
`nll_summary`) so the dataprep tests can assert against it; this module is the
batch driver plus the record shaping that `metrics_explore.py` charts.

```bash
python analysis/model_metrics.py compute --model fish
python analysis/model_metrics.py summarize --rows 3
```

`DEFAULT_DATA_ROOT` is the repo-relative artifact root (`data`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import typer

from dataprep.common import (
    NATS_TO_BITS,
    FeaturizedSequence,
    SpanKind,
    TokenizedSequence,
    _as_numpy,
    audio_frame_metrics,
    nll_summary,
)
from dataprep.expresso import DATASET_NAME

DEFAULT_DATA_ROOT = Path("data")
MODELS = ("miso", "qwen3", "fish")

app = typer.Typer(
    add_completion=False,
    help="Compute and summarize per-frame entropy / NLL metrics for featurized rows.",
)


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
    """Table rows: NLL per group and per codebook, as avg NLL and kbit/s.

    ``nll`` is a ``(frames, num_codebooks)`` slice in nats — pass the whole row or
    one sequence's frames. Emits the ``semantic`` / ``audio`` / ``total`` groups
    from :func:`nll_summary` followed by one row per codebook.
    ``avg_nll_per_codebook`` is always per codebook, so group and codebook rows
    read on the same scale; ``share_pct`` is the row's share of the total bitrate.
    """
    summary = nll_summary(nll, frame_rate)
    total_kbits = summary["total"]["kbits_per_second"]

    def share(kbits: float) -> float:
        return 100.0 * kbits / total_kbits if total_kbits else 0.0

    records = [
        {
            "scope": name,
            "codebook": "—",
            "num_codebooks": values["num_codebooks"],
            "avg_nll_per_codebook": values["avg_nll_per_codebook"],
            "kbits_per_second": values["kbits_per_second"],
            "share_pct": share(values["kbits_per_second"]),
        }
        for name, values in summary.items()
    ]
    records.extend(
        {
            "scope": "semantic" if index == 0 else "audio",
            "codebook": f"{index:02d}",
            "num_codebooks": 1,
            "avg_nll_per_codebook": float(avg_nll),
            "kbits_per_second": float(avg_nll) * NATS_TO_BITS * frame_rate / 1000.0,
            "share_pct": share(float(avg_nll) * NATS_TO_BITS * frame_rate / 1000.0),
        }
        for index, avg_nll in enumerate(nll.mean(axis=0))
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
    row: int,
    *,
    model: str,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    dataset: str = DATASET_NAME,
) -> Path:
    """Score one featurized row and write its ``MODEL_metrics.{npz,json}``."""
    feature_dir = Path(data_root) / dataset / "featurized" / model / str(row)
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
        Path(data_root) / dataset / "tokenized" / model / str(row)
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

    metrics_dir = Path(data_root) / dataset / "metrics" / model / str(row)
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

    # Mirrors the notebook/README table: kbit/s with the avg NLL per codebook.
    print(
        f"{model} row {row}: {len(entropy):,} frames @ {frame_rate:g} Hz, "
        f"{num_codebooks} codebooks | "
        + " | ".join(
            f"{name} {values['kbits_per_second']:.3f} kbit/s "
            f"({values['avg_nll_per_codebook']:.2f} avg nll/codebook)"
            for name, values in summary.items()
        )
        + f" -> {metrics_dir}"
    )
    return metrics_dir


def available_rows(directory: Path, limit: int | None = None) -> list[int]:
    """Sorted numeric row directories under ``directory``, optionally the first ``limit``."""
    rows = sorted(
        int(path.name)
        for path in directory.glob("*")
        if path.is_dir() and path.name.isdigit()
    )
    return rows[:limit] if limit is not None else rows


def model_summary(
    model: str,
    *,
    data_root: str | Path = DEFAULT_DATA_ROOT,
    dataset: str = DATASET_NAME,
    limit: int | None = None,
) -> dict[str, Any] | None:
    """Pool every computed row for ``model`` into one :func:`nll_summary`.

    Concatenating frames (rather than averaging per-row summaries) weights each
    row by its length, so the result is the same as scoring the rows as one
    stream. Returns ``None`` when no metrics have been computed yet.
    """
    metrics_root = Path(data_root) / dataset / "metrics" / model
    if not metrics_root.is_dir():
        return None
    rows = available_rows(metrics_root, limit)
    parts: list[np.ndarray] = []
    frame_rate: float | None = None
    num_codebooks: int | None = None
    for row in rows:
        payload, arrays = load_metrics(metrics_root / str(row), model)
        parts.append(arrays["nll"])
        frame_rate = float(payload["frame_rate"])
        num_codebooks = int(payload["num_codebooks"])
    if not parts:
        return None
    nll = np.concatenate(parts)
    return {
        "model": model,
        "rows": len(rows),
        "frames": int(nll.shape[0]),
        "frame_rate": frame_rate,
        "num_codebooks": num_codebooks,
        "nll_summary": nll_summary(nll, frame_rate),
    }


SUMMARY_COLUMNS = (
    "Model",
    "Codebooks",
    "Frame rate (Hz)",
    "Semantic NLL (nats)",
    "Audio NLL/codebook avg (nats)",
    "Semantic kbit/s",
    "Audio kbit/s",
    "Total kbit/s",
)


def summary_table(summaries: list[dict[str, Any]]) -> str:
    """Markdown table of per-model bitrate and avg NLL, one row per model.

    Cells are space-padded to a fixed per-column width so the table lines up when
    printed to a terminal; it still parses as GitHub-flavored markdown.
    """
    rows = [
        [
            item["model"],
            str(item["num_codebooks"]),
            f"{item['frame_rate']:g}",
            f"{item['nll_summary']['semantic']['avg_nll_per_codebook']:.2f}",
            f"{item['nll_summary']['audio']['avg_nll_per_codebook']:.2f}",
            f"{item['nll_summary']['semantic']['kbits_per_second']:.3f}",
            f"{item['nll_summary']['audio']['kbits_per_second']:.3f}",
            f"{item['nll_summary']['total']['kbits_per_second']:.3f}",
        ]
        for item in summaries
    ]
    widths = [
        max(len(name), *(len(row[index]) for row in rows)) if rows else len(name)
        for index, name in enumerate(SUMMARY_COLUMNS)
    ]

    def line(cells: list[str] | tuple[str, ...]) -> str:
        # First column left-aligned (names); numeric columns right-aligned.
        padded = [
            cell.ljust(widths[0]) if index == 0 else cell.rjust(widths[index])
            for index, cell in enumerate(cells)
        ]
        return "| " + " | ".join(padded) + " |\n"

    rule = (
        "|"
        + "|".join(
            ("-" * (width + 2)) if index == 0 else ("-" * (width + 1) + ":")
            for index, width in enumerate(widths)
        )
        + "|\n"
    )
    return line(SUMMARY_COLUMNS) + rule + "".join(line(row) for row in rows)


@app.command()
def compute(
    model: Optional[str] = typer.Option(
        None, help=f"Model to score (default: every model found). One of {MODELS}."
    ),
    rows: Optional[int] = typer.Option(
        None, "--rows", "-n", help="Score the first N featurized rows (default: all)."
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="Artifact root."),
    dataset: str = typer.Option(DATASET_NAME, help="Dataset slug for artifact paths."),
) -> None:
    """Compute entropy/NLL metrics per model and row, and save them to disk."""
    root = data_root / dataset
    if model is not None and model not in MODELS:
        raise typer.BadParameter(f"Unknown model {model!r}; expected one of {MODELS}")
    if rows is not None and rows < 1:
        raise typer.BadParameter("--rows requires a positive row count")

    models = [model] if model else sorted(
        path.name for path in (root / "featurized").glob("*") if path.is_dir()
    )
    if not models:
        raise typer.BadParameter(f"No model directories under {root / 'featurized'}")

    for name in models:
        featurized = root / "featurized" / name
        found = available_rows(featurized, rows)
        if not found:
            raise typer.BadParameter(f"No featurized rows found under {featurized}")
        for row in found:
            compute_row_metrics(row, model=name, data_root=data_root, dataset=dataset)


@app.command()
def summarize(
    rows: Optional[int] = typer.Option(
        None, "--rows", "-n", help="Summarize the first N computed rows (default: all)."
    ),
    data_root: Path = typer.Option(DEFAULT_DATA_ROOT, help="Artifact root."),
    dataset: str = typer.Option(DATASET_NAME, help="Dataset slug for artifact paths."),
) -> None:
    """Print a summary table pooling all computed rows, one line per model."""
    if rows is not None and rows < 1:
        raise typer.BadParameter("--rows requires a positive row count")

    summaries = [
        item
        for item in (
            model_summary(name, data_root=data_root, dataset=dataset, limit=rows)
            for name in MODELS
        )
        if item is not None
    ]
    if not summaries:
        raise typer.BadParameter(
            f"No computed metrics under {data_root / dataset / 'metrics'}; "
            "run `compute` first"
        )
    counts = ", ".join(
        f"{item['model']} {item['rows']} rows / {item['frames']:,} frames"
        for item in summaries
    )
    typer.echo(f"Pooled over {counts}\n")
    typer.echo(summary_table(summaries))


if __name__ == "__main__":
    app()
