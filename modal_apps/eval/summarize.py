"""Collect scored eval runs off the Modal Volume into one CSV + table.

Reads the ``scored/*.log.console`` captures each run leaves behind, so it needs
no GPU and re-reads nothing that has to be regenerated.

    modal run modal_apps/eval/summarize.py
    modal run modal_apps/eval/summarize.py --dataset-dir seedtts_en --out runs.csv

Per-row error counts come along with the headline metrics: a WER can look
survivable while being driven by insertions, which is over-generation rather
than mis-transcription.
"""

from __future__ import annotations

import modal


VOL_NAME = "seedtts-eval-09-2026"
VOL_ROOT = "/vol"
RES_ROOT = f"{VOL_ROOT}/eval-09-2026"
DEFAULT_DATASET_DIR = "seedtts_en-limit100"

image = modal.Image.debian_slim(python_version="3.11")
app = modal.App("seedtts-eval-summarize")
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

# Metric -> (log basename, pattern). Both spellings of the SIM/UTMOS summary
# lines are accepted; which one is printed depends on the omnivoice version.
FIELDS = [
    ("sim_o", "sim", [r"SIM-o score:\s*([0-9.]+)", r"Average SIM-o:\s*([0-9.]+)"]),
    ("wer_pct", "wer", [r"Seed-TTS WER \(Avg of WERs\):\s*([0-9.]+)%",
                        r"WER \(Weighted\):\s*([0-9.]+)%"]),
    ("wer_weighted_pct", "wer", [r"WER \(Weighted\):\s*([0-9.]+)%"]),
    ("utmos", "mos", [r"UTMOS score:\s*([0-9.]+)", r"Average UTMOS:\s*([0-9.]+)"]),
]


@app.function(image=image, volumes={VOL_ROOT: vol}, timeout=10 * 60)
def collect(dataset_dir: str = DEFAULT_DATASET_DIR) -> list[dict]:
    import os
    import re

    root = f"{RES_ROOT}/{dataset_dir}"
    if not os.path.isdir(root):
        raise RuntimeError(f"{root} missing on volume {VOL_NAME}")

    def last_match(path: str, patterns: list[str]) -> float | None:
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        for pat in patterns:
            hits = re.findall(pat, text, flags=re.MULTILINE)
            if hits:
                return float(hits[-1])
        return None

    rows = []
    for tag in sorted(os.listdir(root)):
        run_dir = f"{root}/{tag}"
        if not os.path.isdir(run_dir):
            continue

        row: dict = {"tag": tag}
        for name, log, patterns in FIELDS:
            row[name] = last_match(f"{run_dir}/scored/{log}.log.console", patterns)

        wer_console = f"{run_dir}/scored/wer.log.console"
        errs = None
        if os.path.exists(wer_console):
            with open(wer_console, encoding="utf-8", errors="replace") as fh:
                hits = re.findall(
                    r"Errors:\s*([0-9.]+) ins,\s*([0-9.]+) del,\s*([0-9.]+) sub"
                    r"\s*/\s*([0-9]+) words", fh.read())
            errs = hits[-1] if hits else None
        row.update(
            ins=float(errs[0]) if errs else None,
            dele=float(errs[1]) if errs else None,
            sub=float(errs[2]) if errs else None,
            words=int(errs[3]) if errs else None,
        )

        gen = f"{run_dir}/generated"
        row["wavs"] = (
            sum(1 for f in os.listdir(gen) if f.endswith(".wav"))
            if os.path.isdir(gen) else 0
        )
        rows.append(row)
    return rows


@app.local_entrypoint()
def main(dataset_dir: str = DEFAULT_DATASET_DIR, out: str = ""):
    import csv
    import pathlib

    rows = collect.remote(dataset_dir=dataset_dir)
    if not rows:
        print(f"no runs under {dataset_dir}")
        return

    cols = ["tag", "wavs", "sim_o", "wer_pct", "utmos",
            "wer_weighted_pct", "ins", "dele", "sub", "words"]
    out_path = pathlib.Path(out or f"agent-workspace/eval_summary_{dataset_dir}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)

    def cell(row: dict, key: str) -> str:
        v = row.get(key)
        if v is None:
            return "-"
        if key in ("sim_o", "utmos"):
            return f"{v:.3f}"
        if key.endswith("_pct"):
            return f"{v:.2f}%"
        return f"{v:g}" if isinstance(v, (int, float)) else str(v)

    show = ["tag", "wavs", "sim_o", "wer_pct", "utmos", "ins", "dele", "sub"]
    widths = {c: max(len(c), max(len(cell(r, c)) for r in rows)) for c in show}
    header = "  ".join(c.ljust(widths[c]) for c in show)
    print(f"\n{dataset_dir}\n")
    print(header)
    print("-" * len(header))
    for row in sorted(rows, key=lambda r: r["tag"]):
        print("  ".join(cell(row, c).ljust(widths[c]) for c in show))
    print(f"\nwrote {out_path} ({len(rows)} runs)")
    return rows
