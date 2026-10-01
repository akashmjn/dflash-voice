"""Summarize fetched eval runs under results/SPLIT/ into SPLIT/summary.csv + a table.

Reads the ``*.log.console`` captures ``seedtts_eval.py`` fetches into
``results/SPLIT/TAG/``; no Modal needed.

    python modal_apps/eval/summarize.py                        # every split
    python modal_apps/eval/summarize.py seedtts_en-limit100

Per-row error counts come along with the headline metrics: a WER can look
survivable while being driven by insertions, which is over-generation rather
than mis-transcription.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import re

RESULTS = pathlib.Path(__file__).parent / "results"

# Metric -> (log basename, pattern). Both spellings of the SIM/UTMOS summary
# lines are accepted; which one is printed depends on the omnivoice version.
FIELDS = [
    ("sim_o", "sim", [r"SIM-o score:\s*([0-9.]+)", r"Average SIM-o:\s*([0-9.]+)"]),
    ("wer_pct", "wer", [r"Seed-TTS WER \(Avg of WERs\):\s*([0-9.]+)%",
                        r"WER \(Weighted\):\s*([0-9.]+)%"]),
    ("wer_weighted_pct", "wer", [r"WER \(Weighted\):\s*([0-9.]+)%"]),
    ("utmos", "mos", [r"UTMOS score:\s*([0-9.]+)", r"Average UTMOS:\s*([0-9.]+)"]),
]
COLS = ["tag", "n", "sim_o", "wer_pct", "utmos",
        "wer_weighted_pct", "ins", "dele", "sub", "words"]
SHOW = ["tag", "n", "sim_o", "wer_pct", "utmos", "ins", "dele", "sub"]


def _read(path: pathlib.Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _last(text: str, patterns: list[str]) -> float | None:
    for pat in patterns:
        hits = re.findall(pat, text, flags=re.MULTILINE)
        if hits:
            return float(hits[-1])
    return None


def collect(split_dir: pathlib.Path) -> list[dict]:
    rows = []
    for run_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        logs = {log: _read(run_dir / f"{log}.log.console") for log in ("sim", "wer", "mos")}
        row: dict = {"tag": run_dir.name}
        for name, log, patterns in FIELDS:
            row[name] = _last(logs[log], patterns)

        errs = re.findall(
            r"Errors:\s*([0-9.]+) ins,\s*([0-9.]+) del,\s*([0-9.]+) sub"
            r"\s*/\s*([0-9]+) words", logs["wer"])
        errs = errs[-1] if errs else None
        row.update(
            ins=float(errs[0]) if errs else None,
            dele=float(errs[1]) if errs else None,
            sub=float(errs[2]) if errs else None,
            words=int(errs[3]) if errs else None,
        )
        # Rows scored: the fewest any scorer processed.
        counts = [int(m) for text in logs.values()
                  for m in re.findall(r"Processed (\d+)/\d+", text)[-1:]]
        row["n"] = min(counts) if counts else None
        rows.append(row)
    return rows


def cell(row: dict, key: str) -> str:
    v = row.get(key)
    if v is None:
        return "-"
    if key in ("sim_o", "utmos"):
        return f"{v:.3f}"
    if key.endswith("_pct"):
        return f"{v:.2f}%"
    return f"{v:g}" if isinstance(v, (int, float)) else str(v)


def summarize(split_dir: pathlib.Path) -> None:
    rows = collect(split_dir)
    if not rows:
        print(f"no runs under {split_dir}")
        return
    out_path = split_dir / "summary.csv"
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLS)
        writer.writeheader()
        writer.writerows(rows)

    widths = {c: max(len(c), max(len(cell(r, c)) for r in rows)) for c in SHOW}
    header = "  ".join(c.ljust(widths[c]) for c in SHOW)
    print(f"\n{split_dir.name}\n")
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(cell(row, c).ljust(widths[c]) for c in SHOW))
    print(f"\nwrote {out_path} ({len(rows)} runs)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("splits", nargs="*",
                        help="split dirs under results/ (default: all)")
    args = parser.parse_args()
    splits = args.splits or sorted(p.name for p in RESULTS.iterdir() if p.is_dir())
    for split in splits:
        summarize(RESULTS / split)


if __name__ == "__main__":
    main()
