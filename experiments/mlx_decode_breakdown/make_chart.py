#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib"]
# ///
"""Render the decode-breakdown figure from `mlx_decode/bench.py` metrics.

Reads each model's `metrics.json` under `mlx_decode/output/` and writes
`assets/mlx-decode-breakdown.png`. Matplotlib is declared inline (PEP 723) so
this stays out of the project extras. Run from the repo root after benchmarking
every model in `MODELS` below:

```bash
uv run experiments/mlx_decode_breakdown/make_chart.py
```
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (short label, metrics subdirectory, native frame rate, audio-decoder params,
#  audio-decoder detail). Params are counted off the loaded MLX weights with
#  quantized matrices unpacked to their logical shapes, and cover the compute path
#  only -- codebook embedding tables are row lookups, not per-iteration weights.
MODELS = [
    ("Qwen3-TTS-0.6B", "qwen3/qwen3-tts-12hz-0.6b-base-8bit", 12.5, "142M", "1 semantic / 15 audio @ 12.5 Hz"),
    ("Qwen3-TTS-1.7B", "qwen3/qwen3-tts-12hz-1.7b-base-8bit", 12.5, "175M", "1 semantic / 15 audio @ 12.5 Hz"),
    ("Voxtral-4B", "voxtral/voxtral-4b-tts-2603-mlx-6bit", 12.5, "369M", "1 semantic / 36 audio @ 12.5 Hz"),
    ("Fish-S2-4B", "fish/fish-audio-s2-pro-8bit", 21.0, "417M", "1 semantic / 9 audio @ 21 Hz"),
    ("CSM-Miso-8B", "miso/misolabs-misotts-8bit", 12.5, "408M", "1 semantic / 31 audio @ 12.5 Hz"),
]

BG = "#1c1c1c"
FG = "#f5f5f5"
MUTED = "#9a9a9a"
BLUE = "#729cbe"
ORANGE = "#cc934c"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = Path(__file__).resolve().parent / "assets" / "mlx-decode-breakdown.png"


def _load(sub: str) -> tuple[float, float, float]:
    """(semantic backbone, audio decoder, total) ms per frame.

    Total exceeds the two parts by the untracked per-step remainder, and is the
    denominator for the share callouts -- matching the README table.
    """
    path = REPO_ROOT / "mlx_decode" / "output" / sub / "metrics.json"
    gen = json.loads(path.read_text())["aggregate"]["generate_ms"]
    return gen["backbone_semantic_per_step"], gen["depth_audio_per_step"], gen["per_step"]


def main() -> None:
    rows = [(label, *_load(sub), hz, params) for label, sub, hz, params, _ in MODELS]

    fig = plt.figure(figsize=(14.5, 8.6), facecolor=BG)
    fig.text(0.035, 0.955, "MLX Inference breakdown per frame", color=FG, fontsize=24, fontweight="bold", va="top")
    fig.text(
        0.035, 0.905,
        "Time per component is broken down. (1) autoregressive 0.6-8B LLM (semantic backbone) and (2) smaller 150-400M model (audio decoder).\n"
        "Time for the audio decoder is reported as % below, and plotted as orange bars.",
        color=MUTED, fontsize=11.5, va="top", linespacing=1.5,
    )

    # Audio-decoder share callouts, one per model, evenly spread across the figure.
    for i, (label, _bb, dp, total, _hz, params) in enumerate(rows):
        x = 0.10 + i * (0.80 / (len(rows) - 1))
        fig.text(x, 0.792, f"{100 * dp / total:.0f}%", color=FG, fontsize=25, fontweight="bold", ha="center")
        fig.text(x, 0.745, f"{label} ({params} decoder)", color=MUTED, fontsize=10, ha="center")

    fig.text(0.035, 0.695, "Time per component", color=FG, fontsize=15, fontweight="bold", va="top")
    fig.text(
        0.035, 0.655,
        "Models run on M1 Max MLX hardware in 8-bit except for Voxtral (6-bit). Audio tokenization configs - "
        "Qwen3: 1 semantic / 15 audio @ 12.5 Hz. Fish: 1 semantic / 9 audio @ 21 Hz.\n"
        "Miso: 1 semantic / 31 audio @ 12.5 Hz. Voxtral: 1 semantic + 36 audio over 7 flow steps @ 12.5 Hz.",
        color=MUTED, fontsize=10, va="top", linespacing=1.5,
    )

    ax = fig.add_axes([0.075, 0.135, 0.885, 0.465])
    ax.set_facecolor(BG)
    labels = [r[0] for r in rows]
    xs = range(len(rows))
    audio_decoder = [r[2] for r in rows]
    semantic_backbone = [r[1] for r in rows]

    ax.bar(xs, audio_decoder, 0.30, color=ORANGE, label="audio decoder")
    ax.bar(xs, semantic_backbone, 0.30, bottom=audio_decoder, color=BLUE, label="semantic backbone")

    # Real-time budget lines: a frame must finish within 1/frame_rate to keep up.
    for hz, style_label in ((12.5, "80 ms @ 12.5 Hz"), (21.0, "47.6 ms @ 21 Hz")):
        budget = 1000.0 / hz
        ax.axhline(budget, color=MUTED, linestyle="--", linewidth=1, alpha=0.65)
        ax.text(-0.34, budget + 2.2, style_label, color=MUTED, fontsize=9.5, ha="left")

    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, color=FG, fontsize=11)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f} ms")
    ax.tick_params(axis="y", colors=MUTED, length=0, pad=6)
    ax.tick_params(axis="x", length=0, pad=9)
    for spine in ("top", "right", "bottom", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="y", color="#3a3a3a", linewidth=0.7, alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(b + d for _, b, d, _, _, _ in rows) * 1.16)

    for i, (_, bb, dp, _, _, _) in enumerate(rows):
        ax.text(i, bb + dp + 2.5, f"{bb + dp:.0f} ms", color=FG, fontsize=10.5, ha="center", fontweight="bold")

    # Sits on the "Time per component" baseline, in figure coords so it tracks the
    # heading rather than the axes (which would push it into the config subtitle).
    handles, keys = ax.get_legend_handles_labels()
    legend = fig.legend(
        handles, keys,
        loc="center left", bbox_to_anchor=(0.225, 0.6835), bbox_transform=fig.transFigure,
        ncol=2, frameon=False, fontsize=10.5, handlelength=1.1, handleheight=1.1,
        handletextpad=0.6, columnspacing=1.8,
    )
    for text in legend.get_texts():
        text.set_color(FG)

    fig.text(
        0.035, 0.025,
        "Takeaway: repeated calls to the smaller (150-400M) audio decoder dominate decode time for every model -- whether it iterates over RVQ codebooks (Qwen3, Fish, Miso)\n"
        "or integrates a flow-matching head (Voxtral) -- inspite of much heavier semantic backbones (0.6B - 8B).",
        color=MUTED, fontsize=10, va="bottom", linespacing=1.5,
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=150, facecolor=BG)
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
