#!/usr/bin/env python3
"""Benchmark the MLX TTS wrappers in this package with optional audio export.

Every model generates one codec frame per autoregressive step, so all timings are
reported per step: backbone (semantic token) and depth decoder (remaining
codebooks) separately, plus codec decode.

```bash
python mlx_decode/bench.py bench --model miso --save-audio
python mlx_decode/bench.py bench --model all      # or a comma-separated list
python mlx_decode/bench.py summarize              # table from saved metrics.json
```
"""

from __future__ import annotations

import builtins
import importlib
import json
import re
import statistics
from pathlib import Path
from typing import Any, Optional

import typer
from rich import print
from tqdm import tqdm

from mlx_decode import GenerationProfile


MODELS = {
    "qwen3": {
        "frame_rate_hz": 12.5,
        "decoder_iterations": 15,
        "model_id": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "generate": {
            "voice": "Ryan",
            "language": "auto",
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "max_tokens": 1024,
        },
    },
    "fish": {
        "frame_rate_hz": 21.0,
        "decoder_iterations": 9,
        "model_id": "mlx-community/fish-audio-s2-pro-8bit",
        "generate": {
            "temperature": 0.7,
            "top_k": 30,
            "top_p": 0.7,
            "max_tokens": 1024,
        },
    },
    "voxtral": {
        "frame_rate_hz": 12.5,
        "decoder_iterations": 7,
        "model_id": "mlx-community/Voxtral-4B-TTS-2603-mlx-6bit",
        # No sampling knobs: argmax semantic code, flow-matched acoustic codes.
        "generate": {
            "voice": "casual_male",
            "max_tokens": 1024,
        },
    },
    "miso": {
        "frame_rate_hz": 12.5,
        "decoder_iterations": 31,
        "model_id": "mlx-community/MisoLabs-MisoTTS-8bit",
        "generate": {
            "temperature": 0.9,
            "top_k": 60,
            "top_p": 1.0,
            "max_tokens": 1024,
        },
    },
}

WARMUP_PROMPT = "Hello, quick mic check."

PROMPTS = [
    "Hello, quick mic check... testing... 1 2 3...",
    "Hello, this is a quick text to speech test on Apple Silicon.",
    "The price is $42.99 — call 555-0123 today!",
    "What is the capital of France, and why is it historically significant?",
    (
        "The quick brown fox jumps over the lazy dog. "
        "Speech synthesis on Apple Silicon should feel fast and natural."
    ),
    (
        "In a world where artificial intelligence transforms how we communicate, "
        "voice synthesis stands at the frontier of human-computer interaction. "
        "Real-time text-to-speech enables assistants, accessibility tools, and "
        "creative applications that were unimaginable a decade ago."
    ),
]

DEFAULT_OUTPUT_DIR = Path("mlx_decode/output")

# Two-speaker priming turns for miso, at its native 24 kHz.
# Miso being a base model appears to be unstable when generating from empty context 
# some preconditioning makes better generations -- at the cost of comparable timings.
WARMUP_DIR = Path(__file__).resolve().parent / "warmup"
MISO_CONTEXT = [
    (0, "Okay we're recording! Let's get into it.", str(WARMUP_DIR / "warmup_spk0.wav")),
    (1, "Sounds good, ready when you are!", str(WARMUP_DIR / "warmup_spk1.wav")),
]


def _load_prompts(path: Path | None) -> list[str]:
    if path is None:
        return PROMPTS
    lines = path.read_text().splitlines()
    return [json.loads(line)["text"] for line in lines if line.strip()]


def _output_dir(output_dir: Path, model: str, model_id: str) -> Path:
    slug = re.sub(r"[^a-z0-9._-]+", "-", model_id.rsplit("/", 1)[-1].lower())
    return output_dir / model / slug


def _save_audio(out_dir: Path, idx: int, result) -> Path:
    from mlx_audio.audio_io import write as audio_write

    path = out_dir / f"prompt_{idx:03d}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_write(path, result.audio, result.sample_rate, format="wav")
    return path


def _ms_stats(timings, decode_s: float) -> dict[str, float]:
    gen = [t.total_s * 1000 for t in timings]
    n = len(gen)
    return {
        "n": n,
        "gen_total": sum(gen),
        "gen_mean": statistics.mean(gen) if n else 0.0,
        "backbone_mean": statistics.mean(t.backbone_semantic_s * 1000 for t in timings)
        if n
        else 0.0,
        "depth_mean": statistics.mean(t.depth_audio_s * 1000 for t in timings)
        if n
        else 0.0,
        "decode_total": decode_s * 1000,
        "decode_mean": decode_s * 1000 / n if n else 0.0,
        "rate": n / (sum(gen) / 1000) if n else 0.0,
    }


def _print_stats(s: dict[str, float]) -> None:
    if not s["n"]:
        return
    print(
        f"  generate: {s['gen_total']:.0f} ms "
        f"({s['gen_mean']:.1f} ms/step, "
        f"backbone_semantic {s['backbone_mean']:.1f} | "
        f"depth_audio {s['depth_mean']:.1f}), "
        f"{s['rate']:.1f} steps/s"
    )
    print(f"  codec decode: {s['decode_total']:.0f} ms ({s['decode_mean']:.1f} ms/step)")


def _timing_metrics(s: dict[str, float]) -> dict[str, Any]:
    return {
        "generate_ms": {
            "total": s["gen_total"],
            "per_step": s["gen_mean"],
            "backbone_semantic_per_step": s["backbone_mean"],
            "depth_audio_per_step": s["depth_mean"],
            "steps_per_s": s["rate"],
        },
        "codec_decode_ms": {"total": s["decode_total"], "per_step": s["decode_mean"]},
    }


SUMMARY_COLUMNS = [
    ("Model", 16),
    ("Frame rate", 10),
    ("Wall RTF", 8),
    ("Audio decoder %", 15),
    ("Total ms", 8),
    ("Semantic backbone (ms)", 22),
    ("Audio decoder (ms)", 18),
    ("Decoder iterations", 18),
    ("ms / iterations", 15),
    ("Frames/s", 8),
]


def _summary_row(
    *, label: str, stats: dict[str, float], mean_rtf: float, spec: dict[str, Any]
) -> list[str]:
    """One markdown row in the shape of experiments/mlx_decode_breakdown."""
    hz, iters = spec.get("frame_rate_hz"), spec.get("decoder_iterations")
    total = stats["gen_mean"]
    return [
        label,
        f"{hz:g} Hz" if hz else "-",
        f"{mean_rtf:.2f}",
        f"{100 * stats['depth_mean'] / total:.0f}%" if total else "-",
        f"{total:.1f}",
        f"{stats['backbone_mean']:.1f}",
        f"{stats['depth_mean']:.1f}",
        str(iters) if iters else "-",
        f"{stats['depth_mean'] / iters:.2f}" if iters else "-",
        f"{stats['rate']:.1f}",
    ]


def _print_summary_table(rows: list[list[str]]) -> None:
    headers = [h for h, _ in SUMMARY_COLUMNS]
    widths = [
        max(w, *(len(r[i]) for r in rows)) for i, (_, w) in enumerate(SUMMARY_COLUMNS)
    ]

    def line(cells: list[str]) -> str:
        return "| " + " | ".join(v.ljust(w) for v, w in zip(cells, widths)) + " |"

    # builtins.print, not rich's -- rich soft-wraps the row at the terminal width.
    builtins.print(line(headers))
    builtins.print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        builtins.print(line(row))


def _run(
    *,
    name: str,
    model_id: str,
    gen_kwargs: dict[str, Any],
    prompts: list[str],
    out_dir: Path,
    save_audio: bool,
    warmup: bool,
    spec: dict[str, Any],
) -> list[str]:
    print(f"Loading {model_id}")
    model = importlib.import_module(f"mlx_decode.{name}").load_model(model_id)
    if warmup:
        list(model.generate(text=WARMUP_PROMPT, **gen_kwargs))

    profiles, results, rows = [], [], []
    for idx, text in enumerate(tqdm(prompts, desc="Benchmarking")):
        profile = GenerationProfile()
        result = list(model.generate(text=text, profile=profile, **gen_kwargs))[0]
        profiles.append(profile)
        results.append(result)

        audio_path = _save_audio(out_dir, idx, result) if save_audio else None
        stats = _ms_stats(profile.step_timings, profile.codec_decode_s)
        rows.append(
            {
                "idx": idx,
                "text": text,
                "audio_path": audio_path.name if audio_path else None,
                "num_steps": stats["n"],
                "audio_duration": result.audio_duration,
                "rtf": result.real_time_factor,
                **_timing_metrics(stats),
            }
        )

        preview = text if len(text) <= 60 else text[:57] + "..."
        print(f"\n[bold]Prompt {idx}[/bold]: {preview!r}")
        if audio_path:
            print(f"  saved: {audio_path}")
        print(
            f"  steps: {stats['n']}  duration: {result.audio_duration}  "
            f"RTF: {result.real_time_factor:.2f}x"
        )
        _print_stats(stats)

    stats = _ms_stats(
        [t for p in profiles for t in p.step_timings],
        sum(p.codec_decode_s for p in profiles),
    )
    mean_rtf = statistics.mean(r.real_time_factor for r in results)
    print(f"\n{'=' * 50}")
    print(f"[bold]Aggregate[/bold]  {len(profiles)} prompts  {stats['n']} steps")
    _print_stats(stats)
    print(f"  mean RTF: {mean_rtf:.2f}x")
    print(f"{'=' * 50}")

    metrics_path = out_dir / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(
            {
                "model": name,
                "model_id": model_id,
                "settings": gen_kwargs,
                "prompts": rows,
                "aggregate": {
                    "num_prompts": len(profiles),
                    "num_steps": stats["n"],
                    "mean_rtf": mean_rtf,
                    **_timing_metrics(stats),
                },
            },
            indent=2,
        )
    )
    print(f"\nSaved metrics: {metrics_path}")
    return _summary_row(
        label=model_id.rsplit("/", 1)[-1], stats=stats, mean_rtf=mean_rtf, spec=spec
    )


app = typer.Typer(
    add_completion=False,
    help="Benchmark the MLX TTS wrappers and save per-step timings.",
)


def _parse_models(value: str) -> list[str]:
    """Comma-separated model names, or "all"."""
    if value.strip() == "all":
        return list(MODELS)
    names = [n.strip() for n in value.split(",") if n.strip()]
    if not names:
        raise typer.BadParameter("No model names given.")
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        raise typer.BadParameter(
            f"Unknown model(s) {unknown}; expected 'all' or any of {tuple(MODELS)}"
        )
    return names


@app.command()
def bench(
    model: str = typer.Option(
        "qwen3", help=f"Comma-separated model names, or 'all'. One of {tuple(MODELS)}."
    ),
    model_id: Optional[str] = typer.Option(
        None, help="Checkpoint id (default: the model's mlx-community build)."
    ),
    max_samples: Optional[int] = typer.Option(
        None, "--max-samples", "-n", help="Benchmark the first N prompts (default: all)."
    ),
    prompts_file: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False, help="JSONL file with a 'text' field per line."
    ),
    warmup: bool = typer.Option(True, help="Generate once before timing."),
    save_audio: bool = typer.Option(False, help="Write generated wav files."),
    context: bool = typer.Option(
        False,
        help="miso only: prime with two-speaker warmup clips for a stable generation. "
        "Adds prompt encode + prefill, so timings are no longer comparable.",
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Root for saved audio and metrics."
    ),
) -> None:
    """Benchmark one or more models over the built-in prompts (or --prompts-file).

    Voice, style and sampling settings are not flags -- edit ``MODELS`` above so
    a run is reproducible from the model name alone.
    """
    names = _parse_models(model)
    if context and names != ["miso"]:
        raise typer.BadParameter(f"--context is miso-only, not supported for {names}")
    if model_id and len(names) > 1:
        raise typer.BadParameter("--model-id names a single checkpoint; benchmark one model.")

    prompts = _load_prompts(prompts_file)[:max_samples]
    rows = []
    for name in names:
        resolved = model_id or MODELS[name]["model_id"]
        gen_kwargs = dict(MODELS[name]["generate"])
        if context:
            gen_kwargs["context"] = MISO_CONTEXT
        rows.append(
            _run(
                name=name,
                model_id=resolved,
                gen_kwargs=gen_kwargs,
                prompts=prompts,
                out_dir=_output_dir(output_dir, name, resolved),
                save_audio=save_audio,
                warmup=warmup,
                spec=MODELS[name],
            )
        )
    print()
    _print_summary_table(rows)


@app.command()
def summarize(
    model: str = typer.Option(
        "all", help=f"Comma-separated model names, or 'all'. One of {tuple(MODELS)}."
    ),
    output_dir: Path = typer.Option(
        DEFAULT_OUTPUT_DIR, help="Root the metrics were written to."
    ),
) -> None:
    """Print the summary table from metrics.json files an earlier run saved.

    Every checkpoint benchmarked under a model's directory gets a row, so the
    per-size variants (e.g. both Qwen3 builds) show up side by side.
    """
    rows = []
    for name in _parse_models(model):
        for path in sorted((output_dir / name).glob("*/metrics.json")):
            data = json.loads(path.read_text())
            agg = data["aggregate"]
            gen = agg["generate_ms"]
            rows.append(
                _summary_row(
                    label=path.parent.name,
                    stats={
                        "gen_mean": gen["per_step"],
                        "backbone_mean": gen["backbone_semantic_per_step"],
                        "depth_mean": gen["depth_audio_per_step"],
                        "rate": gen["steps_per_s"],
                    },
                    mean_rtf=agg["mean_rtf"],
                    spec=MODELS[name],
                )
            )
    if not rows:
        raise typer.BadParameter(f"No metrics.json found under {output_dir}; run bench first.")
    _print_summary_table(rows)


if __name__ == "__main__":
    app()
