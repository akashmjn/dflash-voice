#!/usr/bin/env python3
"""Benchmark MLX TTS backends (Qwen3, Fish Audio) with optional audio export."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

from rich import print
from tqdm import tqdm

DEFAULT_OUTPUT_DIR = Path("benchmark")

BACKENDS = {
    "qwen3": {
        "model": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "voice": "Ryan",
        "language": "auto",
        "warmup": "Hi.",
        "temperature": 0.9,
        "top_k": 50,
        "top_p": 1.0,
        "max_tokens": 1024,
    },
    "fish": {
        "model": "mlx-community/fish-audio-s2-pro-8bit",
        "warmup": "[excited] Hi.",
        "temperature": 0.7,
        "top_k": 30,
        "top_p": 0.7,
        "max_tokens": 1024,
    },
}

PROMPTS = {
    "qwen3": [
        "Hello.",
        "Hello, this is a quick Qwen3 TTS test on Apple Silicon.",
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
    ],
    "fish": [
        "[excited] Hello.",
        "[excited] Hello, this is a quick Fish Audio test on Apple Silicon.",
        "[calm] The price is $42.99 — call 555-0123 today!",
        "[curious] What is the capital of France, and why is it historically significant?",
        (
            "[excited] The quick brown fox jumps over the lazy dog. "
            "[calm] Speech synthesis on Apple Silicon should feel fast and natural."
        ),
        (
            "[excited] In a world where artificial intelligence transforms how we communicate, "
            "[calm] voice synthesis stands at the frontier of human-computer interaction. "
            "[low voice] Real-time text-to-speech enables assistants, accessibility tools, and "
            "creative applications that were unimaginable a decade ago."
        ),
    ],
}


def _load_prompts(path: Path | None, backend: str) -> list[str]:
    if path is not None:
        prompts = []
        for line in path.read_text().splitlines():
            if line.strip():
                prompts.append(json.loads(line)["text"])
        return prompts
    return PROMPTS[backend]


def _model_slug(model_id: str) -> str:
    slug = model_id.rsplit("/", 1)[-1].lower()
    return re.sub(r"[^a-z0-9._-]+", "-", slug)


def _output_dir(output_dir: Path, backend: str, model_id: str) -> Path:
    return output_dir / backend / _model_slug(model_id)


def _save_audio(
    output_dir: Path, backend: str, model_id: str, idx: int, result
) -> Path:
    from mlx_audio.audio_io import write as audio_write

    path = _output_dir(output_dir, backend, model_id) / f"prompt_{idx:03d}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    audio_write(path, result.audio, result.sample_rate, format="wav")
    return path


def _timings(profile, backend: str):
    return profile.frame_timings if backend == "qwen3" else profile.token_timings


def _ms_stats(timings, decode_s: float = 0.0) -> dict[str, float]:
    n = len(timings)
    gen = [t.total_s * 1000 for t in timings]
    bb = [t.backbone_s * 1000 for t in timings]
    dp = [t.depth_decoder_s * 1000 for t in timings]
    dec = decode_s * 1000
    mean = statistics.mean
    return {
        "n": n,
        "gen_total": sum(gen) if n else 0.0,
        "gen_mean": mean(gen) if n else 0.0,
        "backbone_mean": mean(bb) if n else 0.0,
        "depth_mean": mean(dp) if n else 0.0,
        "decode_total": dec,
        "decode_mean": dec / n if n else 0.0,
        "rate": n / (sum(gen) / 1000) if gen else 0.0,
    }


def _preview(text: str) -> str:
    return text if len(text) <= 60 else text[:57] + "..."


def _print_prompt(idx: int, text: str, result, profile, backend: str) -> None:
    unit = "frames" if backend == "qwen3" else "tokens"
    label = "codec frame generate" if backend == "qwen3" else "token generate"
    s = _ms_stats(_timings(profile, backend), profile.codec_decode_s)
    per = "frame" if backend == "qwen3" else "token"

    print(f"\n[bold]Prompt {idx}[/bold]: {_preview(text)!r}")
    print(
        f"  {unit}: {s['n']}  duration: {result.audio_duration}  "
        f"RTF: {result.real_time_factor:.2f}x"
    )
    if s["n"]:
        print(
            f"  {label}: {s['gen_total']:.0f} ms "
            f"({s['gen_mean']:.1f} ms/{per}, "
            f"backbone {s['backbone_mean']:.1f} | depth_decoder {s['depth_mean']:.1f})"
        )
        print(
            f"  codec decode: {s['decode_total']:.0f} ms ({s['decode_mean']:.1f} ms/{per})"
        )


def _print_aggregate(profiles, results, backend: str) -> None:
    unit = "frames" if backend == "qwen3" else "tokens"
    label = "codec frame generate" if backend == "qwen3" else "token generate"
    per = "frame" if backend == "qwen3" else "token"
    timings = [t for p in profiles for t in _timings(p, backend)]
    s = _ms_stats(timings, sum(p.codec_decode_s for p in profiles))

    print(f"\n{'=' * 50}")
    print(f"[bold]Aggregate[/bold]  {len(profiles)} prompts  {s['n']} {unit}")
    if s["n"]:
        print(
            f"  {label}: {s['gen_total']:.0f} ms, "
            f"{s['gen_mean']:.1f} ms/{per} "
            f"(backbone {s['backbone_mean']:.1f} | depth_decoder {s['depth_mean']:.1f}), "
            f"{s['rate']:.1f} {unit}/s"
        )
        print(
            f"  codec decode: {s['decode_total']:.0f} ms ({s['decode_mean']:.1f} ms/{per})"
        )
    print(f"  mean RTF: {statistics.mean(r.real_time_factor for r in results):.2f}x")
    print(f"{'=' * 50}")


def _prompt_metrics(
    idx: int, text: str, result, profile, backend: str, audio_path: Path | None
) -> dict[str, Any]:
    timings = _timings(profile, backend)
    unit = "frame" if backend == "qwen3" else "token"
    s = _ms_stats(timings, profile.codec_decode_s)
    return {
        "idx": idx,
        "text": text,
        "audio_path": audio_path.name if audio_path else None,
        f"num_{unit}s": s["n"],
        "audio_duration": result.audio_duration,
        "rtf": result.real_time_factor,
        "generate_ms": {
            "total": s["gen_total"],
            f"per_{unit}": s["gen_mean"],
            f"backbone_per_{unit}": s["backbone_mean"],
            f"depth_decoder_per_{unit}": s["depth_mean"],
        },
        "codec_decode_ms": {
            "total": s["decode_total"],
            f"per_{unit}": s["decode_mean"],
        },
    }


def _aggregate_metrics(profiles, results, backend: str) -> dict[str, Any]:
    unit = "frame" if backend == "qwen3" else "token"
    timings = [t for p in profiles for t in _timings(p, backend)]
    s = _ms_stats(timings, sum(p.codec_decode_s for p in profiles))
    return {
        "num_prompts": len(profiles),
        f"num_{unit}s": s["n"],
        "mean_rtf": statistics.mean(r.real_time_factor for r in results),
        "generate_ms": {
            "total": s["gen_total"],
            f"per_{unit}": s["gen_mean"],
            f"backbone_per_{unit}": s["backbone_mean"],
            f"depth_decoder_per_{unit}": s["depth_mean"],
            f"{unit}s_per_s": s["rate"],
        },
        "codec_decode_ms": {
            "total": s["decode_total"],
            f"per_{unit}": s["decode_mean"],
        },
    }


def _save_metrics(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved metrics: {path}")


def _run_backend(args, backend: str) -> None:
    if backend == "qwen3":
        from tts_mlx.qwen3 import GenerationProfile, load_model
    else:
        from tts_mlx.fish import GenerationProfile, load_model

    prompts = _load_prompts(args.prompts_file, backend)
    if args.max_samples:
        prompts = prompts[: args.max_samples]

    print(f"Loading {args.model}")
    model = load_model(args.model)

    gen_kw = dict(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stream=False,
    )
    if backend == "qwen3":
        gen_kw.update(voice=args.voice, language=args.language)
    elif args.instruct:
        gen_kw["instruct"] = args.instruct

    if not args.no_warmup:
        list(model.generate(text=BACKENDS[backend]["warmup"], **gen_kw))

    profiles, results, rows = [], [], []
    out_dir = _output_dir(args.output_dir, backend, args.model)
    for idx, text in enumerate(tqdm(prompts, desc="Benchmarking")):
        profile = GenerationProfile()
        result = list(model.generate(text=text, profile=profile, **gen_kw))[0]
        profiles.append(profile)
        results.append(result)
        audio_path = (
            _save_audio(args.output_dir, backend, args.model, idx, result)
            if args.save_audio
            else None
        )
        if audio_path:
            print(f"  saved: {audio_path}")
        rows.append(_prompt_metrics(idx, text, result, profile, backend, audio_path))
        _print_prompt(idx, text, result, profile, backend)

    _print_aggregate(profiles, results, backend)
    _save_metrics(
        out_dir / "metrics.json",
        {
            "backend": backend,
            "model": args.model,
            "settings": gen_kw,
            "prompts": rows,
            "aggregate": _aggregate_metrics(profiles, results, backend),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX TTS benchmark")
    parser.add_argument(
        "--backend",
        choices=["qwen3", "fish"],
        default="qwen3",
        help="TTS backend to benchmark",
    )
    parser.add_argument("--model", help="model id (defaults per backend)")
    parser.add_argument("--voice", help="Qwen3 speaker name")
    parser.add_argument("--language", help="Qwen3 language")
    parser.add_argument("--instruct", help="Fish style instruction")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="write generated wav files under --output-dir",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="directory for saved audio when --save-audio is set (default: ./benchmark)",
    )
    args = parser.parse_args()

    defaults = BACKENDS[args.backend]
    if args.model is None:
        args.model = defaults["model"]
    if args.temperature is None:
        args.temperature = defaults["temperature"]
    if args.top_k is None:
        args.top_k = defaults["top_k"]
    if args.top_p is None:
        args.top_p = defaults["top_p"]
    if args.max_tokens is None:
        args.max_tokens = defaults["max_tokens"]
    if args.backend == "qwen3":
        if args.voice is None:
            args.voice = defaults["voice"]
        if args.language is None:
            args.language = defaults["language"]

    _run_backend(args, args.backend)


if __name__ == "__main__":
    main()
