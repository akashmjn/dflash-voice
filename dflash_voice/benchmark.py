from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from rich import print
from tqdm import tqdm

DEFAULT_MODELS = {
    "qwen3": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "fish": "mlx-community/fish-audio-s2-pro-8bit",
}

QWEN3_PROMPTS = [
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
]

FISH_PROMPTS = [
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
]


def _load_prompts(path: Path | None, backend: str) -> list[str]:
    if path is not None:
        prompts = []
        for line in path.read_text().splitlines():
            if line.strip():
                prompts.append(json.loads(line)["text"])
        return prompts
    return QWEN3_PROMPTS if backend == "qwen3" else FISH_PROMPTS


def _print_qwen3_prompt(idx: int, text: str, result, profile) -> None:
    n = len(profile.frame_timings)
    gen_ms = [t.total_s * 1000 for t in profile.frame_timings]
    bb_ms = [t.backbone_s * 1000 for t in profile.frame_timings]
    dp_ms = [t.depth_s * 1000 for t in profile.frame_timings]
    dec_ms = profile.codec_decode_s * 1000
    preview = text if len(text) <= 60 else text[:57] + "..."

    print(f"\n[bold]Prompt {idx}[/bold]: {preview!r}")
    print(
        f"  frames: {n}  duration: {result.audio_duration}  RTF: {result.real_time_factor:.2f}x"
    )
    if n:
        print(
            f"  codec frame generate: {sum(gen_ms):.0f} ms "
            f"({statistics.mean(gen_ms):.1f} ms/frame, "
            f"backbone {statistics.mean(bb_ms):.1f} | depth {statistics.mean(dp_ms):.1f})"
        )
        print(f"  codec decode: {dec_ms:.0f} ms ({dec_ms / n:.1f} ms/frame)")


def _print_fish_prompt(idx: int, text: str, result, profile) -> None:
    n = len(profile.token_timings)
    gen_ms = [t.total_s * 1000 for t in profile.token_timings]
    slow_ms = [t.slow_s * 1000 for t in profile.token_timings]
    fast_ms = [t.fast_s * 1000 for t in profile.token_timings]
    dec_ms = profile.codec_decode_s * 1000
    preview = text if len(text) <= 60 else text[:57] + "..."

    print(f"\n[bold]Prompt {idx}[/bold]: {preview!r}")
    print(
        f"  tokens: {n}  duration: {result.audio_duration}  RTF: {result.real_time_factor:.2f}x"
    )
    if n:
        print(
            f"  token generate: {sum(gen_ms):.0f} ms "
            f"({statistics.mean(gen_ms):.1f} ms/token, "
            f"slow {statistics.mean(slow_ms):.1f} | fast {statistics.mean(fast_ms):.1f})"
        )
        print(f"  codec decode: {dec_ms:.0f} ms ({dec_ms / n:.1f} ms/token)")


def _print_qwen3_aggregate(profiles, results) -> None:
    timings = [t for p in profiles for t in p.frame_timings]
    n = len(timings)
    gen_ms = [t.total_s * 1000 for t in timings]
    bb_ms = [t.backbone_s * 1000 for t in timings]
    dp_ms = [t.depth_s * 1000 for t in timings]
    dec_ms = sum(p.codec_decode_s for p in profiles) * 1000

    print(f"\n{'=' * 50}")
    print(f"[bold]Aggregate[/bold]  {len(profiles)} prompts  {n} frames")
    if n:
        print(
            f"  codec frame generate: {sum(gen_ms):.0f} ms, "
            f"{statistics.mean(gen_ms):.1f} ms/frame "
            f"(backbone {statistics.mean(bb_ms):.1f} | depth {statistics.mean(dp_ms):.1f}), "
            f"{n / (sum(gen_ms) / 1000):.1f} frames/s"
        )
        print(f"  codec decode: {dec_ms:.0f} ms ({dec_ms / n:.1f} ms/frame)")
    print(f"  mean RTF: {statistics.mean(r.real_time_factor for r in results):.2f}x")
    print(f"{'=' * 50}")


def _print_fish_aggregate(profiles, results) -> None:
    timings = [t for p in profiles for t in p.token_timings]
    n = len(timings)
    gen_ms = [t.total_s * 1000 for t in timings]
    slow_ms = [t.slow_s * 1000 for t in timings]
    fast_ms = [t.fast_s * 1000 for t in timings]
    dec_ms = sum(p.codec_decode_s for p in profiles) * 1000

    print(f"\n{'=' * 50}")
    print(f"[bold]Aggregate[/bold]  {len(profiles)} prompts  {n} tokens")
    if n:
        print(
            f"  token generate: {sum(gen_ms):.0f} ms, "
            f"{statistics.mean(gen_ms):.1f} ms/token "
            f"(slow {statistics.mean(slow_ms):.1f} | fast {statistics.mean(fast_ms):.1f}), "
            f"{n / (sum(gen_ms) / 1000):.1f} tokens/s"
        )
        print(f"  codec decode: {dec_ms:.0f} ms ({dec_ms / n:.1f} ms/token)")
    print(f"  mean RTF: {statistics.mean(r.real_time_factor for r in results):.2f}x")
    print(f"{'=' * 50}")


def _run_qwen3(args) -> None:
    from dflash_voice.tts_mlx.qwen3 import GenerationProfile, load_model

    prompts = _load_prompts(args.prompts_file, "qwen3")
    if args.max_samples:
        prompts = prompts[: args.max_samples]

    print(f"Loading {args.model}")
    model = load_model(args.model)

    gen_kw = dict(
        voice=args.voice,
        language=args.language,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        stream=False,
    )

    if not args.no_warmup:
        list(model.generate(text="Hi.", **gen_kw))

    profiles, results = [], []
    for idx, text in enumerate(tqdm(prompts, desc="Benchmarking")):
        profile = GenerationProfile()
        result = list(model.generate(text=text, profile=profile, **gen_kw))[0]
        profiles.append(profile)
        results.append(result)
        _print_qwen3_prompt(idx, text, result, profile)

    _print_qwen3_aggregate(profiles, results)


def _run_fish(args) -> None:
    from dflash_voice.tts_mlx.fish import GenerationProfile, load_model

    prompts = _load_prompts(args.prompts_file, "fish")
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
    if args.instruct:
        gen_kw["instruct"] = args.instruct

    if not args.no_warmup:
        list(model.generate(text="[excited] Hi.", **gen_kw))

    profiles, results = [], []
    for idx, text in enumerate(tqdm(prompts, desc="Benchmarking")):
        profile = GenerationProfile()
        result = list(model.generate(text=text, profile=profile, **gen_kw))[0]
        profiles.append(profile)
        results.append(result)
        _print_fish_prompt(idx, text, result, profile)

    _print_fish_aggregate(profiles, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLX TTS benchmark")
    parser.add_argument(
        "--backend",
        choices=["qwen3", "fish"],
        default="qwen3",
        help="TTS backend to benchmark",
    )
    parser.add_argument("--model", help="model id (defaults per backend)")
    parser.add_argument("--voice", default="Ryan", help="Qwen3 speaker name")
    parser.add_argument("--language", default="auto", help="Qwen3 language")
    parser.add_argument("--instruct", help="Fish style instruction")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--prompts-file", type=Path)
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    if args.model is None:
        args.model = DEFAULT_MODELS[args.backend]

    if args.backend == "fish":
        if args.temperature == 0.9 and args.top_k == 50 and args.top_p == 1.0:
            args.temperature = 0.7
            args.top_k = 30
            args.top_p = 0.7
        if args.max_tokens == 4096:
            args.max_tokens = 1024
        _run_fish(args)
    else:
        _run_qwen3(args)


if __name__ == "__main__":
    main()
