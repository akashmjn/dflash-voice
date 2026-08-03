#!/usr/bin/env python3
"""Render a two-speaker podcast script with a native mlx-audio TTS backend.

Usage:
    python agent-workspace/demo/demo_tts_podcast.py render SCRIPT.yaml --model miso
    python agent-workspace/demo/demo_tts_podcast.py warmup   # re-roll priming clips

The script is a YAML list of ``{speaker: 0|1, text: ...}`` segments. Segments are
generated one at a time, and each generated turn is fed back as conversational
context for the next one, so the model hears the dialogue it has produced so far:

    [text_1, audio_1, text_2, audio_2, ...] -> text_n -> audio_n

Context is trimmed to ``--context-seconds`` of past audio, oldest
turns dropped first.

Backends use mlx-audio's native ``load_model`` / ``model.generate``, not the
hand-rolled ports in ``benchmark_mlx/``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import typer
import yaml
from mlx_audio.audio_io import write as audio_write
from mlx_audio.tts.utils import load_model
from mlx_audio.utils import load_audio, resample_audio

# Matches benchmark_mlx/bench_tts_mlx.py.
MODELS = {
    "miso": "mlx-community/MisoLabs-MisoTTS-8bit",
    "fish": "mlx-community/fish-audio-s2-pro-8bit",
}

SAMPLING = {
    "miso": {"temperature": 0.9, "top_k": 60, "top_p": 1.0},
    "fish": {"temperature": 0.7, "top_k": 30, "top_p": 0.7},
}

# Miso/Sesame emits 12.5 codec frames per second and has a 2048-frame limit (~160s)
FRAME_RATE = 12.5
DEFAULT_CONTEXT_SECONDS = 45.0
TURN_GAP_S = 0.0

# Miso's Mimi codec rate. Warmup clips are stored at this rate so they load
# without resampling, even though Fish generates them at 44.1kHz.
TARGET_SAMPLE_RATE = 24_000

# Heuristics to regenerate bad turns.
MIN_WORDS_PER_SECOND = 1.5
MAX_WORDS_PER_SECOND = 5.0
MAX_TURN_ATTEMPTS = 3

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "output"
DEFAULT_DEMO_SCRIPT = Path(__file__).resolve().parent / "demo_script.yaml"
# Miso's first turn is unstable when generated from empty context.
WARMUP_DIR = Path(__file__).resolve().parent / "warmup"


@dataclass
class ScriptSegment:
    speaker: int
    text: str

@dataclass
class RenderedTurn:
    speaker: int
    text: str
    audio: mx.array
    sample_rate: int
    gen_seconds: float

    @property
    def duration_s(self) -> float:
        return self.audio.shape[0] / self.sample_rate


def load_script(path: Path) -> list[ScriptSegment]:
    """Parse the YAML script into segments."""
    raw = yaml.safe_load(path.read_text())

    segments = []
    for item in raw["segments"]:
        speaker = int(item.get("speaker", 0))
        assert speaker in (0, 1)
        text = str(item.get("text", "")).strip()
        assert text
        segments.append(ScriptSegment(speaker=speaker, text=text))
    return segments

def load_warmup_prompts(sample_rate: int = TARGET_SAMPLE_RATE) -> list[RenderedTurn]:
    """Read the warmup manifest describing the pre-generated priming clips."""
    manifest = WARMUP_DIR / "prompts.yaml"
    if not manifest.exists():
        raise SystemExit(f"missing warmup manifest {manifest} -- or pass --no-warmup")

    warmup: list[RenderedTurn] = []
    for entry in yaml.safe_load(manifest.read_text())["segments"]:
        path = WARMUP_DIR / entry["audio"]
        if not path.exists():
            raise SystemExit(
                f"missing warmup clip {path} -- see {WARMUP_DIR / 'prompts.yaml'} "
                "for how to regenerate it, or pass --no-warmup"
            )
        # Resamples to the model's rate (clips are 44.1k, Miso is 24k) if provided.
        audio = load_audio(str(path), sample_rate=sample_rate)
        speaker, text = int(entry["speaker"]), entry["text"]
        warmup.append(RenderedTurn(speaker, text, audio, sample_rate, 0.0))

    return warmup

def trim_context(turns: list[RenderedTurn], budget_s: float) -> list[RenderedTurn]:
    """Keep the most recent whole turns fitting within ``budget_s`` of audio."""
    kept: list[RenderedTurn] = []
    total = 0.0
    for turn in reversed(turns):
        if kept and total + turn.duration_s > budget_s:
            break
        kept.append(turn)
        total += turn.duration_s
    return list(reversed(kept))

def is_collapsed(text: str, duration_s: float) -> bool:
    """Heuristic to detect collapsed turn generations (greater risk early on)."""
    words = len(text.split())
    if not words:
        return False
    wps = words / max(duration_s, 1e-6)
    collapse = not (MIN_WORDS_PER_SECOND <= wps <= MAX_WORDS_PER_SECOND)
    if collapse:
        print(f" !! words/second: {wps:.2f} (expected {MIN_WORDS_PER_SECOND}-{MAX_WORDS_PER_SECOND})")
    return collapse

def render_miso(
    model,
    segments: list[ScriptSegment],
    context_seconds: float,
    max_turn_seconds: float,
    sampling: dict,
    save_turns: bool,
    output: Path,
    use_warmup: bool = False,
) -> list[RenderedTurn]:
    """Generate turn by turn, feeding prior turns back as Sesame ``Segment`` context."""
    from mlx_audio.tts.models.sesame.sesame import Segment
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(
        temp=sampling["temperature"], top_k=sampling["top_k"], top_p=sampling["top_p"]
    )
    # The script's speaker labels map onto the model's 0-indexed speaker ids.
    speaker_id = {0: 0, 1: 1}

    def generate_turn(
        seg: ScriptSegment, history: list[RenderedTurn]
    ) -> RenderedTurn:
        context = [
            Segment(speaker=speaker_id[t.speaker], text=t.text, audio=t.audio)
            for t in history
        ]
        # Sampling occasionally emits EOS early and truncates the turn; the same
        # text succeeds on a re-roll, so retry before accepting a short result.
        for attempt in range(1, MAX_TURN_ATTEMPTS + 1):
            tic = time.perf_counter()
            results = list(
                model.generate(
                    text=seg.text,
                    speaker=speaker_id[seg.speaker],
                    context=context,
                    # One call == one turn: don't let mlx-audio re-split on
                    # newlines, and don't let voice_match collapse the multi-turn
                    # context down to a single cloned reference segment.
                    split_pattern=None,
                    voice_match=False,
                    sampler=sampler,
                    max_audio_length_ms=max_turn_seconds * 1000,
                    stream=False,
                )
            )
            gen_s = time.perf_counter() - tic

            audio = mx.concatenate([r.audio for r in results], axis=0)
            mx.eval(audio)
            turn = RenderedTurn(
                seg.speaker, seg.text, audio, results[0].sample_rate, gen_s
            )
            if not is_collapsed(seg.text, turn.duration_s):
                return turn
            if attempt < MAX_TURN_ATTEMPTS:
                print(f"  retrying {attempt}/{MAX_TURN_ATTEMPTS - 1}")
        print(f"  !! still collapsed after {MAX_TURN_ATTEMPTS} attempts")
        return turn

    # Prime the context with pre-generated clips so the first real turn already
    # has a settled voice and a two-speaker layout to imitate. These are context
    # only: they never enter `turns`, so they never reach the output.
    print(f"loading warmup clips from {WARMUP_DIR / 'prompts.yaml'}")
    warmup = load_warmup_prompts(model.sample_rate) if use_warmup else []
    for turn in warmup:
        print(f"[warmup] speaker {turn.speaker} | {turn.duration_s:.2f}s | {turn.text}")

    turns: list[RenderedTurn] = []
    for i, seg in enumerate(segments):
        # Warmup turns stay in the context window, ageing out as real turns
        # accumulate, but never enter `turns` and so never reach the output.
        history = trim_context(warmup + turns, context_seconds)
        ctx_s = sum(t.duration_s for t in history)
        print(
            f"\n[{i}/{len(segments)-1}] speaker {seg.speaker} | "
            f"context: {len(history)} turns, {ctx_s:.1f}s"
        )
        print(f"  {seg.text}")

        turn = generate_turn(seg, history)
        turns.append(turn)
        print(
            f"  -> {turn.duration_s:.2f}s audio in {turn.gen_seconds:.2f}s "
            f"(RTF {turn.gen_seconds / max(turn.duration_s, 1e-6):.2f}x)"
        )

        if save_turns:
            turn_dir = output.with_suffix("")
            turn_dir.mkdir(parents=True, exist_ok=True)
            path = turn_dir / f"turn_{i:03d}_spk{turn.speaker}.wav"
            audio_write(path, turn.audio, turn.sample_rate, format="wav")
            print(f"  saved: {path}")

    return turns


def stitch(turns: list[RenderedTurn], gap_s: float) -> tuple[mx.array, int]:
    sample_rate = turns[0].sample_rate
    gap = mx.zeros((int(gap_s * sample_rate),), dtype=turns[0].audio.dtype)
    pieces: list[mx.array] = []
    for i, turn in enumerate(turns):
        if i and gap_s > 0.0:
            pieces.append(gap)
        pieces.append(turn.audio)
    return mx.concatenate(pieces, axis=0), sample_rate


app = typer.Typer(add_completion=False, help=__doc__.split("\n")[0])


@app.command()
def render(
    script: Path = typer.Argument(DEFAULT_DEMO_SCRIPT, help="YAML script of two-speaker segments"),
    model: str = typer.Option("miso", help="TTS backend"),
    model_id: str = typer.Option(None, help="override the backend's default model id"),
    max_segments: int = typer.Option(None, help="render only the first N segments"),
    context_seconds: float = typer.Option(
        DEFAULT_CONTEXT_SECONDS, help="max seconds of past audio kept as context"
    ),
    max_turn_seconds: float = typer.Option(
        30.0, help="generation cap for a single turn"
    ),
    warmup: bool = typer.Option(
        True, help="prime context with the pre-generated warmup clips"
    ),
    save_turns: bool = typer.Option(False, help="also write each turn as its own wav"),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="output directory (default: ./output)"),
) -> None:
    """Render a two-speaker podcast script to a single stitched wav."""
    if model not in MODELS:
        raise typer.BadParameter(f"unknown backend {model!r}, expected one of {sorted(MODELS)}")
    if model != "miso":
        raise typer.BadParameter(
            f"backend {model!r} is not wired up yet -- only 'miso' is supported"
        )

    segments = load_script(script)
    if max_segments:
        segments = segments[:max_segments]

    resolved_id = model_id or MODELS[model]
    output = output_dir / f"podcast_{model}.wav"

    print(f"loading {resolved_id}")
    tts = load_model(resolved_id)

    tic = time.perf_counter()
    turns = render_miso(
        tts,
        segments,
        context_seconds=context_seconds,
        max_turn_seconds=max_turn_seconds,
        sampling=SAMPLING[model],
        save_turns=save_turns,
        output=output,
        use_warmup=warmup,
    )
    wall_s = time.perf_counter() - tic

    audio, sample_rate = stitch(turns, TURN_GAP_S)
    output.parent.mkdir(parents=True, exist_ok=True)
    audio_write(output, audio, sample_rate, format="wav")

    total_audio_s = audio.shape[0] / sample_rate
    print(f"\n{'=' * 50}")
    print(f"{len(turns)} turns, {total_audio_s:.1f}s audio in {wall_s:.1f}s")
    print(f"overall RTF: {wall_s / max(total_audio_s, 1e-6):.2f}x")
    print(f"peak memory: {mx.get_peak_memory() / 1e9:.2f} GB")
    print(f"saved: {output}")
    print(f"{'=' * 50}")


@app.command("warmup")
def regen_warmup(
    speaker: int = typer.Option(
        None, help="regenerate only this speaker (default: all in the manifest)"
    ),
    model_id: str = typer.Option(MODELS["fish"], help="Fish model to generate with"),
    sample_rate: int = typer.Option(
        TARGET_SAMPLE_RATE, help="resample clips to this rate before saving"
    ),
) -> None:
    """Regenerate the short warmup clips used to prime Miso context with another TTS model.

    Prompts live in warmup/prompts.yaml; edit the bracket tags there to change the
    voices. Clips are written in place, so just re-run this until you like them.

    Clips are saved at ``--sample-rate`` (Miso's 24kHz by default) rather than the
    generating model's rate, so they load without a resample at render time.
    """
    with open(WARMUP_DIR / "prompts.yaml") as f:
        segments = yaml.safe_load(f)["segments"]
        if speaker is not None:
            segments = [s for s in segments if s["speaker"] == speaker]

    print(f"loading {model_id}")
    fish = load_model(model_id)

    for segment in segments:
        path = WARMUP_DIR / segment["audio"]
        prompt = segment["prompt"]
        print(f"\nspeaker {segment['speaker']} | {segment['prompt']}")

        results = list(fish.generate(text=prompt, **SAMPLING["fish"], stream=False))
        audio = mx.concatenate([r.audio for r in results], axis=0)
        mx.eval(audio)

        src_rate = results[0].sample_rate
        if src_rate != sample_rate:
            audio = resample_audio(audio, src_rate, sample_rate)
            mx.eval(audio)
            print(f"  resampled {src_rate} -> {sample_rate} Hz")
        audio_write(path, audio, sample_rate, format="wav")

        duration = audio.shape[0] / sample_rate
        words = len(prompt.split())
        note = "  <== slow, consider re-rolling" if words / duration < 1.5 else ""
        print(
            f"  -> {duration:.2f}s  {words / duration:.2f} wps{note}\n"
            f"  saved: {path}"
        )


if __name__ == "__main__":
    app()
