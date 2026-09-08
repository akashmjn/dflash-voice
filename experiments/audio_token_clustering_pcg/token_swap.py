"""ASG token-swapping ablation for Chatterbox (arXiv 2511.13732, Section 3).

Generates speech, replaces every token with a uniform draw from its group
G(t) = {t' : cos(Emb(t), Emb(t')) > theta}, and renders both sequences with S3Gen.
Groups come from ``cluster.py``'s dumps, so run that first.

Uses the mlx-audio port rather than the torch package -- torchaudio's resampler trips
the MPS conv1d channel cap on references over a few seconds.

Run in the mlx_decode venv, from the repo root:
    python experiments/audio_token_clustering_pcg/token_swap.py [--limit 2]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
import typer

import cluster

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent

MODEL_ID = "mlx-community/Chatterbox-TTS-8bit"

#: The mlx-community checkpoint ships no conds.safetensors, so a voice is required.
#: Also the S3Gen prompt for every render, so it stays disjoint from the target.
DEFAULT_REF_AUDIO = REPO_ROOT / "mlx_decode" / "warmup" / "jensen-30sec.wav"

#: bench.py's prompt list, so generations here match the checked-in renders.
DEFAULT_PROMPTS = [
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

#: bench.py's cbox-ar sampling knobs.
app = typer.Typer(
    add_completion=False,
    help="Swap Chatterbox speech tokens within their Acoustic Similarity Groups.",
)

GENERATE_KWARGS = dict(
    temperature=0.8,
    min_p=0.05,
    top_p=1.0,
    repetition_penalty=1.2,
    cfg_weight=0.5,
    max_tokens=1000,
)


def build_groups(
    thetas: list[float], results_dir: Path, model_slug: str
) -> dict[float, list[np.ndarray]]:
    """Per-seed member arrays at each theta, from ``cluster.py``'s dumps."""
    groups = {}
    for theta in thetas:
        path = cluster.cluster_path(results_dir, model_slug, theta)
        if not path.exists():
            raise SystemExit(
                f"no cluster dump at {path}\n  Run: python "
                f"experiments/audio_token_clustering_pcg/cluster.py --dump-thetas {theta}"
            )
        entry = json.loads(path.read_text())
        groups[theta] = [np.asarray(g) for g in entry["groups"]]
    return groups


def swap_tokens(
    tokens: np.ndarray, groups: list[np.ndarray], rng: np.random.Generator
) -> np.ndarray:
    """Uniform draw from G(t) at every position; t is a member, so some survive."""
    out = tokens.copy()
    for i, token in enumerate(tokens):
        members = groups[token]
        if members.size > 1:
            out[i] = rng.choice(members)
    return out


def generate_tokens(wrapper, text: str, seed: int) -> np.ndarray:
    """The speech tokens T3 emitted, before S3Gen sees them.

    ``generate()`` yields audio, so the tokens only exist inside the call; stubbing
    ``_decode_tokens`` captures them and skips a render this ablation redoes itself.
    """
    from mlx_decode import cbox_ar
    from mlx_audio.tts.models.chatterbox.chatterbox import SPEECH_VOCAB_SIZE

    captured: list[list[int]] = []
    original_decode = cbox_ar._decode_tokens

    def capture(mlx_model, token_ids):
        captured.append(list(token_ids))
        return mx.zeros((1,), dtype=mx.float32)

    cbox_ar._decode_tokens = capture
    try:
        mx.random.seed(seed)
        # Draining the generator is what runs T3; the stub returns empty audio,
        # which generate() drops before yielding.
        list(wrapper.generate(text, **GENERATE_KWARGS))
    finally:
        cbox_ar._decode_tokens = original_decode

    if not captured:
        raise SystemExit(f"T3 emitted no tokens for: {text[:60]!r}")

    hp = wrapper._model.t3.hp
    return np.array(
        [
            t
            for t in captured[0]
            if t not in (hp.start_speech_token, hp.stop_speech_token)
            and t < SPEECH_VOCAB_SIZE
        ],
        dtype=np.int32,
    )


def decode(model, ids: np.ndarray, ref_dict: dict) -> np.ndarray:
    wav = model.s3gen(
        speech_tokens=mx.array(ids[None], dtype=mx.int32),
        ref_dict=ref_dict,
        finalize=True,
    )
    if wav.ndim == 2:
        wav = wav.squeeze(0)
    mx.eval(wav)
    return np.array(wav)


def load_prompts(path: Path | None, limit: int | None) -> list[str]:
    """bench.py's built-ins, or one JSON object per line with a ``text`` field."""
    if path is None:
        return DEFAULT_PROMPTS[:limit]
    lines = path.read_text().splitlines()
    return [json.loads(line)["text"] for line in lines if line.strip()][:limit]


@app.command()
def main(
    thetas: str = typer.Option("0.6,0.45,0.3", help="Comma-separated thetas to swap at."),
    out_dir: Path = typer.Option(
        EXPERIMENT_DIR / "results" / "token_swap", help="output root"
    ),
    limit: Optional[int] = typer.Option(None, help="only the first N prompts"),
    prompts_file: Optional[Path] = typer.Option(
        None, "--prompts",
        help="JSONL with a 'text' field per line (default: bench.py's prompts)",
    ),
    only: Optional[str] = typer.Option(
        None, help="Comma-separated prompt indices to run, e.g. --only 4,5"
    ),
    seed: int = typer.Option(0, help="Sampling and swap RNG seed."),
    model_id: str = typer.Option(MODEL_ID, help="Chatterbox checkpoint to generate with."),
    ref_audio: Path = typer.Option(DEFAULT_REF_AUDIO, help="Reference voice clip."),
    results_dir: Path = typer.Option(
        EXPERIMENT_DIR / "results",
        help="holds token_clusters/MODEL/thetaTT.json, written by cluster.py",
    ),
    cluster_model: str = typer.Option(
        "chatterbox-ar", help="which model's cluster dumps to swap over"
    ),
) -> None:
    thetas = cluster.parse_floats(thetas, option="--thetas")
    try:
        wanted = [int(part) for part in only.split(",") if part.strip()] if only else None
    except ValueError:
        raise typer.BadParameter(f"--only wants comma-separated integers, got {only!r}")

    import soundfile
    from mlx_audio.tts.models.chatterbox.chatterbox import S3GEN_SR
    from mlx_decode import cbox_ar

    prompts = load_prompts(prompts_file, limit)
    indices = wanted if wanted else range(len(prompts))
    if not prompts:
        raise SystemExit("no prompts to run")

    groups = build_groups(thetas, results_dir, cluster_model)
    wrapper = cbox_ar.load_model(model_id, ref_audio=str(ref_audio))
    model = wrapper._model
    ref_dict = model._conds.gen
    rng = np.random.default_rng(seed)

    summary = []
    for idx in indices:
        text = prompts[idx]
        stem = f"prompt_{idx:03d}"
        tokens = generate_tokens(wrapper, text, seed)

        utterance_dir = out_dir / stem
        utterance_dir.mkdir(parents=True, exist_ok=True)
        np.save(utterance_dir / "tokens_original.npy", tokens)
        (utterance_dir / "prompt.txt").write_text(text + "\n")
        soundfile.write(
            utterance_dir / "original.wav", decode(model, tokens, ref_dict), S3GEN_SR
        )

        row = {
            "utterance": stem,
            "text": text,
            "frames": int(tokens.size),
        }
        for theta in thetas:
            swapped = swap_tokens(tokens, groups[theta], rng)
            sizes = np.array([groups[theta][token].size for token in tokens])
            tag = f"theta{theta:.2f}"
            np.save(utterance_dir / f"tokens_swap_{tag}.npy", swapped)
            soundfile.write(
                utterance_dir / f"swap_{tag}.wav",
                decode(model, swapped, ref_dict),
                S3GEN_SR,
            )
            row[tag] = {
                "frac_eligible": round(float((sizes > 1).mean()), 4),
                "frac_changed": round(float((swapped != tokens).mean()), 4),
                "mean_group_size": round(float(sizes.mean()), 2),
            }
        summary.append(row)
        print(f"  {stem}: {tokens.size} frames -> {utterance_dir}")

    payload = {
        "model": "cbox-ar",
        "model_id": model_id,
        "ref_audio": str(ref_audio),
        "thetas": thetas,
        "seed": seed,
        "generate_kwargs": GENERATE_KWARGS,
        "utterances": summary,
    }
    cluster.write_json(out_dir / "summary.json", payload)

    print(f"\n  {'utterance':<24} {'frames':>6}", end="")
    for theta in thetas:
        print(f" {'chg@' + format(theta, '.2f'):>9} {'|G|@' + format(theta, '.2f'):>9}", end="")
    print()
    for row in summary:
        print(f"  {row['utterance']:<24} {row['frames']:>6}", end="")
        for theta in thetas:
            stats = row[f"theta{theta:.2f}"]
            print(
                f" {stats['frac_changed']:>9.3f} {stats['mean_group_size']:>9.1f}",
                end="",
            )
        print()
    print(f"\n  wrote {out_dir / 'summary.json'}")


if __name__ == "__main__":
    app()
