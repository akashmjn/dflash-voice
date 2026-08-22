"""Acoustic token cluster size vs theta for the Chatterbox T3 embedding tables.

Groups follow the Principled Coarse-Graining construction of arXiv 2511.13732:
G(t) = {t' : cos(Emb(t), Emb(t')) > theta}, seeded once per token. Chatterbox has one
speech vocabulary read by one head, so there is a single group table rather than one
per RVQ depth level. See README.md.

```bash
python experiments/audio_token_clustering_pcg/pcg_chatterbox.py
```
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from safetensors import safe_open

#: Trained S3 speech codes. Above this sit BOS/EOS (6561/6562) and untrained padding
#: rows, whose low-norm directions would otherwise dominate any geometry statistic.
LIVE_VOCAB_SIZE = 6561

MODELS = {
    "cbox-ar": ("ResembleAI/chatterbox", "t3_cfg.safetensors"),
    "cbox-flash": ("ResembleAI/chatterbox-flash", "t3_flash.safetensors"),
}

THETAS = [0.9, 0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.3, 0.2, 0.1]


def load_speech_embedding(model: str, *, live_only: bool = True) -> np.ndarray:
    """T3's speech_emb table, trimmed to the trained ids by default."""
    from huggingface_hub import hf_hub_download

    repo_id, filename = MODELS[model]
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with safe_open(path, framework="pt") as handle:
        table = handle.get_tensor("speech_emb.weight").float().numpy()
    return table[:LIVE_VOCAB_SIZE] if live_only else table


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    weights = np.asarray(embeddings, dtype=np.float32)
    unit = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    return unit @ unit.T


def group_stats(similarity: np.ndarray, theta: float) -> dict[str, float]:
    """Size distribution of G_k = {t : cos(Emb[k], Emb[t]) > theta}.

    The diagonal is forced on so theta >= 1 degrades to singletons, not empty groups.
    """
    membership = similarity > theta
    np.fill_diagonal(membership, True)
    sizes = membership.sum(axis=1)
    vocab_size = similarity.shape[0]
    return {
        "theta": theta,
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "p90": float(np.percentile(sizes, 90)),
        "max": int(sizes.max()),
        "frac_singleton": float((sizes == 1).mean()),
        "frac_vocab": float(sizes.mean() / vocab_size),
    }


def anisotropy_stats(similarity: np.ndarray) -> dict[str, float]:
    """Off-diagonal cosine spread: a large mean is a shared direction that inflates
    group size at fixed theta without expressing similarity structure.
    """
    off_diagonal = similarity[~np.eye(similarity.shape[0], dtype=bool)]
    return {
        "mean_cos": float(off_diagonal.mean()),
        "std_cos": float(off_diagonal.std()),
        "p99_cos": float(np.percentile(off_diagonal, 99)),
        "max_cos": float(off_diagonal.max()),
    }


def sweep(model: str, thetas: list[float]) -> dict:
    embeddings = load_speech_embedding(model)
    similarity = cosine_matrix(embeddings)
    return {
        "model": model,
        "vocab": int(embeddings.shape[0]),
        "dim": int(embeddings.shape[1]),
        "anisotropy": anisotropy_stats(similarity),
        "sweep": [group_stats(similarity, theta) for theta in thetas],
    }


def table_agreement() -> dict:
    """How far Flash's speech embeddings moved from AR's.

    Flash is a finetune of AR; the tables differ, but a high Gram cosine means they
    induce the same geometry, so AR's ASGs stay valid over Flash's distributions.
    """
    ar = load_speech_embedding("cbox-ar")
    flash = load_speech_embedding("cbox-flash")

    unit_ar = ar / np.maximum(np.linalg.norm(ar, axis=1, keepdims=True), 1e-12)
    unit_flash = flash / np.maximum(np.linalg.norm(flash, axis=1, keepdims=True), 1e-12)
    per_token = (unit_ar * unit_flash).sum(axis=1)

    # Gram cosine: same geometry, beyond per-row drift.
    gram_ar = cosine_matrix(ar)
    gram_flash = cosine_matrix(flash)
    off = ~np.eye(gram_ar.shape[0], dtype=bool)
    a, b = gram_ar[off], gram_flash[off]
    gram_cos = float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    return {
        "per_token_cos_mean": float(per_token.mean()),
        "per_token_cos_min": float(per_token.min()),
        "gram_cos": gram_cos,
    }


def print_sweep(result: dict) -> None:
    print(f"\n{result['model']}  vocab={result['vocab']}  dim={result['dim']}")
    print(f"\n  {'theta':>6} {'mean':>9} {'median':>7} {'p90':>7} {'max':>6}"
          f" {'singleton':>10} {'%vocab':>8}")
    for row in result["sweep"]:
        print(f"  {row['theta']:>6.2f} {row['mean']:>9.1f} {row['median']:>7.0f}"
              f" {row['p90']:>7.0f} {row['max']:>6d} {row['frac_singleton']:>10.3f}"
              f" {100 * row['frac_vocab']:>7.2f}%")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    parser.add_argument("--thetas", nargs="+", type=float, default=THETAS)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).with_name("results"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        result = sweep(model, args.thetas)
        print_sweep(result)
        path = args.out_dir / f"{model.replace('cbox-', 'chatterbox-')}.json"
        path.write_text(json.dumps(result, indent=2))
        print(f"\n  wrote {path}")

    if set(args.models) == set(MODELS):
        agreement = table_agreement()
        print("\nAR vs Flash speech_emb:")
        for key, value in agreement.items():
            print(f"  {key:20s} {value:.5f}")


if __name__ == "__main__":
    main()
