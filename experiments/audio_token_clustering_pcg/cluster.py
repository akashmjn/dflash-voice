"""Acoustic Similarity Groups for the Chatterbox T3 embedding tables.

Groups follow the Principled Coarse-Graining construction of arXiv 2511.13732:
G(t) = {t' : cos(Emb(t), Emb(t')) > theta}, seeded once per token. Groups overlap, and
cosine is reflexive so no token is ever orphaned.

Chatterbox has one speech vocabulary read by one head, so there is a single group table
rather than one per RVQ depth level. The construction helpers are also imported by
``token_swap.py`` and by ``hacks/audio_token_clustering_pcg/pcg_qwen3.py``, which builds
one table per codebook. See README.md.

```bash
python experiments/audio_token_clustering_pcg/cluster.py
```
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer
from safetensors import safe_open

#: Trained S3 speech codes. Above this sit BOS/EOS (6561/6562) and untrained padding
#: rows, whose low-norm directions would otherwise dominate any geometry statistic.
LIVE_VOCAB_SIZE = 6561

MODELS = {
    "cbox-ar": ("ResembleAI/chatterbox", "t3_cfg.safetensors"),
    "cbox-flash": ("ResembleAI/chatterbox-flash", "t3_flash.safetensors"),
}

THETAS = "0.9,0.8,0.7,0.6,0.5,0.45,0.4,0.3,0.2,0.1"

#: Thetas the mappings are dumped at, spanning the README's usable band. Narrower
#: than THETAS: members grow as mean group size x vocab, megabytes at permissive theta.
DUMP_THETAS = "0.6,0.5,0.45,0.4,0.3"

app = typer.Typer(
    add_completion=False,
    help="Build Acoustic Similarity Groups from the Chatterbox T3 embedding tables.",
)


# ---------------------------------------------------------------------------
# Group construction
# ---------------------------------------------------------------------------


def cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity over the rows of an embedding table."""
    weights = np.asarray(embeddings, dtype=np.float32)
    unit = weights / np.maximum(np.linalg.norm(weights, axis=1, keepdims=True), 1e-12)
    return unit @ unit.T


def membership(similarity: np.ndarray, theta: float) -> np.ndarray:
    """Boolean (V, V) membership, `M[k, t]` true when token t is in group k.

    The diagonal is forced on so theta >= 1 degrades to singletons, not empty groups.
    """
    matrix = similarity > theta
    np.fill_diagonal(matrix, True)
    return matrix


def group_stats(similarity: np.ndarray, theta: float) -> dict[str, float]:
    """Size distribution of the groups at one theta."""
    sizes = membership(similarity, theta).sum(axis=1)
    return {
        "theta": theta,
        "mean": float(sizes.mean()),
        "median": float(np.median(sizes)),
        "max": int(sizes.max()),
        "frac_singleton": float((sizes == 1).mean()),
        "frac_vocab": float(sizes.mean() / similarity.shape[0]),
    }


def gram_cosine(a_table: np.ndarray, b_table: np.ndarray) -> float:
    """Agreement between two embedding geometries, ignoring dimension.

    Compares off-diagonal cosine matrices, so tables of different widths are
    comparable: a high value means both induce the same neighbourhoods, and one
    model's groups stay valid over the other's distributions.
    """
    gram_a, gram_b = cosine_matrix(a_table), cosine_matrix(b_table)
    off = ~np.eye(gram_a.shape[0], dtype=bool)
    a, b = gram_a[off], gram_b[off]
    return float((a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cluster_mapping(similarity: np.ndarray, theta: float) -> dict:
    """Group members per seed, plus the equal split weight per token.

    Overlap means a token's mass would be double-counted across the groups holding
    it, so PCG carries a weight `w[k, t] = 1 / |S(t)|` that partitions it —
    `owner_counts` is that `|S(t)|`, which is all the weights need since the split
    is equal. It is a column sum, so it does not fall out of the row-major groups.
    """
    matrix = membership(similarity, theta)
    _, members = np.nonzero(matrix)  # row-major, so already grouped by seed
    sizes = matrix.sum(axis=1)
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    return {
        "theta": theta,
        "vocab": int(matrix.shape[0]),
        "groups": [members[lo:hi].astype(int).tolist()
                   for lo, hi in zip(bounds[:-1], bounds[1:])],
        "owner_counts": matrix.sum(axis=0).astype(int).tolist(),
    }


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def cluster_path(out_dir: Path, model_slug: str, theta: float) -> Path:
    """Where one theta's mapping lives: `token_clusters/MODEL/thetaTT.json`."""
    return out_dir / "token_clusters" / model_slug / f"theta{theta:.2f}.json"


def write_json(path: Path, payload: dict) -> Path:
    """Write `payload` to `path`, creating the parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return path


def parse_floats(value: str, *, option: str) -> list[float]:
    """Comma-separated floats, the repo's convention for list-valued options."""
    try:
        return [float(part) for part in value.split(",") if part.strip()]
    except ValueError:
        raise typer.BadParameter(f"{option} wants comma-separated floats, got {value!r}")


def print_sweep(rows: list[dict], *, label: str) -> None:
    print(f"\n  {label}")
    print(f"  {'theta':>6} {'mean':>9} {'median':>7} {'max':>6}"
          f" {'singleton':>10} {'%vocab':>8}")
    for row in rows:
        print(f"  {row['theta']:>6.2f} {row['mean']:>9.1f} {row['median']:>7.0f}"
              f" {row['max']:>6d} {row['frac_singleton']:>10.3f}"
              f" {100 * row['frac_vocab']:>7.2f}%")


# ---------------------------------------------------------------------------
# Chatterbox entrypoint
# ---------------------------------------------------------------------------


def load_speech_embedding(model: str) -> np.ndarray:
    """T3's speech_emb table, trimmed to the trained ids."""
    from huggingface_hub import hf_hub_download

    repo_id, filename = MODELS[model]
    path = hf_hub_download(repo_id=repo_id, filename=filename)
    with safe_open(path, framework="pt") as handle:
        table = handle.get_tensor("speech_emb.weight").float().numpy()
    return table[:LIVE_VOCAB_SIZE]


def slug(model: str) -> str:
    return model.replace("cbox-", "chatterbox-")


@app.command()
def main(
    models: str = typer.Option(
        ",".join(MODELS),
        help=f"Comma-separated checkpoints: {' / '.join(MODELS)}",
    ),
    thetas: str = typer.Option(
        THETAS, help="Comma-separated thetas to report sweep stats at."
    ),
    dump_thetas: str = typer.Option(
        DUMP_THETAS, help="Comma-separated thetas to write cluster mappings for."
    ),
    out_dir: Path = typer.Option(
        Path(__file__).parent / "results", help="output root"
    ),
) -> None:
    names = [part for part in models.split(",") if part.strip()]
    for model in names:
        if model not in MODELS:
            raise typer.BadParameter(f"model must be one of {' / '.join(MODELS)}")
    thetas = parse_floats(thetas, option="--thetas")
    dump_thetas = parse_floats(dump_thetas, option="--dump-thetas")

    tables = {}
    for model in names:
        embeddings = load_speech_embedding(model)
        similarity = cosine_matrix(embeddings)
        tables[model] = embeddings

        sweep = [group_stats(similarity, theta) for theta in thetas]
        print_sweep(sweep, label=f"{model}  vocab={embeddings.shape[0]}"
                                 f"  dim={embeddings.shape[1]}")
        summary = write_json(
            out_dir / "summary" / f"{slug(model)}.json",
            {"model": model, "vocab": int(embeddings.shape[0]),
             "dim": int(embeddings.shape[1]), "sweep": sweep},
        )
        print(f"\n  wrote {summary}")
        for theta in dump_thetas:
            path = write_json(
                cluster_path(out_dir, slug(model), theta),
                {"model": model, **cluster_mapping(similarity, theta)},
            )
            print(f"  wrote {path}")

    if len(tables) == 2:
        a, b = (tables[name] for name in MODELS)
        print(f"\nAR vs Flash speech_emb Gram cosine: {gram_cosine(a, b):.4f}")


if __name__ == "__main__":
    app()
