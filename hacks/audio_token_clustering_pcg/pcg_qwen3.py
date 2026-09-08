"""Acoustic token clusters for the Qwen3-TTS codec embedding tables.

Qwen3-TTS is a two-stage RQ-transformer, so unlike Chatterbox there is one group table
per codebook: the talker's codebook 0, plus 15 `code_predictor` depth levels. Group
construction is imported from
``experiments/audio_token_clustering_pcg/cluster.py``. See README.md.

Unmaintained: the clustering workstream moved to Chatterbox, so this is kept for the
per-codebook numbers in README.md rather than actively run.

```bash
python hacks/audio_token_clustering_pcg/pcg_qwen3.py
```
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import typer
from safetensors import safe_open

# Group construction stayed in experiments/ when this moved to hacks/.
sys.path.insert(
    0,
    str(Path(__file__).resolve().parents[2] / "experiments" / "audio_token_clustering_pcg"),
)

import cluster  # noqa: E402

#: Every codebook holds 2048 codes. The talker's head is 3072 wide, but ids >= 2048 are
#: near-zero-norm reserved slots that read as singletons and skew every group statistic.
LIVE_VOCAB_SIZE = 2048

NUM_DEPTH_LEVELS = 15

#: The 8-bit MLX checkpoints; codec embeddings are stored unquantized in both, and
#: these are what the acceptance experiment already benchmarks against.
MODELS = {
    "qwen3-0.6b": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "qwen3-1.7b": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
}

THETAS = "0.9,0.8,0.7,0.6,0.5,0.45,0.4,0.3,0.2,0.1"

#: Theta the dumped cluster mappings are built at, mid-band per the README.
DUMP_THETA = 0.4

app = typer.Typer(
    add_completion=False,
    help="Build Acoustic Similarity Groups for the Qwen3-TTS codec embedding tables.",
)


def load_codec_embeddings(model: str) -> list[np.ndarray]:
    """The 16 codec embedding tables: codebook 0 from the talker, then cb1..cb15.

    Read straight from the checkpoint rather than instantiating the model.
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(repo_id=MODELS[model], filename="model.safetensors")
    tables = []
    with safe_open(path, framework="pt") as handle:
        cb0 = handle.get_tensor("talker.model.codec_embedding.weight")
        tables.append(cb0.float().numpy()[:LIVE_VOCAB_SIZE])
        for level in range(NUM_DEPTH_LEVELS):
            key = f"talker.code_predictor.model.codec_embedding.{level}.weight"
            tables.append(handle.get_tensor(key).float().numpy()[:LIVE_VOCAB_SIZE])
    return tables


@app.command()
def main(
    models: str = typer.Option(
        ",".join(MODELS),
        help=f"Comma-separated checkpoints: {' / '.join(MODELS)}",
    ),
    thetas: str = typer.Option(
        THETAS, help="Comma-separated thetas to report sweep stats at."
    ),
    dump_theta: float = typer.Option(
        DUMP_THETA, help="theta to write cluster mappings for"
    ),
    out_dir: Path = typer.Option(
        Path(__file__).parent / "results", help="output root"
    ),
) -> None:
    names = [part for part in models.split(",") if part.strip()]
    for model in names:
        if model not in MODELS:
            raise typer.BadParameter(f"model must be one of {' / '.join(MODELS)}")
    thetas = cluster.parse_floats(thetas, option="--thetas")


    loaded = {}
    for model in names:
        tables = loaded[model] = load_codec_embeddings(model)
        similarities = [cluster.cosine_matrix(table) for table in tables]

        codebooks, mappings = [], []
        for index, similarity in enumerate(similarities):
            sweep = [cluster.group_stats(similarity, theta) for theta in thetas]
            codebooks.append({"codebook": index, "sweep": sweep})
            mappings.append({
                "codebook": index,
                **cluster.cluster_mapping(similarity, dump_theta),
            })

        print(f"\n{model}  vocab={LIVE_VOCAB_SIZE}  codebooks={len(tables)}")
        for entry in codebooks:
            cluster.print_sweep(entry["sweep"], label=f"cb{entry['codebook']}")

        summary = cluster.write_json(
            out_dir / "summary" / f"{model}.json",
            {"model": model, "vocab": LIVE_VOCAB_SIZE, "codebooks": codebooks},
        )
        clusters = cluster.write_json(
            cluster.cluster_path(out_dir, model, dump_theta),
            {"model": model, "theta": dump_theta, "codebooks": mappings},
        )
        print(f"\n  wrote {summary}\n  wrote {clusters}")

    if len(loaded) == 2:
        small, large = (loaded[name] for name in MODELS)
        print("\n0.6B vs 1.7B codec_embedding (Gram cosine):")
        for index, (a, b) in enumerate(zip(small, large)):
            print(f"  cb{index:<3} {cluster.gram_cosine(a, b):.4f}")


if __name__ == "__main__":
    app()
