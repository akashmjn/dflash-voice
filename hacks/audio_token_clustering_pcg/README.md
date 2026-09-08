# Qwen3-TTS token clustering (unmaintained)

The same acoustic-token clustering analysis (see the
[Chatterbox writeup](../../experiments/audio_token_clustering_pcg/README.md) for the method) applied
to the RVQ case. [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) has a
talker LM emitting codebook 0 over time, plus a `code_predictor` running 15 depth levels per
frame. Each has its own 2048-entry embedding table, so there are 16 group tables rather than
one.

## Cluster size vs θ, per codebook

Mean group size, from the 1.7B:

| θ        | cb0      | cb1       | cb3      | cb7     | cb11    | cb15    |
| -------- | -------- | --------- | -------- | ------- | ------- | ------- |
| 0.60     | 4.4      | 35.1      | 1.8      | 1.0     | 1.0     | 1.0     |
| 0.50     | 12.7     | 84.6      | 7.9      | 1.2     | 1.0     | 1.0     |
| **0.45** | **20.8** | **122.7** | **17.2** | **1.7** | **1.0** | **1.0** |
| **0.40** | **34.8** | **172.4** | **34.6** | **3.0** | **1.2** | **1.0** |
| 0.30     | 107.9    | 312.5     | 108.0    | 12.4    | 4.8     | 1.1     |
| 0.20     | 381.1    | 518.7     | 265.3    | 75.0    | 100.8   | 20.5    |

**Group size varies with depth at a fixed θ.** At θ=0.40 it spans ~2.5 orders of magnitude:
cb1 holds 172 tokens (8.4% of vocabulary, zero singletons) while cb15 is 99.8% singletons —
and a singleton group is provably identical to ordinary speculative decoding.

The gradient follows what the codebooks carry. Early levels hold semantic content the LM
reorganizes into clusters; deep levels are near-pure acoustic residual left close to
orthogonal, so almost no pairs clear any given threshold. Codebook 0 behaves like Chatterbox
— one table, smooth curve, negligible singletons by θ=0.45 — which is unsurprising given it
is the one autoregressive-over-time head in the model.

## One group table covers both sizes

The geometry agrees closely enough across sizes to share group tables — Gram cosine 0.98 at
cb0, 0.99 at cb1, dipping to 0.85 mid-stack — so the 0.6B's groups stay valid over the
1.7B's distributions. Full sweeps are in `results/summary/qwen3-0.6b.json` and
`results/summary/qwen3-1.7b.json` under this directory.

## Reproduce

Needs `numpy` and `safetensors`; checkpoints download to the HF cache on first run. From the
repo root:

```bash
python hacks/audio_token_clustering_pcg/pcg_qwen3.py
```

The script reads the 8-bit MLX checkpoints, whose codec embeddings are stored unquantized.
It accepts `--models`, `--thetas`, and `--dump-theta` (default 0.4). Each model's sweep is
written to `results/summary/MODEL.json`; its cluster mappings are written to
`results/token_clusters/MODEL/thetaTT.json` as a `codebooks` list of 16 entries, one per
codebook. The per-entry fields are described in the
[Chatterbox writeup](../../experiments/audio_token_clustering_pcg/README.md).

Group construction is imported from that experiment's `cluster.py`, so this script depends on
`experiments/audio_token_clustering_pcg/` staying in place.
