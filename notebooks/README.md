# Notebooks

Analysis of TTS model predictions over codebooks - predictive entropy, teacher-forced NLL (CE loss) of ground-truth audio codes on samples from the Expresso dataset.

## Reproduce

Run from the repo root. Each step feeds the next:

**1. Dataprep** — tokenize + featurize a model's rows into `data/DATASET/featurized/MODEL/ROW/` (DATASET defaults to `expresso`):

```bash
python -m dataprep.prepare --model miso --stage all
```

**2. Compute metrics (negative log likelihood - NLL, predictive entropy)** — read the featurized rows summarize model logits and write to `data/DATASET/metrics/MODEL/ROW/MODEL_metrics.{npz,json}`:

```bash
python notebooks/frame_metrics.py --model miso
```

Runs every featurized row by default; pass `--rows N` for just the first N.

**3. Explore** — open the marimo notebook and pick a model / row:

```bash
marimo edit notebooks/metrics_explore.py
```



## What the notebook shows

- **Global view** — averaged metrics per-codebook and as a grouped time series over the whole file.
- **Per-segment explorer** — the same views for one selected sequence in the file.

Global row view — miso

Along with entropy plotted above, the Teacher-forced NLL (CE loss) is also computed — how many bits each model spends to reproduce the ground-truth audio codes. Lower value means better predicted/compressed.


Pooled over the first 3 Expresso rows (`--rows 3`):

| Model | Codebooks | Frame rate (Hz) | Semantic kbit/s (avg NLL) | Audio kbit/s (avg NLL per codebook) | Total kbit/s (avg NLL per codebook) |
| ----- | --------- | --------------- | ---------------------------- | -------------------------------------- | -------------------------------------- |
| miso  | 32        | 12.5            | 0.031 (1.75)                 | 2.357 (4.22)                           | 2.389 (4.14)                           |
| qwen3 | 16        | 12.5            | 0.150 (8.29)                 | 1.524 (5.63)                           | 1.673 (5.80)                           |
| fish  | 10        | 21              | 0.139 (4.60)                 | 1.358 (4.98)                           | 1.498 (4.94)                           |


For multiple audio codebooks we average `avg NLL per codebook = avg_K (NLL_k); K=num_codebooks` (so each group's figure is the per-codebook cost of one frame, in nats), while for semantic codes this is directly the NLL/CE loss. This is then normalized to `kbits/s` computed as  `kbits/s = avg NLL per codebook × log2(e) × frame_rate x num_codebooks / 1000`, multiplying the count back in so the semantic and audio bitrates sum to the total.

This normalization accounts for differences in frame rate and number of codebooks making it more comparable. All three land in the same 1.5–2.4 kbit/s band despite differing codebook counts and frame rates. miso spends the most (~1.6× fish) but spreads it over 32 codebooks, so its per-codebook cost is the lowest of the three; it also has by far the cheapest semantic stream, predicting codebook 0 at 1.75 nats where qwen3 needs 8.29.