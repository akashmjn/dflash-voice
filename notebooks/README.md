# Notebooks

Analysis of TTS model predictions over codebooks - predictive entropy, teacher-forced NLL (CE) of ground-truth codes on samples from the Expresso dataset.

## Reproduce

Run from the repo root. Each step feeds the next:

**1. Dataprep** — tokenize + featurize a model's rows into `data/expresso/MODEL_featurized/ROW/`:

```bash
python -m dataprep.prepare --model miso --stage all
```

**2. Compute metrics (negative log likelihood - NLL, predictive entropy)** — read the featurized rows summarize model logits and write to `data/expresso/metrics/ROW/MODEL_metrics.{npz,json}`:

```bash
python notebooks/frame_metrics.py --model miso
```

**3. Explore** — open the marimo notebook and pick a model / row:

```bash
marimo edit notebooks/metrics_explore.py
```



## What the notebook shows

- **Global view** — averaged metrics per-codebook and as a grouped time series over the whole file.
- **Per-segment explorer** — the same views for one selected sequence in the file.

Global row view — miso

Along with entropy plotted above, the Teacher-forced NLL is also computed — how many bits each model spends to reproduce the ground-truth audio codes. Lower value means better predicted/compressed.


| Model | Codebooks | Frame rate | Semantic kbit/s (nats/frame) | Audio kbit/s (nats/frame) | Total kbit/s (nats/frame) |
| ----- | --------- | ---------- | ---------------------------- | ------------------------- | ------------------------- |
| miso  | 32        | 12.5 Hz    | 0.034 (1.87)                 | 5.246 (290.91)            | 5.280 (292.78)            |
| qwen3 | 16        | 12.5 Hz    | 0.151 (8.36)                 | 1.545 (85.67)             | 1.696 (94.04)             |
| fish  | 10        | 21 Hz      | 0.131 (4.32)                 | 1.373 (45.31)             | 1.504 (49.63)             |


The NLL in nats/frame is normalized to kbits/s computed as  (`nats × log2(e) × frame_rate`). This accounts for differences in frame rate and number of codebooks making it more comparable. qwen3 and fish land within ~13% of each other despite a 2× gap in nats/frame. miso is the outlier at ~3.5× the bitrate, spread over 32 codebooks.