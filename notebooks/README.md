# Notebooks

Predictive-entropy analysis of TTS codebooks over featurized Expresso rows.

## Reproduce

Run from the repo root. Each step feeds the next:

**1. Dataprep** — tokenize + featurize a model's rows into `data/expresso/MODEL_featurized/ROW/`:

```bash
python -m dataprep.prepare --model miso --stage all
```

**2. Compute entropy** — read the featurized rows and write
`data/expresso/entropy/ROW/MODEL_entropy.{npz,json}` (raw per-frame entropy):

```bash
python notebooks/entropy_utils.py --model miso
```

**3. Explore** — open the marimo notebook and pick a model / row:

```bash
marimo edit notebooks/entropy_explore.py
```

## What the notebook shows

- **Global row view** — per-codebook mean entropy, and the semantic / audio-half
  group time series over the whole file (x-axis in seconds).
- **Per-segment explorer** — the same two views for one selected sequence, with
  the time series x-axis in frame index (seconds available on hover / in the header).

Charts hold raw per-frame values; smoothing is done in the chart (Altair loess).

![Global row view — miso](screenshots/global_miso.png)
