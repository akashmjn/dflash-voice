# Dataprep

Turns conversational speech datasets into per-model token sequences and
teacher-forced features — the shared input for downstream work like fine-tuning
and entropy analysis. Source audio and transcripts stay model-independent; each
model gets its own artifact directory per dataset row.

The tokenize / featurize stages and the records below are dataset-agnostic; a
per-dataset loader is the only dataset-specific piece. Expresso
(`expresso.py`) is the one wired up today.

## Pipeline

```text
raw (audio + transcript)  ──tokenize──▶  tokenized  ──featurize──▶  featurized
```

Run a model through both stages from the repo root:

```bash
python -m dataprep.prepare --model miso --stage all   # or --stage tokenize / featurize
```

- **tokenize** — codec + text into a flat `(L, C+1)` token/mask sequence, with
  spans marking text / audio / special regions.
- **featurize** — replay each sequence under teacher forcing to capture the
  model's per-step logits and hidden states.

Per-model tokenization + featurization backends live in `miso.py` /
`qwen3.py` / `fish.py`; dataset loading lives in a loader like `expresso.py`.

## Records

The pipeline passes small, self-describing dataclasses instead of raw tensors.
All are defined in `common.py` — see it for exact fields and on-disk format.

| Record | What it holds |
| --- | --- |
| `Segment` | One speaker turn — transcript metadata only. |
| `SpanKind`, `TokenSequenceSpan` | Region kind (`text` / `audio` / `special`) and its `[start, end)` range on a sequence. |
| `TokenizedSequenceLayout` | Constant-per-model geometry: channel map (`num_codebooks`, text channel), which token column each head is scored against, and the featurized hidden/logit widths. |
| `TokenizedSequence` | Model-ready `(L, C+1)` tokens/mask, plus its layout and spans. |
| `FeaturizedSequence` | Teacher-forced `{logits, hiddens}` of length `L-1`; index `i` predicts `tokens[i+1]`. |

Consumers branch on spans rather than model-specific framing rules. For an audio
span `[s, e)`, the predictions live at features `[s-1, e-1)` — use
`FeaturizedSequence.feature_slice_for_targets`.

`common.py` also scores a featurized sequence against its ground-truth codes:
`audio_frame_metrics` returns per-frame entropy and NLL for each codebook (using
`layout.head_targets` to pick each head's target column), and `nll_summary`
reduces that NLL to the `semantic` / `audio` / `total` groups in nats per frame
and kbit/s. `notebooks/frame_metrics.py` drives these across rows.

## On-disk layout

```text
data/DATASET/                   # e.g. data/expresso/
  raw/ROW/
    audio.wav                 # channel-first; transcript times in seconds
    transcript_segments.json
    MODEL_codebooks.pt        # temporary per-model codec dump, (F, C) per channel
  MODEL_tokenized/ROW/
    sequences.pt              # ragged list[{tokens, mask}] shaped (L, C+1)
    metadata.json             # one layout for the row + per-sequence length/spans
  MODEL_featurized/ROW/
    features.pt               # ragged list[{logits, hiddens}] of length L-1
    metadata.json             # same shape; layout also carries hidden/logit dims
    kv_context.pt             # only with --dump-kv
  metrics/ROW/                # written by notebooks/frame_metrics.py
    MODEL_metrics.{npz,json}
```

## Environments

MisoTTS pins Transformers 4.49 while the tested MLX stack pins Transformers
5.6 / `huggingface-hub` 1.5, so `dataprep` and `dataprep-mlx` (and `tts_mlx`)
conflict — install only one extra per environment:

```bash
# Miso stack
uv pip install -e ".[dataprep]"
# For local Miso development, replace the installed package:
uv pip install -e ../MisoTTS

# Qwen3 / Fish MLX stack (replaces the Miso pins above)
uv pip install -e ".[dataprep-mlx]"
```
