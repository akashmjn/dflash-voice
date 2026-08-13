## Turning speech datasets into teacher-forced training data

Tokenizes conversational speech into per-model token sequences, replays them under teacher forcing to
capture the model's logits/hiddens, and packs the result into WebDataset shards that `train/` reads.
Audio and transcripts stay model-independent; each model gets its own artifacts.

```text
raw (audio + transcript) ──tokenize──▶ tokenized ──featurize──▶ featurized ──shard──▶ wds shards
```

```bash
uv pip install -e ".[dataprep-mlx]"           # or ".[dataprep-miso]" — see Environments
python -m dataprep.cli prepare --model miso   # whole dataset, straight to shards
```

`cli.py` has two verbs; compute stages live in `pipeline.py`, sharding in `shards.py`.


| Verb      | What it does                                                                              |
| --------- | ----------------------------------------------------------------------------------------- |
| `prepare` | Streams the dataset into WebDataset shards. Nothing per-row hits disk, memory stays flat. |
| `inspect` | Writes every intermediate per row so you can open a `sequences.pt`. A few rows only.      |


```bash
python -m dataprep.cli prepare --model miso --rows 60 --slug expresso-rows60
python -m dataprep.cli inspect --model miso --rows 3 --stage tokenize
```

Shards land in `data/sharded_wds/SLUG/`, named per run (`--slug`, default `DATASET-rowsN`) since a
shard set is defined by the run that produced it. Overwriting one needs `--force`. Failed rows are
logged to `failures.jsonl` and skipped, so a multi-hour run does not die on one bad row.

The maintained tokenize/featurize backend is `miso.py`; dataset loading is in a loader like
`expresso.py` (the one wired up today). Everything else is dataset-agnostic.

`qwen3` and `fish` live in [`mlx_backends/`](./mlx_backends/README.md) and are **deprecated,
MLX-only, and unverified** — kept only so `experiments/expresso_nll_entropy/` stays reproducible.
They warn on use, are skipped by the default test run, and are not expected to survive pipeline
changes.

### Notes on the design

`prepare` is one pull-driven chain:

```text
HF dataset ──▶ raw example stream ──▶ shard_prepare ──▶ sample stream ──▶ tars
```

Only lightweight `DecodedExample` records stream off the hub; `shard_prepare` drives tokenize and
featurize from inside its write loop, so the forward pass runs only for rows it writes. That is what
lets a row filter — `skip_rows`, which a resumed run needs — drop a row for the cost of one loader
read instead of a full forward pass.

Rows stream in dataset order, and samples are written in the order they arrive. `IterableDataset.shuffle`
would prefetch from several of the 36 audio files at once and stall the first row for minutes; in-order
streaming starts in ~10s. Writing in order is also what keeps a run resumable — where a sample lands
depends only on `(row, seq_id)`, never on buffer state — so mixing is left to the training dataloader,
which shuffles both shard order and a sample window. `--shuffle-buffer` trades that away for a
write-time reservoir; it defaults to off.

Train/val is assigned by hashing the row id, so every sequence from a row lands on the same side —
one speaker's turns in a recording cannot straddle the split, and the assignment survives shuffling
and restarts. Over very few rows this can put everything on one side; `--split-ratio` only tracks the
requested fraction once there are enough rows.

`--include-logits` adds the teacher's per-head distributions as float16 `logits.npy`. Off by default
at ~16x the size of the hiddens (2.5 GB → ~46 GB for the 60-row set), and only worth it when
distilling against the full distribution rather than ground-truth codes. float16 is exact here:
verified bit-identical across all 464 sequences of the 10 analysis rows.

## Records

Stages pass small self-describing dataclasses, all defined in `common.py` — see it for exact fields
and on-disk format.


| Record                          | What it holds                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------- |
| `Segment`                       | One speaker turn — transcript metadata only.                                                       |
| `TokenSpanKind`, `TokenSequenceSpan` | Region kind (`text` / `audio` / `special`) and its `[start, end)` range.                           |
| `TokenizedSequenceLayout`       | Per-model geometry: channel map, which token column each head scores against, hidden/logit widths. |
| `TokenizedSequence`             | Model-ready `(L, C+1)` tokens, plus layout and spans. Spans are the only record of which columns are live.       |
| `FeaturizedSequence`            | Teacher-forced `{logits, hiddens}` of length `L-1`; index `i` predicts `tokens[i+1]`.              |
| `ShardSample`                   | One sequence serialized to `.npy` bytes, ready for a WebDataset shard.                             |


Consumers branch on spans rather than model-specific framing rules. For an audio span `[s, e)` the
predictions live at features `[s-1, e-1)` — use `FeaturizedSequence.feature_slice_for_targets` rather
than reimplementing the offset. Samples also carry `head_targets`, since scoring a head means pairing
it with the right token column (Fish head 0 targets the semantic token, not the audio code).

`common.py` also scores a featurized sequence: `audio_frame_metrics` gives per-frame entropy and NLL
per codebook, and `nll_summary` reduces it to `semantic` / `audio` / `total` in nats per frame and
kbit/s.

## On-disk layout

```text
data/DATASET/                 # e.g. data/expresso/ — override with --dataset
  raw/ROW/                    # audio.wav (channel-first), transcript_segments.json,
                              # MODEL_codebooks.pt (temporary per-model codec dump)
  tokenized/MODEL/ROW/        # sequences.pt — ragged list[{tokens}] of (L, C+1)
                              # metadata.json — row layout + per-sequence length/spans
  featurized/MODEL/ROW/       # features.pt — ragged list[{logits, hiddens}] of length L-1
                              # metadata.json, kv_context.pt (only with --dump-kv)
  metrics/MODEL/ROW/          # MODEL_metrics.{npz,json}, written by the analysis notebooks

data/sharded_wds/SLUG/        # dataset_info.json (dims, frame rate, split totals)
                              # shards.json (per-shard sample/frame counts)
                              # train/MODEL_train_00000.tar, val/MODEL_val_00000.tar
```

`data/expresso/` keeps rows 0-9, the set the `metrics/` analysis covers. Shard sets are standalone
artifacts — `expresso-rows60` was built from 60 rows and cannot be rebuilt from what is on disk now.

## Environments

MisoTTS pins Transformers 4.49, the MLX stack pins Transformers 5.6 / `huggingface-hub` 1.5, and
chatterbox-tts pins Transformers 5.2 / torch 2.6 — all three conflict, so install only one extra per
environment:

```bash
uv pip install -e ".[dataprep-mlx]"         # deprecated Qwen3 / Fish backends
uv pip install -e ".[dataprep-miso]"        # Miso (replaces the pins above)
uv pip install -e ".[dataprep-chatterbox]"  # Chatterbox AR + Flash (tokenize only)
uv pip install -e ../MisoTTS                # to use a locally cloned MisoTTS
```

## Chatterbox backend (tokenize only)

Prepares finetuning data for both Chatterbox AR (500M English) and Chatterbox Flash, which share a
tokenizer — Flash subclasses the same `T3` and differs only by an input-only `[MASK]` embedding row.
There is no featurizer yet, so `prepare` and `--stage featurize` are rejected up front:

```bash
python -m dataprep.cli inspect --model chatterbox --rows 3 --stage tokenize
```

Unlike the RVQ backends, Chatterbox is two-stream: one S3 codebook (25 Hz, vocab 6561) plus a
separate text vocabulary, laid out as `[cond | SOT text EOT | SOS y EOS]` with speech in column 0 and
text in column 1. The leading 34 frames are a reserved, unsupervised placeholder for T3's fixed
conditioning prefix (1 speaker + 32 perceiver + 1 emotion), keeping grid positions equal to model
positions. The 256-d speaker embedding is precomputed to `embedding_context.pt`; prompt speech ids
are sliced from the sequence's own audio at collate time.

Segments are encoded one at a time from their own waveform, since S3's encoder is bidirectional.
Segments whose audio runs far longer than their transcript implies are rejected to `failures.jsonl` —
Expresso has at least one 105 s segment labelled "Thank you.".

