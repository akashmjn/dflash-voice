# modal_apps/

Modal entrypoints that run this repo's modules on cloud GPUs. Scripts only — the
logic lives in `dataprep/`, `train/`, etc., and is mounted at run time.

```text
modal_apps/
├── dataprep/  shard generation, one app per dataset
├── eval/      (tbd)
└── train/     (tbd)
```

## Setup

```bash
modal secret create hf-modal-0814 HF_TOKEN=hf_...
```

The token needs **write** scope — a read-scoped one authenticates fine and then
fails at the bucket sync, at the end of the run.

## Dataprep: Creating processed WebDataset shards

Streams an HF dataset from the hub, runs `dataprep.cli prepare` on a GPU, and syncs the
WebDataset shards to the
[dflash-voice-dataprep-081426](https://huggingface.co/buckets/akashmjn/dflash-voice-dataprep-081426) HF storage bucket.  

This runs `tokenize` to create train-ready (text, audio) token sequences and `featurize` to dump teacher-forced (hiddens, logits) for experimentation. 

One app per dataset; both run the same pipeline with `--model chatterbox`.

**Expresso (~30h)** — `--rows` caps utterances, not source rows, since the
loader cuts each multi-turn row into one segment per turn.

```bash
modal run modal_apps/dataprep/cbox-expresso30h-shards.py --rows 10   # test run
modal run modal_apps/dataprep/cbox-expresso30h-shards.py --rows 0    # full dataset
modal run modal_apps/dataprep/cbox-expresso30h-shards.py --no-sync
```

**Emilia (~10h)** — sized by tar glob rather than `--rows`: one EN tar is ~1.7h
across ~37 speakers, so six tars is ~10h with enough speakers for a meaningful
val split. `amphion/Emilia-Dataset` is gated, so the `hf-modal-0814` secret's
token needs access to it.

```bash
modal run modal_apps/dataprep/cbox-emilia10h-shards.py --rows 10     # smoke test
modal run modal_apps/dataprep/cbox-emilia10h-shards.py               # the full ~10h
modal run modal_apps/dataprep/cbox-emilia10h-shards.py --data-files 'Emilia/EN/EN-B0000[0-5]*.tar'
```

Tar names carry six digits (`EN-B000000`), so the last-digit bracket above is
6 tars and `EN-B0000[0-5]*.tar` is 60 (~100h).

Environment comes from this repo's `pyproject.toml` via the `dataprep-chatterbox` extra.  
`dataprep/` is mounted rather than baked in — editing it re-runs without an image rebuild, while  
changing `pyproject.toml` triggers one.

Notes:
- Scalability: Writes to local container ephemeral storage, and syncs after completion. Careful with >100h data.
  With `--include-logits` the shards are ~16x larger — the 30h Expresso run produced ~50GB, so budget
  roughly 1.7GB per hour of audio and check it against the container's disk before scaling up.
- **An empty** `val/` **split.** `assign_split` hashes the speaker's source recording, so a smoke run
  over few speakers can put all of them on one side. The Emilia app warns when `val` comes out empty;
  at ~222 speakers (6 tars) it does not, landing near 0.96 against a 0.95 target — speaker-level
  hashing trades an exact ratio for no voice leaking into val.
- **Rows in** `failures.jsonl`**.** `prepare` skips bad utterances rather than aborting. A systematic
  decode bug shows up as *every* utterance failing, not a handful, so a small nonzero count is healthy.
- **Shard sets cannot be appended to.** Resume is keyed on the string `seq_id`; a rerun needs a new
  `--slug` (or `--force`).


### Throughput

Benchmark for single-process, single-GPU run on Expresso dataset (30h), from
`cbox-expresso30h-shards.py`. Measured before the Segment refactor; the Emilia
run has no numbers yet.

```yaml
rows : None  
slug : expresso-cbox-full  
gpu : Tesla T4  
sequences : 14984  
frames : 2689572  
train_frames : 2534779  
val_frames : 154793  
weights_s : 15.3  
prepare_s : 4283.9  
sync_s : 54.8  
shard_mb : 49709.2  
frames_per_s : 627.8
```

Processed at ~25x realtime in ~1.2 hours. Total ~2.7M tokens.
