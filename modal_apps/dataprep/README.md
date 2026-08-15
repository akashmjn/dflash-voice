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
modal run modal_apps/dataprep/cbox-expresso-shards.py --rows 10   # test run
modal run modal_apps/dataprep/cbox-expresso-shards.py --rows 0    # full dataset
modal run modal_apps/dataprep/cbox-expresso-shards.py --no-sync
```

**Emilia (1 tar, ~68h)** — the glob picks language and subset; `--rows` picks
the size. One YODAS EN tar is ~68h across ~25k utterances, so the default run is
~85min of GPU and ~110 GiB of shards. `amphion/Emilia-Dataset` is gated, so the
`hf-modal-0814` secret's token needs access to it.

```bash
modal run modal_apps/dataprep/cbox-emilia-shards.py --rows 10     # smoke test
modal run modal_apps/dataprep/cbox-emilia-shards.py --rows 3800   # ~10h
modal run modal_apps/dataprep/cbox-emilia-shards.py               # the full tar
```

**Size the run before launching it** — see [SKILL_SHARDSIZING.md](SKILL_SHARDSIZING.md).
It verifies a tar pattern against the hub (a five-digit bracket silently matches
nothing) and measures hours/speakers, so `--rows` is a number rather than a guess.

Environment comes from this repo's `pyproject.toml` via the `dataprep-chatterbox` extra.  
`dataprep/` is mounted rather than baked in — editing it re-runs without an image rebuild, while  
changing `pyproject.toml` triggers one.

Notes:
- **Ephemeral disk.** Shards are written to container storage and synced only after `prepare`
  finishes, so peak disk is the whole set. With `--include-logits` that is ~1.62 GiB per hour of
  audio (measured: the 30h Expresso run produced ~49 GiB), against Modal's **512 GiB default** — so
  one 68h tar uses ~21%, and ~300h is where it starts to bind. Overrunning it rejects writes as an
  `OSError` mid-run. See [SKILL_SHARDSIZING.md](SKILL_SHARDSIZING.md) step 3 before scaling past that.
- **An empty** `val/` **split.** `assign_split` hashes the speaker's source recording, so a smoke run
  over few speakers can put all of them on one side. The Emilia app warns when `val` comes out empty;
  at ~150 speakers / 92 sources (a ~10h YODAS slice) it does not, landing near 0.957 against a 0.95
  target — speaker-level hashing trades an exact ratio for no voice leaking into val.
- **Rows in** `failures.jsonl`**.** `prepare` skips bad utterances rather than aborting. A systematic
  decode bug shows up as *every* utterance failing, not a handful, so a small nonzero count is healthy.
- **Shard sets cannot be appended to.** Resume is keyed on the string `seq_id`; a rerun needs a new
  `--slug` (or `--force`).


### Throughput

Benchmark for single-process, single-GPU run on Expresso dataset (30h), from
`cbox-expresso-shards.py`. Measured before the Segment refactor; the Emilia
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
