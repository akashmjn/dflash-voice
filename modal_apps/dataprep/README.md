# modal_apps/

Modal entrypoints that run this repo's modules on cloud GPUs. Scripts only — the
logic lives in `dataprep/`, `train/`, etc., and is mounted at run time.

```text
modal_apps/
├── data/     dataprep pipelines (shard generation)
├── eval/     (tbd)
└── train/    (tbd)
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

```bash
modal run modal_apps/dataprep/modal_prepare_shards.py --rows 10       # test run ~1hr input
modal run modal_apps/dataprep/modal_prepare_shards.py --rows 0        # full dataset
modal run modal_apps/dataprep/modal_prepare_shards.py --no-sync
```

Environment comes from this repo's `pyproject.toml` via the `dataprep-chatterbox` extra.  
`dataprep/` is mounted rather than baked in — editing it re-runs without an image rebuild, while  
changing `pyproject.toml` triggers one.

Notes:
- Scalability: Writes to local container ephemeral storage, and syncs after completion. Careful with >100h data.
- **An empty** `val/` **split.** `assign_split` hashes each row id independently, so
at 10 rows every row can land in train.
- **Rows in** `failures.jsonl`**.** `prepare` skips bad rows rather than aborting.


### Throughput

Benchmark for single-process, single-GPU run on Expresso dataset (30h).

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
