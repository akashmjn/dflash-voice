---
name: size-emilia-run
description: Size an Emilia dataprep run before launching it on a Modal GPU — verify a tar glob matches real files, measure hours/speakers in those tars, and convert to a --rows count. Use when picking --data-files or --rows for modal_apps/dataprep, when a run's size or runtime is unclear, or when a glob appears to match nothing.
---

# Sizing an Emilia run

A tar glob picks *language and subset*, not size. One EN tar is ~68h across ~25k
utterances, so the smallest non-empty glob already overshoots a 10h test by ~7x.
Runs are sized with `--rows`; this is how you pick the number.

Catches the two glob mistakes that otherwise cost a GPU hour: one that silently
matches nothing, and one that matches far more than intended.

## Setup

`amphion/Emilia-Dataset` is gated: without an `HF_TOKEN` that has access,
`load_dataset` fails on auth rather than on a missing file. Load it from `.env`
rather than pasting it anywhere:

```bash
set -a && source .env && set +a
```

Step 2 imports `dataprep.shards`, so install the repo and run from its root:

```bash
uv pip install -e ".[dataprep-chatterbox]"
```

Reuse whichever dataprep venv already exists — the extras are mutually exclusive
(see CLAUDE.md), and either works here. Nothing below loads a backend or needs a
GPU.

## Step 1 — does the pattern match real files?

Seconds, no download. Ask the hub for the file list and filter it:

```bash
python -c "
import fnmatch, os
from huggingface_hub import HfApi
pattern = 'Emilia-YODAS/EN/EN-B00000[0-5].tar'
files = HfApi().list_repo_files('amphion/Emilia-Dataset', repo_type='dataset', token=os.environ['HF_TOKEN'])
matched = sorted(f for f in files if fnmatch.fnmatch(f, pattern))
print(f'{len(matched)} matched')
print(*matched[:10], sep='\n')
"
```

**Zero matches is the common outcome, and it is a typo, not an empty subset.**
Tar names carry **six** digits — `EN-B000000.tar` — so a five-digit bracket like
`EN-B0000[0-5].tar` matches nothing. When the count is 0, print the real
neighbours and compare digit widths:

```python
print(*sorted(f for f in files if f.startswith('Emilia-YODAS/EN/'))[:5], sep='\n')
```

Known counts: 1140 tars in `Emilia/EN`, 1362 in `Emilia-YODAS/EN`.

`load_dataset(..., streaming=True).n_shards` reports the same count once the
stream is open, and is a good assertion inside a run.

## Step 2 — what is actually in them?

Streams metadata only. `--limit`-style capping matters: reading one tar to the
end takes a few minutes, and a few thousand rows already projects well.

```bash
python -c "
import collections, itertools, os
from datasets import load_dataset
from dataprep.shards import split_key

pattern, limit = 'Emilia-YODAS/EN/EN-B000000.tar', 4000
stream = load_dataset('amphion/Emilia-Dataset', data_files={'train': pattern},
                      split='train', streaming=True, token=os.environ['HF_TOKEN'])
n_shards = stream.n_shards
# Clearing the schema avoids the inferred Audio feature, which would pull in
# torchcodec. See dataprep.datasources.emilia._undecoded_audio.
stream._info.features = None

rows = stream if limit <= 0 else itertools.islice(stream, limit)
count = seconds = 0
speakers, sources = collections.Counter(), collections.Counter()
for row in rows:
    meta = row['json']
    count += 1
    seconds += float(meta.get('duration', 0.0))
    speakers[str(meta['speaker'])] += 1
    sources[split_key(str(meta['speaker']))] += 1

hours = seconds / 3600
print(f'shards {n_shards}  utterances {count}  hours {hours:.2f}')
print(f'speakers {len(speakers)}  sources {len(sources)}')
print(f'for ~10h: --rows {int(count / hours * 10)}')
"
```

Judge the train/val split on `sources`, not `speakers`: `split_key` strips a
trailing `_SPEAKER_\d+`, so YODAS's diarized `EN_tKvmUvxYZXI_SPEAKER_00/01`
collapse into one bucket and never straddle the split.

## Measured figures

Both EN subsets, one full tar read to the end:

| subset | hours/tar | utterances/tar | speakers | sources |
|---|---|---|---|---|
| `Emilia/EN` | 69.4 | 24932 | 1000 | 1000 |
| `Emilia-YODAS/EN` | 68.1 | 25482 | 820 | 469 |

So ~10h is **~3600 rows** (`Emilia/EN`) or **~3800 rows** (`Emilia-YODAS/EN`).
At that size YODAS gives ~150 speakers over 92 sources, and the split lands at
0.957 against a 0.95 target.

## Step 3 — turn hours into wall-clock and disk

Chatterbox dataprep on a T4 runs at **~48x realtime**. With `--include-logits`,
shards cost **~1.62 GiB per hour** of audio — measured, not estimated: the 30h
Expresso run produced 49709 MB (~18.5 kB/frame).

| audio | GPU time | shards | of the 512 GiB quota |
|---|---|---|---|
| 10h | ~13 min | ~16 GiB | 3% |
| 30h (Expresso, measured) | ~38 min | ~49 GiB | 9% |
| 68h (1 YODAS tar) | ~85 min | ~110 GiB | 21% |
| 100h | ~2h 5min | ~162 GiB | 32% |
| 200h | ~4h 10min | ~324 GiB | 63% |

**A container gets 512 GiB of ephemeral disk by default**, so nothing under
~300h needs anything done about it. Two details that keep it that roomy: Emilia
is streamed rather than downloaded, and the HF cache is a Volume, so weights and
tars never land on container disk. Only `--data-root` does.

Peak disk is the *whole* shard set, though — `sync_bucket` runs after `prepare`
finishes, so nothing drains mid-run. Overrunning the quota rejects writes as an
`OSError` partway through, after the GPU time is already spent.

Past ~300h, raise it with `ephemeral_disk=` (max 3.0 TiB) — but note disk is
billed by inflating the memory request 20:1, so asking for 512 GiB forces a
~25 GiB memory request this workload has no use for. Leave it unset below that.

## Checklist before `modal run`

- [ ] Step 1 returned the tar count you expected — not 0, not 60.
- [ ] `--rows` is set from step 2, not guessed.
- [ ] Projected GPU time fits the app's `timeout`.
- [ ] Projected shard size fits the 512 GiB ephemeral disk (~300h of audio).
- [ ] The slug is new — shard sets cannot be appended to (resume is keyed on
      `seq_id`); a rerun needs a fresh `--slug` or `--force`.
