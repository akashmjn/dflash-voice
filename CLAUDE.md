# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research repo exploring speculative decoding / block-diffusion for parallel RVQ audio token generation in TTS. Structured as core modules (dataprep, mlx_decode) plus a growing list of investigations documented under `experiments/`.

Pipeline: **dataprep** (tokenize + teacher-forced featurize) → **experiments** (NLL/entropy analysis, MLX profiling) → **mlx_decode** (ported inference for benchmarking) → **train** (depth-decoder training, `dev` branch only).

## Project layout

```text
dflash-voice/
├── dataprep/              tokenize + featurize pipeline
│   ├── common.py            shared dataclasses (Segment, TokenizedSequence, FeaturizedSequence, ...)
│   ├── cli.py                CLI entrypoint (python -m dataprep.cli <prepare|inspect>)
│   ├── pipeline.py            tokenize + featurize compute stages
│   ├── shards.py              streams sequences into WebDataset shards
│   ├── expresso.py            dataset loader (only one wired up)
│   ├── miso.py                the maintained tokenize/featurize backend
│   ├── mlx_backends/           DEPRECATED qwen3.py / fish.py (MLX-only, unverified)
│   └── tests/
├── mlx_decode/            ported MLX inference loops per model, for benchmarking
│   ├── bench.py               benchmark CLI entrypoint
│   ├── miso.py / qwen3.py / fish.py
│   └── tests/
├── experiments/           analysis writeups, each with its own README
│   ├── expresso_nll_entropy/    NLL/entropy analysis (marimo notebooks)
│   └── mlx_decode_breakdown/    decode-time breakdown writeup
├── demo/                  two-speaker podcast demo (mlx-audio APIs directly)
├── train/                 depth-decoder training — dev branch only, empty on main
├── agent-workspace/       gitignored scratch space for agents (see Agent workflow)
├── data/                  gitignored dataprep input/output artifacts (see below)
└── tmp/                   gitignored scratch outputs
```

Data flows left to right: `dataprep/` (raw → tokenized → featurized, under `data/`) feeds `experiments/` and `train/`; `mlx_decode/` is a separate inference/benchmarking track that `demo/` and `experiments/mlx_decode_breakdown/` build on.

### data/ layout

Populated by `dataprep.cli`: `inspect` writes the per-row raw/tokenized/featurized trees, `prepare` streams straight to `sharded_wds/`. Rows are indexed by integer position in the source dataset; each model gets its own subtree under `MODEL/`.

```text
data/
├── dataprep_logs/
│   └── failures.jsonl              # per-row failures across prepare runs
├── expresso/                       # DATASET, defaults to dataprep.expresso.DATASET_NAME
│   ├── raw/ROW/                    # model-independent source, e.g. raw/0/
│   │   ├── audio.wav                 channel-first; transcript times in seconds
│   │   ├── transcript_segments.json
│   │   └── MODEL_codebooks.pt        temporary per-model codec dump, (F, C) per channel
│   ├── tokenized/MODEL/ROW/
│   │   ├── sequences.pt             ragged list[{tokens, mask}] shaped (L, C+1)
│   │   └── metadata.json             layout for the row + per-sequence length/spans
│   ├── featurized/MODEL/ROW/
│   │   ├── features.pt              ragged list[{logits, hiddens}] of length L-1
│   │   ├── metadata.json              same shape; layout also carries hidden/logit dims
│   │   └── kv_context.pt             only with --dump-kv
│   └── metrics/MODEL/ROW/           written by experiments/expresso_nll_entropy/model_metrics.py
│       ├── MODEL_metrics.npz
│       └── MODEL_metrics.json
└── sharded_wds/SLUG/               written by dataprep/shards.py, e.g. sharded_wds/expresso-rows60/
    ├── dataset_info.json
    ├── shards.json
    ├── train/MODEL_train_NNNNN.tar
    └── val/MODEL_val_NNNNN.tar
```

`MODEL` directories in practice include per-size variants (e.g. `qwen3-0.6b`, `qwen3-1.7b`) alongside the bare `qwen3`/`miso`/`fish` names.

## Environments (mutually exclusive extras)

`dataprep-miso` pins Transformers 4.49 (via MisoTTS); `dataprep-mlx`/`mlx_decode` pin Transformers 5.6 + huggingface-hub 1.5. Only install one extra per venv.

```bash
uv pip install -e ".[dataprep-mlx]"    # deprecated Qwen3 / Fish dataprep + MLX inference
uv pip install -e ".[dataprep-miso]"   # Miso dataprep (needs MisoTTS package)
uv pip install -e ".[mlx_decode]"      # MLX inference/benchmarking only
uv pip install -e ".[dev]"             # pytest, modal
```

Requires Apple Silicon (MLX) for anything touching `mlx_decode` or the MLX dataprep backends.

## Commands

```bash
# dataprep: stream Expresso into WebDataset shards, from repo root
# (--model qwen3|fish also parse but are deprecated; see dataprep/mlx_backends/README.md)
python -m dataprep.cli prepare --model miso [--rows N] [--slug NAME]
# per-row intermediates on disk instead, for inspection (a few rows only)
python -m dataprep.cli inspect --model miso --rows 3 [--stage <tokenize|featurize>]

# MLX inference benchmark
python mlx_decode/bench.py --model <qwen3|fish|miso>

# tests — `-m 'not expensive and not deprecated'` is the pytest default
# (skips full-model-loading tests and the deprecated qwen3/fish backends)
pytest -v mlx_decode/tests/test_decode_parity.py
pytest -v dataprep/tests/
pytest -v -m expensive dataprep/tests/test_miso_entropy.py
pytest -v -m deprecated dataprep/tests/       # qwen3/fish MLX backends

# demo: two-speaker podcast render
python demo/demo_tts_podcast.py render --model miso --max-segments 6

# experiment repro, from experiments/expresso_nll_entropy/
python model_metrics.py compute --model miso
python model_metrics.py summarize --rows 10
marimo edit experiments/expresso_nll_entropy/metrics_explore.py
```

## Architecture

**dataprep/** — turns speech datasets into per-model token sequences and teacher-forced features: `raw (audio + transcript) --tokenize--> tokenized --featurize--> featurized --shard--> wds shards`. `cli.py` is the entrypoint, `pipeline.py` the compute stages, `shards.py` the WebDataset writers — `prepare` streams rows straight into shards (deterministic sample order, flat memory), `inspect` writes per-row intermediates for a handful of rows. `miso` is the only maintained backend; `qwen3`/`fish` are deprecated under `dataprep/mlx_backends/`
(MLX-only, unverified, warn on use, skipped by default in tests) and should not constrain pipeline
changes. `prepare` is one pull-driven chain, `HF dataset -> DecodedExample stream -> shard_prepare -> sample stream`: `shards.shard_prepare` drives `pipeline.stream_prepared_samples` from inside its write loop, so the forward pass runs only for rows actually written and `skip_rows` can drop a row cheaply. Train/val is assigned by hashing the row id so a row's sequences never straddle the split. `common.py` holds the shared dataclasses passed between stages instead of raw tensors:
- `Segment` — one speaker turn, metadata only.
- `TokenizedSequence` — model-ready `(L, C+1)` tokens/mask + `TokenizedSequenceLayout` (per-model geometry) + spans (`SpanKind`: text/audio/special).
- `FeaturizedSequence` — teacher-forced `{logits, hiddens}`, length `L-1`; index `i` predicts `tokens[i+1]`. For audio span `[s, e)`, predictions live at `[s-1, e-1)` — use `feature_slice_for_targets` rather than reimplementing the offset.
- `ShardSample` — one sequence serialized to `.npy` bytes for a WebDataset shard.
- `audio_frame_metrics`/`nll_summary` score a featurized sequence into `semantic`/`audio`/`total` NLL (nats/frame, kbit/s).

**mlx_decode/** — vendored/ported MLX TTS inference loop (from `mlx-audio` 0.4.4) as single-file modules per model, reusing its weights/`nn.Module`s but reimplementing prompt construction, autoregression, and codec decode so timing breaks down per step.

**experiments/** — self-contained writeups, each with its own README, depending on dataprep output under `data/`. See repo README for a summary of the investigations.

**demo/** — `demo_tts_podcast.py` uses mlx-audio's native APIs directly (not `mlx_decode`). Only `miso` supports cross-turn context; `--model fish` is not wired up (no `Segment`/context equivalent yet).

**train/** (dev branch only) — depth-decoder training (`cli.py`, `dataset.py`, `model.py`, `convert.py`; `MisoRVQDepthDecoder` + WebDataset dataloader + NLL eval). Removed from `main` deliberately (`reorg: train on dev only`); check `git log --oneline main..dev` before assuming it's present locally.

## Agent workflow

- For new investigations, use `agent-workspace/` as scratch space by default. If the work should be preserved, create a new branch and put it under `agent-workspace/` or a directory under `experiments/` instead. `agent-workspace/` is gitignored, but individual files still get force-added (`git add -f`) when something there needs to be committed — don't assume everything under it is untracked.

### Branch policy: dev/main

`dev` is the working branch; `main` is the public-facing subset. They are **not kept in sync by merging** — `main` only ever receives content via explicit path promotion, never `git merge dev`. This is deliberate: some paths (e.g. `train/`) are meant to stay dev-only until they're cleaned up for release, and a plain merge would eventually resurrect them on `main` once anything else in that path changes on `dev`.

- **Feature/investigation work**: branch off `dev`, do the work, squash-merge the branch back into `dev` as one clean commit. This is the `claude/dev-feature` skill.
- **Promoting to `main`**: never merge `dev` into `main`. Instead, checkout only the specific paths that are ready for `main` from `dev` (an explicit allowlist), so dev-only paths like `train/` are never touched. This is the `claude/release` skill.
- **Direct edits to `main`**: should be rare (trivial fixes only — typos, small doc corrections). If one happens, cherry-pick that single commit onto `dev` immediately, in the same session, so it isn't lost on the next promotion. If the "minor tweak" turns out to be more than trivial, redo it properly on `dev` instead.

## Notes

- `data/`, `tmp/`, `agent-workspace/`, `demo/output/` are gitignored — artifacts there won't show up in `git status`.
- Modal (`MODAL_TOK_KEY`/`MODAL_TOK_SECRET` in `.env`) is used for remote execution — see `agent-workspace/modal_hello.py`.
