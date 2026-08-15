"""Run `dataprep.cli prepare` over ~10h of Emilia on a Modal GPU, sync to HF bucket.

A scale test for the Emilia loader before committing to the ~100h run: same
pipeline as the Expresso app, different source. Timings are reported per phase
(weights / prepare / sync).

Size comes from the tar glob, not `--rows`: one EN tar is ~1.7h across ~37
speakers, so six tars is roughly 10h with enough speakers for the train/val
split to mean something. `--rows` stays available to cut a run short.

`amphion/Emilia-Dataset` is gated -- the `hf-modal-0814` secret must carry an
HF_TOKEN with access, or `load_dataset` fails on auth.

    modal run modal_apps/dataprep/cbox-emilia10h-shards.py --rows 10   # smoke test
    modal run modal_apps/dataprep/cbox-emilia10h-shards.py             # the full ~10h
    modal run modal_apps/dataprep/cbox-emilia10h-shards.py --no-sync
"""

from __future__ import annotations

import json
import pathlib
import time

import modal


# ==============================

REPO_ROOT = pathlib.Path(__file__).parent.parent.parent

BUCKET = "akashmjn/dflash-voice-dataprep-081426"
DEFAULT_GPU = "T4:1"
MINUTES = 90

MODEL = "chatterbox"
DATASET = "emilia"
DEFAULT_SLUG = "emilia10h-cbox"

#: Six EN tars at ~1.7h each. Tar names carry six digits (EN-B000000), so the
#: bracket sits in the last one; `EN-B0000[0-5]*.tar` is 60 tars (~100h).
DATA_FILES = "Emilia/EN/EN-B00000[0-5].tar"

REMOTE_ROOT = "/root/repo"
DATA_ROOT = f"{REMOTE_ROOT}/data"
HF_CACHE = "/root/.cache/huggingface"
HF_SECRET_NAME = "hf-modal-0814"

# ==============================


# torch brings its own CUDA runtime as wheels, so no devel base image is needed.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1")
    .add_local_file(REPO_ROOT / "pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml", copy=True)
    .uv_pip_install(f"{REMOTE_ROOT}[dataprep-chatterbox]")
    .env({"PYTHONPATH": REMOTE_ROOT, "HF_XET_HIGH_PERFORMANCE": "1"})
    # dataprep/ is mounted, so edits skip the rebuild. add_local_* must come last.
    .add_local_dir(REPO_ROOT / "dataprep", f"{REMOTE_ROOT}/dataprep")
)

app = modal.App("cbox-emilia10h-shards-0815")

# ~2GB of Chatterbox weights; only the first run pays the download.
hf_cache = modal.Volume.from_name("cbox-hf-cache-0814", create_if_missing=True)


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={HF_CACHE: hf_cache},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=4 * 60 * MINUTES,
)
def prepare_shards(
    rows: int = 0,
    data_files: str = DATA_FILES,
    slug: str | None = None,
    sync: bool = True,
) -> dict:
    import subprocess
    import sys

    # rows <= 0 drops --rows, so the glob alone decides the size.
    limit = rows if rows > 0 else None
    resolved_slug = slug or (
        f"{DEFAULT_SLUG}-rows{limit}" if limit else f"{DEFAULT_SLUG}-full"
    )
    shard_dir = pathlib.Path(DATA_ROOT) / "sharded_wds" / resolved_slug

    print("=== torch / device check", flush=True)
    import torch

    print(f"torch      : {torch.__version__}")
    print(f"cuda avail : {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        # A CPU wheel runs fine, just ~50x slower, so the benchmark would look
        # valid and mean nothing.
        raise RuntimeError("CUDA not available -- refusing to benchmark on CPU")
    print(f"gpu        : {torch.cuda.get_device_name(0)}")

    # Load here so prepare_s below is compute, not a cold-volume download.
    print("\n=== loading weights", flush=True)
    t0 = time.perf_counter()
    from dataprep.pipeline import load_tokenizer

    load_tokenizer(MODEL, device="cuda")
    weights_s = time.perf_counter() - t0
    hf_cache.commit()
    print(f"weights loaded in {weights_s:.1f}s")

    cmd = [
        sys.executable, "-m", "dataprep.cli", "prepare",
        "--model", MODEL,
        "--dataset", DATASET,
        "--data-files", data_files,
        "--slug", resolved_slug,
        "--data-root", DATA_ROOT,
        "--device", "cuda",
        "--include-logits",
        # The default (32) pads sequences to bound the shape count, working
        # around an MPS allocator issue that does not apply on CUDA.
        "--bucket-frames", "0",
    ]
    if limit is not None:
        cmd += ["--rows", str(limit)]

    print(f"\n=== prepare\n{' '.join(cmd)}\n", flush=True)
    t0 = time.perf_counter()
    subprocess.run(cmd, cwd=REMOTE_ROOT, check=True)
    prepare_s = time.perf_counter() - t0

    shard_bytes = sum(p.stat().st_size for p in shard_dir.rglob("*.tar") if p.is_file())

    # One Emilia row is one utterance is one sequence, so sequences tracks
    # utterances directly; frames are tokenized at the codec's rate.
    info = json.loads((shard_dir / "dataset_info.json").read_text(encoding="utf-8"))
    train_frames = info["total_train_frames"]
    val_frames = info["total_val_frames"]
    total_frames = train_frames + val_frames
    total_sequences = info["total_train_sequences"] + info["total_val_sequences"]

    failures = pathlib.Path(DATA_ROOT) / "dataprep_logs" / "failures.jsonl"
    failure_count = (
        sum(1 for line in failures.read_text(encoding="utf-8").splitlines() if line.strip())
        if failures.exists()
        else 0
    )

    sync_s = 0.0
    if sync:
        from huggingface_hub import sync_bucket

        destination = f"hf://buckets/{BUCKET}/{resolved_slug}"
        print(f"\n=== sync -> {destination}", flush=True)
        t0 = time.perf_counter()
        sync_bucket(str(shard_dir), destination)
        sync_s = time.perf_counter() - t0

    stats = {
        "rows": limit,
        "data_files": data_files,
        "slug": resolved_slug,
        "gpu": torch.cuda.get_device_name(0),
        "sequences": total_sequences,
        "frames": total_frames,
        "train_frames": train_frames,
        "val_frames": val_frames,
        "failures": failure_count,
        "weights_s": round(weights_s, 1),
        "prepare_s": round(prepare_s, 1),
        "sync_s": round(sync_s, 1),
        "shard_mb": round(shard_bytes / 1e6, 1),
        "frames_per_s": round(total_frames / prepare_s, 1) if prepare_s else None,
    }

    print("\n=== timings")
    for key, value in stats.items():
        print(f"{key:14s}: {value}")
    if not val_frames:
        print("\nWARNING: val split is empty -- too few distinct speakers")
    if failure_count:
        print(f"WARNING: {failure_count} utterances failed, see {failures}")
    return stats


@app.local_entrypoint()
def main(
    rows: int = 0,  # 0 or less lets the tar glob decide the size
    data_files: str = DATA_FILES,
    slug: str | None = None,
    sync: bool = True,
):
    stats = prepare_shards.remote(
        rows=rows, data_files=data_files, slug=slug, sync=sync
    )
    if sync:
        print(f"\nhttps://huggingface.co/buckets/{BUCKET}")
    return stats
