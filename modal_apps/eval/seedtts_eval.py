"""Zero-shot TTS eval on Seed-TTS test-en, on Modal: Chatterbox TTS (AR).

Follows ``scripts/run_eval.sh`` from resemble-ai/chatterbox-flash broken into stages:

    download   CPU   eval datasets + scorers into the Volume
    generate   GPU   ChatterboxTTS.generate -> one 24kHz wav per row, baseline
    upload     CPU   locally generated wavs into a run dir, in place of generate
    score      GPU   SIM-o (omnivoice) + WER (Whisper-large-v3) + UTMOS
    fetch      local copy a run's scored/ logs into results/SPLIT/TAG/

A Modal Volume is used for input/output storage with the following layout:

    download/            shared inputs
    eval-09-2026/seedtts_en-limit100/TAG/     (-limitN only for subset runs)
    ├── generated/       one 24kHz {id}.wav per row
    ├── scored/          sim.log / wer.log / mos.log
    └── test_list.jsonl  the row list this run used

``score`` is followed by ``fetch``, so scored runs land next to this file under
``results/seedtts_en-limit100/TAG/`` for ``summarize.py``:

    modal run modal_apps/eval/seedtts_eval.py --limit 100 --tag TAG
    modal run modal_apps/eval/seedtts_eval.py --tag full      # 1088 rows
    modal run modal_apps/eval/seedtts_eval.py --stage score --tag TAG --limit 100
    modal run modal_apps/eval/seedtts_eval.py --stage fetch --tag TAG --limit 100

Use `--stage upload` to score predecoded audio (e.g. locally generated via mlx_decode).

    modal run modal_apps/eval/seedtts_eval.py --stage upload --limit 100 \\
        --tag 0930-1800-TAG \\
        --wav-dir agent-workspace/mlx_decode_output/seedtts-100/DECODE_SLUG \\
        --test-list data/seedtts/prompts/seedtts_test_en_100.jsonl
"""

from __future__ import annotations

import pathlib
import time

import modal


# ==============================

DEFAULT_GPU = "L4:1"  # Ada; bf16-capable, unlike the Turing T4 used for dataprep
TIMEOUT_MINUTES = 30
HF_SECRET_NAME = "hf-modal-0814"

EVAL_DATA_REPO = "k2-fsa/TTS_eval_datasets"
EVAL_MODEL_REPO = "k2-fsa/TTS_eval_models"

# Must be named exactly "download": the k2-fsa JSONLs hard-code ref_audio under
# that prefix, resolved against this dir's parent.
MODAL_VOLUME_NAME = "seedtts-eval-09-2026"
VOL_MOUNT_ROOT = "/vol"
DOWNLOAD_DIR = f"{VOL_MOUNT_ROOT}/download"
REF_AUDIO_ROOT = VOL_MOUNT_ROOT
TTS_EVAL_DATA_DIR = f"{DOWNLOAD_DIR}/tts_eval_datasets"
TTS_EVAL_MODEL_DIR = f"{DOWNLOAD_DIR}/tts_eval_models"
# Row count scopes the dataset dir, so subsets never share a parent.
RES_ROOT = f"{VOL_MOUNT_ROOT}/eval-09-2026"
DATASET = "seedtts_en"
# Local mirror of each run's scored/ logs, read by summarize.py.
LOCAL_RESULTS = pathlib.Path(__file__).parent / "results"

TEST_JSONL = f"{TTS_EVAL_DATA_DIR}/seedtts_test_en.jsonl"

# ==============================

# chatterbox-tts hard-pins torch==2.6.0, which the wheels around it have no
# matching ABI for. uv_pip_install ignores upstream's [tool.uv]
# override-dependencies, so those versions are passed explicitly. Not /tmp:
# uv reads this file in the next build step.
UV_OVERRIDES_PATH = "/opt/uv-overrides.txt"
UV_OVERRIDES = [
    "torch>=2.7,<2.8",
    "torchaudio>=2.7,<2.8",
    "torchvision>=0.22,<0.23",
    "transformers>=4.46",
]

# chatterbox-flash is here for its WER scorer, not its TTS model.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("ffmpeg", "libsndfile1", "git", "tar")
    .run_commands(
        # One arg per line: printf '%s' would write the escapes literally.
        "printf '%s\\n' " + " ".join(f"'{o}'" for o in UV_OVERRIDES)
        + f" > {UV_OVERRIDES_PATH}",
        f"cat {UV_OVERRIDES_PATH}",  # visible in the build log
    )
    .uv_pip_install(
        "torch>=2.7,<2.8",
        "torchaudio>=2.7,<2.8",
        "git+https://github.com/resemble-ai/chatterbox-flash.git",
        "omnivoice>=0.1.5",
        "jiwer>=3.0,<4",  # omnivoice uses jiwer.compute_measures, gone in 4.0
        "zhon>=2.0",
        "whisper-normalizer>=0.0.10",
        "s3prl>=0.4",  # omnivoice SIM-o loads WavLM via torch.hub, needs s3prl
        "hf-transfer",
        extra_options=f"--overrides {UV_OVERRIDES_PATH}",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("cbox-ar-seedtts-eval")

# ~7.5GB: scorers + testset. Model weights live in the HF cache.
vol = modal.Volume.from_name(MODAL_VOLUME_NAME, create_if_missing=True)


def _run_paths(tag: str, limit: int = 0) -> dict:
    """Per-run directory layout, keyed by row count then tag."""
    dataset_dir = f"{DATASET}-limit{limit}" if limit > 0 else DATASET
    run_dir = f"{RES_ROOT}/{dataset_dir}/{tag}"
    return {
        "dataset_dir": dataset_dir,
        "run_dir": run_dir,
        "generated": f"{run_dir}/generated",
        "scored": f"{run_dir}/scored",
        "test_list": f"{run_dir}/test_list.jsonl",
    }


@app.function(
    image=image,
    volumes={VOL_MOUNT_ROOT: vol},
    timeout=TIMEOUT_MINUTES * 60,
)
def upload(
    tag: str,
    limit: int,
    wavs: list[tuple[str, bytes]],
    test_list_text: str,
) -> dict:
    """Plant externally generated wavs into a run dir, in place of ``generate``.

    Scoring only needs ``{id}.wav`` under ``generated/`` plus the row list, so
    audio from any decoder can be scored by the same path.
    """
    import os

    paths = _run_paths(tag, limit)
    os.makedirs(paths["generated"], exist_ok=True)
    for name, blob in wavs:
        with open(f"{paths['generated']}/{name}", "wb") as fh:
            fh.write(blob)
    with open(paths["test_list"], "w", encoding="utf-8") as fh:
        fh.write(test_list_text)
    vol.commit()

    rows = sum(1 for line in test_list_text.splitlines() if line.strip())
    print(f"uploaded {len(wavs)} wavs + {rows} rows -> {paths['run_dir']}", flush=True)
    return {"run_dir": paths["run_dir"], "wavs": len(wavs), "rows": rows}


@app.function(
    image=image,
    volumes={VOL_MOUNT_ROOT: vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=TIMEOUT_MINUTES * 60,
)
def download() -> dict:
    """Stage 1: datasets + scorers into the Volume (idempotent)."""
    import os
    import pathlib
    import subprocess

    from huggingface_hub import snapshot_download

    os.makedirs(TTS_EVAL_DATA_DIR, exist_ok=True)
    os.makedirs(TTS_EVAL_MODEL_DIR, exist_ok=True)

    print("=== seedtts test list + testset", flush=True)
    for fname in ("seedtts_test_en.jsonl", "seedtts_testset.tar.gz"):
        if not os.path.exists(f"{TTS_EVAL_DATA_DIR}/{fname}"):
            snapshot_download(
                EVAL_DATA_REPO,
                repo_type="dataset",
                local_dir=TTS_EVAL_DATA_DIR,
                allow_patterns=[fname],
            )
            print(f"downloaded {fname}", flush=True)

    sentinel = pathlib.Path(f"{TTS_EVAL_DATA_DIR}/.seedtts_testset.extracted")
    if not sentinel.exists():
        print("extracting seedtts_testset.tar.gz", flush=True)
        subprocess.run(
            ["tar", "-xzf", f"{TTS_EVAL_DATA_DIR}/seedtts_testset.tar.gz",
             "-C", TTS_EVAL_DATA_DIR],
            check=True,
        )
        sentinel.touch()

    print("=== eval scorers", flush=True)
    models_done = pathlib.Path(f"{TTS_EVAL_MODEL_DIR}/.downloaded")
    if not models_done.exists():
        # Only the scorers used here; skips hubert, paraformer-zh and
        # whisper-d-v1a (~8GB less).
        snapshot_download(
            EVAL_MODEL_REPO,
            local_dir=TTS_EVAL_MODEL_DIR,
            allow_patterns=[
                "speaker_similarity/**",
                "wer/whisper-large-v3/**",
                "mos/**",
            ],
        )
        models_done.touch()

    vol.commit()

    def _du(path: str) -> float:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return round(total / 1e9, 2)

    with open(TEST_JSONL, encoding="utf-8") as fh:
        rows = sum(1 for _ in fh)

    stats = {
        "test_rows": rows,
        "data_gb": _du(TTS_EVAL_DATA_DIR),
        "scorers_gb": _du(TTS_EVAL_MODEL_DIR),
    }
    print(f"\n{stats}", flush=True)
    return stats


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={VOL_MOUNT_ROOT: vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=TIMEOUT_MINUTES * 60,
)
def generate(
    limit: int = 0,
    tag: str = "seedtts_en",
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    min_p: float = 0.05,
    top_p: float = 1.0,
    repetition_penalty: float = 1.2,
) -> dict:
    """Stage 3a: synthesize one wav per row.

    Chatterbox ships no batched eval CLI, so this drives
    ``ChatterboxTTS.generate`` row by row.
    """
    import json
    import os

    import torch
    import torchaudio

    print("=== gpu check", flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device")
    gpu_name = torch.cuda.get_device_name(0)
    print(f"gpu: {gpu_name}", flush=True)

    paths = _run_paths(tag, limit)
    res_dir = paths["generated"]
    os.makedirs(res_dir, exist_ok=True)

    # Kept in the run dir so each run is self-describing.
    test_list = paths["test_list"]
    with open(TEST_JSONL, encoding="utf-8") as src:
        rows = src.readlines()
    if limit > 0:
        rows = rows[:limit]
    with open(test_list, "w", encoding="utf-8") as dst:
        dst.writelines(rows)
    print(f"test list: {len(rows)} rows -> {test_list}", flush=True)

    print("\n=== loading ChatterboxTTS", flush=True)
    t0 = time.perf_counter()
    from chatterbox.tts import ChatterboxTTS

    model = ChatterboxTTS.from_pretrained(device="cuda")
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s (sr={model.sr})", flush=True)

    parsed = [json.loads(line) for line in rows]

    # Preflight: a path miss would otherwise silently skip every row.
    probe = os.path.join(REF_AUDIO_ROOT, parsed[0]["ref_audio"])
    print(f"ref_audio[0] = {parsed[0]['ref_audio']}", flush=True)
    print(f"  resolved   = {probe}", flush=True)
    print(f"  exists     = {os.path.exists(probe)}", flush=True)

    print(f"\n=== generate ({len(parsed)} rows)", flush=True)
    failures: list[dict] = []
    audio_s = 0.0
    t0 = time.perf_counter()
    for i, row in enumerate(parsed):
        out_path = os.path.join(res_dir, f"{row['id']}.wav")
        if os.path.exists(out_path):  # resume
            continue
        ref = os.path.join(REF_AUDIO_ROOT, row["ref_audio"])
        try:
            wav = model.generate(
                row["text"],
                audio_prompt_path=ref,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
                min_p=min_p,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            torchaudio.save(out_path, wav.cpu(), model.sr)
            audio_s += wav.shape[-1] / model.sr
        except Exception as exc:
            failures.append({"id": row["id"], "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [{i}] FAILED {row['id']}: {exc}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(parsed)}  {time.perf_counter() - t0:.0f}s", flush=True)
    generate_s = time.perf_counter() - t0

    wavs = [f for f in os.listdir(res_dir) if f.endswith(".wav")]
    vol.commit()

    stats = {
        "tag": tag,
        "model": "chatterbox-ar",
        "gpu": gpu_name,
        "temperature": temperature,
        "exaggeration": exaggeration,
        "cfg_weight": cfg_weight,
        "requested_rows": len(parsed),
        "wavs": len(wavs),
        "failures": len(failures),
        "load_s": round(load_s, 1),
        "generate_s": round(generate_s, 1),
        "audio_s": round(audio_s, 1),
        # RTF < 1 is faster than realtime.
        "rtf": round(generate_s / audio_s, 4) if audio_s else None,
        "x_realtime": round(audio_s / generate_s, 1) if generate_s else None,
        "s_per_utt": round(generate_s / len(wavs), 2) if wavs else None,
    }
    print("\n=== generate timings")
    for k, v in stats.items():
        print(f"{k:18s}: {v}")
    if failures:
        print(f"\nfirst failures: {failures[:3]}")

    if not wavs:
        raise RuntimeError("0 wavs generated")
    return stats


@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={VOL_MOUNT_ROOT: vol},
    secrets=[modal.Secret.from_name(HF_SECRET_NAME)],
    timeout=TIMEOUT_MINUTES * 60,
)
def score(tag: str = "seedtts_en", limit: int = 0, nj_per_gpu: int = 2) -> dict:
    """Stage 3b: SIM-o + WER + UTMOS over the generated wavs."""
    import os
    import re
    import subprocess
    import sys

    paths = _run_paths(tag, limit)
    res_dir = paths["generated"]
    if not os.path.isdir(res_dir):
        raise RuntimeError(f"{res_dir} missing -- run generate first")

    # Written by generate.
    test_list = paths["test_list"]
    if not os.path.exists(test_list):
        raise RuntimeError(f"{test_list} missing -- run generate first")
    os.makedirs(paths["scored"], exist_ok=True)

    def _run(name: str, cmd: list[str], log: str) -> None:
        print(f"\n=== {name}\n{' '.join(cmd)}\n", flush=True)
        t0 = time.perf_counter()
        # cwd: omnivoice's sim.py uses sample["ref_audio"] verbatim, so the
        # JSONL's "download/..." paths need the parent of download/ as cwd.
        proc = subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, cwd=VOL_MOUNT_ROOT)
        # --decode-path holds the scorer's per-utterance detail; keep it.
        with open(f"{log}.console", "w", encoding="utf-8") as fh:
            fh.write(proc.stdout)
        print(proc.stdout[-3000:], flush=True)
        print(f"[{name}] {time.perf_counter() - t0:.1f}s rc={proc.returncode}", flush=True)

    sim_log, wer_log, mos_log = (
        f"{paths['scored']}/{name}.log" for name in ("sim", "wer", "mos")
    )

    _run("SIM-o", [
        sys.executable, "-m", "omnivoice.eval.speaker_similarity.sim",
        "--wav-path", res_dir, "--test-list", test_list,
        "--decode-path", sim_log, "--model-dir", TTS_EVAL_MODEL_DIR,
        "--nj-per-gpu", str(nj_per_gpu),
    ], sim_log)

    _run("WER", [
        sys.executable, "-m", "chatterbox_flash.eval.wer_seedtts",
        "--wav-path", res_dir, "--test-list", test_list,
        "--decode-path", wer_log, "--model-dir", TTS_EVAL_MODEL_DIR,
        "--lang", "en", "--nj-per-gpu", str(nj_per_gpu),
        "--text-norm", "omnivoice",
    ], wer_log)

    _run("UTMOS", [
        sys.executable, "-m", "omnivoice.eval.mos.utmos",
        "--wav-path", res_dir, "--test-list", test_list,
        "--decode-path", mos_log, "--model-dir", TTS_EVAL_MODEL_DIR,
        "--nj-per-gpu", str(nj_per_gpu),
    ], mos_log)

    vol.commit()

    # run_eval.sh's patterns, incl. its WER fallbacks.
    def _metric(log: str, pattern: str) -> float | None:
        # Summary lines are in the console capture, not the decode file.
        log = f"{log}.console"
        if not os.path.exists(log):
            return None
        hit = None
        with open(log, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = re.search(pattern, line)
                if m:
                    hit = float(m.group(1))
        return hit

    wer = None
    for pat in (r"Seed-TTS WER \(Avg of WERs\):\s*([0-9.]+)%",
                r"WER \(Weighted\):\s*([0-9.]+)%",
                r"^WER:\s*([0-9.]+)%"):
        wer = _metric(wer_log, pat)
        if wer is not None:
            break

    results = {
        "tag": tag,
        "sim_o": _metric(sim_log, r"SIM-o score:\s*([0-9.]+)")
        or _metric(sim_log, r"Average SIM-o:\s*([0-9.]+)"),
        "wer_pct": wer,
        "utmos": _metric(mos_log, r"UTMOS score:\s*([0-9.]+)")
        or _metric(mos_log, r"Average UTMOS:\s*([0-9.]+)"),
    }
    print("\n=== scores")
    for k, v in results.items():
        print(f"{k:10s}: {v}")
    return results


def fetch(tag: str, limit: int) -> pathlib.Path:
    """Copy a run's scored/ logs off the Volume into results/SPLIT/TAG/."""
    paths = _run_paths(tag, limit)
    remote = paths["scored"].removeprefix(f"{VOL_MOUNT_ROOT}/")
    local = LOCAL_RESULTS / paths["dataset_dir"] / tag
    local.mkdir(parents=True, exist_ok=True)
    entries = vol.listdir(remote)
    for entry in entries:
        name = entry.path.rsplit("/", 1)[-1]
        (local / name).write_bytes(b"".join(vol.read_file(entry.path)))
    print(f"fetched {len(entries)} files: {remote} -> {local}")
    return local


@app.local_entrypoint()
def main(
    limit: int = 0,
    tag: str = "seedtts_en",
    stage: str = "all",  # all | download | generate | score | upload | fetch
    temperature: float = 0.8,
    exaggeration: float = 0.5,
    wav_dir: str = "",
    test_list: str = "",
    timeout_min: int = TIMEOUT_MINUTES,  # minutes, per function
):
    gen_stats: dict = {}
    scores: dict = {}

    if stage in ("all", "download"):
        print(">>> stage 1: download")
        print(download.with_options(timeout=timeout_min * 60).remote())

    if stage == "upload":
        import json
        import pathlib

        if not wav_dir or not test_list:
            raise SystemExit("--stage upload needs --wav-dir and --test-list")
        rows = [
            line for line in pathlib.Path(test_list).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if limit > 0:
            rows = rows[:limit]
        ids = {json.loads(line)["id"] for line in rows}

        # mlx_decode nests wavs a level below the model dir, hence rglob.
        found = {p.stem: p for p in pathlib.Path(wav_dir).rglob("*.wav")}
        missing = sorted(ids - found.keys())
        if missing:
            raise SystemExit(
                f"{len(missing)} of {len(ids)} rows have no wav under {wav_dir}: "
                f"{missing[:3]}"
            )

        print(f">>> upload: {len(ids)} wavs from {wav_dir}")
        print(upload.with_options(timeout=timeout_min * 60).remote(
            tag=tag, limit=limit,
            wavs=[(f"{i}.wav", found[i].read_bytes()) for i in sorted(ids)],
            test_list_text="\n".join(rows) + "\n",
        ))

    if stage in ("all", "generate"):
        print(">>> stage 3a: generate")
        gen_stats = generate.with_options(timeout=timeout_min * 60).remote(
            limit=limit, tag=tag,
            temperature=temperature, exaggeration=exaggeration,
        )

    if stage in ("all", "score", "upload"):
        print(">>> stage 3b: score")
        scores = score.with_options(timeout=timeout_min * 60).remote(tag=tag, limit=limit)

    if stage in ("all", "score", "upload", "fetch"):
        fetch(tag, limit)

    print("\n" + "=" * 52)
    print(f"  Chatterbox TTS (AR) / Seed-TTS test-en ({tag})")
    print("=" * 52)
    if gen_stats:
        print(f"  model      : {gen_stats['model']}  gpu={gen_stats['gpu']}")
        print(f"  config     : temp={gen_stats['temperature']} "
              f"exag={gen_stats['exaggeration']} cfg={gen_stats['cfg_weight']}")
        print(f"  wavs       : {gen_stats['wavs']} / {gen_stats['requested_rows']}"
              f"  ({gen_stats['failures']} failed)")
        print(f"  generate   : {gen_stats['generate_s']}s "
              f"({gen_stats['s_per_utt']}s/utt, {gen_stats['x_realtime']}x realtime)")
        print(f"  RTF        : {gen_stats['rtf']}")
    if scores:
        print(f"  SIM-o      : {scores['sim_o']}")
        print(f"  WER        : {scores['wer_pct']}%")
        print(f"  UTMOS      : {scores['utmos']}")
    print("=" * 52)
    return {"generate": gen_stats, "scores": scores}
