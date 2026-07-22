"""Fish Speech S2 Pro — Fast-AR depth-step ablation (offline, vLLM-Omni, on Modal).

Measures how throughput scales as the number of Fast-AR (depth audio decoder) forward
passes per frame is reduced 9 -> 1, at several concurrency levels, on one H100.
This is a COMPUTE ablation, not a quality one: truncating the depth loop produces
garbage audio and that is expected. No quality metric is reported.

The question the sweep answers: is the Fast-AR cost weight-traffic/FLOP bound
(throughput ~linear in step count) or kernel-launch bound (flat-then-cliff)?
Back-of-envelope ceiling on the LM stage going 9 -> 1 is ~1.8-1.9x; a measured
stage-0 speedup materially above ~2x is a BUG SIGNAL (almost certainly the
sequence-length confound, brief §5.1), not a result.

Architecture (verified against vllm-omni @ c93359b):
  * Stage 0 `slow_ar`  : Qwen3-4B backbone semantic, 1 semantic token / frame. CUDA graphs ON
                         (deploy `enforce_eager: false`).
  * Fast AR            : depth audio decoder (4-layer residual predictor), run graph-safe
                         INSIDE the stage-0 graph via
                         `FishSpeechSlowARForConditionalGeneration.talker_mtp`.
                         The depth loop lives in `FishSpeechFastAR.forward`
                         (`for step in range(1, num_cb)`), patched by
                         patch_depth_loop.diff to `range(1, FISH_DEPTH_STEPS+1)`.
  * Stage 1 `dac_decoder`: RVQ codes -> 44.1 kHz audio. Untouched by this ablation.

Because the depth-step count is baked at CUDA-graph capture, FISH_DEPTH_STEPS is a
static env var read at import time and we launch a FRESH PROCESS per sweep point.
Never sweep inside one process (brief §5.2).

Usage:
    modal run benchmark_gpu/fish_depth_ablation.py                 # full sweep
    modal run benchmark_gpu/fish_depth_ablation.py --dry-run       # build image + 1 warmup at depth=9
    modal run benchmark_gpu/fish_depth_ablation.py --depth-steps 1,9 --concurrency 1,4 --graphs on,off

Deliverables written to the results Volume (sync down with `modal volume get`):
    results.csv   — one row per (depth_steps, concurrency, cudagraphs, repeat)
    FINDINGS.md   — graphed-vs-eager curves, measured 9->1 stage-0 speedup vs ceiling, verdict
"""

from __future__ import annotations

import json
import os

import modal

# ---------------------------------------------------------------------------
# Pins (from vllm-omni recipes/fishaudio/Fish-Speech-S2-Pro.md)
# ---------------------------------------------------------------------------
VLLM_VERSION = "0.19.0"
VLLM_OMNI_COMMIT = "c93359bb354a6aa5c14d062430cb85b2c4db251e"
MODEL_ID = "fishaudio/s2-pro"

# Fixed sweep constants (brief §6).
N_FRAMES = 250          # pinned frame count per request (~12 s of audio at ~21 Hz)
SEED = 42
GPU_MEM_UTIL = 0.85
WARMUP_REPEATS = 1      # discarded (triggers compile / CUDA-graph capture)
TIMED_REPEATS = 3       # >= 3 timed repeats per cell
PROMPT_TEXT = (
    "In a world where artificial intelligence transforms how we communicate, "
    "voice synthesis stands at the frontier of human-computer interaction."
)

# Full experiment matrix (brief §6). Overridable from the CLI.
DEFAULT_DEPTH_STEPS = (1, 3, 5, 7, 9)
DEFAULT_CONCURRENCY = (1, 4, 16, 32)
DEFAULT_GRAPHS = ("on", "off")   # on == CUDA graphs; off == enforce_eager

HF_CACHE_DIR = "/root/hf_cache"
RESULTS_DIR = "/root/results"
PATCH_PATH = "/root/patch/patch_depth_loop.diff"
EAGER_YAML = "/root/patch/deploy_fish_eager.yaml"

app = modal.App("fish-depth-ablation")

hf_cache = modal.Volume.from_name("fish-hf-cache", create_if_missing=True)
results_vol = modal.Volume.from_name("fish-depth-results", create_if_missing=True)

_here = os.path.dirname(__file__)

# ---------------------------------------------------------------------------
# Image: vLLM 0.19.0 + vllm-omni @ pinned commit + fish-speech, then apply the
# depth-loop patch to the installed package on disk (brief §7). Patching at
# build time means every EngineCore subprocess the sweep spawns sees it — a
# driver-side monkeypatch would run in the parent and the engine would never
# see it (brief §4).
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "libportaudio2", "portaudio19-dev", "libsndfile1")
    .pip_install(
        f"vllm=={VLLM_VERSION}",
        f"vllm-omni @ git+https://github.com/vllm-project/vllm-omni.git@{VLLM_OMNI_COMMIT}",
        "fish-speech",
        "soundfile",
        "huggingface-hub",
    )
    # Ship the patch + no-async deploy config into the image, then apply the patch
    # to the installed vllm_omni package.
    .add_local_file(
        os.path.join(_here, "patch_depth_loop.diff"), PATCH_PATH, copy=True
    )
    .add_local_file(
        os.path.join(_here, "deploy_fish_eager.yaml"), EAGER_YAML, copy=True
    )
    .run_commands(
        # Locate the installed package root and apply the patch against it.
        'PKG=$(python -c "import vllm_omni, os; print(os.path.dirname(os.path.dirname(vllm_omni.__file__)))") && '
        f'echo "vllm_omni package root: $PKG" && '
        f'git -C "$PKG" apply --verbose {PATCH_PATH} && '
        # Fail the build loudly if the patched marker isn't present.
        'grep -q "FISH_DEPTH_STEPS" '
        '"$PKG/vllm_omni/model_executor/models/fish_speech/fish_speech_fast_ar.py" && '
        'echo "PATCH APPLIED OK"'
    )
    .env({"HF_HOME": HF_CACHE_DIR, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"})
)


# ===========================================================================
# In-container helpers (run on the H100). Imports are inside functions so the
# local `modal run` invocation doesn't need vllm on macOS.
# ===========================================================================


def _resolve_deploy_config(graphs: str) -> str:
    """Return the stage-config YAML path to hand to Omni.

    `graphs == "on"`  -> the pinned package's default deploy/fish_qwen3_omni.yaml
                         (enforce_eager:false, CUDA graphs ON).
    `graphs == "off"` -> our deploy_fish_eager.yaml (enforce_eager:true), identical
                         otherwise. The gap between the two curves is the
                         launch-overhead contribution (brief §5.3).

    NOTE: async_chunk STAYS true in both — the Fish pipeline has no sync path, so
    async_chunk:false fails validation. Stage-0 time is isolated via the
    orchestrator's per-stage timers (log_stats=True) and the in-engine
    _DEPTH_LOOP_SECONDS accumulator, NOT by sequential stage execution (brief §5.4;
    see README for the caveat).

    We always bump stage-0 max_num_seqs to the max concurrency (32) so the
    scheduler can actually batch (brief §6).
    """
    import shutil

    import vllm_omni
    import yaml

    pkg_root = os.path.dirname(vllm_omni.__file__)
    default_yaml = os.path.join(pkg_root, "deploy", "fish_qwen3_omni.yaml")

    out = os.path.join("/root", f"deploy_{graphs}.yaml")
    if graphs == "off":
        shutil.copy(EAGER_YAML, out)
    else:
        shutil.copy(default_yaml, out)

    # Bump stage-0 max_num_seqs so batching to the requested concurrency is possible.
    with open(out) as f:
        cfg = yaml.safe_load(f)
    for stage in cfg.get("stages", []):
        if stage.get("stage_id") == 0:
            stage["max_num_seqs"] = max(int(stage.get("max_num_seqs", 4)), max(DEFAULT_CONCURRENCY))
            stage["gpu_memory_utilization"] = GPU_MEM_UTIL
            sp = stage.setdefault("default_sampling_params", {})
            sp["seed"] = SEED
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f)
    return out


def _build_prompt(text: str) -> dict:
    """Build one text-only Fish prompt (mirrors examples/.../fish_speech/end2end.py)."""
    from transformers import AutoTokenizer

    from vllm_omni.model_executor.models.fish_speech.prompt_utils import (
        build_fish_text_only_prompt_ids,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    prompt_ids, normalized_text = build_fish_text_only_prompt_ids(tokenizer, text)
    return {
        "prompt_token_ids": prompt_ids,
        "additional_information": {"text": [normalized_text]},
    }


def _stage0_sampling_params():
    """Stage-0 sampling params that PIN the frame count (brief §5.1, mandatory).

    max_tokens == min_tokens == N and ignore_eos=True force exactly N frames per
    request, neutralising the sequence-length bias (fewer frames misread as a
    speedup). We also clear stop_token_ids in case ignore_eos alone doesn't
    override the pipeline's stop on <|im_end|> (151645).
    """
    from vllm.sampling_params import SamplingParams

    return SamplingParams(
        max_tokens=N_FRAMES,
        min_tokens=N_FRAMES,
        ignore_eos=True,
        stop_token_ids=[],
        temperature=0.8,
        top_k=30,
        top_p=0.9,
        seed=SEED,
    )


def _stage1_sampling_params():
    """Stage-1 (DAC decoder) sampling params — greedy, mirrors the deploy default.

    Passed explicitly rather than None because per-element None in the stage list
    isn't guaranteed to be coerced to the stage default in this vllm-omni build.
    """
    from vllm.sampling_params import SamplingParams

    return SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=65536,
        seed=SEED,
    )


def _run_one_cell(depth_steps: int, concurrency: int, graphs: str) -> dict:
    """Run one sweep cell in THIS process (env already has FISH_DEPTH_STEPS set).

    Returns a dict of per-repeat metrics. The engine has already imported the
    patched module, so FISH_DEPTH_STEPS is frozen; we verify it from the logs.
    """
    import time

    import torch

    from vllm_omni import Omni

    assert os.environ.get("FISH_DEPTH_STEPS") == str(depth_steps), (
        "FISH_DEPTH_STEPS env must be set before this process imports vllm_omni"
    )

    stage_cfg = _resolve_deploy_config(graphs)
    prompt = _build_prompt(PROMPT_TEXT)
    sp0 = _stage0_sampling_params()
    sp1 = _stage1_sampling_params()

    omni = Omni(model=MODEL_ID, stage_configs_path=stage_cfg, log_stats=True)

    def _generate_batch(n: int) -> dict:
        """Submit n identical prompts in one generate() call; return timing + frame counts."""
        prompts = [dict(prompt) for _ in range(n)]
        # Per-stage sampling params: [stage0, stage1]. Only stage 0 needs the
        # frame pin; stage 1 mirrors the DAC decoder's greedy default.
        sp_list = [sp0, sp1]

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        frame_counts: list[int] = []
        for out in omni.generate(prompts, sampling_params_list=sp_list, use_tqdm=False):
            ro = out.request_output
            if ro is None or not ro.outputs:
                continue
            # Count stage-0 emitted tokens for the exact-N assertion. The exact
            # attribute path is confirmed on first GPU run; fall back gracefully.
            n_tok = _extract_frame_count(out)
            if n_tok is not None:
                frame_counts.append(n_tok)
        torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        peak_mem_gb = torch.cuda.max_memory_allocated() / 1e9
        return {"wall_s": wall_s, "peak_mem_gb": peak_mem_gb, "frame_counts": frame_counts}

    # Warmup (discarded) — triggers compile / CUDA-graph capture (brief §6).
    for _ in range(WARMUP_REPEATS):
        _generate_batch(concurrency)

    repeats = []
    for r in range(TIMED_REPEATS):
        m = _generate_batch(concurrency)
        total_frames = concurrency * N_FRAMES
        # ASSERT exactly N frames per request (brief §5.1 / §9). Fail loudly.
        bad = [c for c in m["frame_counts"] if c != N_FRAMES]
        frames_ok = len(m["frame_counts"]) == concurrency and not bad
        repeats.append(
            {
                "repeat": r,
                "wall_s": m["wall_s"],
                "frames_per_s": total_frames / m["wall_s"],
                "frames_per_s_per_req": N_FRAMES / m["wall_s"],
                "peak_mem_gb": m["peak_mem_gb"],
                "frames_ok": frames_ok,
                "observed_frame_counts": m["frame_counts"],
            }
        )
        if not frames_ok:
            raise RuntimeError(
                f"Frame-count assertion FAILED (depth={depth_steps}, conc={concurrency}, "
                f"graphs={graphs}, repeat={r}): expected {concurrency}x{N_FRAMES}, "
                f"got {m['frame_counts']}. Refusing to report — see brief §5.1."
            )

    return {
        "depth_steps": depth_steps,
        "concurrency": concurrency,
        "cudagraphs": graphs,
        "n_frames": N_FRAMES,
        "repeats": repeats,
        "depth_loop_seconds": _read_depth_loop_accum(),
    }


def _extract_frame_count(stage_output) -> int | None:
    """Best-effort per-request stage-0 frame (token) count from an OmniRequestOutput.

    The exact field is verified on first GPU run; several plausible paths are tried
    so the harness surfaces a clear count (or None -> assertion catches it).
    """
    ro = getattr(stage_output, "request_output", None)
    if ro is None or not getattr(ro, "outputs", None):
        return None
    out0 = ro.outputs[0]
    for attr in ("token_ids", "codes"):
        val = getattr(out0, attr, None)
        if val is not None:
            try:
                return len(val)
            except TypeError:
                pass
    mm = getattr(out0, "multimodal_output", None)
    if isinstance(mm, dict) and "codes" in mm:
        codes = mm["codes"]
        try:
            # codes: [B, num_codebooks, T] or list of frame chunks.
            return int(codes[0].shape[-1]) if hasattr(codes[0], "shape") else len(codes)
        except (IndexError, TypeError, AttributeError):
            return None
    return None


def _read_depth_loop_accum() -> dict:
    """Read the in-engine depth-loop accumulator (host-side; eager arm only)."""
    try:
        from vllm_omni.model_executor.models.fish_speech import fish_speech_fast_ar as far

        return {
            "effective_depth_steps": int(far.FISH_DEPTH_STEPS),
            "depth_loop_seconds": float(far._DEPTH_LOOP_SECONDS),
            "depth_loop_calls": int(far._DEPTH_LOOP_CALLS),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": repr(exc)}


# ===========================================================================
# Modal functions — ONE function invocation == ONE fresh process == ONE cell.
# ===========================================================================


@app.function(
    image=image,
    gpu="H100",
    volumes={HF_CACHE_DIR: hf_cache, RESULTS_DIR: results_vol},
    timeout=60 * 60,
)
def run_cell(depth_steps: int, concurrency: int, graphs: str) -> dict:
    """Run one (depth_steps, concurrency, graphs) cell in a fresh container/process.

    FISH_DEPTH_STEPS MUST be set before vllm_omni is imported. Because this runs
    in a fresh container, setting it here (before any vllm_omni import) is safe;
    the patched module reads it at import time.
    """
    os.environ["FISH_DEPTH_STEPS"] = str(depth_steps)
    result = _run_one_cell(depth_steps, concurrency, graphs)
    # Persist raw per-cell JSON immediately (crash-resilient).
    os.makedirs(RESULTS_DIR, exist_ok=True)
    fname = f"cell_d{depth_steps}_c{concurrency}_{graphs}.json"
    with open(os.path.join(RESULTS_DIR, fname), "w") as f:
        json.dump(result, f, indent=2)
    results_vol.commit()
    return result


@app.function(
    image=image,
    gpu="H100",
    volumes={HF_CACHE_DIR: hf_cache, RESULTS_DIR: results_vol},
    timeout=60 * 30,
)
def dry_run_cell() -> dict:
    """Build check: run a single warmup request at depth=9 and confirm it works."""
    os.environ["FISH_DEPTH_STEPS"] = "9"
    stage_cfg = _resolve_deploy_config("on")
    from vllm_omni import Omni

    omni = Omni(model=MODEL_ID, stage_configs_path=stage_cfg, log_stats=True)
    prompt = _build_prompt(PROMPT_TEXT)
    sp_list = [_stage0_sampling_params(), _stage1_sampling_params()]
    n_out = 0
    for out in omni.generate([prompt], sampling_params_list=sp_list, use_tqdm=False):
        if out.request_output and out.request_output.outputs:
            n_out += 1
    return {"dry_run_ok": n_out > 0, "n_outputs": n_out, "accum": _read_depth_loop_accum()}


# ===========================================================================
# Aggregation into results.csv + FINDINGS.md (runs in-container so it can read
# the Volume).
# ===========================================================================


@app.function(image=image, volumes={RESULTS_DIR: results_vol}, timeout=60 * 10)
def aggregate() -> str:
    import csv
    import glob
    import io
    import statistics as stats

    results_vol.reload()
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "cell_*.json"))):
        with open(path) as f:
            cell = json.load(f)
        for rep in cell["repeats"]:
            rows.append(
                {
                    "depth_steps": cell["depth_steps"],
                    "concurrency": cell["concurrency"],
                    "cudagraphs": cell["cudagraphs"],
                    "n_frames": cell["n_frames"],
                    "repeat": rep["repeat"],
                    "wall_s": round(rep["wall_s"], 4),
                    "frames_per_s": round(rep["frames_per_s"], 2),
                    "frames_per_s_per_req": round(rep["frames_per_s_per_req"], 2),
                    "peak_mem_gb": round(rep["peak_mem_gb"], 2),
                    "frames_ok": rep["frames_ok"],
                    "depth_loop_seconds": round(
                        cell.get("depth_loop_seconds", {}).get("depth_loop_seconds", 0.0), 4
                    ),
                    "effective_depth_steps": cell.get("depth_loop_seconds", {}).get(
                        "effective_depth_steps"
                    ),
                }
            )

    # results.csv
    buf = io.StringIO()
    if rows:
        w = csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    csv_path = os.path.join(RESULTS_DIR, "results.csv")
    with open(csv_path, "w") as f:
        f.write(buf.getvalue())

    # FINDINGS.md — median frames/s per (graphs, depth) at concurrency 1, plus the
    # 9->1 stage-0 speedup and the bug-signal check.
    def _median_fps(graphs: str, depth: int, conc: int) -> float | None:
        vals = [
            r["frames_per_s"]
            for r in rows
            if r["cudagraphs"] == graphs and r["depth_steps"] == depth and r["concurrency"] == conc
        ]
        return stats.median(vals) if vals else None

    lines = ["# Fish Speech S2 Pro — Fast-AR depth-step ablation: FINDINGS\n"]
    for graphs in DEFAULT_GRAPHS:
        lines.append(f"\n## CUDA graphs = {graphs} (concurrency=1)\n")
        lines.append("| depth_steps | median frames/s |")
        lines.append("|---|---|")
        for d in DEFAULT_DEPTH_STEPS:
            fps = _median_fps(graphs, d, 1)
            lines.append(f"| {d} | {fps:.1f} |" if fps is not None else f"| {d} | (missing) |")
        f9, f1 = _median_fps(graphs, 9, 1), _median_fps(graphs, 1, 1)
        if f9 and f1:
            speedup = f1 / f9
            verdict = ""
            if speedup > 2.0:
                verdict = "  ⚠️ **> 2x — BUG SIGNAL** (check sequence-length confound, brief §5.1)"
            lines.append(f"\n**9→1 throughput speedup: {speedup:.2f}x** (predicted ceiling ~1.8x).{verdict}")

    lines.append(
        "\n## Verdict\n"
        "- If throughput scales ~linearly with depth_steps ⇒ **compute/weight-traffic bound** ⇒ "
        "removing depth steps is the real win.\n"
        "- If flat-then-cliff ⇒ **kernel-launch bound** ⇒ CUDA-graph the depth loop instead.\n"
        "- The graphed-vs-eager gap is the launch-overhead contribution.\n"
        "- (Fill the one-line verdict after inspecting the two curves above.)\n"
    )
    findings_path = os.path.join(RESULTS_DIR, "FINDINGS.md")
    with open(findings_path, "w") as f:
        f.write("\n".join(lines))
    results_vol.commit()
    return f"Wrote {csv_path} ({len(rows)} rows) and {findings_path}"


# ===========================================================================
# Local entrypoint — orchestrates the sweep (one remote call per cell).
# ===========================================================================


def _parse_csv_ints(s: str) -> tuple[int, ...]:
    return tuple(int(x) for x in s.split(",") if x.strip())


@app.local_entrypoint()
def main(
    dry_run: bool = False,
    depth_steps: str = "",
    concurrency: str = "",
    graphs: str = "",
):
    """Kick off the sweep on Modal.

    --dry-run              build the image and run a single depth=9 warmup request.
    --depth-steps 1,3,9    override the depth-step axis.
    --concurrency 1,4      override the concurrency axis.
    --graphs on,off        override the CUDA-graph axis.
    """
    if dry_run:
        res = dry_run_cell.remote()
        print("dry_run result:", res)
        return

    depths = _parse_csv_ints(depth_steps) if depth_steps else DEFAULT_DEPTH_STEPS
    concs = _parse_csv_ints(concurrency) if concurrency else DEFAULT_CONCURRENCY
    graph_axis = tuple(g.strip() for g in graphs.split(",")) if graphs else DEFAULT_GRAPHS

    cells = [(d, c, g) for g in graph_axis for d in depths for c in concs]
    print(f"Sweeping {len(cells)} cells: depths={depths} conc={concs} graphs={graph_axis}")

    # Fan out one fresh process per cell (brief §5.2). starmap preserves isolation:
    # each invocation is a fresh container that sets FISH_DEPTH_STEPS before import.
    for result in run_cell.starmap(cells):
        r0 = result["repeats"][0]
        print(
            f"  depth={result['depth_steps']} conc={result['concurrency']} "
            f"graphs={result['cudagraphs']}: "
            f"{r0['frames_per_s']:.1f} frames/s (frames_ok={r0['frames_ok']})"
        )

    summary = aggregate.remote()
    print(summary)
    print("Sync results down with: modal volume get fish-depth-results / ./benchmark_gpu/out")
