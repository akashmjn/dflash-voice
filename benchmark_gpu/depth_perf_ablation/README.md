# Fish Speech S2 Pro — Fast-AR depth-step ablation (offline, vLLM-Omni)

Measures how throughput scales as the number of **Fast-AR (depth audio decoder)** forward passes per
frame is reduced from **9 → 1**, at several concurrency levels, on a single **H100** (Modal). This
is a **compute ablation, not a quality one** — truncating the depth loop produces garbage audio,
which is expected and acceptable. No quality metric is reported.

## The question

Fish S2 Pro is Dual-AR: a **Slow AR** (Qwen3-4B backbone semantic) predicts one semantic token per frame, then a
**Fast AR** (depth audio decoder; 4-layer, ~400M) walks the **9 residual codebooks** depth-wise — one forward each. So
per frame: **1 Slow AR + 9 Fast AR forwards.** We sweep the Fast-AR step count and ask:

- Throughput ~**linear** in step count ⇒ **compute/weight-traffic bound** ⇒ removing steps is the win.
- **Flat-then-cliff** ⇒ **kernel-launch bound** ⇒ CUDA-graphing the depth loop is the win.

Back-of-envelope: Slow AR reads ~4B params/frame, Fast AR ~0.4B × 9 ≈ 3.6B, so depth is ~47% of
LM-stage weight traffic → the ceiling going 9→1 on the LM stage is ~**1.8–1.9×** (less end-to-end).
**A measured stage-0 speedup materially above ~2× is a BUG SIGNAL** (almost certainly the
sequence-length confound), not a result — the harness flags it.

## What's here

| File | Purpose |
|------|---------|
| `fish_depth_ablation.py` | Modal app: patched image, sweep driver (one fresh process per cell), CSV/FINDINGS aggregation. |
| `patch_depth_loop.diff` | On-disk patch to `vllm_omni/.../fish_speech/fish_speech_fast_ar.py`. Reviewable in isolation. |
| `deploy_fish_eager.yaml` | `enforce_eager: true` variant of the packaged deploy config, for the `graphs=off` arm. |

## Pins (from vllm-omni `recipes/fishaudio/Fish-Speech-S2-Pro.md`)

- vLLM **0.19.0**
- vllm-omni commit **`c93359bb354a6aa5c14d062430cb85b2c4db251e`**
- Model **`fishaudio/s2-pro`** (~11 GB, cached in a Modal Volume), `num_codebooks=10` (9 residual + 1
  semantic — **do not** change it; the codec expects all 10).

## How the patch works (`patch_depth_loop.diff`)

The Fast-AR depth loop is `for step in range(1, num_cb)` in `FishSpeechFastAR.forward`. The patch:

1. Reads a module-level constant **once at import**:
   `FISH_DEPTH_STEPS = min(int(os.environ.get("FISH_DEPTH_STEPS", "9")), 9)` (9 = unmodified baseline).
2. Bounds the loop to `range(1, min(FISH_DEPTH_STEPS+1, num_cb))` and **zero-inits** `all_codes`
   (was `torch.empty`) so the un-run residual codebooks are token id 0 (valid; the DAC decoder
   consumes all 10 layers without a shape error).
3. Logs the **effective** `FISH_DEPTH_STEPS` from *inside* the engine process, and accumulates
   host-side depth-loop time (`_DEPTH_LOOP_SECONDS`).

**Why import-time + one process per cell:** the Fast AR runs *graph-safe inside stage 0's CUDA
graph* (`FishSpeechSlowARForConditionalGeneration.talker_mtp_graph_safe = True`, stage 0
`enforce_eager: false`). The step count is therefore **baked at graph capture** — a per-request or
runtime-mutable count would be silently ignored. So the value is static, read at import, and the
driver launches a **fresh container per sweep point**.

**Why patch on disk at image-build time:** vLLM v1 runs the engine in a separate EngineCore process.
A monkeypatch in the driver runs in the parent and the engine never sees it. The image applies the
diff to the installed package (`git apply` in `.run_commands`), so every subprocess sees it.

## Running

```bash
uv pip install -e ".[dev]"       # thin driver deps (modal, soundfile); vllm is image-side
modal token new                  # once, if not already authenticated

# 1) Build the image and smoke-test: 1 warmup request at depth=9.
modal run benchmark_gpu/fish_depth_ablation.py --dry-run

# 2) Full sweep (5 depths × 4 concurrencies × 2 graph modes = 40 cells).
modal run benchmark_gpu/fish_depth_ablation.py

# Subset:
modal run benchmark_gpu/fish_depth_ablation.py --depth-steps 1,9 --concurrency 1,4 --graphs on,off

# 3) Pull results down.
modal volume get fish-depth-results / ./benchmark_gpu/out
```

## Experiment matrix (brief §6)

| Axis | Values |
|---|---|
| `FISH_DEPTH_STEPS` | 1, 3, 5, 7, 9 |
| Concurrency (prompts per `generate()` call) | 1, 4, 16, 32 |
| CUDA graphs | on (default config), off (`deploy_fish_eager.yaml`) |

Fixed: same prompt, `max_tokens = min_tokens = N = 250` frames (~12 s at ~21 Hz), `ignore_eos=True`,
`seed=42`, `gpu_memory_utilization=0.85`. Stage-0 `max_num_seqs` is bumped to 32 so the scheduler can
batch. Per cell: 1 discarded warmup + **≥3 timed repeats**; median + spread reported.

## Guardrails baked in

- **Frame pinning (mandatory, brief §5.1):** stage-0 `max_tokens=min_tokens=N`, `ignore_eos=True`,
  `stop_token_ids=[]`. The harness **asserts every request emitted exactly N frames** and aborts the
  cell loudly otherwise — truncated depth degenerates the Slow AR to early EOS, and fewer frames
  would be misread as a speedup.
- **Bug-signal check:** `FINDINGS.md` flags any 9→1 throughput speedup > 2× as a likely confound.
- **Patch-inert baseline:** at `DEPTH_STEPS=9`, `min(9+1,10)=10` ⇒ the loop is byte-identical to the
  original and the zero-fill slice is empty — so depth=9 must match the unpatched reference within
  noise (acceptance criterion).

## Known caveats (confirm on first GPU run)

1. **Stage-0 isolation is NOT via sequential stages.** The Fish pipeline only defines
   `async_chunk_process_next_stage_input_func` (no `sync_process_input_func`), so `async_chunk:false`
   fails pipeline validation — we cannot force sequential stage-0/stage-1 execution. Stage-0 time is
   isolated instead via (a) the orchestrator's per-stage `stage_first_ts`/`stage_last_ts` timers
   (`log_stats=True`, `vllm_omni/metrics/stats.py`) and (b) the in-engine `_DEPTH_LOOP_SECONDS`
   accumulator. Under CUDA graphs the host-side depth timer under-reads true kernel time, so the
   graphs-on stage-0 signal leans on the orchestrator timers; the eager arm's `_DEPTH_LOOP_SECONDS`
   is the cleaner depth-only number. **Wire the stage-0 duration from `OrchestratorMetrics` into the
   CSV once its exact accessor is confirmed on the box.**
2. **Per-request frame count** (`_extract_frame_count`) tries several `OmniRequestOutput` field paths;
   confirm the real one on first run so the exact-N assertion is meaningful (it currently fails
   closed — a `None` count trips the assertion).
3. **`ignore_eos` vs `stop_token_ids`:** we clear `stop_token_ids` in the stage-0 sampling params as
   a belt-and-suspenders; verify N frames actually land.
4. **Image base / Python:** built on `nvidia/cuda:12.8.0-devel` + Python 3.12. If the pinned
   vllm/vllm-omni wheels aren't available for 3.12, drop `add_python` to 3.10 (recipe says 3.10+).

## Deliverables produced

- `results.csv` — one row per `(depth_steps, concurrency, cudagraphs, repeat)`.
- `FINDINGS.md` — median frames/s per (graphs, depth) curve, the measured 9→1 stage-0 speedup vs the
  ~1.8× ceiling, and a one-line **compute-bound vs launch-bound** verdict.

## Follow-on (not in scope)

A later workstream benchmarks concurrent throughput against a **live** vLLM-Omni
`/v1/audio/speech` endpoint. The timing/metric definitions here (frames/sec at fixed frame count,
stage-0 vs end-to-end) are kept identical so the two sets of numbers are directly comparable.
