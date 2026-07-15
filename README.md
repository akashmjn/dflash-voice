# dflash-voice

Readable MLX TTS inference and benchmarking for **Qwen3-TTS** and **Fish Audio S2 Pro**, ported from [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.4. Inference loop is ported to understand key components and benchmark per-frame timing (backbone_semantic vs depth_audio).

## Current state

| Area | Status |
|------|--------|
| **Qwen3-TTS** (`tts_mlx/qwen3.py`) | Base preset-voice generate loop ported; streaming supported |
| **Fish S2 Pro** (`tts_mlx/fish.py`) | DualAR generate loop ported; inline `[tag]` control supported |
| **Parity tests** | `tests/test_*_generate.py` compare output against mlx-audio reference |

Weights and `nn.Module` definitions still come from mlx-audio; `tts_mlx` owns prompt construction, autoregression, and codec decode in annotated, single-file modules.

## Install

```bash
uv pip install -e ".[tts_mlx,dev]"
```

Pinned deps match mlx-audio's tested stack (`mlx-lm==0.31.1`, `transformers==5.6.0`).

## Quick start

```bash
# Will download models to HF_CACHE on first run
python benchmark_mlx/bench_tts_mlx.py --backend qwen3
python benchmark_mlx/bench_tts_mlx.py --backend fish
```

GPU / vLLM-Omni Fish depth-step ablation (Modal H100): see [`benchmark_gpu/README.md`](benchmark_gpu/README.md).

## Benchmark results (aggregate)

6 prompts per model, 8-bit MLX checkpoints, 64GB M1 Max Apple Silicon. **Gen RTF** = codec-frame generation speed vs native frame rate; **Wall RTF** = end-to-end including codec decode.

![TTS MLX benchmark aggregate](docs/benchmark-time-per-frame.png)

| Model | Native | Backbone (semantic codes) | Depth (audio codes) | Depth % | Depth iters | ms / depth iter | Total ms | Codec frames/s | Gen RTF | Wall RTF |
|-------|--------|---------------------------|---------------------|---------|-------------|-----------------|----------|----------------|---------|----------|
| Qwen3 1.7B 8bit | 12.5 Hz | 9.4 ms | 17.0 ms | 63% | 15 | 1.13 ms | 26.9 ms | 37.2 | 2.98× | 2.52× |
| Fish S2 Pro 8bit | 21 Hz | 21.9 ms | 20.4 ms | 48% | 9 | 2.26 ms | 42.3 ms | 23.7 | 1.13× | 0.92× |

Real-time budgets: Qwen3 @ 12.5 Hz → 80 ms/frame; Fish @ 21 Hz → 47.6 ms/frame.

Raw metrics: `benchmark_mlx/output/qwen3/qwen3-tts-12hz-1.7b-base-8bit/metrics.json`, `benchmark_mlx/output/fish/fish-audio-s2-pro-8bit/metrics.json` (gitignored — regenerate with `benchmark_mlx/bench_tts_mlx.py`).
