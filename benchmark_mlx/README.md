## Lightweight vendored/ported TTS inference for benchmarking

This module ports the MLX TTS inference loop from [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.4 to a hackable, 
single-file module - allowing us to benchmark backbone vs depth decoding times.

```bash
uv pip install -e ".[benchmark_mlx]"
# downloads models to HF_CACHE on first run
python benchmark_mlx/bench_tts_mlx.py --model qwen3
python benchmark_mlx/bench_tts_mlx.py --model fish
python benchmark_mlx/bench_tts_mlx.py --model miso
```

Key components of the inference loop are ported: (prompt construction, autoregression, codec decode) while leveraging weights and `nn.Module` definitions from mlx-audio.
pytests verify parity of the ported inference loop with reference `mlx-audio` implementations, on both generated audio and wall-clock generation time.

```bash
uv pip install -e ".[dev]"
pytest -v benchmark_mlx/tests/test_tts_mlx.py
```
