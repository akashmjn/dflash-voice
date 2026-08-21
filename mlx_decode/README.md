## Lightweight vendored/ported TTS inference for benchmarking

This module ports the MLX TTS inference loop from [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.4 to a hackable, 
single-file module - allowing us to benchmark semantic backbone vs audio decoder times.

```bash
uv pip install -e ".[mlx_decode]"
# downloads models to HF_CACHE on first run
python mlx_decode/bench.py bench --model <qwen3|voxtral|fish|miso|all>
```

`voxtral` is the odd one out: its audio decoder is a flow-matching acoustic head that emits every
acoustic codebook at once, rather than an autoregressive RVQ chain, so `depth_audio` timings there
cover Euler denoising steps instead of per-codebook decodes.

Key components of the inference loop are ported: (prompt construction, autoregression, codec decode) while leveraging weights and `nn.Module` definitions from mlx-audio.
pytests verify parity of the ported inference loop with reference `mlx-audio` implementations, on both generated audio and wall-clock generation time.

```bash
uv pip install -e ".[dev]"
pytest -v mlx_decode/tests/test_decode_parity.py
```
