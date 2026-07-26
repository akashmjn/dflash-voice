## Lightweight vendored/ported TTS inference for benchmarking

This module ports the MLX TTS inference loop from [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.4 to a lightweight, readable single-file module.
This helps easily understand various key components (prompt construction, autoregression, codec decode) while leveraging weights and `nn.Module` definitions from mlx-audio.

pytests verify parity with reference `mlx-audio` implementations, on both generated audio and wall-clock generation time.

```bash
pytest -v tts_mlx/tests/test_tts_mlx.py
```
