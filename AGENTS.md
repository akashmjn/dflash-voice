## Cursor Cloud specific instructions

This repo is a local TTS research CLI (`dflash-voice`). There is no web server. Install and run commands are in `README.md`, `mlx_decode/README.md`, and `dataprep/README.md`. Use the project virtualenv at `.venv` (`uv` is on `PATH`).

- On Linux, the `mlx` wheel does not include `libmlx.so`. Install the CPU backend with `uv pip install "mlx[cpu]"` (package `mlx-cpu`) after the project extras. `mlx.core` then defaults to `Device(cpu, 0)`. Synthesis of `mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit` takes about 11 seconds per 12.5 Hz frame (about 190× real time).
- Keep one extra set per environment. This VM uses `mlx_decode`, `dataprep-mlx`, and `dev`. `dataprep-miso` pins a different Transformers release; see `dataprep/README.md`.
- The `dataprep-mlx` extra installs the current PyTorch CUDA wheel. `torch.cuda.is_available()` is false here, and PyTorch ops used by dataprep fall back to CPU.
- The repo has no lint configuration. `python -m compileall mlx_decode dataprep demo experiments` checks syntax.
- `pytest` uses `-m 'not expensive'` and still collects `mlx_decode/tests/test_decode_parity.py`, which downloads multi-gigabyte checkpoints and times full utterances. On this CPU VM, run `pytest dataprep/tests/test_tokenize.py -k qwen3`. That test encodes the checked-in Expresso fixture with the 0.6B Qwen3 speech tokenizer.
- `dataprep/tests/test_featurize.py` teacher-forces about 297 frames as one lazy MLX graph. On a 16 GB CPU VM the process passed 6 GB RSS and was still growing, so run that test on a machine with more memory.
- A short bench of the 0.6B Qwen3 checkpoint, with a one-line JSONL file `{"text":"Hi."}`:

```bash
.venv/bin/python mlx_decode/bench.py bench --model qwen3 --model-id mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit -n 1 --prompts-file prompts.jsonl --no-warmup --save-audio
```

The default prompt list and warmup pass generate many more frames at the same CPU rate.
