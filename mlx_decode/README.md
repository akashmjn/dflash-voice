## Lightweight vendored/ported TTS inference for benchmarking

This module ports the MLX TTS inference loop from [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.4 to a hackable, 
single-file module - allowing us to benchmark semantic backbone vs audio decoder times.

```bash
uv pip install -e ".[mlx_decode]"
# downloads models to HF_CACHE on first run
# see models.yaml for available models
python mlx_decode/bench.py bench --model <MODEL|all>
python mlx_decode/bench.py summarize --model <MODEL|all>
```

### Usage for Chatterbox Seed-TTS evals

```bash
python -m dataprep.cli fetch --dataset seedtts --rows 100
python mlx_decode/bench.py bench --model MODEL \
  --prompts-file data/seedtts/prompts/seedtts_test_en_100.jsonl \
  --ref-audio-dir data/seedtts/ref_audio -n 100 \
  --output-dir OUTPUT_DIR
```



### Notes

RVQ models like qwen3/fish/miso have separate semantic/audio token decoders. Inference times for other models are reported as follows:
- `voxtral`: `depth_audio` timings cover flow-matching Euler denoising steps instead of per-codebook decodes.
- `cbox-*`: Chatterbox models report zero `depth_audio` since it is built on a flat single-codebook tokenizer decoded by the backbone.

