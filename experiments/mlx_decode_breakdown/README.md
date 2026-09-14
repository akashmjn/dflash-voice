# MLX Inference breakdown

We measure per-codec-frame decode times of OSS TTS models ranging from 0.5B (Chatterbox) to 8B (CSM/Miso) parameters. This is broken down by semantic backbone (an autoregressive LLM predicting semantic codes) and audio decoder (a smaller model predicting the remaining audio codes).

We see that overhead from repeated forward passes of 140-400M param audio decoders takes up more than 50% of inference time, inspite of heavier semantic backbones (0.6B - 8B).

![MLX decode breakdown chart](./assets/mlx_decode_breakdown.png)


| Model            | Frame rate | Wall RTF | Audio decoder % | Total ms | Semantic backbone (ms) | Audio decoder (ms) | Decoder iterations | ms / iterations | Frames/s |
| ---------------- | ---------- | -------- | --------------- | -------- | ---------------------- | ------------------ | ------------------ | --------------- | -------- |
| Qwen3 1.7B 8bit  | 12.5 Hz    | 0.42     | 65%             | 29.2     | 9.6                    | 19.1               | 15                 | 1.27            | 34.2     |
| Voxtral 4B 6bit  | 12.5 Hz    | 0.77     | 72%             | 59.3     | 15.9                   | 42.9               | 7                  | 6.13            | 16.8     |
| MisoTTS 8B 8bit  | 12.5 Hz    | 1.53     | 74%             | 110.4    | 29.0                   | 81.3               | 31                 | 2.62            | 9.1      |
| Qwen3 0.6B 8bit  | 12.5 Hz    | 0.38     | 71%             | 25.7     | 7.0                    | 18.2               | 15                 | 1.21            | 38.9     |
| Fish S2 Pro 8bit | 21 Hz      | 1.15     | 47%             | 44.5     | 23.5                   | 21.0               | 9                  | 2.33            | 22.5     |
| Chatterbox 8bit  | 25 Hz      | 0.47     | 0%              | 9.4      | 9.0                    | 0.0                | -                  | -               | 106.9    |


6 prompts per model, 8-bit MLX checkpoints (Voxtral 6-bit), on M1 Max Apple Silicon. **Wall RTF** = end-to-end time / audio duration, including codec decode (`mean_rtf`); <1 is faster than realtime. **Audio decoder %** is the audio decoder's share of total per-frame time.

**Decoder iterations** counts the sequential passes through the audio decoder per frame: RVQ codebooks after the semantic one for the autoregressive models, and flow-matching Euler steps (`n_denoising_steps - 1`) for Voxtral, whose single pass emits every acoustic codebook at once.

Chatterbox (`cbox-ar` in the bench harness) is the control case: a 0.5B single-codebook backbone with no audio decoder at all. Note it is the English-only Chatterbox variant (704-token text vocabulary), not the 23-language build.

## Reproduce

Use `[mlx_decode](../../mlx_decode/README.md)` benchmark harness. Run from the repo root:

```bash
uv pip install -e ".[mlx_decode]"
# downloads models to HF_CACHE on first run
python mlx_decode/bench.py bench --model all
# the 0.6B Qwen3 row
python mlx_decode/bench.py bench --model qwen3 --model-id mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit
# reprint the table above from saved metrics.json files
python mlx_decode/bench.py summarize
```

Then regenerate the chart above from the per-model `metrics.json`. Matplotlib is
declared inline in the script, so it needs no project extra:

```bash
uv run experiments/mlx_decode_breakdown/make_chart.py
```

Audio-decoder parameter counts in the chart are counted off the loaded MLX weights and cover the compute path each depth iteration re-reads: Qwen3 `code_predictor`, Voxtral `acoustic_transformer`, Fish `fast_layers/norm/output`, Miso `decoder + projection + audio_head`. Codebook embedding tables are excluded -- the summed embedding only reaches the backbone once the frame is complete.