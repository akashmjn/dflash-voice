# dflash-voice

The goal of this project is to speed up TTS inference, starting with RVQ (residual vector quantization) audio codec generation. This forms a surprisingly large bottleneck (i.e. orange bars below), complicating inference especially when running locally.

![TTS MLX benchmark aggregate](docs/benchmark-time-per-frame.png)

## Current state

Readable MLX TTS inference and benchmarking for **Qwen3-TTS** and **Fish Audio S2 Pro**, ported from [mlx-audio](https://github.com/Blaizzy/mlx-audio) 0.4.4.

Inference loop is ported to understand key components and benchmark per-frame timing (backbone_semantic vs depth_audio). Weights and `nn.Module` definitions still come from mlx-audio; `tts_mlx` owns prompt construction, autoregression, and codec decode in annotated, single-file modules.

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

# Compare output of ported inference loop against mlx-audio reference
pytest tests/test_*_generate.py
```

## Benchmark results (per-codec-frame breakdown)

6 prompts per model, 8-bit MLX checkpoints, 64GB M1 Max Apple Silicon. **Gen RTF** = codec-frame generation speed vs native frame rate; **Wall RTF** = end-to-end including codec decode.

Real-time budgets: Qwen3 @ 12.5 Hz → 80 ms/frame; Fish @ 21 Hz → 47.6 ms/frame.

| Model | Native | Backbone (semantic codes) | Depth (audio codes) | Depth % | Depth iters | ms / depth iter | Total ms | Codec frames/s | Gen RTF | Wall RTF |
|-------|--------|---------------------------|---------------------|---------|-------------|-----------------|----------|----------------|---------|----------|
| Qwen3 1.7B 8bit | 12.5 Hz | 9.4 ms | 17.0 ms | 63% | 15 | 1.13 ms | 26.9 ms | 37.2 | 2.98× | 2.52× |
| Fish S2 Pro 8bit | 21 Hz | 21.9 ms | 20.4 ms | 48% | 9 | 2.26 ms | 42.3 ms | 23.7 | 1.13× | 0.92× |

Raw metrics: (gitignored — regenerate with `benchmark_mlx/bench_tts_mlx.py`).

## Why 

This started after noticing an expensive memory bottleneck for audio tokens mentioned in the [Sesame CSM blog post](https://www.sesame.com/blog/crossing-the-uncanny-valley-of-voice). Why should audio tokens be comparably expensive to predict vs language tokens? Especially given the lower information density.

![TTS MLX benchmark aggregate](docs/entropy-per-codebook.png)

Taking a closer look at the predictive entropy over 32 [Mimi](https://huggingface.co/kyutai/mimi) RVQ codebooks for the [MisoTTS](https://github.com/MisoLabsAI/MisoTTS) depth audio decoder (essentially an 8B repro of the CSM model) confirms this. The first 7 codebook tokens have quite low entropy/information content - as low as 0.75 bits. Do we really always need 32x300M param forward passes?

![TTS MLX benchmark aggregate](docs/entropy-per-frame.png)

From an information theory lens: we are seeing variable rate of information density, both across depth (RVQ audio codes) and across time (semantic codes). Most modern TTS models (e.g. Qwen3 TTS, Fish Audio S2) have converged to a autoregressive backbone and depth models predicting both.

Given what we've seen above, and inspired by speculative decoding and flow matching, it would be nice to get more bang for buck per model compute. Why not spend less compute on the easy stuff?

More to come here soon. Feel free to reach me [here](https://akashmjn.me/) if you've thoughts here!

> P.S.: repo naming was originally motivated by speculative decoding methods like [DFlash](https://github.com/z-lab/dflash) for TTS models. However turns out specdec for TTS is complicated by the dual-RVQ (semantic + audio) codec structure used by most models. The current focus is on a narrower bottleneck: speeding up/simplifying audio codec generation to begin. Will rename appropriately :)

## Citation

If you use this repository, please cite:

```bibtex
@misc{mahajan2026dflashvoice,
  title        = {dflash-voice: Speeding up RVQ audio codec generation for TTS},
  author       = {Mahajan, Akash},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/akashmjn/dflash-voice}
}
```

### Related work

```bibtex
@misc{sesame2024csm,
  title        = {Crossing the uncanny valley of voice},
  author       = {{Sesame}},
  year         = {2024},
  howpublished = {Blog post},
  url          = {https://www.sesame.com/blog/crossing-the-uncanny-valley-of-voice}
}

@article{Qwen3-TTS,
  title   = {Qwen3-TTS Technical Report},
  author  = {Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal = {arXiv preprint arXiv:2601.15621},
  year    = {2026}
}

@misc{liao2026fishaudios2technical,
  title         = {Fish Audio S2 Technical Report},
  author        = {Shijia Liao and Yuxuan Wang and Songting Liu and Yifan Cheng and Ruoyi Zhang and Tianyu Li and Shidong Li and Yisheng Zheng and Xingwei Liu and Qingzheng Wang and Zhizhuo Zhou and Jiahua Liu and Xin Chen and Dawei Han},
  year          = {2026},
  eprint        = {2603.08823},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SD},
  url           = {https://arxiv.org/abs/2603.08823}
}

@techreport{kyutai2024moshi,
  title       = {Moshi: a speech-text foundation model for real-time dialogue},
  author      = {Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and Am\'elie Royer and Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
  year        = {2024},
  eprint      = {2410.00037},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS},
  url         = {https://arxiv.org/abs/2410.00037}
}
```
