# dflash-voice: Accelerating RVQ audio codec generation

The goal of this project is to speed up TTS and multimodal voice LLM inference, starting with RVQ (residual vector quantization) audio codec generation. This forms a surprisingly large bottleneck — RVQ audio-code (depth decoder) generation dominates the per-codec-frame cost — complicating inference especially when running locally.

See section [Why](#why) below for more details on the motivation and in-progress approach being explored to fixing this.


## MLX inference benchmark breakdown

A per-codec-frame timing breakdown across Qwen3, Fish, and Miso 8-bit MLX checkpoints shows depth-decoder RVQ audio-code generation dominating each frame's cost.

> For the full results table and chart, see [experiments/mlx_decode_breakdown](experiments/mlx_decode_breakdown/README.md).

See [mlx_decode](mlx_decode/README.md) for the benchmark harness and how to reproduce these results. You will need an Apple Silicon laptop with MLX support.


## Why

> To reproduce these results, see [experiments/expresso_nll_entropy](experiments/expresso_nll_entropy/README.md).

This started after noticing an expensive memory bottleneck for audio tokens mentioned in the [Sesame CSM blog post](https://www.sesame.com/blog/crossing-the-uncanny-valley-of-voice). Why should audio tokens be comparably expensive to predict vs language tokens? Especially given the lower information density.

Taking a closer look at the predictive entropy over 32 [Mimi](https://huggingface.co/kyutai/mimi) RVQ codebooks for the [MisoTTS](https://github.com/MisoLabsAI/MisoTTS) depth audio decoder (8B repro of the CSM model) confirms this. The first 7 codebook tokens have quite low entropy/information content - as low as 0.75 bits. Do we really always need 32x300M param forward passes to generate 32 RVQ audio tokens?

![Entropy vs RVQ codebook depth](docs/MisoCSM-codebook-entropy.png)

From an information theory lens: there is clearly a varying rate of information density, both across depth (RVQ audio codebooks - first plot) and across time (codec frames - plotted below). Most modern TTS models (e.g. Qwen3 TTS, Fish Audio S2) have converged to an autoregressive 1-4B LLM `backbone_semantic` predicting semantic codes across time and smaller 100-400M `depth_audio` decoders predicting audio RVQ codebooks across depth.

Given what we've seen above, and inspired by speculative decoding and flow matching, it would be nice to get more bang for buck per model forward pass. Why not spend less compute on the easy stuff?

Specifically, I am exploring both discrete and continuous approaches to generating RVQ audio tokens faster.

1. Discrete: block-diffusion inspired by speculative decoding methods like [DFlash](https://github.com/z-lab/dflash)
2. Continuous: single-step flow matching inspired by [MeanFlows](https://github.com/haidog-yaqub/MeanFlow)

More to come here soon. Feel free to [connect/reach me](https://akashmjn.me/) if you've any thoughts!

> P.S.: Speculative decoding for TTS is complicated by the dual-RVQ (semantic backbone + audio depth decoder) codec structure used by most SoTA models. So the initial project focus is on a narrower bottleneck to begin: training models to speed up/simplify RVQ audio codec generation. Will revisit/rename repo appropriately based on progress :)


## Citation

If you use this repository, please cite:

```bibtex
@misc{mahajan2026dflashvoice,
  title        = {dflash-voice: Accelerating RVQ audio codec generation for TTS},
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

