# Exploring efficient audio generation 

Local inference of popular open-source TTS/omni models is bottlenecked (> 50%) by repeated forward passes of smaller models to generate audio tokens vs larger LLM-based backbones that handle semantics/prosody.

![MLX decode breakdown chart](./experiments/mlx_decode_breakdown/assets/mlx-decode-breakdown.png)

This project contains ongoing explorations on two directions to speed this up:
1. Discrete: Speculative decoding over audio tokens on a single-codebook TTS model [Chatterbox](https://github.com/resemble-ai/chatterbox). We look at acceptance rates for various off-the-shelf target/draft pairings, and explore a principled way to relax acceptance criteria to handle nuances of audio tokens.
2. Continuous: [MeanFlows](https://arxiv.org/abs/2505.13447) distillation of velocity fields from the [Voxtral TTS](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) audio token flow decoder, reducing NFEs (number of function evaluations) from 14 to 2 for upto 2x faster local inference (ongoing).

The exploration helps understand the critical role of audio tokenization in both. Only single-codebook FSQ models are straightforward to implement speculative decoding over, while MeanFlows only applies to models with flow-based decoders generating multiple FSQ codes. Models with multi-layer RVQ tokenizers can't be used with either. Ultimately, we see why continuous-based approaches are better positioned for local efficiency - as seen in [Pocket TTS](https://arxiv.org/abs/2509.06926).

To reproduce the benchmark above, you will need an Apple Silicon laptop with MLX support. The `dev` branch contains work in progress on dataprep, finetuning models and other experiments under `experiments/`.

More to come here soon. Feel free to [connect/reach me](https://akashmjn.me/) if you've any thoughts!
