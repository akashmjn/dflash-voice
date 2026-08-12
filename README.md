# dflash-voice: Exploring specdec and block-diffusion for TTS

Blockwise diffusion language models are at a flash point - with [DFlash](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/#dflash-parallel-drafting-with-kv-injection) seeing impressive 6+ token speculative decoding acceptance lengths  on the latest LLMs (DeepseekV4, Qwen3.5, Inkling). Parallel canvas-style generation makes a perfect fit for fast, high-interactivity models (e.g. voice, function calling etc).

Like with text, tokenization for audio is tricky but critical, as leading OSS TTS models need to generate upwards of 200+ tokens/s. Tokenizer design i.e. RVQ/FSQ and single-codebook/multi-layer-codebook greatly influences applicability of specdec and block-diffusion parallel audio generation.

This repo contains some early explorations/experiments:
1. [Motivation](./experiments/expresso_nll_entropy/): We pick a few TTS models (Qwen3, Fish S2, CSM/Miso) and find they compress audio down to 1.4-2.2 kbit/s. Information density is unevenly distributed, motivating parallel generation of low-information tokens to increase model throughput.
2. [MLX inference breakdown](./experiments/mlx_decode_breakdown/): We see that repeated forward passes of 100-300M param RVQ audio decoders take up more than 50% of inference time, inspite of heavier LLM backbones (1.7B - 8B).

Repo also contains code to reproduce above analyses - you will need an Apple Silicon laptop with MLX support.

More to come here soon. Feel free to [connect/reach me](https://akashmjn.me/) if you've any thoughts!
