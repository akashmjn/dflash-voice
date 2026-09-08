# Estimating specdec acceptance stats offline

What do specdec acceptance rates look like for TTS models that decode audio tokens?

Here we look the  [Chatterbox](https://github.com/resemble-ai/chatterbox) model family, whose autoregressive backbones decode a single stream of FSQ tokens. This is the closest to an LLM making it the most tractable for specdec, unlike multi-layer RVQ models like Qwen3-TTS with dual autoregressive decoders.

## Method

For efficiency, acceptance stats are simulated by running **one specdec step** from offline logit dumps. Both models are  
teacher-forced over the same ground-truth sequence, so every frame `t` yields a pair of next-token distributions for `x_t+1`. We then run a specdec draft + verification step over all frames in parallel - accepting `x` with probability `min(1, target(x) / draft(x))` .

```
α = accepted tokens / total frames
τ(γ) = (1 − α^(γ+1)) / (1 − α)
```

`τ(γ)` is the expected tokens per specdec step extrapolated from `α` to draft block size `γ` (assuming i.i.d. acceptance).

Measured on the [Expresso](https://huggingface.co/datasets/Zackh/expresso-contextual) dataset rows 0–9: 457 sequences, 93,272 frames (~3700s). Vocab size is 6561 FSQ + 1 EOS token.

## Models

The target is  [Chatterbox AR](https://github.com/resemble-ai/chatterbox) (Llama-style backbone, ~503M, EN-only model). Three drafts, all sharing FSQ code tokens from the [S3 tokenizer V2](https://github.com/xingchensong/S3Tokenizer):

- [Flash](https://huggingface.co/ResembleAI/chatterbox-flash) — block-diffusion dLLM, ~503M backbone. The speech stream is cut into blocks of `B`; each cumulative block conditions a parallel prediction of the next block of mask tokens, simulating blockwise autoregressive inference (see `dump_logits.py`). Near-identical in size to the target, so there is no compute win — it is here to read accept rates over FSQ tokens between two models sharing a tokenizer.
- [Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)  — causal, GPT-2 medium, **362M** backbone (0.72× the target).
- [Nano](https://huggingface.co/ResembleAI/chatterbox-nano)  — causal, GPT-2 small, **130M** backbone (0.26× the target).

## Results

Comparing all drafts on the same frames, against a 500M param Chatterbox-AR target with an NLL of 4.356 nats/frame. Flash refers to a single step of block-diffusion model at varying masked block sizes.

*(these are with off-the-shelf model as draft, a dedicated draft model would presumably offer a better tradeoff)*


| Draft        | α (accept rate) | τ(γ=2) (accept length) | τ(γ=4) (accept length) | Draft NLL | NLL Δ vs Target |
| ------------ | --------------- | ---------------------- | ---------------------- | --------- | --------------- |
| Flash B=1    | 0.769           | 2.36                   | 3.16                   | 4.419     | +0.063          |
| Flash B=2    | 0.630           | 2.03                   | 2.44                   | 4.826     | +0.470          |
| Flash B=4    | 0.500           | —                      | 1.94                   | 5.304     | +0.949          |
| Flash B=8    | 0.385           | —                      | —                      | 5.837     | +1.481          |
| Turbo (350M) | 0.646           | 2.06                   | 2.51                   | 4.751     | +0.395          |
| Nano (130M)  | 0.614           | 1.99                   | 2.36                   | 4.891     | +0.535          |


**Flash needs special training to be a reasonable draft:** acceptance falls off sharply with block size as the model is trained to run upto 10 unmasking steps per block of 16 tokens - unlike DFlash draft models trained with an objective prioritizing early parallel proposals in a single step.

**Takeaway.** Off-the-shelf nano looks like a reasonable draft at 0.26× the backbone it still accepts α ≈ 0.61.

Using coarse-grained acceptance criteria based on audio token clusters using the PCG method, we see:


| Draft               | α (accept rate) | τ(γ=2) (accept length) | τ(γ=4) (accept length) |
| ------------------- | --------------- | ---------------------- | ---------------------- |
| Turbo (350M) θ=0.6  | 0.682           | 2.15                   | 2.68                   |
| Turbo (350M) θ=0.45 | 0.716           | 2.23                   | 2.86                   |
| Nano (130M) θ=0.6   | 0.654           | 2.08                   | 2.54                   |
| Nano (130M) θ=0.45  | 0.689           | 2.16                   | 2.72                   |


This yields slightly higher accept rates of 0.65-0.69. See [audio token clustering (PCG)](../audio_token_clustering_pcg/) for more details. This is a 35-40% speedup at gamma=2. We'll verify speedup and audio quality in a full decoding implementation next.

## Caveats

- **Teacher-forced prefix.** Each block conditions on ground-truth history, not on what the previous block actually drafted. This provides a quick estimate.
- **Fixed audio prompts.** Every chatterbox model requires an audio prompt for the reference speaker. Tests above use one reference utterance per speaker, checked in under `expresso_speaker_prompts/` (`ex01` cut from row 7, `ex02` from row 8) as Expresso is the same two speakers throughout.

## Reproduce

You need extra `uv pip install -e ".[dataprep-chatterbox]"`  (from repo root).

Then from the repo root, `--model` selects the target/draft to dump, `--draft` selects the draft to score:

```bash
python experiments/specdec_offline_acceptance/dump_logits.py --rows 10 --model ar
python experiments/specdec_offline_acceptance/dump_logits.py --rows 10 --model flash --block-sizes 1,2,4,8
python experiments/specdec_offline_acceptance/dump_logits.py --rows 10 --model turbo   # and nano
python experiments/specdec_offline_acceptance/simulate_acceptance.py --rows 10 --draft turbo
python experiments/specdec_offline_acceptance/simulate_acceptance.py --rows 10 --draft turbo --theta 0.6
```

Tokens come from `data/DATASET/tokenized/chatterbox/ROW/`; logits land in `specdec_offline/chatterbox-{ar,flash,turbo,nano}/ROW/`. Results write to `results/chatterbox_{bN,turbo,nano}.json` (gitignored).

`--theta` switches to coarse-grained verification, reading the ASG clusters
`experiments/audio_token_clustering_pcg/cluster.py` dumps — run that first with
`--dump-thetas` covering the θ you want to score at. `--cluster-results` and
`--cluster-model` point at a different dump tree or model.

Rough runtimes for 10 rows (M1 Max MacBook, PyTorch MPS backend): ~8 min for AR, ~4 min each for Turbo and Nano, and ~25 min for Flash across all four block sizes. `simulate_acceptance.py` is cheap: runs in well under 30s.