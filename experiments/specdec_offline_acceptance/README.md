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

Measured on the [Expresso](https://huggingface.co/datasets/Zackh/expresso-contextual) dataset rows 0–9: 457 sequences, 93,272 frames (~3700s). Vocab size is 6562 FSQ + EOS tokens.

## Models

The target is  [Chatterbox AR](https://github.com/resemble-ai/chatterbox) (Llama-style backbone, ~503M, EN-only model). Three drafts, all sharing FSQ code tokens from the [S3 tokenizer V2](https://github.com/xingchensong/S3Tokenizer):

-  [Flash](https://huggingface.co/ResembleAI/chatterbox-flash) — block-diffusion dLLM, ~503M backbone. The speech stream is cut into blocks of `B`; each cumulative block conditions a parallel prediction of the next block of mask tokens, simulating blockwise autoregressive inference (see `dump_logits.py`). Near-identical in size to the target, so there is no compute win — it is here to read accept rates over FSQ tokens between two models sharing a tokenizer.
-  [Turbo](https://huggingface.co/ResembleAI/chatterbox-turbo)  — causal, GPT-2 medium, **362M** backbone (0.72× the target).
-  [Nano](https://huggingface.co/ResembleAI/chatterbox-nano)  — causal, GPT-2 small, **130M** backbone (0.26× the target).

## Results

All drafts on the same frames, against a Chatterbox-AR target NLL of 4.356 nats/frame; `Δ` is the draft's excess over it. τ(γ) extrapolates accepted length to draft block size `γ`; for Flash, γ below its dump block `B` is blanked, since it cannot draft a narrower block.


| Draft     | α (accept rate) | τ(γ=2) (accept length) | τ(γ=4) (accept length) | Draft NLL | NLL Δ vs Target |
| --------- | --------------- | ---------------------- | ---------------------- | --------- | --------------- |
| Flash B=1 | 0.769           | 2.36                   | 3.16                   | 4.419     | +0.063          |
| Flash B=2 | 0.630           | 2.03                   | 2.44                   | 4.826     | +0.470          |
| Flash B=4 | 0.500           | —                      | 1.94                   | 5.304     | +0.949          |
| Flash B=8 | 0.385           | —                      | —                      | 5.837     | +1.481          |
| Turbo     | 0.646           | 2.06                   | 2.51                   | 4.751     | +0.395          |
| Nano      | 0.614           | 1.99                   | 2.36                   | 4.891     | +0.535          |


**Flash acceptance falls off sharply with block size,** and τ peaks at B=2 — widening the block past that costs acceptance faster than it gains positions. B=1 is the control: it matches autoregressive token-by-token prediction. Turbo and Nano are plain causal dumps with no block to sweep.

**Acceptance tracks draft quality (NLL).** Turbo models the audio better than Nano (NLL 4.75 vs 4.89) and is accepted more often (α 0.65 vs 0.61). All drafts are slightly worse than the target's own 4.356.

**Takeaway.** Nano looks like a reasonable draft for the size on offer: at 0.26× the backbone it still accepts α ≈ 0.61, which is where a real speedup comes from. Flash's near-target α at B=1 buys nothing, since it is the same size as the target — a block-diffusion draft would only pay off if trained as a draft against a smaller backbone. This is an offline read of acceptance stats, not an end-to-end speedup measurement.

## Caveats

- **One denoising step per block.** Cbox-Flash is trained to run up to 10 unmasking steps per block of 16 tokens at inference. Here we are only measuring acceptance after one step - which is a lower bound. Effective draft models like DFlash for LLMs are explicitly trained with an objective matching the inference setting, weighting earlier tokens higher.
- **Teacher-forced prefix.** Each block conditions on ground-truth history, not on what the previous block actually drafted.
- **Fixed audio prompts.** Every chatterbox model requires an audio prompt for the reference speaker. Test above conditions on one reference utterance per speaker, checked in under `expresso_speaker_prompts/` (`ex01` cut from row 7, `ex02` from row 8) as Expresso is the same two speakers throughout.



## Reproduce

You need extra `uv pip install -e ".[dataprep-chatterbox]"`  (from repo root).

Then from the repo root, `--model` selects the target/draft to dump, `--draft` selects the draft to score:

```bash
python experiments/specdec_offline_acceptance/dump_logits.py --rows 10 --model ar
python experiments/specdec_offline_acceptance/dump_logits.py --rows 10 --model flash --block-sizes 1,2,4,8
python experiments/specdec_offline_acceptance/dump_logits.py --rows 10 --model turbo   # and nano
python experiments/specdec_offline_acceptance/simulate_acceptance.py --rows 10 --draft turbo
```

Tokens come from `data/DATASET/tokenized/chatterbox/ROW/`; logits land in `specdec_offline/chatterbox-{ar,flash,turbo,nano}/ROW/`. Results write to `results/chatterbox_{bN,turbo,nano}.json` (gitignored).

[Qwen3-TTS results](qwen3.md) cover the same measurement on a multi-codebook model.