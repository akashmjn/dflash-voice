# Offline specdec acceptance ratio estimation

How well does [Chatterbox Flash](https://huggingface.co/ResembleAI/chatterbox-flash), a block-diffusion decoder, draft for [Chatterbox AR](https://huggingface.co/ResembleAI/chatterbox)? Both are single-codebook, so there is one token sequence to speculate on and no separate audio token decoder (RVQ models like Qwen3-TTS).

## Method

For efficiency, acceptance stats are simulated by running **one specdec step** from offline logit
dumps. Both models are teacher-forced over the same ground-truth sequence, so every frame `t`
yields a pair of next-token distributions for `x_t+1`. Sample from the draft, verify against the
target — accept `x` with probability `min(1, target(x) / draft(x))` — over all frames in parallel:

```
α = accepted tokens / frames
τ(γ) = (1 − α^(γ+1)) / (1 − α)
```

`τ(γ)` is the expected tokens per verify round at draft block `γ`, assuming acceptance is i.i.d.  
across the γ positions.



Chatterbox-Flash is a block diffusion dLLM: the speech stream is cut into blocks of `B`, and each cumulative speech block serves as context to predict a block of mask tokens in parallel - simulating blockwise autoregressive inference (see `chatterbox_dump_logits.py`).

> Note that Cbox-Flash (draft) and Cbox-AR (target) models near-identical in size, so there is no compute win on offer. We are getting a sense for what accept rates over FSQ audio tokens looks like for two models sharing the same tokenizer.



## Results

Acceptance rates for different masked block sizes `B` are relatively low, dropping off sharply with block size.


| Block `B` | α (accept rate) | τ(γ=B) (accept length) | Draft NLL | NLL Δ vs Target |
| --------- | --------------- | ---------------------- | --------- | --------------- |
| 1         | 0.763           | 1.76                   | 4.363     | +0.084          |
| 2         | 0.625           | 2.02                   | 4.780     | +0.500          |
| 4         | 0.498           | 1.93                   | 5.262     | +0.983          |
| 8         | 0.384           | 1.62                   | 5.786     | +1.507          |


Measured on Expresso dataset rows 0–9: 457 sequences, 93,272 frames. Vocab size is 6562 FSQ + EOS tokens. Target model (Cbox-AR) NLL is 4.279 nats/frame; `Δ` is the draft's excess over it. 

**B=1 is the control.** This closely matches autoregressive token-by-token predictions.

**τ peaks at B=2 and falls after.** Widening the block costs acceptance faster than it gains positions past B=2.

## Caveats

- **One denoising step per block.** Cbox-Flash is trained to run up to 10 unmasking steps per block of 16 tokens at inference. Here we are only measuring acceptance after one step - which is a lower bound. DFlash draft models for LLMs are explicitly trained with an objective matching the inference setting, weighting earlier tokens higher.
- **Teacher-forced prefix.** Each block conditions on ground-truth history, not on what the previous block actually drafted.



## Reproduce

Dataprep featurizes the AR checkpoint only (the `chatterbox` artifact), so Flash logits are
dumped separately into their own `chatterbox-flash` artifact, off the tokenized dump the two
share. That needs the `dataprep-chatterbox` extra; the acceptance script itself needs only
torch, on CPU.

```bash
python experiments/specdec_offline_acceptance/chatterbox_dump_logits.py --rows 10 --block-sizes 1,2,4,8
python experiments/specdec_offline_acceptance/simulate_acceptance.py --rows 10 --block-size 4
```

`--block-sizes` takes a CSV; each size writes `featurized/chatterbox-flash/ROW/features_bN.pt`,
which `--block-size` reads back into `results/chatterbox_bN.json` (gitignored).

Both scripts assume dataprep's `data/DATASET/{tokenized,featurized}/MODEL/ROW/` layout — AR under
`chatterbox`, Flash under `chatterbox-flash`, sharing one tokenized directory. See [dataprep](../../dataprep/README.md#on-disk-layout) for the layout.

[Qwen3-TTS results](qwen3.md) cover the same measurement on a multi-codebook model.