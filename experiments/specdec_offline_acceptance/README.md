# Speculative decoding acceptance

Speculative decoding is only worth it if a small draft model's tokens survive verification by the target often enough. We measure the acceptance ratio `α` — how often a sampled draft token is accepted — and the expected tokens per verify round it implies:

```
τ(γ) = (1 − α^(γ+1)) / (1 − α)
```

for a draft block of `γ` tokens, assuming acceptance is i.i.d. across the γ positions. `γ` is not measured; it is the block size you would pick, and `τ` is what that choice yields at the measured `α`.

Measured **offline** against pre-dumped teacher-forced logits, so at every frame the draft `p` and target `q` share a ground-truth prefix.

## Qwen3-TTS: 0.6B drafting 1.7B

[Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) drafting [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base).   
  
> Note: this is just an offline simulation to get a feel for acceptance rates on audio tokens. No real compute/efficiency benefits as both models (especially audio decoders) are v similar size.


| Axis                         | α         | τ(γ=1) | τ(γ=2) | τ(γ=3) | τ(γ=4) | τ(γ=5) | τ(γ=7) |
| ---------------------------- | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Semantic backbone (cb0)      | **0.762** | 1.76   | 2.34   | 2.79   | 3.12   | 3.38   | 3.73   |
| Audio depth decoder (cb1–15) | **0.660** | 1.66   | 2.10   | 2.38   | 2.57   | 2.70   | 2.83   |


Every audio-bearing sequence in Expresso rows 0–9: 465 sequences, 55,695 frames per axis (s.e. ≈ 0.002). Full output in `results/qwen3.json`.

**Acceptance is flat with depth.** Per-level α, cb1 → cb15:


| cb1   | cb2   | cb3   | cb4   | cb5   | cb6   | cb7   | cb8   | cb9   | cb10  | cb11  | cb12  | cb13  | cb14  | cb15  |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.674 | 0.674 | 0.665 | 0.650 | 0.649 | 0.657 | 0.650 | 0.654 | 0.657 | 0.661 | 0.662 | 0.658 | 0.662 | 0.660 | 0.661 |


Both load the same frozen codec (16 codebooks × 2048 entries @ 12.5 Hz), so a token id means the same acoustic thing in each and no vocab remapping is needed. It is a two-stage RQ-transformer, giving two axes to speculate on: a 28-layer talker LM stepping over *time* (codebook 0), and a 5-layer `code_predictor` stepping over *codebooks* 1–15 within each frame.

## Caveats

- **Teacher-forced logits.** Real speculative decoding re-runs the target on the *drafted* prefix. Exact at γ=1, a prefix-matched approximation beyond it.
- **τ is idealized.** Acceptance is mildly correlated across depth levels, so the i.i.d. assumption behind the closed form does not hold exactly.
- **One draft/target pair, 10 rows of one dataset.** Rows are unbalanced — row 7 alone contributes 179 of the 465 sequences.



## Reproduce

Needs teacher-forced logit dumps for both variants under `data/expresso/featurized/qwen3-{0.6b,1.7b}/`. The dataprep backend that produced them is not part of the current pipeline; the script assumes they exist.

```bash
python experiments/specdec_offline_acceptance/acceptance_qwen3.py --axis both --rows 10
```

Flags: `--axis` selects `cb0`/`depth`/`both`, `--rows` sets how many dataset rows to pool.