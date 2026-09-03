# Bonus: Qwen3-TTS, 0.6B drafting 1.7B

The same offline acceptance measurement (see [README](README.md) for the method) applied to a
multi-codebook model, where Chatterbox's single axis becomes two.

[Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) drafting
[Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base). Both are
AR — there is no block masking here, so `τ` is reported across γ rather than at a block size.

| Axis                         | α         | τ(γ=1) | τ(γ=2) | τ(γ=3) | τ(γ=4) | τ(γ=5) | τ(γ=7) |
| ---------------------------- | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Semantic backbone (cb0)      | **0.758** | 1.76   | 2.33   | 2.77   | 3.10   | 3.35   | 3.69   |
| Audio depth decoder (cb1–15) | **0.659** | 1.66   | 2.09   | 2.38   | 2.57   | 2.69   | 2.83   |

Expresso rows 0–9: 465 sequences, 55,695 frames per axis (s.e. ≈ 0.002). Full output in
`results/qwen3.json`.

It is a two-stage RQ-transformer, giving two axes to speculate on: a 28-layer talker LM stepping
over *time* (codebook 0), and a 5-layer `code_predictor` stepping over *codebooks* 1–15 within
each frame. Both sizes load the same frozen codec (16 codebooks × 2048 entries @ 12.5 Hz), so a
token id means the same acoustic thing in each and no vocab remapping is needed.

**Acceptance is flat with depth.** Per-level α, cb1 → cb15:

| cb1   | cb2   | cb3   | cb4   | cb5   | cb6   | cb7   | cb8   | cb9   | cb10  | cb11  | cb12  | cb13  | cb14  | cb15  |
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
| 0.674 | 0.674 | 0.665 | 0.650 | 0.649 | 0.657 | 0.650 | 0.654 | 0.657 | 0.661 | 0.662 | 0.658 | 0.662 | 0.660 | 0.661 |

Caveats match the main writeup: teacher-forced logits (exact at γ=1, a prefix-matched
approximation beyond), and rows are unbalanced — row 7 alone contributes 179 of the 465 sequences.

## Reproduce

Needs teacher-forced dumps for both variants under `data/expresso/featurized/qwen3-{0.6b,1.7b}/`.
The dataprep backend that produced them is not part of the current pipeline; the script assumes
they exist.

```bash
python experiments/specdec_offline_acceptance/qwen3.py --axis both --rows 10
```

`--axis` selects `cb0`/`depth`/`both`, `--rows` sets how many dataset rows to pool.
