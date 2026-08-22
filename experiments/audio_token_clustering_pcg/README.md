# Clustering audio tokens for coarse-grained speculative decoding

Speculative decoding accepts a draft token only on an exact match against the target
model's distribution. For audio that test is too strict: a speech codebook entry is a
point in a continuous acoustic space, so neighbouring ids are often near-interchangeable
and a draft gets rejected over a difference nobody can hear.

[Principled Coarse-Graining (PCG)](https://arxiv.org/abs/2511.13732) verifies at the
level of *groups* of acoustically similar tokens rather than individual ids. Groups are
read straight off the target model's embedding table — one seeded per token,
`G(t) = {t' : cos(Emb(t), Emb(t')) > θ}` — so nothing is trained, and the single
threshold `θ` trades output fidelity for acceptance.

We measure how `θ` controls group size on [Chatterbox](https://huggingface.co/ResembleAI/chatterbox),
a 0.5B single-codebook TTS model, to find the usable range before building a full
speculative decoding loop. Too small and PCG reduces to ordinary speculative decoding;
too large and it accepts drafts that no longer sound like the target.

## Cluster size vs θ

Groups built from the `speech_emb` table of Chatterbox AR (vocab 6561, dim 1024). Bold
marks θ∈[0.38, 0.45], the band the paper reports as optimal for a 65k
[X-codec2](https://huggingface.co/HKUSTAudio/xcodec2) vocabulary.


| θ        | mean size | singleton frac | median | max     | % of vocab |
| -------- | --------- | -------------- | ------ | ------- | ---------- |
| 0.80     | 1.0       | 0.997          | 1      | 3       | 0.02%      |
| 0.70     | 1.3       | 0.840          | 1      | 8       | 0.02%      |
| 0.60     | 4.1       | 0.269          | 4      | 31      | 0.06%      |
| 0.50     | 12.9      | 0.032          | 12     | 75      | 0.20%      |
| **0.45** | **23.6**  | **0.009**      | **22** | **111** | **0.36%**  |
| **0.40** | **43.5**  | **0.003**      | **42** | **154** | **0.66%**  |
| 0.30     | 148.0     | 0.001          | 145    | 320     | 2.26%      |
| 0.20     | 504.5     | 0.000          | 513    | 966     | 7.69%      |


**Singleton frac** is the share of tokens grouped only with themselves — where PCG
provably reduces to ordinary speculative decoding. **% of vocab** is mean group size over
the 6561-token vocabulary, for comparison against other codecs. Full sweep in
`results/chatterbox-ar.json`.

**θ∈[0.4, 0.5] is the usable band.** At θ=0.40 a group holds 43.5 tokens, 0.66% of the  
vocabulary, with almost no singletons. Matching the paper's reported mean of ~140 tokens  
by vocabulary *fraction* lands near θ≈0.47, by absolute count near θ≈0.31. The curve turns  
sharply above θ=0.6 — 0.7→0.6→0.5 gives 1.3→4.1→12.9 — and above θ≈0.75 every group is a  
singleton, so the method costs construction and indexing to do nothing.

## One group table covers both Chatterbox variants

Speculative decoding needs draft and target to share a token vocabulary, and the groups
must describe both. We compared Chatterbox AR against
[Chatterbox Flash](https://huggingface.co/ResembleAI/chatterbox-flash), its faster finetuned sibling. Vocab embeddings have indeed been finetuned   
but it preserved the geometry: Gram cosine 0.9979, and group-size curves within 1% at every θ.

> [!NOTE] Groups are built over ids 0–6560, the trained speech codes. Ids 6561/6562 are   
> BOS/EOS and 6563–8193 appear to be untrained vocab padding.



## What this does not measure

Group size bounds what PCG can do; it does not show that it works. The open question is
**acceptance rate** — how often the coarse-grained test accepts a draft — which needs
draft and target logits over real frames, and then speech quality at the chosen θ to
confirm the relaxed distribution still sounds right.

## Reproduce

Needs `numpy` and `safetensors` only; checkpoints download to the HF cache on first run.
From the repo root:

```bash
python experiments/audio_token_clustering_pcg/pcg_chatterbox.py
```

Per-model group sizes land in `results/`. Flags: `--models` selects
`cbox-ar`/`cbox-flash`, `--thetas` sets the sweep points.