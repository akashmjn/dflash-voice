# Clustering audio tokens

Are audio tokens materially different from text tokens?

[Principled Coarse-Graining (PCG) (Apple, 2026)](https://arxiv.org/abs/2511.13732) showed that unlike text, audio tokens are interchangeable with minimal downstream impact. This means that in a speculative decoding setup, tokens are rejected over a difference that is perceptually inaudible.

Here we implement the clustering algorithm and the token-swapping ablation on [Chatterbox](https://huggingface.co/ResembleAI/chatterbox) - a 0.5B TTS model with a single ~6.5k size FSQ token vocab.

### Method

Acoustic similarity groups (ASG) are created by clustering the target model's embedding table — `G(t) = {t' : cos(Emb(t), Emb(t')) > θ}`. At decoding, target/draft verification is modified so that exactness of the sampled token distribution is guaranteed at the level of groups (ASG) rather than individual tokens. See `experiments/specdec_offline_acceptance` for more.

The single threshold `θ` trades output fidelity for acceptance - in the limit `θ=1` each group consists of each vocab token, reducing to ordinary speculative decoding.

### Cluster size vs θ

Groups are built from the `speech_emb` table of Chatterbox AR [vocab 6561 (excluding eos), dim 1024]. Bold marks θ∈[0.45], the band the paper reports as optimal for a 65k [X-codec2](https://huggingface.co/HKUSTAudio/xcodec2) vocabulary.


| θ        | mean size | singleton frac | median | max     | % of vocab |
| -------- | --------- | -------------- | ------ | ------- | ---------- |
| 0.80     | 1.0       | 0.997          | 1      | 3       | 0.02%      |
| 0.70     | 1.3       | 0.840          | 1      | 8       | 0.02%      |
| 0.60     | 4.1       | 0.269          | 4      | 31      | 0.06%      |
| 0.50     | 12.9      | 0.032          | 12     | 75      | 0.20%      |
| **0.45** | **23.6**  | **0.009**      | **22** | **111** | **0.36%**  |
| 0.40     | 43.5      | 0.003          | 42     | 154     | 0.66       |
| 0.30     | 148.0     | 0.001          | 145    | 320     | 2.26%      |
| 0.20     | 504.5     | 0.000          | 513    | 966     | 7.69%      |


Full sweeps are saved to `results/summary/chatterbox-ar.json`.

> NOTE: Groups are built over ids 0–6560, the trained speech codes. Ids 6561/6562 are  
> BOS/EOS and 6563–8193 appear to be untrained vocab padding.

## Audio Token Swapping

We reproduce the paper's token-swapping ablation here, which shows that at θ=0.60, the swapped audio **sounds indistinguishable from the unswapped baseline despite 76.9% of tokens changing.**


| θ        | mean group size | singleton frac | max group size | tokens swapped |
| -------- | --------------- | -------------- | -------------- | -------------- |
| **0.60** | **4.1**         | **0.269**      | **31**         | **76.9%**      |
| 0.45     | 23.6            | 0.009          | 111            | 94.9%          |
| 0.30     | 148.0           | 0.001          | 320            | 98.5%          |


An utterance is generated with the Chatterbox AR model, and every token belonging to a group of size > 1 is replaced by a uniform draw from its group. Replacement stats above are reported on six utterances of 1,068 frames total.

The same audio prompt (speaker embedding, context) is used when detokenizing back to audio with the S3Gen the swap is the only thing separating the two renders.

## Reproduce

Cluster generation needs `numpy` and `safetensors`; checkpoints download to the HF cache on first run. From the repo root:

```bash
uv pip install -e ".[mlx_decode]"
python experiments/audio_token_clustering_pcg/cluster.py --dump-thetas 0.6,0.45,0.3
```

The script writes the θ sweep to `results/summary/MODEL.json`, and ASG token clusters to
`results/token_clusters/MODEL/thetaTT.json` — one file per θ, since a mapping runs to
megabytes and nothing needs more than one threshold at a time.

The swapping ablation runs as below:

```bash
python experiments/audio_token_clustering_pcg/token_swap.py --thetas 0.6,0.45,0.3
```

Each utterance gets its own directory under `results/token_swap/`, holding both audio (wav)
and token sequences (npy); the tree is gitignored. `demo/` carries prompt_004's unswapped
render next to its θ=0.60 swap, as a checked-in before/after pair.

## Reusing the saved clusters

(see `experiments/specdec_offline_acceptance/coarse_acceptance.py` for usage)

Each θ gets its own file, `results/token_clusters/MODEL/thetaTT.json`. There is one group per token in the 6561-token speech vocabulary, stored as a sparse adjacency list:

```json
{"model": "cbox-ar", "theta": 0.45, "vocab": 6561,
 "groups": [[0, 3, 6, 9, 27, ...], ...], "owner_counts": [...]}
```

`groups[k]` is `G(k)`, listing the tokens acoustically similar to `k` (including `k`
itself, since cosine is reflexive). In the θ=0.45 dump `groups[0]` holds 20 ids and mean
group size is 23.6, over 154,915 members total.

Groups overlap, so a token belongs to several. `owner_counts[t]` is how many groups hold
token `t`. It is a count *down* the columns, so unlike the groups themselves it cannot be read off a single row. 
