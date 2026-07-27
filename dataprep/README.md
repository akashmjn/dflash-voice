# Dataprep artifacts

The pipeline keeps source conversations model-independent and writes one
model-specific artifact directory per Expresso row.

```text
data/
  expresso/
    raw/ROW/
      audio.wav
      transcript_segments.json
      MODEL_codebooks.pt         # temporary per-model codec dump
    MODEL_tokenized/ROW/
      sequences.pt              # list[{tokens, mask}]
      metadata.json             # layout + spans per sequence
    MODEL_featurized/ROW/
      features.pt               # list[{logits, hiddens}] length L-1
      metadata.json
      kv_context.pt             # only with --dump-kv
    MODEL_entropy/ROW/          # from notebooks/entropy_compute.py
      entropy.npz
      entropy.json
```

`audio.wav` is channel-first when loaded by `dataprep.expresso`; transcript
times are seconds and channels are zero-based. Per-model `MODEL_codebooks.pt`
under `raw/` is a temporary codec dump (`(F, C)` per channel). `sequences.pt`
stores ragged `{tokens, mask}` entries shaped `(L, C+1)`. Layout
(`num_codebooks`, `text_channel`) and spans (`text` / `audio` / `special`) live
in `metadata.json` so consumers do not need model-specific framing rules.

Featurization replays each `TokenizedSequence` under teacher forcing.
`features.pt` stores ragged `{logits, hiddens}` of length `L-1`, where index
`i` predicts `tokens[i+1]`. Use audio spans to select supervised regions.

## Environments

MisoTTS pins Transformers 4.49 while the tested MLX stack pins Transformers
5.6 / `huggingface-hub` 1.5, so `dataprep` and `dataprep-mlx` (and `tts_mlx`)
conflict — install only one extra in a given environment:

```bash
# Miso stack
uv pip install -e ".[dataprep]"
# For local Miso development, replace the installed package:
uv pip install -e ../MisoTTS

# Qwen3 / Fish MLX stack (replaces the Miso pins above)
uv pip install -e ".[dataprep-mlx]"
```
