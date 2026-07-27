# Dataprep artifacts

The pipeline keeps source conversations model-independent and writes one
model-specific artifact directory per Expresso row.

```text
data/
  expresso/
    raw/ROW/
      audio.wav
      transcript_segments.json
    MODEL_tokenized/ROW/
      codebooks.pt
      sequences.pt
      masks.pt
      metadata.json
    MODEL_featurized/ROW/
      logits.pt
      hiddens.pt
      metadata.json
      kv_context.pt  # only with --dump-kv
    MODEL_entropy/ROW/          # from notebooks/entropy_compute.py
      entropy.npz               # per-frame entropy + positions
      entropy.json              # means, spans, offsets into the npz
```

`audio.wav` is channel-first when loaded by `dataprep.expresso`; transcript
times are seconds and channels are zero-based. `codebooks.pt` stores a list of
signed integer `(F, C)` tensors, one per source channel. `sequences.pt` stores
variable-length `(L, C+1)` tensors and `masks.pt` stores matching boolean
tensors. The final sequence channel is the text lane for Miso and Qwen. Fish uses
its native layout transposed to seq-major form: channel 0 contains text or offset
semantic IDs and channels 1–10 contain raw VQ IDs at audio positions.

`metadata.json` records sequence shapes and maps each text/audio span back to
the dataset row, speaker, channel, timestamps, and codec-frame range. Miso
adds an all-zero audio EOS frame; Qwen uses its configured codec EOS; Fish
framing comes from `Conversation.encode_for_inference`.

Featurization replays each saved `TokenizedSequence` with ground-truth audio
tokens. `logits.pt` is a list of dictionaries keyed by codebook because the
semantic and residual vocabularies can differ. Hidden states and logits contain
only real audio target frames.

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
