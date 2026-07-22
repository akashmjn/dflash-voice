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
```

`audio.wav` is channel-first when loaded by `dataprep.expresso`; transcript
times are seconds and channels are zero-based. `codebooks.pt` stores a list of
signed integer `(C, F)` tensors, one per source channel. `sequences.pt` stores
variable-length `(C+1, L)` tensors and `masks.pt` stores matching boolean
tensors. The final sequence row is the text lane for Miso and Qwen. Fish uses
its native layout: row 0 contains text or offset semantic IDs and rows 1–10
contain raw VQ IDs at audio positions.

`metadata.json` records sequence shapes and maps each text/audio span back to
the dataset row, speaker, channel, timestamps, and codec-frame range. Miso
adds an all-zero audio EOS frame; Qwen uses its configured codec EOS; Fish
framing comes from `Conversation.encode_for_inference`.

## Environments

MisoTTS pins Transformers 4.49 while the tested MLX stack pins Transformers
5.6, so they intentionally use separate uv environments:

```bash
uv sync --group dataprep --no-default-groups
# For local Miso development, replace the installed package:
uv pip install -e ../MisoTTS

uv sync --group dataprep-mlx --no-default-groups
```

## Prepare three rows

Run the single-segment Miso forward check before the complete Miso dataset:

```bash
DFLASH_RUN_MISO_FORWARD=1 uv run --group dataprep \
  pytest dataprep/tests/test_miso_forward.py -m integration
uv run --group dataprep python -m dataprep.prepare \
  --model miso --rows 0 1 2 --verify-decode
```

Then use a `dataprep-mlx` environment for Qwen3 and Fish:

```bash
uv run --group dataprep-mlx python -m dataprep.prepare \
  --model qwen3 --rows 0 1 2 --verify-decode
uv run --group dataprep-mlx python -m dataprep.prepare \
  --model fish --rows 0 1 2 --verify-decode
```

Downloaded and generated files are ignored by Git; this README is tracked.
