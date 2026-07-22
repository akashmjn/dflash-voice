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
signed integer `(F, C)` tensors, one per source channel. `sequences.pt` stores
variable-length `(L, C+1)` tensors and `masks.pt` stores matching boolean
tensors. The final sequence channel is the text lane for Miso and Qwen. Fish uses
its native layout transposed to seq-major form: channel 0 contains text or offset
semantic IDs and channels 1–10 contain raw VQ IDs at audio positions.

`metadata.json` records sequence shapes and maps each text/audio span back to
the dataset row, speaker, channel, timestamps, and codec-frame range. Miso
adds an all-zero audio EOS frame; Qwen uses its configured codec EOS; Fish
framing comes from `Conversation.encode_for_inference`.

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

## Prepare (debug)

`--debug` prepares the first N Expresso rows (default 3) and writes the current
per-row raw/tokenized intermediates. Full-dataset parquet export is not
implemented yet and requires omitting `--debug`.

Run the single-segment Miso forward check before a debug prepare (Miso env):

```bash
DFLASH_RUN_MISO_FORWARD=1 pytest dataprep/tests/test_miso_forward.py -m integration
python -m dataprep.prepare --model miso --debug --verify-decode
```

Then switch to the MLX dataprep extra for Qwen3 and Fish:

```bash
uv pip install -e ".[dataprep-mlx]"
python -m dataprep.prepare --model qwen3 --debug --verify-decode
python -m dataprep.prepare --model fish --debug --verify-decode
```

Downloaded and generated files are ignored by Git; this README is tracked.
