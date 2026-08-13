# dataprep MLX backends (deprecated)

**Status: deprecated, experimental, unmaintained.** Kept for reference and for
reproducing the numbers in `experiments/expresso_nll_entropy/`. Do not build new
work on these.

| Backend | Module     | Codebooks | Frame rate | Status                                  |
| ------- | ---------- | --------- | ---------- | --------------------------------------- |
| `qwen3` | `qwen3.py` | 16        | 12.5 Hz    | experimental, known correctness gap      |
| `fish`  | `fish.py`  | 10        | 21 Hz      | experimental, unverified                 |

The maintained dataprep backend is `dataprep/miso.py` (PyTorch).

## Why these are deprecated

- **MLX-only.** Every tensor here is `mlx.core`, so they run on Apple Silicon
  only. The pipeline is moving to PyTorch, and these were never ported.
- **Not verified.** `test_featurize.py` guards its "confidently wrong" check to
  `miso` only, because qwen3 codebook 0 trips it — NLL 4.01 nats against a
  predictive entropy of 0.97. That gap is the signature of logits being scored
  against the wrong targets or of the teacher-forced pass leaking future
  context. It was never chased down. Fish was never audited for the same class
  of bug.
- **Feature drift.** They don't support `bucket_frames` (the MPS memory
  workaround `load_tokenizer` applies for miso), and they have not been exercised
  against the streaming `shard_prepare` path — only the per-row `inspect` path
  they were written for.

Because of the correctness gap, treat any NLL/entropy figures these produce as
indicative only. The published comparison in `experiments/expresso_nll_entropy/`
carries the same caveat.

## Isolation

These modules are deliberately kept off every default path so that ongoing
pipeline work never has to account for them:

- Imported lazily, only inside `dataprep.pipeline.load_tokenizer`, and only when
  the model is `qwen3` or `fish`. Calling either emits a `DeprecationWarning`.
- Their tests are marked `deprecated` and skipped by the default `addopts`.
  Nothing in a normal `pytest` run touches them.
- `dataprep/tests/test_shards.py`, the shard pipeline, and `train/` are miso-only.

If a pipeline change breaks these backends, that is expected: fix them only if
you intend to revive them, otherwise let them fail.

## Running them anyway

```bash
uv pip install -e ".[dataprep-mlx]"
```

```bash
python -m dataprep.cli inspect --model qwen3 --rows 3
```

```bash
pytest -v -m deprecated dataprep/tests/
```

## Reviving one

1. Port the tokenize/featurize path off `mlx.core` onto torch, following
   `dataprep/miso.py`.
2. Fix the codebook-0 scoring gap above and drop the `model == "miso"` guard in
   `dataprep/tests/test_featurize.py` so the excess-NLL assertion covers it.
3. Add `bucket_frames` support, then exercise it through `shard_prepare` rather
   than only through `inspect`.
4. Re-record the fixtures under `dataprep/tests/fixtures/segment0/expected/` and
   drop the `deprecated` marks.
