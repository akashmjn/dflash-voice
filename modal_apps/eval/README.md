# modal_apps/eval/

Zero-shot TTS evals on Modal GPUs.

## Seed-TTS test-en

[`cbox-ar-seedtts.py`](cbox-ar-seedtts.py) — Chatterbox TTS (AR), scored with SIM-o / WER / UTMOS. Modelled on [scripts/run_eval.sh](https://github.com/resemble-ai/chatterbox-flash/blob/master/scripts/run_eval.sh) from resemble-ai/chatterbox-flash, cut to stage 3 (Seed-TTS test-en); the scoring
half still uses that repo's model-agnostic `eval.wer_seedtts`.

```bash
modal run modal_apps/eval/cbox-ar-seedtts.py --limit 100 --tag 0917-1800-cbox_ar
modal run modal_apps/eval/cbox-ar-seedtts.py --tag TAG                 # 1088 rows
modal run modal_apps/eval/cbox-ar-seedtts.py --stage score --tag TAG --limit 100
```

`download` (cpu) → `generate` (L4) → `score` (L4), over the persistent
`seedtts-eval-09-2026` volume, so a failure in one doesn't re-pay for the others.
L4 rather than dataprep's T4: Turing has no bf16.

See the module docstring for the volume layout and per-run knobs.

### Gotchas

- **The `download/` directory name is load-bearing.** The k2-fsa JSONLs hard-code
  `ref_audio` under a `download/` prefix, resolved against that dir's parent.
- **Scorers resolve `ref_audio` against their CWD** — omnivoice's `sim.py` uses
  `sample["ref_audio"]` verbatim — so `score` runs them with `cwd=/vol`.
- **Scorer summary lines vary by version**: this omnivoice prints `SIM-o score:` /
  `UTMOS score:`, not the `Average ...` lines `run_eval.sh` greps.

### Results

100 rows, L4, generate defaults:

| run | model | SIM-o | WER | UTMOS | s/utt | RTF |
| --- | --- | --- | --- | --- | --- | --- |
| `0917-1800-cbox_ar` | Chatterbox TTS (AR) | 0.579 | **1.09%** | 4.03 | 3.97 | 1.17 |
| `0917-1730-cbox_flash` | Chatterbox-Flash | 0.682 | 49.98% | 3.83 | 1.59 | 0.30 |

Reproducing generation for the Flash model apprears buggy, though those results can be safely ignored as we don't need this model.
