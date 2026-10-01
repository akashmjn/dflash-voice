# modal_apps/eval/

Quick TTS evals on Modal GPUs.

## Seed-TTS test-en

Scores SIM-o / WER / UTMOS. Modelled on [scripts/run_eval.sh](https://github.com/resemble-ai/chatterbox-flash/blob/master/scripts/run_eval.sh) from resemble-ai/chatterbox-flash, cut to stage 3 (Seed-TTS test-en).

```bash
modal run modal_apps/eval/seedtts_eval.py.py --limit 100 --tag TAG
modal run modal_apps/eval/seedtts_eval.py.py --limit 100 --stage score --tag TAG 
```

See the module docstring for the volume layout and per-run knobs.

## Scoring MLX decode results run locally

```bash
modal run modal_apps/eval/seedtts_eval.py.py --stage upload --limit 100 --tag TAG \
--wav-dir agent-workspace/mlx_decode_output/seedtts-100/DECODE_SLUG \
--test-list data/seedtts/prompts/seedtts_test_en_100.jsonl
```

### Gotchas

- **The `download/` directory name is hard-coded.** The eval JSONL files in k2-fsa repo hard-code `ref_audio` under this prefix.
- **Scorers resolve `ref_audio` against their CWD** — omnivoice's `sim.py` uses
  `sample["ref_audio"]` verbatim — so `score` runs them with `cwd=/vol`.
- **Scorer summary lines vary by version**: this omnivoice prints `SIM-o score:` /
  `UTMOS score:`, not the `Average ...` lines `run_eval.sh` greps.

