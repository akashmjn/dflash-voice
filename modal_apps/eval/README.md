# modal_apps/eval/

Quick TTS evals on Modal GPUs.

## Seed-TTS test-en

[`seedtts_eval.py`](seedtts_eval.py) scores SIM-o / WER / UTMOS, following [scripts/run_eval.sh](https://github.com/resemble-ai/chatterbox-flash/blob/master/scripts/run_eval.sh) from resemble-ai/chatterbox-flash in stages: `download` → `generate` (PyTorch Chatterbox AR baseline) or `upload` (predecoded wavs) → `score` → `fetch`. See its docstring for the Volume layout.

```bash
# Baseline: decodes prompts with Chatterbox AR baseline before scoring
modal run modal_apps/eval/seedtts_eval.py --limit 100 --tag TAG   # baseline; omit --limit for all 1088 rows

# Scoring predecoded audio
modal run modal_apps/eval/seedtts_eval.py --stage upload --limit 100 --tag TAG \
  --wav-dir agent-workspace/mlx_decode_output/seedtts-100/DECODE_SLUG \
  --test-list data/seedtts/prompts/seedtts_test_en_100.jsonl
```

## Results

`fetch` runs after `score`, copying the run's `scored/` logs to [results/](results/)`SPLIT/TAG/` (`SPLIT` e.g. `seedtts_en-limit100`).

```bash
modal run modal_apps/eval/seedtts_eval.py --stage fetch --limit 100 --tag TAG  # re-fetch an older run
python modal_apps/eval/summarize.py [SPLIT ...]   # -> results/SPLIT/summary.csv
```

### Gotchas

- **The `download/` directory name is hard-coded.** The k2-fsa eval JSONLs hard-code `ref_audio` under this prefix.
- **Scorers resolve `ref_audio` against their CWD**, so `score` runs them with `cwd=/vol`.
- **Scorer summary lines vary by version**: this omnivoice prints `SIM-o score:` / `UTMOS score:`, not the `Average ...` lines `run_eval.sh` greps.
