# modal_apps/eval/

Quick TTS evals on Modal GPUs.

## Seed-TTS test-en

`[seedtts_eval.py](seedtts_eval.py)` scores SIM-o / WER / UTMOS, following [scripts/run_eval.sh](https://github.com/resemble-ai/chatterbox-flash/blob/master/scripts/run_eval.sh) from resemble-ai/chatterbox-flash in stages: `download` → `generate` (PyTorch Chatterbox AR baseline) or `upload` (predecoded wavs) → `score` → `fetch`. See its docstring for the Volume layout.

```bash
# Baseline: decodes prompts with Chatterbox AR baseline before scoring
modal run modal_apps/eval/seedtts_eval.py --limit 500 --tag TAG   # baseline; omit --limit for all 1088 rows

# Scoring predecoded audio
modal run modal_apps/eval/seedtts_eval.py --stage upload --limit 500 --tag TAG \
  --wav-dir agent-workspace/mlx_decode_output/seedtts-500/DECODE_SLUG \
  --test-list data/seedtts/prompts/seedtts_test_en_100.jsonl
```



## Results

`fetch` runs after `score`, copying the run's `scored/` logs to [results/](results/)`SPLIT/TAG/` (`SPLIT` e.g. `seedtts_en-limit100`).

```bash
modal run modal_apps/eval/seedtts_eval.py --stage fetch --limit 500 --tag TAG  # re-fetch an older run
python modal_apps/eval/summarize.py [SPLIT ...]   # -> results/SPLIT/summary.csv
```

Results below are on a 500-row subset of the full 1088-row test set used in `k2-fsa/TTS_eval_datasets`. We can closely reproduce Table 1 in [paper](https://arxiv.org/pdf/2605.30748), except for an unexplained gap in `sim_o` of 0.685 (original) vs 0.594 when rerunning `model.generate` from the [pypi chatterbox-tts](https://github.com/resemble-ai/chatterbox-flash/blob/74e05baa8ce574bf2cc571702391a21f1b0d48c5/pyproject.toml#L34) package.


| **tag**                          | **sim_o** | **wer_pct** | **utmos** |
|:---------------------------------|:---------:|:-----------:|:---------:|
| **ground_truth**                 |  0.734    |    2.42     |   3.52    |
| chatterbox_tts_500M-reported     |  0.685    |    2.20     |   4.10    |
| pypi-chatterbox_tts_500M-repro   |  0.594    |    2.40     |   4.04    |
| mlxaudio-chatterbox_tts_fp16_500M|  0.579    |    2.29     |   4.00    |


### Gotchas

- **The** `download/` **directory name is hard-coded.** The k2-fsa eval JSONLs hard-code `ref_audio` under this prefix.
- **Scorers resolve** `ref_audio` **against their CWD**, so `score` runs them with `cwd=/vol`.
- **Scorer summary lines vary by version**: this omnivoice prints `SIM-o score:` / `UTMOS score:`, not the `Average ...` lines `run_eval.sh` greps.
