# rvq_decoder_train

Unmaintained: the Miso RVQ depth-decoder experiment.

Model, checkpoint converter, and a small wall-clock trainer for the depth
decoder lifted from Sesame CSM-8B. Superseded by Chatterbox-Flash finetuning
and kept for reference; nothing else in the repo imports it.

`dataset.FramePackingIterableDataset` lives here rather than under `train/`
because it packs frames across sequence boundaries into flat batches with no
sequence axis -- the depth decoder's per-frame geometry. Finetuning work that
needs sequences wants a different loader.

Not a package: modules import each other as siblings, so run it from this
directory rather than as `python -m`. The trainer is `trainer.py`, not
`train.py` -- the latter would be shadowed by the repo-root `train/` package
whenever the repo root is on `sys.path`.

```bash
cd hacks/rvq_decoder_train
python cli.py convert
python cli.py eval --checkpoint ../../tmp/miso_depth_decoder.safetensors
python cli.py train --preset smoke --minutes 5
pytest tests/
```

Needs the `train` extra (`uv pip install -e ".[train]"`), and shards under
`data/sharded_wds/` from `python -m dataprep.cli prepare`. Paths default to
repo-root-relative, so pass `--data-root ../../data` when running from here.
