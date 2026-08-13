"""Deprecated MLX dataprep backends: Qwen3-TTS and Fish S2.

Unmaintained and experimental. These were the MLX-based tokenize/featurize
backends used for the one-off NLL/entropy comparison in
``experiments/expresso_nll_entropy/``; ``miso`` is the maintained (PyTorch)
backend that the shard pipeline and training path actually use.

See ``README.md`` in this directory for status, the known correctness gap, and
what it would take to revive them. Imports are lazy on purpose -- nothing here
is imported unless ``load_tokenizer`` is called with ``qwen3`` or ``fish``, so
these modules cannot break a pipeline run or a default test run.
"""

from __future__ import annotations

DEPRECATION_NOTE = (
    "dataprep.mlx_backends ({model}) is deprecated and unmaintained: MLX-only, "
    "not verified against the current pipeline, and excluded from the default "
    "test run. The maintained dataprep backend is 'miso'. "
    "See dataprep/mlx_backends/README.md."
)

__all__ = ["DEPRECATION_NOTE"]
