"""Unmaintained: the Miso RVQ depth-decoder experiment.

Model, checkpoint converter, and a small wall-clock trainer for the depth
decoder lifted from Sesame CSM-8B. Superseded by Chatterbox-Flash finetuning
and kept for reference; nothing else in the repo imports it.

``dataset.FramePackingIterableDataset`` belongs here rather than under ``train``
because it packs frames across sequence boundaries into flat batches with no
sequence axis -- the depth decoder's per-frame geometry. Finetuning work that
needs sequences wants a different loader.
"""

from __future__ import annotations

DEPRECATION_NOTE = (
    "train.rvq_decoder is unmaintained: the Miso RVQ depth-decoder experiment, "
    "superseded by Chatterbox-Flash finetuning."
)

__all__ = ["DEPRECATION_NOTE"]
