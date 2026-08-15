"""Small helpers shared by the dataprep records.

Array coercion between numpy and torch, the two padding/key conventions the
records rely on, and the NLL scoring applied to a featurized sequence. No
dataclasses live here -- see ``dataprep.types``, which imports this module, so
type references back to it stay under ``TYPE_CHECKING``.
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from dataprep.types import FeaturizedSequence

NATS_TO_BITS = 1.0 / math.log(2.0)


def bucket_length(length: int, bucket_frames: int) -> int:
    """Round ``length`` up to a multiple of ``bucket_frames`` (0 disables)."""
    if bucket_frames <= 0:
        return length
    blocks = -(-length // bucket_frames)  # ceil
    return blocks * bucket_frames


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    if type(value).__module__.startswith("mlx."):
        import mlx.core as mx

        if str(value.dtype) == "mlx.core.bfloat16":
            value = value.astype(mx.float32)
        mx.eval(value)
        return np.asarray(value)
    return np.asarray(value)


def _as_torch(value: Any):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if (
        type(value).__module__.startswith("mlx.")
        and str(value.dtype) == "mlx.core.bfloat16"
    ):
        import mlx.core as mx

        value = value.astype(mx.float32)
    return torch.from_numpy(np.asarray(value)).cpu()


def _as_torch_tree(value: Any):
    if isinstance(value, dict):
        return {key: _as_torch_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_torch_tree(item) for item in value]
    if value is None:
        return None
    return _as_torch(value)


_KEY_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")


def _wds_key(seq_id: str) -> str:
    """Sanitize a segment id into a WebDataset sample key."""
    key = _KEY_UNSAFE.sub("_", seq_id)
    if not key:
        raise ValueError(f"segment id {seq_id!r} sanitizes to an empty WDS key")
    return key


def audio_frame_metrics(
    features: FeaturizedSequence, tokens: Any, num_codebooks: int
) -> dict[str, Any]:
    """Per-frame predictive entropy and ground-truth NLL (both nats) per codebook.

    Returns ``(F, num_codebooks)`` torch tensors for ``entropy`` / ``nll`` plus the
    audio-frame ``positions`` (F,), concatenated over the audio spans of
    ``features``. ``tokens`` is the source ``(L, C+1)`` array. The ground-truth
    column for each head comes from ``layout.head_targets``, so a semantic LM head
    (Fish head 0) is scored against the semantic token rather than the audio code.
    """
    from dataprep.types import TokenSpanKind

    tokens = _as_numpy(tokens)
    columns = features.layout.head_targets
    entropy_parts: list[Any] = []
    nll_parts: list[Any] = []
    position_parts: list[Any] = []
    for span in features.spans_of(TokenSpanKind.AUDIO):
        pred = features.feature_slice_for_targets(span.start, span.end)
        targets = torch.as_tensor(tokens[span.start : span.end], dtype=torch.long)
        entropy_cb, nll_cb = [], []
        for index in range(num_codebooks):
            # _as_numpy, not np.asarray: in-memory MLX logits may be bfloat16.
            logits = torch.as_tensor(_as_numpy(features.logits[index][pred])).float()
            log_probs = torch.log_softmax(logits, dim=-1)
            target = targets[:, columns[index]]
            entropy_cb.append(-(log_probs.exp() * log_probs).sum(dim=-1))
            nll_cb.append(-log_probs.gather(1, target[:, None]).squeeze(1))
        entropy_parts.append(torch.stack(entropy_cb, dim=1))
        nll_parts.append(torch.stack(nll_cb, dim=1))
        position_parts.append(torch.arange(span.start, span.end, dtype=torch.int32))
    if not entropy_parts:
        raise ValueError("Sequence has no audio spans")
    return {
        "entropy": torch.cat(entropy_parts, dim=0),
        "nll": torch.cat(nll_parts, dim=0),
        "positions": torch.cat(position_parts, dim=0),
    }


def nll_summary(nll: Any, frame_rate: float) -> dict[str, dict[str, float]]:
    """Teacher-forced CE for the ``semantic`` / ``audio`` / ``total`` code groups.

    ``nll`` is ``(frames, num_codebooks)`` in nats. Each group reports
    ``avg_nll_per_codebook`` — the NLL averaged over both frames and the group's
    codebooks, so it is the per-codebook cost of one frame and stays comparable
    across models with different codebook counts — plus ``num_codebooks`` for
    that group and the bitrate it implies::

        kbits_per_second = avg_nll * log2(e) * frame_rate * num_codebooks / 1000

    Multiplying the count back in makes kbit/s the group's whole share of the
    stream, so ``semantic`` and ``audio`` kbit/s sum to ``total``.
    """
    nll = _as_numpy(nll)
    groups = {
        "semantic": nll[:, :1],
        "audio": nll[:, 1:],
        "total": nll,
    }
    summary = {}
    for name, values in groups.items():
        count = int(values.shape[1])
        avg_nll = float(values.mean()) if count else 0.0
        summary[name] = {
            "avg_nll_per_codebook": avg_nll,
            "num_codebooks": count,
            "kbits_per_second": avg_nll * NATS_TO_BITS * frame_rate * count / 1000.0,
        }
    return summary
