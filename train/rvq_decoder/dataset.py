"""WebDataset-backed iterable dataset for frame-level audio codec token prediction.

Each WDS sample (produced by dataprep.shards) contains:
  hiddens.npy  — float16 (F, hidden_dim)
  targets.npy  — int16   (F, num_codebooks)
  meta.json    — provenance dict
  kv.npy       — float16 (F, layers, 2, heads, kv_dim)  [optional]

``FramePackingIterableDataset`` accumulates audio frames from consecutive
sequences and yields fixed-size batches without padding or masking.  With
``drop_last=False`` the final batch may hold fewer than ``batch_frames`` rows.
The yielded dict has:
  hiddens:  (batch_frames, hidden_dim)   float32
  targets:  (batch_frames, num_codebooks) int64
  semantic: (batch_frames,)              int64   == targets[:, 0]
  kv:       (batch_frames, layers, 2, heads, kv_dim)  float32  [if use_kv=True]

Usage::

  from train.rvq_decoder.dataset import FramePackingIterableDataset
  ds = FramePackingIterableDataset("data/sharded_wds/DATASET/train", batch_frames=2048)
  # OR  ds = FramePackingIterableDataset("data/sharded_wds/DATASET/train/miso_train_{00000..00012}.tar", batch_frames=2048)

  for batch in ds:
      loss = model(batch["hiddens"], batch["semantic"], batch["targets"])

The default is a training stream: shuffled and resampled, so it never ends —
stop it on a step budget. Pass ``eval_mode=True`` for a single ordered pass over
the split instead.
"""

from __future__ import annotations

import io
import json
import os
import warnings
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.utils.data


class FramePackingIterableDataset(torch.utils.data.IterableDataset):
    """Packs audio frames from WDS sequences into fixed-size batches.

    Args:
        source: shard dataset directory (e.g. ``data/sharded_wds/<slug>/train``) containing 
            ``*.tar`` files. Also accepts a native WebDataset brace pattern like
            ``path/to/dataset-{000000..000054}.tar``.
        batch_frames: number of audio frames per yielded batch.
        shuffle_buffer: number of WDS samples to shuffle over (0 = no shuffle).
        resampled: if True, shards are sampled with replacement (infinite stream,
            good for training). If False, each shard is visited once (for validation).
        use_kv: if True, expect and include ``kv.npy`` in each batch.
        drop_last: if True (default), trailing frames that do not fill a whole
            batch are discarded. Set False for single-pass evaluation, where
            dropping up to ``batch_frames - 1`` frames would silently shrink the
            split; the final batch is then smaller than ``batch_frames``.
        eval_mode: if True, ignore the three arguments above and score the split
            exactly once, in shard order: no shard shuffle, no sample buffer, no
            resampling, and every trailing frame kept. This is the only
            configuration whose metrics are comparable between runs.
    """

    def __init__(
        self,
        source,
        *,
        batch_frames: int = 2048,
        shuffle_buffer: int = 1000,
        resampled: bool = True,
        use_kv: bool = False,
        drop_last: bool = True,
        eval_mode: bool = False,
    ):
        super().__init__()
        self.source = source
        self.shard_urls = _resolve_shard_urls(source)
        self.batch_frames = batch_frames
        self.use_kv = use_kv
        self.eval_mode = eval_mode
        # A single clean pass; the shuffle/resample/drop_last arguments are
        # whatever the caller asked for only outside eval_mode.
        self.shuffle_buffer = 0 if eval_mode else shuffle_buffer
        self.resampled = False if eval_mode else resampled
        self.drop_last = False if eval_mode else drop_last

    def __iter__(self) -> Iterator[dict]:
        pipeline = _wds_pipeline(
            self.shard_urls,
            shuffle_buffer=self.shuffle_buffer,
            resampled=self.resampled,
        )

        buf_h: list[np.ndarray] = []  # list of (F_i, H) float16
        buf_t: list[np.ndarray] = []  # list of (F_i, K) int16
        buf_kv: list[np.ndarray] = []  # list of (F_i, ...) float16
        buf_frames = 0

        for sample in pipeline:
            if "hiddens.npy" not in sample or "targets.npy" not in sample:
                continue

            h = _decode_npy(sample["hiddens.npy"])   # (F, H) float16
            t = _decode_npy(sample["targets.npy"])   # (F, K) int16

            if self.use_kv:
                if "kv.npy" not in sample:
                    warnings.warn(
                        f"use_kv=True but sample {sample.get('__key__')} has no kv.npy; skipping.",
                        stacklevel=2,
                    )
                    continue
                kv = _decode_npy(sample["kv.npy"])  # (F, layers, 2, heads, kv_dim)
                buf_kv.append(kv)

            buf_h.append(h)
            buf_t.append(t)
            buf_frames += h.shape[0]

            while buf_frames >= self.batch_frames:
                yield self._drain(buf_h, buf_t, buf_kv)
                buf_frames -= self.batch_frames

        if not self.drop_last and buf_frames > 0:
            yield self._drain(buf_h, buf_t, buf_kv, n=buf_frames)

    def _drain(
        self,
        buf_h: list[np.ndarray],
        buf_t: list[np.ndarray],
        buf_kv: list[np.ndarray],
        *,
        n: int | None = None,
    ) -> dict:
        n = self.batch_frames if n is None else n
        h_cat = np.concatenate(buf_h, axis=0)  # (total, H)
        t_cat = np.concatenate(buf_t, axis=0)  # (total, K)

        batch_h = torch.from_numpy(h_cat[:n].astype(np.float32))  # (n, H)
        batch_t = torch.from_numpy(t_cat[:n].astype(np.int64))    # (n, K)

        # Rewrite buffers with remainder.
        rem_h = h_cat[n:]
        rem_t = t_cat[n:]
        buf_h.clear()
        buf_t.clear()
        if rem_h.shape[0]:
            buf_h.append(rem_h)
            buf_t.append(rem_t)

        out = {
            "hiddens": batch_h,
            "targets": batch_t,
            "semantic": batch_t[:, 0],
        }

        if self.use_kv:
            kv_cat = np.concatenate(buf_kv, axis=0)  # (total, layers, 2, heads, dim)
            out["kv"] = torch.from_numpy(kv_cat[:n].astype(np.float32))
            rem_kv = kv_cat[n:]
            buf_kv.clear()
            if rem_kv.shape[0]:
                buf_kv.append(rem_kv)

        return out


def make_dataloader(
    source,
    *,
    batch_frames: int = 2048,
    shuffle_buffer: int = 1000,
    resampled: bool = True,
    use_kv: bool = False,
    drop_last: bool = True,
    eval_mode: bool = False,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """Convenience wrapper: returns a DataLoader ready for accelerate.prepare()."""
    dataset = FramePackingIterableDataset(
        source,
        batch_frames=batch_frames,
        shuffle_buffer=shuffle_buffer,
        resampled=resampled,
        use_kv=use_kv,
        drop_last=drop_last,
        eval_mode=eval_mode,
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=None,   # dataset yields full batches already
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
    )


def _resolve_shard_urls(source) -> str | list[str]:
    """Normalize ``source`` into something ``wds.WebDataset`` accepts.

    A directory (``data/sharded_wds/<slug>/train``) becomes the sorted list of
    ``*.tar`` files it holds; anything else — a brace/glob pattern, a single tar
    path, or an explicit list — is passed through untouched.
    """
    if not isinstance(source, (str, os.PathLike)):
        return list(source)

    path = Path(source)
    if not path.is_dir():
        # A brace/glob pattern or a single tar is the caller's business; a plain
        # path that simply is not there is a typo worth catching before the
        # model loads, rather than a webdataset error several seconds later.
        if path.suffix != ".tar" and not any(c in str(source) for c in "{*?["):
            raise FileNotFoundError(f"no shard directory at {source}")
        return str(source)

    shards = sorted(str(p) for p in path.glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"no .tar shards under {path}")
    return shards


def _decode_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


def _wds_pipeline(shard_urls: str | list[str], *, shuffle_buffer: int, resampled: bool):
    """Build a webdataset pipeline that yields raw sample dicts.

    Accepts either a list of *.tar file paths or a native WebDataset brace pattern like
    ``path/to/dataset-{000000..000054}.tar``.
    """
    import webdataset as wds

    shardshuffle = 100 if shuffle_buffer > 0 else False
    dataset = wds.WebDataset(shard_urls, resampled=resampled, shardshuffle=shardshuffle)
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(shuffle_buffer)
    return dataset

