"""Pack featurized sequences into WebDataset tar shards.

Each shard holds ~``samples_per_shard`` sequences. Per sample:
  {key}.hiddens.npy  — float16 (F, hidden_dim)   audio-span hiddens only
  {key}.targets.npy  — int16   (F, num_codebooks) audio codebook tokens
  {key}.meta.json    — lightweight provenance
  {key}.logits.npy   — float16 (F, num_codebooks, vocab)  optional
  {key}.kv.npy       — float16 (F, layers, 2, heads, kv_dim)  optional

``shard_prepare`` takes a ``DecodedExample`` stream and drives tokenize/featurize
itself, so the forward pass runs only for rows it writes. The dataset is never
held in memory: the train/val split is a per-row hash and shards roll over as
they fill.

Samples are written in the order they arrive. That is what keeps a run
resumable -- where a sample lands depends only on ``(row, seq_id)``, and
buffered shuffle is left to the training dataloader.

Library module: the command line lives in ``dataprep.cli``.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch

from dataprep.common import (
    FeaturizedSequence,
    ShardSample,
    SpanKind,
    TokenizedSequence,
    _as_numpy,
)
from dataprep.expresso import DATASET_NAME, DecodedExample


DATA_ROOT = Path("data")
SHARD_DIRNAME = "sharded_wds"


class ShardWriter:
    """Write samples to numbered tar shards, rolling over every N samples.

    This avoids needing the full sample list up front, keeping only the
    open shard, so a run's memory does not grow with the dataset.
    """

    def __init__(self, output_dir: Path, *, shard_prefix: str, samples_per_shard: int):
        self.output_dir = output_dir
        self.shard_prefix = shard_prefix
        self.samples_per_shard = samples_per_shard
        self.shard_index: list[dict] = []
        self._sink = None
        self._shard_num = 0
        self._in_shard = 0
        self._shard_frames = 0
        output_dir.mkdir(parents=True, exist_ok=True)

    def write(self, sample: ShardSample) -> None:
        import webdataset as wds

        if self._sink is None:
            path = self.output_dir / f"{self.shard_prefix}_{self._shard_num:05d}.tar"
            self._sink = wds.TarWriter(str(path))
            self._path = path
        self._sink.write(sample.to_wds())
        self._in_shard += 1
        self._shard_frames += sample.audio_frames
        if self._in_shard >= self.samples_per_shard:
            self._close_shard()

    def _close_shard(self) -> None:
        if self._sink is None:
            return
        self._sink.close()
        self.shard_index.append(
            {
                "shard": self._path.name,
                "num_samples": self._in_shard,
                "total_frames": self._shard_frames,
            }
        )
        self._sink = None
        self._shard_num += 1
        self._in_shard = 0
        self._shard_frames = 0

    def close(self) -> None:
        self._close_shard()

    @property
    def num_samples(self) -> int:
        return sum(entry["num_samples"] for entry in self.shard_index)

    @property
    def total_frames(self) -> int:
        return sum(entry["total_frames"] for entry in self.shard_index)


def shard_prepare(
    raw_examples: Iterable[DecodedExample],
    *,
    model: str,
    tokenizer,
    wds_root: Path,
    split_ratio: float = 0.95,
    samples_per_shard: int = 250,
    include_kv: bool = False,
    include_logits: bool = False,
    shuffle_seed: int = 42,
    shuffle_buffer: int = 0,
    log_root: Path | None = None,
    pack_segments: bool = False,
    progress: bool = True,
    skip_rows: set[int] | None = None,
) -> dict:
    """Tokenize, featurize and shard a stream of decoded rows.

    Owns the compute rather than consuming ready-made samples, so tokenize and
    featurize run per sample written and ``skip_rows`` can drop a row before it
    costs a forward pass.

    Nothing scales with dataset size: the train/val split is a per-row hash and
    shards are written as samples arrive.

    Sample order is deterministic by default -- shard position is a function of
    ``(row, seq_id)``. ``shuffle_buffer`` trades that for write-time mixing; see
    :func:`shuffle_stream`.

    ``wds_root`` is the shard directory itself (see :func:`shard_root`).
    """
    from tqdm import tqdm

    from dataprep.pipeline import stream_prepared_samples

    samples = stream_prepared_samples(
        raw_examples,
        model=model,
        tokenizer=tokenizer,
        log_root=log_root,
        pack_segments=pack_segments,
        include_kv=include_kv,
        include_logits=include_logits,
        skip_rows=skip_rows,
    )

    writers = {
        split: ShardWriter(
            wds_root / split,
            shard_prefix=f"{model}_kv_{split}" if include_kv else f"{model}_{split}",
            samples_per_shard=samples_per_shard,
        )
        for split in ("train", "val")
    }

    first_meta: dict | None = None
    bar = tqdm(desc="Writing samples", unit="seq", disable=not progress)
    try:
        for sample in shuffle_stream(
            samples, buffer_size=shuffle_buffer, seed=shuffle_seed
        ):
            if first_meta is None:
                first_meta = sample.meta()
            split = assign_split(
                sample.row, split_ratio=split_ratio, seed=shuffle_seed
            )
            writers[split].write(sample)
            bar.update(1)
            bar.set_postfix(
                train=writers["train"].num_samples, val=writers["val"].num_samples
            )
    finally:
        bar.close()
        for writer in writers.values():
            writer.close()

    if first_meta is None:
        raise ValueError("no samples to export")

    train, val = writers["train"], writers["val"]
    wds_root.mkdir(parents=True, exist_ok=True)
    (wds_root / "shards.json").write_text(
        json.dumps({"train": train.shard_index, "val": val.shard_index}, indent=2),
        encoding="utf-8",
    )

    # Totals are only knowable once the stream is exhausted, so this is written
    # last; train/cli.py reads it for the eval progress bar.
    dataset_info = {
        "model": model,
        "hidden_dim": first_meta["hidden_dim"],
        "num_codebooks": first_meta["num_codebooks"],
        "frame_rate": first_meta["frame_rate"],
        "vocab_size": first_meta.get("vocab_size"),
        "has_kv": include_kv,
        "has_logits": include_logits,
        "total_train_sequences": train.num_samples,
        "total_val_sequences": val.num_samples,
        "total_train_frames": train.total_frames,
        "total_val_frames": val.total_frames,
    }
    (wds_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2), encoding="utf-8"
    )
    print(
        f"Done. Train: {len(train.shard_index)} shard(s), {train.total_frames} frames. "
        f"Val: {len(val.shard_index)} shard(s), {val.total_frames} frames."
    )
    return dataset_info


def shuffle_stream(
    samples: Iterator[ShardSample], *, buffer_size: int, seed: int = 42
) -> Iterator[ShardSample]:
    """Reservoir-shuffle an iterator through a fixed-size buffer.

    Off by default: sequences arrive grouped by row, so shuffling here spreads 
    a row's turns across shards, making written order depend on buffer state, 
    blocking ability to easily resume on cancel/failure.
    """
    if buffer_size <= 1:
        yield from samples
        return

    rng = random.Random(seed)
    buffer: list[ShardSample] = []
    for sample in samples:
        if len(buffer) < buffer_size:
            buffer.append(sample)
            continue
        # Evict a random resident, then seat the newcomer in its place.
        index = rng.randrange(buffer_size)
        yield buffer[index]
        buffer[index] = sample
    rng.shuffle(buffer)
    yield from buffer


def build_sample(
    seq,
    feat,
    *,
    row: int,
    seq_id: int,
    model: str,
    frame_rate: float,
    include_kv: bool = False,
    include_logits: bool = False,
) -> ShardSample | None:
    """Serialize one tokenized/featurized pair into a :class:`ShardSample`.

    Returns None when the sequence has no audio span. Serializing here rather
    than at write time is deliberate: the streaming exporter holds a shuffle
    buffer of these, and fp16 ``.npy`` bytes are far smaller than the live
    fp32 tensors they came from.

    ``include_logits`` adds the teacher's per-head distributions. They are ~16x
    the size of the hiddens (32 heads x 2051 vocab vs one 4096-wide vector), so
    they are off by default and only worth paying for when distilling against
    the teacher's full distribution rather than the ground-truth codes.
    """
    audio_spans = seq.spans_of(SpanKind.AUDIO)
    if not audio_spans:
        return None
    # Miso: one audio span per sequence; multi-span packing not supported yet.
    if len(audio_spans) > 1:
        raise ValueError(f"Row {row} seq {seq_id}: multi-span sequences not yet supported")

    span = audio_spans[0]
    s, e = span.start, span.end
    F = e - s  # number of audio frames
    hidden_dim = int(feat.hiddens.shape[-1])
    num_codebooks = seq.layout.num_codebooks

    # hiddens[s-1:e-1] — teacher-forcing offset: position i predicts token i+1
    h_audio = feat.hiddens[s - 1 : e - 1].to(torch.float16).numpy()  # (F, H)
    # tokens[s:e, 0:num_codebooks] — first num_codebooks columns are audio
    targets = seq.tokens[s:e, :num_codebooks].to(torch.int16).numpy()  # (F, K)

    assert h_audio.shape == (F, hidden_dim), f"hiddens shape mismatch row {row} seq {seq_id}"
    assert targets.shape == (F, num_codebooks), f"targets shape mismatch row {row} seq {seq_id}"

    sample = ShardSample(
        row=row,
        seq_id=seq_id,
        source_dataset_id=span.source_dataset_id,
        segment_id=span.segment_id,
        audio_frames=F,
        frame_rate=frame_rate,
        model=model,
        hidden_dim=hidden_dim,
        num_codebooks=num_codebooks,
        hiddens=_npy_bytes(h_audio),
        targets=_npy_bytes(targets),
    )

    if include_logits:
        logits = _stack_head_logits(
            feat, s, e, num_codebooks=num_codebooks
        )
        assert logits.shape[:2] == (F, num_codebooks), (
            f"logits shape mismatch row {row} seq {seq_id}: {logits.shape}"
        )
        sample.logits = _npy_bytes(logits)
        sample.vocab_size = int(logits.shape[-1])
        sample.head_targets = list(seq.layout.head_targets)

    if include_kv and feat.kv_cache is not None:
        # kv_cache shape depends on model; store as-is in float16.
        # Expected: list of (keys, values) per layer → stack to (layers, 2, ...)
        # then slice to audio-span frames.
        kv = _extract_kv_audio_slice(feat.kv_cache, s, e)
        if kv is not None:
            sample.kv = _npy_bytes(kv)

    return sample


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def _stack_head_logits(feat, s: int, e: int, *, num_codebooks: int) -> np.ndarray:
    """Stack per-head logits over an audio span into one ``(F, K, V)`` fp16 array.

    ``FeaturizedSequence`` keeps logits as ``{head: (L-1, V)}``. Stacking makes
    one contiguous blob per sample instead of K arrays, and reuses
    ``feature_slice_for_targets`` for the teacher-forcing offset so the values
    line up with ``audio_frame_metrics`` rather than re-deriving the shift here.

    fp16 is exact for the logits this repo produces -- verified bit-identical
    over all 464 sequences of the 10 analysis rows, NLL included -- so there is
    nothing to gain from a wider dtype.
    """
    pred = feat.feature_slice_for_targets(s, e)
    stacked = torch.stack(
        [torch.as_tensor(_as_numpy(feat.logits[k][pred])) for k in range(num_codebooks)],
        dim=1,
    )  # (F, K, V)
    return stacked.to(torch.float16).numpy()


def _extract_kv_audio_slice(kv_cache, s: int, e: int) -> np.ndarray | None:
    """Extract audio-span frames from kv_cache, return float16 numpy array or None."""
    try:
        import torch
        # kv_cache: list[(keys, values)] per layer, each (batch, heads, seq, dim)
        layers = []
        for keys, values in kv_cache:
            k = torch.as_tensor(keys).float()
            v = torch.as_tensor(values).float()
            # Slice sequence dimension (dim 2) to audio span
            k_slice = k[0, :, s - 1 : e - 1, :]  # (heads, F, dim)
            v_slice = v[0, :, s - 1 : e - 1, :]
            # Stack as (2, heads, F, dim) → transpose to (F, 2, heads, dim)
            kv_layer = torch.stack([k_slice, v_slice], dim=0).permute(2, 0, 1, 3)
            layers.append(kv_layer)
        # (layers, F, 2, heads, dim) → (F, layers, 2, heads, dim)
        kv = torch.stack(layers, dim=0).permute(1, 0, 2, 3, 4)
        return kv.to(torch.float16).numpy()
    except Exception:
        return None


def shard_root(slug: str, *, data_root: Path = DATA_ROOT) -> Path:
    """Directory holding one shard set: ``data/sharded_wds/SLUG/``.
    """
    return data_root / SHARD_DIRNAME / slug


def assign_split(row: int, *, split_ratio: float, seed: int = 42) -> str:
    """Deterministically route a whole row to 'train' or 'val'.

    Hashing the row id rather than slicing a shuffled list has two properties a
    streaming exporter needs. Every sequence from a row lands on the same side,
    so a speaker's other turns in the same recording cannot leak from train into
    val and flatter the eval. And the answer depends only on the row id, so it
    survives shuffling, restarts, and interrupted runs.

    blake2b, not hash(): PYTHONHASHSEED randomizes str/int hashing per process,
    which would reshuffle the split between runs.
    """
    digest = hashlib.blake2b(f"{seed}:{row}".encode(), digest_size=8).digest()
    # Map the digest onto [0, 1) and compare against the train fraction.
    fraction = int.from_bytes(digest, "big") / float(1 << 64)
    return "train" if fraction < split_ratio else "val"



def next_shard_num(split_dir: Path, *, shard_prefix: str) -> int:
    """First unused shard number in ``split_dir``.

    A resumed run must not reopen ``00000``; numbering continues past whatever
    survived the interrupted run so existing shards are never overwritten.
    """
    highest = -1
    for path in split_dir.glob(f"{shard_prefix}_*.tar"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1
