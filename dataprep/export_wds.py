"""Convert per-row featurized PT artifacts to WebDataset tar shards.

Each shard contains ~``samples_per_shard`` sequences. Per sample:
  {key}.hiddens.npy  — float16 (F, hidden_dim)   audio-span hiddens only
  {key}.targets.npy  — int16   (F, num_codebooks) audio codebook tokens
  {key}.meta.json    — lightweight provenance
  {key}.kv.npy       — float16 (F, layers, 2, heads, kv_dim)  optional

Run::

  python -m dataprep.export_wds --model miso --data-root data --rows 0 1 2
  python -m dataprep.export_wds --model miso --data-root data --debug 3  # rows 0..2
"""

from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from dataprep.common import (
    FeaturizedSequence,
    SpanKind,
    TokenizedSequence,
)
from dataprep.expresso import DATASET_NAME


def _npy_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr)
    return buf.getvalue()


def iter_row_samples(
    row: int,
    *,
    data_root: Path,
    model: str,
    dataset: str = DATASET_NAME,
    include_kv: bool = False,
) -> Iterator[dict]:
    """Yield one WDS sample dict per sequence in ``row``."""
    tok_dir = data_root / dataset / "tokenized" / model / str(row)
    feat_dir = data_root / dataset / "featurized" / model / str(row)

    sequences, tok_meta = TokenizedSequence.load_all(tok_dir)
    features, feat_meta = FeaturizedSequence.load_all(feat_dir)

    if len(sequences) != len(features):
        raise ValueError(
            f"Row {row}: {len(sequences)} tokenized vs {len(features)} featurized sequences"
        )

    frame_rate = float(feat_meta["frame_rate"])
    hidden_dim = int(features[0].hiddens.shape[-1])
    num_codebooks = sequences[0].layout.num_codebooks

    for seq_id, (seq, feat) in enumerate(zip(sequences, features)):
        audio_spans = seq.spans_of(SpanKind.AUDIO)
        if not audio_spans:
            continue
        # Miso: one audio span per sequence; multi-span packing not supported yet.
        if len(audio_spans) > 1:
            raise ValueError(
                f"Row {row} seq {seq_id}: multi-span sequences not yet supported"
            )
        span = audio_spans[0]
        s, e = span.start, span.end
        F = e - s  # number of audio frames

        # hiddens[s-1:e-1] — teacher-forcing offset: position i predicts token i+1
        h_audio = feat.hiddens[s - 1 : e - 1].to(torch.float16).numpy()  # (F, H)
        # tokens[s:e, 0:num_codebooks] — first num_codebooks columns are audio
        targets = seq.tokens[s:e, :num_codebooks].to(torch.int16).numpy()  # (F, K)

        assert h_audio.shape == (F, hidden_dim), f"hiddens shape mismatch row {row} seq {seq_id}"
        assert targets.shape == (F, num_codebooks), f"targets shape mismatch row {row} seq {seq_id}"

        meta = {
            "row": row,
            "seq_id": seq_id,
            "source_dataset_id": span.source_dataset_id,
            "segment_id": span.segment_id,
            "audio_frames": F,
            "frame_rate": frame_rate,
            "model": model,
            "hidden_dim": hidden_dim,
            "num_codebooks": num_codebooks,
            "has_kv": False,
        }

        sample = {
            "__key__": f"{row:08d}_{seq_id:05d}",
            "hiddens.npy": _npy_bytes(h_audio),
            "targets.npy": _npy_bytes(targets),
            "meta.json": json.dumps(meta).encode(),
        }

        if include_kv and feat.kv_cache is not None:
            # kv_cache shape depends on model; store as-is in float16.
            # Expected: list of (keys, values) per layer → stack to (layers, 2, ...)
            # then slice to audio-span frames.
            kv = _extract_kv_audio_slice(feat.kv_cache, s, e)
            if kv is not None:
                sample["kv.npy"] = _npy_bytes(kv)
                meta["has_kv"] = True
                sample["meta.json"] = json.dumps(meta).encode()

        yield sample


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


def write_shards(
    all_samples: list[dict],
    *,
    output_dir: Path,
    shard_prefix: str,
    samples_per_shard: int = 250,
) -> list[dict]:
    """Write samples to tar shards; return shard index entries."""
    import webdataset as wds

    output_dir.mkdir(parents=True, exist_ok=True)
    shard_index = []
    for shard_num, start in enumerate(range(0, len(all_samples), samples_per_shard)):
        chunk = all_samples[start : start + samples_per_shard]
        shard_path = output_dir / f"{shard_prefix}_{shard_num:05d}.tar"
        total_frames = 0
        with wds.TarWriter(str(shard_path)) as sink:
            for sample in chunk:
                sink.write(sample)
                meta = json.loads(sample["meta.json"])
                total_frames += meta["audio_frames"]
        shard_index.append(
            {
                "shard": str(shard_path.name),
                "num_samples": len(chunk),
                "total_frames": total_frames,
            }
        )
    return shard_index


def export(
    rows: list[int],
    *,
    model: str,
    data_root: Path,
    output_root: Path,
    dataset: str = DATASET_NAME,
    split_ratio: float = 0.95,
    samples_per_shard: int = 250,
    include_kv: bool = False,
    shuffle_seed: int = 42,
) -> None:
    from tqdm import tqdm

    wds_root = output_root / dataset / "wds" / model

    # Collect all samples across rows, then shuffle before shard assignment.
    all_samples: list[dict] = []
    for row in tqdm(rows, desc="Collecting samples", unit="row"):
        for sample in iter_row_samples(
            row, data_root=data_root, model=model, dataset=dataset, include_kv=include_kv
        ):
            all_samples.append(sample)

    rng = random.Random(shuffle_seed)
    rng.shuffle(all_samples)

    split_idx = int(len(all_samples) * split_ratio)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    train_dir = wds_root / "train"
    val_dir = wds_root / "val"
    shard_prefix = f"{model}_kv_train" if include_kv else f"{model}_train"
    val_prefix = f"{model}_kv_val" if include_kv else f"{model}_val"

    print(
        f"Writing {len(train_samples)} train / {len(val_samples)} val samples "
        f"to {wds_root}"
    )

    train_index = write_shards(
        train_samples,
        output_dir=train_dir,
        shard_prefix=shard_prefix,
        samples_per_shard=samples_per_shard,
    )
    val_index = write_shards(
        val_samples,
        output_dir=val_dir,
        shard_prefix=val_prefix,
        samples_per_shard=samples_per_shard,
    )

    # Write index and dataset info.
    shards_info = {"train": train_index, "val": val_index}
    (wds_root / "shards.json").write_text(
        json.dumps(shards_info, indent=2), encoding="utf-8"
    )

    # Infer dataset-level metadata from the first sample.
    first_meta = json.loads(all_samples[0]["meta.json"])
    total_train_frames = sum(s["total_frames"] for s in train_index)
    total_val_frames = sum(s["total_frames"] for s in val_index)
    dataset_info = {
        "model": model,
        "hidden_dim": first_meta["hidden_dim"],
        "num_codebooks": first_meta["num_codebooks"],
        "frame_rate": first_meta["frame_rate"],
        "has_kv": include_kv,
        "total_train_sequences": len(train_samples),
        "total_val_sequences": len(val_samples),
        "total_train_frames": total_train_frames,
        "total_val_frames": total_val_frames,
    }
    (wds_root / "dataset_info.json").write_text(
        json.dumps(dataset_info, indent=2), encoding="utf-8"
    )
    print(
        f"Done. Train: {len(train_index)} shard(s), {total_train_frames} frames. "
        f"Val: {len(val_index)} shard(s), {total_val_frames} frames."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export featurized Miso sequences to WebDataset tar shards."
    )
    parser.add_argument("--model", choices=("miso", "qwen3", "fish"), default="miso")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--dataset",
        default=DATASET_NAME,
        help=f"Dataset slug for the on-disk artifact directory (default: {DATASET_NAME}).",
    )
    parser.add_argument(
        "--rows",
        nargs="?",
        const=3,
        type=int,
        default=None,
        metavar="N",
        help="Export first N rows (default 3).",
    )
    parser.add_argument(
        "--split-ratio",
        type=float,
        default=0.95,
        help="Fraction of sequences to assign to train split.",
    )
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=250,
        help="Target number of sequences per tar shard.",
    )
    parser.add_argument(
        "--include-kv",
        action="store_true",
        help="Include KV cache slices in shards (requires kv_context.pt).",
    )
    parser.add_argument("--shuffle-seed", type=int, default=42)
    args = parser.parse_args()

    rows = list(range(args.rows))
    output_root = args.output_root or args.data_root
    export(
        rows,
        model=args.model,
        data_root=args.data_root,
        output_root=output_root,
        dataset=args.dataset,
        split_ratio=args.split_ratio,
        samples_per_shard=args.samples_per_shard,
        include_kv=args.include_kv,
        shuffle_seed=args.shuffle_seed,
    )


if __name__ == "__main__":
    main()
