from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from tqdm import tqdm

from dataprep.common import FeaturizedSequence, Segment, TokenizedSequence
from dataprep.expresso import download_expresso, load_raw_example


DEFAULT_MODELS = {
    "qwen3": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "fish": "mlx-community/fish-audio-s2-pro-8bit",
}


def _as_torch(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if (
        type(value).__module__.startswith("mlx.")
        and str(value.dtype) == "mlx.core.bfloat16"
    ):
        import mlx.core as mx

        value = value.astype(mx.float32)
    return torch.from_numpy(np.asarray(value)).cpu()


def load_tokenizer(model: str, model_id: str | None = None, device: str | None = None):
    if model == "miso":
        from dataprep.miso import MisoAudioCodec, MisoFeaturizer, MisoTokenizer

        return MisoTokenizer(
            audio_codec=MisoAudioCodec(device=device),
            featurizer=MisoFeaturizer(device=device),
        )
    if model == "qwen3":
        from dataprep.qwen3 import Qwen3Tokenizer

        return Qwen3Tokenizer(model_id or DEFAULT_MODELS["qwen3"])
    if model == "fish":
        from dataprep.fish import FishTokenizer

        return FishTokenizer(model_id or DEFAULT_MODELS["fish"])
    raise ValueError(f"Unknown model {model!r}")


def save_codebooks(
    path: Path,
    channel_codes: Sequence[Any],
    *,
    sample_rate: int,
    frame_rate: float,
    num_codebooks: int,
) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "channels": [_as_torch(codes).long() for codes in channel_codes],
            "sample_rate": sample_rate,
            "frame_rate": frame_rate,
            "num_codebooks": num_codebooks,
        },
        path,
    )


def slice_segment_codes(
    segments: Sequence[Segment],
    channel_codes: Sequence[Any],
    *,
    frame_rate: float,
) -> dict[int, Any]:
    codes_by_segment: dict[int, Any] = {}
    for segment in segments:
        codes = channel_codes[segment.source_audio_channel_id]
        start, end = segment.frame_bounds(
            frame_rate=frame_rate, max_frames=int(codes.shape[0])
        )
        codes_by_segment[segment.segment_id] = codes[start:end]
    return codes_by_segment


def _pack_segments(
    segments: Sequence[Segment],
    tokenizer,
    *,
    audio_codes: Mapping[int, Any],
) -> list[TokenizedSequence]:
    """Greedily group consecutive segments up to the model sequence limit."""
    chunks: list[TokenizedSequence] = []
    pending: list[Segment] = []
    for segment in segments:
        candidate = [*pending, segment]
        try:
            tokenizer.apply_chat_template(candidate, audio_codes=audio_codes)
        except ValueError as error:
            if "exceeds" not in str(error) or not pending:
                raise
            chunks.append(
                tokenizer.apply_chat_template(pending, audio_codes=audio_codes)
            )
            pending = [segment]
            tokenizer.apply_chat_template(pending, audio_codes=audio_codes)
        else:
            pending = candidate
    if pending:
        chunks.append(tokenizer.apply_chat_template(pending, audio_codes=audio_codes))
    return chunks


def build_sequences(
    segments: Sequence[Segment],
    tokenizer,
    *,
    audio_codes: Mapping[int, Any],
    pack_segments: bool = False,
) -> list[TokenizedSequence]:
    if pack_segments:
        return _pack_segments(segments, tokenizer, audio_codes=audio_codes)
    return [
        tokenizer.apply_chat_template([segment], audio_codes=audio_codes)
        for segment in tqdm(segments, desc="tokenize", unit="seg", leave=False)
    ]


def prepare_row(
    row_dir: str | Path,
    *,
    model: str,
    tokenizer,
    output_root: str | Path = "data",
    pack_segments: bool = False,
) -> Path:
    example, audio = load_raw_example(row_dir)
    row = example.row
    row_dir = Path(row_dir)
    tqdm.write(
        f"Tokenizing row {row}: {example.num_channels} channel(s), "
        f"{len(example.segments)} segment(s)"
    )

    channel_codes = [
        tokenizer.audio_codec.encode(audio[channel], example.sample_rate)
        for channel in tqdm(
            range(example.num_channels),
            desc=f"row {row} encode",
            unit="ch",
            leave=False,
        )
    ]
    codebooks = row_dir / f"{model}_codebooks.pt"
    save_codebooks(
        codebooks,
        channel_codes,
        sample_rate=tokenizer.audio_codec.sample_rate,
        frame_rate=tokenizer.audio_codec.frame_rate,
        num_codebooks=tokenizer.audio_codec.num_codebooks,
    )

    segments = [
        Segment.from_transcript_item(raw, source_dataset_id=example.row)
        for raw in example.segments
    ]
    sequences = build_sequences(
        segments,
        tokenizer,
        audio_codes=slice_segment_codes(
            segments,
            channel_codes,
            frame_rate=tokenizer.audio_codec.frame_rate,
        ),
        pack_segments=pack_segments,
    )
    output_dir = Path(output_root) / "expresso" / f"{model}_tokenized" / str(row)
    TokenizedSequence.save_all(
        output_dir,
        sequences,
        metadata={
            "model": model,
            "dataset": "Zackh/expresso-contextual",
            "row": example.row,
            "sample_rate": tokenizer.audio_codec.sample_rate,
            "frame_rate": tokenizer.audio_codec.frame_rate,
            "num_codebooks": tokenizer.audio_codec.num_codebooks,
            "num_channels": example.num_channels,
            "num_segments": len(segments),
            "pack_segments": pack_segments,
            "codebooks": str(codebooks),
        },
    )
    tqdm.write(f"Tokenizing row {row}: wrote {len(sequences)} sequence(s) to {output_dir}")
    return output_dir


def featurize_row(
    row: int,
    *,
    model: str,
    featurizer,
    data_root: str | Path = "data",
    dump_kv: bool = False,
) -> Path:
    input_dir = Path(data_root) / "expresso" / f"{model}_tokenized" / str(row)
    sequences, source_metadata = TokenizedSequence.load_all(input_dir)
    tqdm.write(
        f"Featurizing row {row}: {len(sequences)} sequence(s)"
        + (" with KV cache" if dump_kv else "")
    )
    features = [
        featurizer.featurize(sequence, include_kv=dump_kv)
        for sequence in tqdm(
            sequences, desc=f"row {row} featurize", unit="seq", leave=False
        )
    ]
    for sequence, feature in zip(sequences, features):
        feature.validate(sequence_length=sequence.length)

    feature_dir = Path(data_root) / "expresso" / f"{model}_featurized" / str(row)
    FeaturizedSequence.save_all(
        feature_dir,
        features,
        metadata={
            "model": model,
            "dataset": source_metadata["dataset"],
            "row": row,
            "frame_rate": source_metadata["frame_rate"],
            "num_codebooks": featurizer.num_codebooks,
            "semantic_logits_masked": model == "fish",
            "kv_saved": dump_kv,
        },
    )
    tqdm.write(f"Featurizing row {row}: wrote {feature_dir}")
    return feature_dir


def resolve_rows(*, debug: int | None) -> list[int]:
    if debug is None:
        raise NotImplementedError(
            "Full-dataset parquet export is not implemented yet; pass --debug [N]"
        )
    if debug < 1:
        raise ValueError("--debug requires a positive row count")
    return list(range(debug))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare Expresso codebooks and flat TTS sequences."
    )
    parser.add_argument("--model", choices=("miso", "qwen3", "fish"), required=True)
    parser.add_argument(
        "--stage",
        choices=("tokenize", "featurize", "all"),
        default="tokenize",
    )
    parser.add_argument(
        "--debug",
        nargs="?",
        const=3,
        type=int,
        default=None,
        metavar="N",
        help="Prepare the first N Expresso rows (default: 3).",
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--dump-kv",
        action="store_true",
        help="Save the slow-AR layer KV cache during featurization.",
    )
    parser.add_argument(
        "--pack-segments",
        action="store_true",
        help="Pack consecutive segments up to the model limit.",
    )
    args = parser.parse_args()

    rows = resolve_rows(debug=args.debug)
    raw_root = args.data_root / "expresso" / "raw"
    missing = [
        row
        for row in rows
        if not (raw_root / str(row) / "transcript_segments.json").exists()
    ]
    if missing:
        download_expresso(missing, root=raw_root)

    tokenizer = load_tokenizer(
        args.model, model_id=args.model_id, device=args.device
    )
    paths: list[Path] = []
    if args.stage in ("tokenize", "all"):
        for row in tqdm(rows, desc=f"Tokenizing {args.model}", unit="row"):
            paths.append(
                prepare_row(
                    raw_root / str(row),
                    model=args.model,
                    tokenizer=tokenizer,
                    output_root=args.data_root,
                    pack_segments=args.pack_segments,
                )
            )
    if args.stage in ("featurize", "all"):
        for row in tqdm(rows, desc=f"Featurizing {args.model}", unit="row"):
            paths.append(
                featurize_row(
                    row,
                    model=args.model,
                    featurizer=tokenizer.featurizer,
                    data_root=args.data_root,
                    dump_kv=args.dump_kv,
                )
            )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
