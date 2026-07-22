from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dataprep.expresso import download_expresso, load_raw_example
from dataprep.tokenizer import Segment, TokenizedSequence


DEFAULT_MODELS = {
    "qwen3": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-8bit",
    "fish": "mlx-community/fish-audio-s2-pro-8bit",
}


def _as_torch(value: Any):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.from_numpy(np.asarray(value)).cpu()


def segment_frame_bounds(
    segment: dict[str, Any],
    *,
    frame_rate: float,
    max_frames: int,
) -> tuple[int, int]:
    start = max(0, int(math.floor(float(segment["start"]) * frame_rate)))
    end = min(max_frames, int(math.ceil(float(segment["end"]) * frame_rate)))
    if end <= start:
        raise ValueError(f"Segment {segment.get('segment_id')} maps to empty codec frame range")
    return start, end


def load_tokenizer(model: str, model_id: str | None = None, device: str | None = None):
    if model == "miso":
        from dataprep.miso import MisoAudioCodec, MisoTokenizer

        return MisoTokenizer(audio_codec=MisoAudioCodec(device=device))
    if model == "qwen3":
        from dataprep.qwen3 import Qwen3Tokenizer

        return Qwen3Tokenizer(model_id or DEFAULT_MODELS["qwen3"])
    if model == "fish":
        from dataprep.fish import FishTokenizer

        return FishTokenizer(model_id or DEFAULT_MODELS["fish"])
    raise ValueError(f"Unknown model {model!r}")


def _build_segments(example, channel_codes: list[Any], tokenizer) -> list[Segment]:
    segments = []
    speaker_ids: dict[str, int] = {}
    for raw in example.segments:
        channel = int(raw["channel"])
        if channel < 0 or channel >= len(channel_codes):
            raise ValueError(f"Invalid channel {channel} for row {example.row}")
        speaker = str(raw.get("speaker", "speaker"))
        speaker_id = speaker_ids.setdefault(speaker, len(speaker_ids))
        codes = channel_codes[channel]
        start_frame, end_frame = segment_frame_bounds(
            raw,
            frame_rate=tokenizer.audio_codec.frame_rate,
            max_frames=int(codes.shape[0]),
        )
        metadata = {
            "row": example.row,
            "segment_id": int(raw["segment_id"]),
            "speaker": speaker,
            "speaker_id": speaker_id,
            "channel": channel,
            "start_sec": float(raw["start"]),
            "end_sec": float(raw["end"]),
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        segments.append(
            Segment(
                text=str(raw["text"]),
                speaker=speaker_id,
                audio_codes=codes[start_frame:end_frame],
                metadata=metadata,
            )
        )
    return segments


def _pack_segments(segments: Sequence[Segment], tokenizer) -> list[TokenizedSequence]:
    """Greedily group consecutive segments up to the model sequence limit."""
    chunks: list[TokenizedSequence] = []
    pending: list[Segment] = []
    for segment in segments:
        candidate = [*pending, segment]
        try:
            tokenizer.apply_chat_template(candidate)
        except ValueError as error:
            if "exceeds" not in str(error) or not pending:
                raise
            chunks.append(tokenizer.apply_chat_template(pending))
            pending = [segment]
            tokenizer.apply_chat_template(pending)
        else:
            pending = candidate
    if pending:
        chunks.append(tokenizer.apply_chat_template(pending))
    return chunks


def _build_sequences(
    segments: Sequence[Segment],
    tokenizer,
    *,
    pack_segments: bool = False,
) -> list[TokenizedSequence]:
    if pack_segments:
        return _pack_segments(segments, tokenizer)
    return [tokenizer.apply_chat_template([segment]) for segment in segments]


def _span_payload(span) -> dict[str, Any]:
    return {
        "segment_index": span.segment_index,
        "start": span.start,
        "end": span.end,
        "kind": span.kind,
        **span.metadata,
    }


def prepare_row(
    row_dir: str | Path,
    *,
    model: str,
    tokenizer,
    output_root: str | Path = "data",
    verify_decode: bool = False,
    pack_segments: bool = False,
) -> Path:
    import torch

    example, audio = load_raw_example(row_dir)
    output_dir = Path(output_root) / "expresso" / f"{model}_tokenized" / str(example.row)
    output_dir.mkdir(parents=True, exist_ok=True)

    channel_codes = [
        tokenizer.audio_codec.encode(audio[channel], example.sample_rate)
        for channel in range(example.num_channels)
    ]
    torch.save(
        {
            "channels": [_as_torch(codes).long() for codes in channel_codes],
            "sample_rate": tokenizer.audio_codec.sample_rate,
            "frame_rate": tokenizer.audio_codec.frame_rate,
            "num_codebooks": tokenizer.audio_codec.num_codebooks,
        },
        output_dir / "codebooks.pt",
    )

    segments = _build_segments(example, channel_codes, tokenizer)
    sequences = _build_sequences(segments, tokenizer, pack_segments=pack_segments)
    torch.save([_as_torch(item.tokens).long() for item in sequences], output_dir / "sequences.pt")
    torch.save([_as_torch(item.mask).bool() for item in sequences], output_dir / "masks.pt")

    metadata = {
        "model": model,
        "dataset": "Zackh/expresso-contextual",
        "row": example.row,
        "sample_rate": tokenizer.audio_codec.sample_rate,
        "frame_rate": tokenizer.audio_codec.frame_rate,
        "num_codebooks": tokenizer.audio_codec.num_codebooks,
        "num_channels": example.num_channels,
        "num_segments": len(segments),
        "pack_segments": pack_segments,
        "sequences": [
            {
                "sequence_id": index,
                "shape": list(sequence.tokens.shape),
                "spans": [_span_payload(span) for span in sequence.spans],
            }
            for index, sequence in enumerate(sequences)
        ],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if verify_decode:
        for channel, codes in enumerate(channel_codes):
            check_frames = max(1, int(round(tokenizer.audio_codec.frame_rate * 5)))
            decoded = np.asarray(tokenizer.audio_codec.decode(codes[:check_frames]))
            if decoded.size == 0 or not np.isfinite(decoded).all():
                raise RuntimeError(f"{model} row {example.row} channel {channel} decode was empty/non-finite")
    return output_dir


def prepare_rows(
    rows: Sequence[int],
    *,
    model: str,
    data_root: str | Path = "data",
    model_id: str | None = None,
    device: str | None = None,
    verify_decode: bool = False,
    pack_segments: bool = False,
) -> list[Path]:
    raw_root = Path(data_root) / "expresso" / "raw"
    missing = [row for row in rows if not (raw_root / str(row) / "transcript_segments.json").exists()]
    if missing:
        download_expresso(missing, root=raw_root)
    tokenizer = load_tokenizer(model, model_id=model_id, device=device)
    return [
        prepare_row(
            raw_root / str(row),
            model=model,
            tokenizer=tokenizer,
            output_root=data_root,
            verify_decode=verify_decode,
            pack_segments=pack_segments,
        )
        for row in rows
    ]


def resolve_prepare_rows(*, debug: int | None) -> list[int]:
    """Select which Expresso rows to prepare.

    Debug mode writes per-row raw/tokenized intermediates for the first N rows.
    A non-debug full-dataset parquet export will be added later.
    """
    if debug is None:
        raise NotImplementedError(
            "Full-dataset parquet export is not implemented yet; pass --debug [N] "
            "to prepare the first N rows with per-row intermediate files."
        )
    if debug < 1:
        raise ValueError("--debug requires a positive row count")
    return list(range(debug))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Expresso codebooks and flat TTS sequences.")
    parser.add_argument("--model", choices=("miso", "qwen3", "fish"), required=True)
    parser.add_argument(
        "--debug",
        nargs="?",
        const=3,
        type=int,
        default=None,
        metavar="N",
        help=(
            "Prepare the first N Expresso rows (default: 3) and write per-row raw "
            "and tokenized intermediate files under data/. Without this flag, a "
            "full-dataset parquet export will be used (not implemented yet)."
        ),
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--verify-decode", action="store_true")
    parser.add_argument(
        "--pack-segments",
        action="store_true",
        help="Pack consecutive segments into sequences up to the model limit (default: one segment per sequence).",
    )
    args = parser.parse_args()
    paths = prepare_rows(
        resolve_prepare_rows(debug=args.debug),
        model=args.model,
        data_root=args.data_root,
        model_id=args.model_id,
        device=args.device,
        verify_decode=args.verify_decode,
        pack_segments=args.pack_segments,
    )
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
