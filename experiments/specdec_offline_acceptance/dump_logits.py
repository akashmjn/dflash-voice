"""Dump Chatterbox logits over the tokenized rows, for offline acceptance.

Writes into `data/DATASET/specdec_offline/chatterbox-{ar,flash,turbo,nano}/ROW/` --
outside dataprep's `featurized/` tree, which model_metrics.py globs for models to
score. The `ar` dump is the target, re-featurized on reference prompts so it is
conditioned like the drafts it is compared against; dataprep's own `featurized/`
tree prompts each utterance with its own audio. Model loading and the teacher-forced
forward live in `chatterbox_utils.py`.

Flash dumps bare tensors per block size (`features_bN.pt`); the causal drafts dump
standard `FeaturizedSequence` rows carrying their own spans.

```bash
python experiments/specdec_offline_acceptance/dump_logits.py --rows 10
python experiments/specdec_offline_acceptance/dump_logits.py --model turbo --rows 10
python experiments/specdec_offline_acceptance/dump_logits.py --rows 3 --block-sizes 1,2,4,8
```
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator, Optional

import torch
import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from chatterbox_utils import (  # noqa: E402
    DRAFTS,
    Draft,
    featurize_block_masked,
    featurize_causal_draft,
    featurize_target_ar,
    load_causal_draft,
    load_causal_tokenizer,
    load_flash_t3,
    load_speaker_references,
    reference_sources,
    sequence_segments,
    tokenize_for_causal_draft,
)
from dataprep.chatterbox import (  # noqa: E402
    AudioPrompt,
    ChatterboxFeaturizer,
    _default_device,
)
from dataprep.types import (  # noqa: E402
    FeaturizedSequence,
    TokenizedSequence,
)

#: Tokens are shared with AR, so the input artifact differs from the output one.
TOKENIZED_ARTIFACT = "chatterbox"

#: Dumps land here rather than in dataprep's ``featurized/`` tree, which
#: model_metrics.py globs for models to score.
SPECDEC_STAGE = "specdec_offline"

#: The target; every other name in DRAFTS is a draft for it.
AR_MODEL = "ar"

app = typer.Typer(
    add_completion=False,
    help="Teacher-force a Chatterbox model over tokenized dumps, for offline logits.",
)


def dump_suffix(block_size: int) -> str:
    """Filename suffix for a dump; ``simulate_acceptance.py --block-size`` reads it back."""
    return f"_b{block_size}"


def dump_flash_draft(
    row_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    model: "T3",
    references: dict[str, AudioPrompt],
    *,
    device: str,
    dtype: torch.dtype,
    block_size: int,
) -> int:
    sequences, metadata = TokenizedSequence.load_all(row_dir)
    segments = sequence_segments(row_dir, raw_dir)

    featurized = [
        featurize_block_masked(
            model, sequence, references[segment["speaker"]],
            device=device, dtype=dtype, block_size=block_size,
        )
        for sequence, segment in zip(sequences, segments)
    ]

    suffix = dump_suffix(block_size)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        [{"logits": {0: logits}, "hiddens": hiddens} for logits, hiddens in featurized],
        out_dir / f"features{suffix}.pt",
    )
    payload = dict(metadata)
    payload["model"] = "chatterbox-flash"
    payload["decode"] = {
        "mode": "blockwise",
        "block_size": block_size,
        "speaker_references": reference_sources(),
    }
    (out_dir / f"metadata{suffix}.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return len(featurized)


def dump_ar_target(
    row_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    featurizer: "ChatterboxFeaturizer",
    references: dict[str, AudioPrompt],
) -> int:
    """Teacher-force the AR target over one row, prompted on the speakers' references."""
    sequences, _ = TokenizedSequence.load_all(row_dir)
    segments = sequence_segments(row_dir, raw_dir)
    featurized = [
        featurize_target_ar(featurizer, sequence, references[segment["speaker"]])
        for sequence, segment in zip(sequences, segments)
    ]
    FeaturizedSequence.save_all(
        out_dir,
        featurized,
        metadata={
            "model": "chatterbox-ar",
            "decode": {"mode": "causal", "speaker_references": reference_sources()},
        },
    )
    return len(featurized)


def dump_causal_draft(
    row_dir: Path,
    raw_dir: Path,
    out_dir: Path,
    model: "T3",
    tokenizer: "PreTrainedTokenizerBase",
    references: dict[str, AudioPrompt],
    spec: Draft,
    draft: str,
    *,
    device: str,
    dtype: torch.dtype = torch.float32,
) -> int:
    """Teacher-force Turbo/Nano over one row, into a standard FeaturizedSequence dump."""
    sequences, prompts = tokenize_for_causal_draft(
        row_dir, raw_dir, model.hp, tokenizer, references
    )
    featurized = [
        featurize_causal_draft(model, sequence, reference, device=device, dtype=dtype)
        for sequence, reference in zip(sequences, prompts)
    ]
    FeaturizedSequence.save_all(
        out_dir,
        featurized,
        metadata={
            "model": f"chatterbox-{draft}",
            "decode": {
                "mode": "causal",
                "speech_cond_prompt_len": spec.prompt_len,
                "speaker_references": reference_sources(),
            },
        },
    )
    return len(featurized)


@app.command()
def main(
    model: str = typer.Option(
        "flash",
        help="ar (the target), flash (blockwise draft), or turbo/nano (causal drafts).",
    ),
    rows: int = typer.Option(10, help="Dump rows 0..N-1."),
    row_start: int = typer.Option(0, help="Skip rows below this, to resume a run."),
    block_sizes: str = typer.Option(
        "1,2,4", help="Comma-separated block sizes to dump. Flash only."
    ),
    artifact: str = typer.Option(
        TOKENIZED_ARTIFACT, help="Tokenized subdir to read (shared with AR)."
    ),
    out_subdir: Optional[str] = typer.Option(
        None, help="specdec_offline subdir to write (default: chatterbox-MODEL)."
    ),
    dataset: str = typer.Option("expresso", help="Dataset slug for artifact paths."),
    device: Optional[str] = typer.Option(None, help="Defaults to mps/cuda/cpu."),
) -> None:
    """Dump target or draft logits into the specdec_offline row dirs."""
    if model != AR_MODEL and model not in DRAFTS:
        raise typer.BadParameter(
            f"unknown model {model!r}; expected {AR_MODEL} or {', '.join(DRAFTS)}"
        )
    spec = DRAFTS.get(model)
    out_artifact = out_subdir or f"chatterbox-{model}"
    data = REPO_ROOT / "data" / dataset
    resolved = _default_device(device)

    def row_dirs() -> Iterator[tuple[int, Path, Path]]:
        for row in range(row_start, rows):
            row_dir = data / "tokenized" / artifact / str(row)
            if not row_dir.exists():
                raise SystemExit(f"no tokenized dump at {row_dir}")
            yield row, row_dir, data / SPECDEC_STAGE / out_artifact / str(row)

    if model == AR_MODEL:
        print(f"loading Chatterbox AR on {resolved}")
        featurizer = ChatterboxFeaturizer(device=resolved)
        references = load_speaker_references()
        for row, row_dir, out_dir in row_dirs():
            count = dump_ar_target(
                row_dir, data / "raw" / str(row), out_dir, featurizer, references
            )
            print(f"  row {row}: {count} sequence(s) -> {out_dir}/features.pt")
        return

    if not spec.is_flash:
        print(f"loading Chatterbox {model} on {resolved}")
        model = load_causal_draft(spec, resolved)
        tokenizer = load_causal_tokenizer(spec)
        references = load_speaker_references()
        for row, row_dir, out_dir in row_dirs():
            count = dump_causal_draft(
                row_dir, data / "raw" / str(row), out_dir, model, tokenizer,
                references, spec, model, device=resolved,
            )
            print(f"  row {row}: {count} sequence(s) -> {out_dir}/features.pt")
        return

    try:
        sizes = [int(part) for part in block_sizes.split(",") if part.strip()]
    except ValueError:
        raise typer.BadParameter(f"wants comma-separated integers, got {block_sizes!r}")

    print(f"loading Chatterbox Flash on {resolved}")
    model = load_flash_t3(spec, resolved)
    references = load_speaker_references()
    for block_size in sizes:
        print(f"block size {block_size}:")
        for row, row_dir, out_dir in row_dirs():
            count = dump_flash_draft(
                row_dir, data / "raw" / str(row), out_dir, model, references,
                device=resolved, dtype=torch.float32, block_size=block_size,
            )
            name = f"features{dump_suffix(block_size)}.pt"
            print(f"  row {row}: {count} sequence(s) -> {out_dir}/{name}")


if __name__ == "__main__":
    app()
