"""seed-tts eval set: fetch only, no ``Segment`` stream.

Unlike the other loaders this is evaluation data for ``mlx_decode/bench.py``,
not training material -- each row pairs a text with its own reference clip, so
there is nothing to tokenize into shards. ``fetch_seedtts`` just lands the
prompts and the wavs somewhere the bench can read them.

The wavs ship inside a 1 GB tarball under
``seedtts_testset/en/prompt_wavs/NAME.wav``, while the prompts spell the same
file as ``download/tts_eval_datasets/.../prompt_wavs/NAME.wav``. The bench
resolves a row against ``--ref-audio-dir`` by basename, so extraction
flattens the tree and neither path has to be rewritten.
"""

from __future__ import annotations

import json
import tarfile
from pathlib import Path
from typing import Any, Iterable

REPO_ID = "k2-fsa/TTS_eval_datasets"
PROMPTS_FILE = "seedtts_test_en.jsonl"
TARBALL = "seedtts_testset.tar.gz"

#: The clip each row is conditioned on, shared across rows. The archive also
#: holds a ``zh`` half, which nothing here reads.
PROMPT_WAVS = "seedtts_testset/en/prompt_wavs/"

#: The human recording of each row's target text, one per row and named by row
#: id -- the "ground truth" row of the chatterbox-flash quality table, and the
#: upper bound a scoring run compares generated audio against.
GROUND_TRUTH_WAVS = "seedtts_testset/en/ground_truth_wavs/"

DATASET_NAME = "seedtts"


def _ref_names(prompts_path: Path, limit: int | None) -> set[str] | None:
    """Basenames cited by the first ``limit`` prompts, or None for all of them.

    Rows reuse clips -- the first 50 cite 32 distinct wavs -- so this is what
    keeps a small ``--limit`` from extracting far more than it needs.
    """
    if limit is None:
        return None
    names: set[str] = set()
    with prompts_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            row = json.loads(line)
            names.add(Path(row["ref_audio"]).name)
    return names


def _wanted(
    members: Iterable[tarfile.TarInfo], prefix: str, keep: set[str] | None
):
    """Yield the members under ``prefix`` to extract, ignoring the rest."""
    for member in members:
        if not member.isfile() or not member.name.startswith(prefix):
            continue
        if keep is None or Path(member.name).name in keep:
            yield member


def _row_ids(prompts_path: Path, limit: int | None) -> list[str]:
    """Row ids in file order, capped at ``limit``.

    Ground truth is one wav per row named by id, and the bench writes its
    output under the same name, so a split is just the first N ids.
    """
    ids = []
    with prompts_path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            ids.append(json.loads(line)["id"])
    return ids


def _extract(
    tarball: str, audio_dir: Path, prefix: str, keep: set[str] | None
) -> int:
    """Extract matching members flat into ``audio_dir``, skipping what is there."""
    extracted = 0
    with tarfile.open(tarball, mode="r:gz") as archive:
        for member in _wanted(archive, prefix, keep):
            target = audio_dir / Path(member.name).name
            if target.exists():
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            target.write_bytes(source.read())
            extracted += 1
    return extracted


def fetch_seedtts(
    dest: Path, *, limit: int | None = None, token: str | None = None
) -> dict[str, Any]:
    """Download the seed-tts English eval set into ``dest``.

    Lands ``prompts/seedtts_test_en.jsonl`` and a flat ``ref_audio/`` beside it.
    Both the hub download and the extraction are skip-if-present, so re-running
    after an interrupted fetch costs nothing.
    """
    from huggingface_hub import hf_hub_download

    prompts_dir = dest / "prompts"
    audio_dir = dest / "ref_audio"
    prompts_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    cached_prompts = hf_hub_download(
        REPO_ID, PROMPTS_FILE, repo_type="dataset", token=token
    )
    prompts_path = prompts_dir / PROMPTS_FILE
    prompts_path.write_bytes(Path(cached_prompts).read_bytes())

    keep = _ref_names(prompts_path, limit)
    # A tarball member is only reachable by reading everything before it, so
    # checking for the files first avoids paying the 1 GB scan on a re-run.
    missing = keep is None or any(not (audio_dir / name).exists() for name in keep)

    extracted = 0
    if missing:
        tarball = hf_hub_download(REPO_ID, TARBALL, repo_type="dataset", token=token)
        extracted = _extract(tarball, audio_dir, PROMPT_WAVS, keep)

    return {
        "prompts": prompts_path,
        "ref_audio": audio_dir,
        "extracted": extracted,
        "wavs": len(list(audio_dir.glob("*.wav"))),
    }


def fetch_ground_truth(
    dest: Path, *, rows: int, token: str | None = None
) -> dict[str, Any]:
    """Download the human recordings for the first ``rows`` prompts.

    Lands a self-contained ``seedtts-{rows}-ground_truth/`` holding one wav per
    row, named by row id -- the same name the bench writes its generated audio
    under, so a scoring run joins the two on filename alone.
    """
    from huggingface_hub import hf_hub_download

    audio_dir = dest / f"seedtts-{rows}-ground_truth"
    audio_dir.mkdir(parents=True, exist_ok=True)

    prompts_path = dest / "prompts" / PROMPTS_FILE
    if not prompts_path.exists():
        cached = hf_hub_download(REPO_ID, PROMPTS_FILE, repo_type="dataset", token=token)
        prompts_path.parent.mkdir(parents=True, exist_ok=True)
        prompts_path.write_bytes(Path(cached).read_bytes())

    ids = _row_ids(prompts_path, rows)
    keep = {f"{row_id}.wav" for row_id in ids}
    missing = [name for name in keep if not (audio_dir / name).exists()]

    extracted = 0
    if missing:
        tarball = hf_hub_download(REPO_ID, TARBALL, repo_type="dataset", token=token)
        extracted = _extract(tarball, audio_dir, GROUND_TRUTH_WAVS, keep)

    present = sorted(p.name for p in audio_dir.glob("*.wav"))
    return {
        "ground_truth": audio_dir,
        "requested": len(keep),
        "extracted": extracted,
        "wavs": len(present),
        "missing": sorted(set(keep) - set(present)),
    }
