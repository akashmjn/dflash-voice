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

#: Members to extract. The archive also holds ``en/ground_truth_wavs`` (the
#: reference renditions) and a ``zh`` half, neither of which the bench reads.
PROMPT_WAVS = "seedtts_testset/en/prompt_wavs/"

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


def _wanted(members: Iterable[tarfile.TarInfo], keep: set[str] | None):
    """Yield the prompt wavs to extract, ignoring the rest of the archive."""
    for member in members:
        if not member.isfile() or not member.name.startswith(PROMPT_WAVS):
            continue
        if keep is None or Path(member.name).name in keep:
            yield member


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
        with tarfile.open(tarball, mode="r:gz") as archive:
            for member in _wanted(archive, keep):
                target = audio_dir / Path(member.name).name
                if target.exists():
                    continue
                source = archive.extractfile(member)
                if source is None:
                    continue
                target.write_bytes(source.read())
                extracted += 1

    return {
        "prompts": prompts_path,
        "ref_audio": audio_dir,
        "extracted": extracted,
        "wavs": len(list(audio_dir.glob("*.wav"))),
    }
