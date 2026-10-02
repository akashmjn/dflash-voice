"""EmergentTTS-Eval as a harder seed-tts: fetch only, like ``seedtts.py``.

bosonai/EmergentTTS-Eval holds 1,645 English texts in six categories (Emotions,
Paralinguistics, Foreign Words, Syntactic Complexity, Questions, Complex
Pronunciation), each grown from a seed prompt over ``evolution_depth`` 0-3
rounds of LLM rewriting. It ships text only: the ``audio`` column is the
gpt-4o-mini-tts baseline used for its win-rate judge, not a human recording,
so there is no ground truth and no speaker reference.

``fetch_emergenttts`` rewrites it into seedtts_test_en.jsonl's shape (``id``,
``text``, ``ref_audio``, ``ref_text``) so the bench and the Modal scorers take
it unchanged. Each row borrows a seed-tts prompt clip as its voice, picked by
hashing the row id, and keeps the clip's path verbatim -- run
``fetch --dataset seedtts`` (no ``--rows``) to land every clip it can cite.
Rows are interleaved across (category, depth) so the first N stay balanced,
the same "split is the first N rows" convention seed-tts uses.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from dataprep.datasources import seedtts

REPO_ID = "bosonai/EmergentTTS-Eval"
PROMPTS_FILE = "emergenttts_en.jsonl"
DATASET_NAME = "emergenttts"

#: Text columns only; reading ``audio`` would decode ~1,600 baseline clips.
COLUMNS = ("category", "text_to_synthesize", "evolution_depth", "language")


def _slug(category: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def load_records(token: str | None = None) -> list[dict[str, Any]]:
    """The hub's rows in file order, text columns only."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    files = sorted(
        name
        for name in HfApi().list_repo_files(REPO_ID, repo_type="dataset", token=token)
        if name.endswith(".parquet")
    )
    if not files:
        raise RuntimeError(f"no parquet files in {REPO_ID}")
    records: list[dict[str, Any]] = []
    for name in files:
        path = hf_hub_download(REPO_ID, name, repo_type="dataset", token=token)
        records.extend(pq.read_table(path, columns=list(COLUMNS)).to_pylist())
    return records


def load_references(seedtts_prompts: Path) -> list[dict[str, str]]:
    """Distinct (ref_audio, ref_text) pairs from a seed-tts prompts file, sorted."""
    refs: dict[str, str] = {}
    with seedtts_prompts.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            refs.setdefault(row["ref_audio"], row["ref_text"])
    return [{"ref_audio": a, "ref_text": t} for a, t in sorted(refs.items())]


def _interleave(groups: dict[Any, list]) -> Iterable:
    """Round-robin over groups in key order, so any prefix stays balanced."""
    queues = [list(groups[key]) for key in sorted(groups)]
    while any(queues):
        for queue in queues:
            if queue:
                yield queue.pop(0)


def build_rows(
    records: list[dict[str, Any]], refs: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """seed-tts-shaped rows, balanced-prefix ordered.

    ``id`` keeps the hub index so a row maps back to the source (and to its
    baseline clip) after the reordering.
    """
    if not refs:
        raise ValueError("need at least one reference clip to pair rows with")
    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for index, record in enumerate(records):
        depth = int(record["evolution_depth"])
        category = _slug(record["category"])
        row_id = f"emergenttts_{category}_d{depth}_{index:04d}"
        digest = int(hashlib.sha1(row_id.encode()).hexdigest(), 16)
        ref = refs[digest % len(refs)]
        groups[(category, depth)].append(
            {
                "id": row_id,
                "text": record["text_to_synthesize"],
                "ref_audio": ref["ref_audio"],
                "ref_text": ref["ref_text"],
                "category": record["category"],
                "evolution_depth": depth,
                "language": record.get("language") or "en",
                "source_index": index,
            }
        )
    return list(_interleave(groups))


def fetch_emergenttts(
    dest: Path,
    *,
    limit: int | None = None,
    seedtts_root: Path,
    token: str | None = None,
) -> dict[str, Any]:
    """Write ``prompts/emergenttts_en.jsonl`` (and ``_N`` for a limit) under ``dest``.

    Reads the seed-tts prompts under ``seedtts_root`` for the voices, fetching
    that JSONL first if it is missing (the wavs are left to ``fetch``).
    """
    from huggingface_hub import hf_hub_download

    seed_prompts = seedtts_root / "prompts" / seedtts.PROMPTS_FILE
    if not seed_prompts.exists():
        cached = hf_hub_download(
            seedtts.REPO_ID, seedtts.PROMPTS_FILE, repo_type="dataset", token=token
        )
        seed_prompts.parent.mkdir(parents=True, exist_ok=True)
        seed_prompts.write_bytes(Path(cached).read_bytes())

    rows = build_rows(load_records(token), load_references(seed_prompts))
    prompts_dir = dest / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for name, subset in [(PROMPTS_FILE, rows)] + (
        [(PROMPTS_FILE.replace(".jsonl", f"_{limit}.jsonl"), rows[:limit])]
        if limit is not None
        else []
    ):
        path = prompts_dir / name
        with path.open("w", encoding="utf-8") as handle:
            for row in subset:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        written[name] = path

    counts: dict[str, int] = defaultdict(int)
    for row in rows[:limit]:
        counts[row["category"]] += 1
    return {
        "prompts": list(written.values()),
        "rows": len(rows),
        "refs": len({row["ref_audio"] for row in rows}),
        "by_category": dict(sorted(counts.items())),
    }
