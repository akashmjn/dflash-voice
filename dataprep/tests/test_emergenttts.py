"""Offline tests for the EmergentTTS-Eval fetcher: the hub reads are faked."""

from __future__ import annotations

import json
from collections import Counter

from dataprep.datasources import emergenttts, seedtts

CATEGORIES = ("Emotions", "Complex Pronunciation", "Foreign Words")


def _records(per_group: int = 4) -> list[dict]:
    return [
        {
            "category": category,
            "text_to_synthesize": f"{category} {depth} {i}",
            "evolution_depth": depth,
            "language": "en",
        }
        for category in CATEGORIES
        for depth in range(4)
        for i in range(per_group)
    ]


REFS = [
    {"ref_audio": f"download/prompt_wavs/spk{i}.wav", "ref_text": f"ref {i}"}
    for i in range(5)
]


def test_prefix_is_balanced_across_category_and_depth():
    rows = emergenttts.build_rows(_records(), REFS)
    assert len(rows) == len(CATEGORIES) * 4 * 4
    first = Counter((row["category"], row["evolution_depth"]) for row in rows[:12])
    assert set(first.values()) == {1} and len(first) == 12


def test_rows_are_seedtts_shaped_and_stable():
    rows = emergenttts.build_rows(_records(), REFS)
    again = emergenttts.build_rows(_records(), REFS)
    assert rows == again
    assert len({row["id"] for row in rows}) == len(rows)
    for row in rows:
        assert {"id", "text", "ref_audio", "ref_text"} <= row.keys()
        assert {"ref_audio": row["ref_audio"], "ref_text": row["ref_text"]} in REFS
        assert row["id"].startswith("emergenttts_") and " " not in row["id"]
    by_index = {row["source_index"]: row["text"] for row in rows}
    assert by_index[0] == _records()[0]["text_to_synthesize"]


def test_fetch_writes_full_and_limited_splits(tmp_path, monkeypatch):
    seed_root = tmp_path / "seedtts"
    seed_prompts = seed_root / "prompts" / seedtts.PROMPTS_FILE
    seed_prompts.parent.mkdir(parents=True)
    # Rows reuse clips; the references are deduped.
    seed_prompts.write_text(
        "".join(
            json.dumps({"id": f"r{i}", "text": "t", **REFS[i % len(REFS)]}) + "\n"
            for i in range(12)
        )
    )
    monkeypatch.setattr(emergenttts, "load_records", lambda token=None: _records())

    result = emergenttts.fetch_emergenttts(
        tmp_path / "emergenttts", limit=24, seedtts_root=seed_root
    )
    full, limited = result["prompts"]
    assert len(full.read_text().splitlines()) == 48
    lines = [json.loads(line) for line in limited.read_text().splitlines()]
    assert len(lines) == 24 and limited.name == "emergenttts_en_24.jsonl"
    assert result["by_category"] == {c: 8 for c in CATEGORIES}
    assert result["refs"] <= len(REFS)
