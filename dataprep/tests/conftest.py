from __future__ import annotations

import json
from pathlib import Path

import pytest
import soundfile as sf

from dataprep.common import Segment
from dataprep.prepare import slice_segment_codes

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "segment0"


@pytest.fixture(scope="session")
def segment0():
    meta = json.loads((FIXTURE_DIR / "segment.json").read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(FIXTURE_DIR / "audio.wav", dtype="float32")
    segment = Segment(
        source_dataset_id=0,
        source_audio_channel_id=0,
        segment_id=0,
        text=meta["text"],
        speaker=meta["speaker"],
        start_sec=float(meta["start_sec"]),
        end_sec=float(meta["end_sec"]),
    )
    return {
        "meta": meta,
        "audio": audio,
        "sample_rate": int(sample_rate),
        "segment": segment,
    }


@pytest.fixture(scope="session")
def expected_tokenized():
    expected = {}
    for path in (FIXTURE_DIR / "expected").glob("*.json"):
        if path.stem.endswith("_featurize"):
            continue
        expected[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return expected


@pytest.fixture(scope="session")
def expected_featurized():
    expected = {}
    for path in (FIXTURE_DIR / "expected").glob("*_featurize.json"):
        expected[path.stem.removesuffix("_featurize")] = json.loads(
            path.read_text(encoding="utf-8")
        )
    return expected


@pytest.fixture(scope="session")
def miso_entropy_reference():
    return json.loads((FIXTURE_DIR / "entropy" / "miso.json").read_text(encoding="utf-8"))


def tokenize_segment(segment0, tokenizer):
    segment = segment0["segment"]
    channel_codes = [
        tokenizer.audio_codec.encode(segment0["audio"], segment0["sample_rate"])
    ]
    audio_codes = slice_segment_codes(
        [segment],
        channel_codes,
        frame_rate=tokenizer.audio_codec.frame_rate,
    )
    return tokenizer.apply_chat_template([segment], audio_codes=audio_codes)
