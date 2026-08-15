"""Offline tests for the Emilia loader.

``load_dataset`` is faked, so these run with no network and no gated access.
They cover the two things that can actually be wrong: the two metadata schemas
spell the id differently, and the ``mp3`` column arrives as either raw bytes or
an ``Audio`` dict depending on feature inference.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from dataprep.datasources import emilia

#: Both schemas in amphion/Emilia-Dataset: Emilia proper spells the id ``id``
#: and carries ``wav``; Emilia-YODAS spells it ``_id`` and has no ``wav``.
EN_META = {
    "id": "EN_B00000_S00000_W000000",
    "wav": "EN_B00000/EN_B00000_S00000/mp3/EN_B00000_S00000_W000000.mp3",
    "text": "the birch canoe slid on the smooth planks",
    "duration": 3.0,
    "speaker": "EN_B00000_S00000",
    "language": "en",
    "dnsmos": 3.2,
}
YODAS_META = {
    "_id": "DE_B000000_S00109_W000023",
    "text": "und das ist auch gut so",
    "duration": 3.0,
    "speaker": "DE_wSq11gYbUgU_SPEAKER_04",
    "language": "de",
    "dnsmos": 3.1,
    "phone_count": 21,
}


def _wav_bytes(sample_rate: int = 24000, seconds: float = 3.0) -> bytes:
    """A real encoded waveform, so soundfile does the decoding for real.

    WAV rather than mp3 keeps this off libsndfile's optional mp3 support; the
    decode path under test is format-agnostic.
    """
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    tone = (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, tone, sample_rate, format="WAV")
    return buffer.getvalue()


class _FakeInfo:
    """Stands in for ``DatasetInfo``; only ``features`` is ever touched."""

    def __init__(self):
        # Non-None to start with, so clearing it is observable.
        self.features = {"mp3": "Audio(decode=True)"}


class _FakeStream:
    """Minimal stand-in for a streaming ``IterableDataset``.

    Carries ``_info`` because the loader clears the inferred feature schema.
    """

    def __init__(self, rows, info=None):
        self._rows = list(rows)
        self._info = info or _FakeInfo()

    def take(self, n):
        return _FakeStream(self._rows[:n], info=self._info)

    def __iter__(self):
        return iter(self._rows)


@pytest.fixture
def fake_load_dataset(monkeypatch):
    """Patch ``datasets.load_dataset`` and record the kwargs it was called with."""
    calls = {}

    def factory(rows):
        def fake(dataset, **kwargs):
            stream = _FakeStream(rows)
            calls.update(kwargs, dataset=dataset, returned=stream)
            return stream

        monkeypatch.setattr("datasets.load_dataset", fake)
        return calls

    return factory


def test_stream_emilia_reads_en_schema(fake_load_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    rows = [{"json": EN_META, "mp3": _wav_bytes(), "__key__": "EN_B00000_S00000_W000000"}]
    calls = fake_load_dataset(rows)

    segments = list(emilia.stream_emilia(data_files="Emilia/EN/EN-B000000.tar"))

    assert len(segments) == 1
    segment = segments[0]
    assert segment.id == "EN_B00000_S00000_W000000"
    assert segment.speaker == "EN_B00000_S00000"
    assert segment.text == EN_META["text"]
    assert segment.sample_rate == 24000
    assert segment.audio.ndim == 1, "audio must be mono (samples,)"
    assert segment.audio.dtype == np.float32
    # Duration must match the metadata, or the codes supervise the wrong text.
    assert segment.audio.shape[0] / segment.sample_rate == pytest.approx(
        EN_META["duration"], abs=0.05
    )
    # The glob must reach load_dataset: it is what limits the download.
    assert calls["data_files"] == {"train": "Emilia/EN/EN-B000000.tar"}
    assert calls["streaming"] is True


def test_stream_emilia_reads_yodas_underscore_id(fake_load_dataset, monkeypatch):
    """Emilia-YODAS spells the id ``_id`` and ships no ``wav`` key."""
    monkeypatch.setenv("HF_TOKEN", "test-token")
    rows = [{"json": YODAS_META, "mp3": _wav_bytes()}]
    fake_load_dataset(rows)

    segment = next(iter(emilia.stream_emilia(data_files="Emilia-YODAS/DE/*.tar")))

    assert segment.id == "DE_B000000_S00109_W000023"
    assert segment.speaker == "DE_wSq11gYbUgU_SPEAKER_04"


def test_stream_emilia_accepts_decoded_audio_dict(fake_load_dataset, monkeypatch):
    """``datasets`` hands over an Audio dict when it infers the feature."""
    monkeypatch.setenv("HF_TOKEN", "test-token")
    rows = [
        {
            "json": EN_META,
            "mp3": {"array": np.zeros(2400, dtype=np.float32), "sampling_rate": 24000},
        }
    ]
    fake_load_dataset(rows)

    segment = next(iter(emilia.stream_emilia()))

    assert segment.audio.shape == (2400,)
    assert segment.sample_rate == 24000


def test_stream_emilia_clears_the_inferred_feature_schema(fake_load_dataset, monkeypatch):
    """Leaving the inferred Audio feature in place makes datasets require torchcodec."""
    monkeypatch.setenv("HF_TOKEN", "test-token")
    rows = [{"json": EN_META, "mp3": _wav_bytes(seconds=0.1)}]
    calls = fake_load_dataset(rows)

    segments = list(emilia.stream_emilia())

    assert len(segments) == 1
    # The stream load_dataset returned is the one the loader must have cleared.
    assert calls["returned"]._info.features is None


def test_stream_emilia_limit_counts_utterances(fake_load_dataset, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "test-token")
    rows = [{"json": EN_META, "mp3": _wav_bytes(seconds=0.1)} for _ in range(5)]
    fake_load_dataset(rows)

    assert len(list(emilia.stream_emilia(limit=2))) == 2


def test_stream_emilia_requires_a_token(monkeypatch):
    """The dataset is gated, so a missing token must say so, not 404."""
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(ValueError, match="gated"):
        next(iter(emilia.stream_emilia()))


def test_utterance_id_rejects_metadata_without_one():
    with pytest.raises(KeyError):
        emilia._utterance_id({"text": "no id here"})
