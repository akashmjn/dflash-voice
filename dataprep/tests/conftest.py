from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import soundfile as sf

from dataprep.common import Segment

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "segment0"

#: The package each backend needs, and the extra that installs it. Presence
#: only: a backend whose package is installed can still fail at runtime.
BACKEND_REQUIREMENTS = {
    "miso": ("generator", "dataprep-miso"),
    "chatterbox": ("chatterbox", "dataprep-chatterbox"),
}


#: Mimi's 30 s encode chunk exceeds the MPS conv1d output-channel cap, so every
#: miso test that encodes audio raises NotImplementedError on Apple silicon.
#: Not a dataprep bug: the same encode succeeds under ``load_tokenizer("miso",
#: device="cpu")``, which these tests do not thread a device through yet.
MPS_CONV1D_CAP = "miso audio encode exceeds the MPS conv1d output cap (works on cpu)"


def mps_only_backend(model: str) -> bool:
    """True when ``model`` would auto-select MPS and hit :data:`MPS_CONV1D_CAP`."""
    if model != "miso":
        return False
    import torch

    return not torch.cuda.is_available() and torch.backends.mps.is_available()


def backend_available(model: str) -> bool:
    package, _ = BACKEND_REQUIREMENTS[model]
    try:
        return importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        # find_spec raises, rather than returning None, on a half-installed package.
        return False


def pytest_collection_modifyitems(config, items):
    """Skip backend tests this venv or this machine cannot run.

    Two reasons: the backend's extra is not installed, or it is miso on Apple
    silicon (see :data:`MPS_CONV1D_CAP`). A test opts in by parametrizing over
    ``model`` or by carrying ``@pytest.mark.backend("miso")``. Modules that
    import a backend at top level need their own ``importorskip``: this runs
    after the module is imported.
    """
    for item in items:
        models = set()
        if hasattr(item, "callspec"):
            models.add(item.callspec.params.get("model"))
        for marker in item.iter_markers(name="backend"):
            models.update(marker.args)
        for model in models & BACKEND_REQUIREMENTS.keys():
            if not backend_available(model):
                package, extra = BACKEND_REQUIREMENTS[model]
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"{model}: no {package!r} module; needs the {extra} extra"
                    )
                )
            elif mps_only_backend(model):
                item.add_marker(pytest.mark.skip(reason=MPS_CONV1D_CAP))


@pytest.fixture(scope="session")
def monkeypatch_session():
    """Session-scoped monkeypatch; the built-in fixture is function-scoped."""
    with pytest.MonkeyPatch.context() as patcher:
        yield patcher


@pytest.fixture(scope="session")
def segment0():
    meta = json.loads((FIXTURE_DIR / "segment.json").read_text(encoding="utf-8"))
    audio, sample_rate = sf.read(FIXTURE_DIR / "audio.wav", dtype="float32")
    segment = Segment(
        id="segment0",
        text=meta["text"],
        speaker=meta["speaker"],
        audio=audio,
        sample_rate=int(sample_rate),
    )
    return {
        "meta": meta,
        "audio": audio,
        "sample_rate": int(sample_rate),
        "segment": segment,
    }


@pytest.fixture(scope="session")
def expected_tokenized():
    """Per-model gold for ``test_tokenize_segment0*``.

    ``chatterbox.json`` is dumped by external script using offical 
    Chatterbox Flash inference repo. Audio resample is torchaudio not librosa,
    used by official repo, disagrees on ~6% of S3 token ids on this clip.
    """
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
    """Tokenize the one-segment fixture."""
    segment = segment0["segment"]
    audio_codes = tokenizer.audio_codec.encode(segment.audio, segment.sample_rate)
    return tokenizer.apply_chat_template(segment, audio_codes=audio_codes)


def featurize_segment(segment0, tokenizer, sequence):
    """Featurize a sequence, supplying a context to backends that need one.

    Chatterbox conditions on a speaker embedding that has no token id, so it
    takes the ``SequenceEmbeddingContext`` tokenize produced.
    """
    if not hasattr(tokenizer, "embedding_context"):
        return tokenizer.featurizer.featurize(sequence)
    context = tokenizer.embedding_context(segment0["segment"])
    return tokenizer.featurizer.featurize(sequence, context=context)
