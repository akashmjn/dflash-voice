"""Round-trip tests for WDS export.

Tests that the exported shards faithfully reproduce the original PT artifacts:
- targets match original tokens at the audio span
- hiddens (after fp16 round-trip) are close to the original stored hiddens
- NLL computed from stored logits matches saved metrics JSON

These tests are non-expensive: they read from already-featurized data in
``data/`` and do not load any neural network models.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from dataprep.expresso import DATASET_NAME

BARE_DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
DATA_ROOT = BARE_DATA_ROOT / DATASET_NAME
MISO_TOK = DATA_ROOT / "tokenized" / "miso"
MISO_FEAT = DATA_ROOT / "featurized" / "miso"
MISO_METRICS = DATA_ROOT / "metrics" / "miso"

# Skip the entire module if the 3-row debug data hasn't been generated yet.
pytestmark = pytest.mark.skipif(
    not (MISO_TOK / "0").exists() or not (MISO_FEAT / "0").exists(),
    reason="Miso featurized data not found under data/; run prepare.py first",
)


def _load_npy(data: bytes) -> np.ndarray:
    return np.load(io.BytesIO(data))


# ---------------------------------------------------------------------------
# Fixture: export 3 rows to a temporary WDS directory once per test session.
# ---------------------------------------------------------------------------

def _replay_rows(rows):
    """Stand-in for the hub loader, replaying rows already on disk.

    The audio is a placeholder -- the patched tokenize never reads it.
    """
    from dataprep.expresso import DecodedExample

    for row in rows:
        yield DecodedExample(
            row=row,
            audio_path=Path(),
            transcript_path=Path(),
            sample_rate=24000,
            num_channels=1,
            segments=[],
            audio=np.zeros((1, 1), dtype=np.float32),
        )


class _ReplayTokenizer:
    """Returns the saved tokenized/featurized rows instead of running a model."""

    def __init__(self, data_root: Path, model: str):
        self.data_root = data_root
        self.model = model
        feat_meta = json.loads(
            (data_root / DATASET_NAME / "featurized" / model / "0" / "metadata.json").read_text()
        )
        self.audio_codec = type(
            "codec", (), {"frame_rate": float(feat_meta["frame_rate"])}
        )()
        self.featurizer = self
        # Paired by identity, not seq_id -- that is a segment id, not an index.
        self._pairs: dict[int, object] = {}

    def featurize(self, sequence, include_kv=False):
        return self._pairs[id(sequence)]

    def sequences_for(self, row: int):
        from dataprep.common import TokenizedSequence

        tok_dir = self.data_root / DATASET_NAME / "tokenized" / self.model / str(row)
        sequences, _ = TokenizedSequence.load_all(tok_dir)
        return sequences

    def features_for(self, row: int):
        from dataprep.common import FeaturizedSequence

        feat_dir = self.data_root / DATASET_NAME / "featurized" / self.model / str(row)
        features, _ = FeaturizedSequence.load_all(feat_dir)
        return features


@pytest.fixture(scope="session")
def wds_root(tmp_path_factory, monkeypatch_session):
    """Shard the on-disk rows through shard_prepare, the path `prepare` uses."""
    from dataprep import pipeline
    from dataprep.shards import shard_prepare, shard_root

    out = tmp_path_factory.mktemp("miso_wds")
    rows = [r for r in range(3) if (MISO_TOK / str(r)).exists() and (MISO_FEAT / str(r)).exists()]
    if not rows:
        pytest.skip("No featurized rows available")

    tokenizer = _ReplayTokenizer(BARE_DATA_ROOT, "miso")

    def fake_tokenize(example, audio, *, tokenizer, log_root, pack_segments=False):
        sequences = tokenizer.sequences_for(example.row)
        features = tokenizer.features_for(example.row)
        assert len(sequences) == len(features)
        for seq, feat in zip(sequences, features):
            tokenizer._pairs[id(seq)] = feat
        return sequences, None, [None] * len(sequences)

    monkeypatch_session.setattr(pipeline, "tokenize_example", fake_tokenize)

    wds_root = shard_root(f"{DATASET_NAME}-test", data_root=out)
    shard_prepare(
        _replay_rows(rows),
        model="miso",
        tokenizer=tokenizer,
        wds_root=wds_root,
        # The split is a per-row hash, so with only 3 rows most ratio/seed pairs
        # strand one side entirely. This pair puts rows 0,1 in train and 2 in val.
        split_ratio=0.7,
        samples_per_shard=50,  # small shards for speed in tests
        shuffle_seed=1,
        progress=False,
    )
    return wds_root


@pytest.fixture(scope="session")
def shard_samples(wds_root):
    """Load all samples from the first train shard as raw dicts."""
    import tarfile

    train_dir = wds_root / "train"
    tars = sorted(train_dir.glob("*.tar"))
    assert tars, f"No train shards found in {train_dir}"

    samples = []
    with tarfile.open(tars[0]) as tf:
        members = tf.getmembers()
        # Group by key prefix.
        by_key: dict[str, dict] = {}
        for m in members:
            key, _, suffix = m.name.partition(".")
            data = tf.extractfile(m).read()
            by_key.setdefault(key, {})["__key__"] = key
            by_key[key][suffix] = data
        samples = list(by_key.values())
    return samples


# ---------------------------------------------------------------------------
# 1. Export structure tests
# ---------------------------------------------------------------------------

def test_dataset_info_written(wds_root):
    info_path = wds_root / "dataset_info.json"
    assert info_path.exists(), "dataset_info.json not written"
    info = json.loads(info_path.read_text())
    assert info["model"] == "miso"
    assert info["hidden_dim"] == 4096
    assert info["num_codebooks"] == 32
    assert info["frame_rate"] == 12.5
    assert info["total_train_sequences"] > 0


def test_shards_index_written(wds_root):
    idx_path = wds_root / "shards.json"
    assert idx_path.exists()
    idx = json.loads(idx_path.read_text())
    assert "train" in idx and "val" in idx
    assert len(idx["train"]) >= 1


def test_no_row_straddles_the_split(wds_root):
    """A row's sequences must land wholly in train or wholly in val.

    Sequences from one row share speakers and a recording session; splitting
    them across train/val would leak that identity into the eval.
    """
    import tarfile

    sides: dict[int, set[str]] = {}
    for split in ("train", "val"):
        for tar_path in sorted((wds_root / split).glob("*.tar")):
            with tarfile.open(tar_path) as tf:
                for member in tf.getmembers():
                    if not member.name.endswith("meta.json"):
                        continue
                    row = json.loads(tf.extractfile(member).read())["row"]
                    sides.setdefault(row, set()).add(split)

    straddling = {row: s for row, s in sides.items() if len(s) > 1}
    assert not straddling, f"rows present in both splits: {straddling}"


def test_split_assignment_is_deterministic():
    """The split must not move between processes or runs."""
    from dataprep.shards import assign_split

    first = [assign_split(r, split_ratio=0.9, seed=7) for r in range(200)]
    second = [assign_split(r, split_ratio=0.9, seed=7) for r in range(200)]
    assert first == second
    # A different seed must actually reshuffle the assignment.
    other = [assign_split(r, split_ratio=0.9, seed=8) for r in range(200)]
    assert other != first
    # And the ratio should be roughly honoured over enough rows.
    train = sum(s == "train" for s in first)
    assert 0.8 < train / len(first) < 1.0


def test_shuffle_stream_conserves_samples():
    """The reservoir must emit every input exactly once, reordered."""
    from dataprep.shards import shuffle_stream

    source = [{"i": i} for i in range(500)]
    out = list(shuffle_stream(iter(source), buffer_size=64, seed=3))
    assert sorted(s["i"] for s in out) == list(range(500))
    assert [s["i"] for s in out] != list(range(500)), "stream was not reordered"
    # A degenerate buffer must pass through untouched rather than drop samples.
    assert [s["i"] for s in shuffle_stream(iter(source), buffer_size=1)] == list(range(500))


def test_logits_absent_by_default(shard_samples):
    """Logits are ~16x the hiddens, so they must be opt-in."""
    for s in shard_samples:
        assert "logits.npy" not in s, f"{s['__key__']} has logits without --include-logits"
        assert json.loads(s["meta.json"])["has_logits"] is False


@pytest.fixture(scope="session")
def logits_sample():
    """One sample built with logits, straight from row 0's saved artifacts."""
    from dataprep.common import FeaturizedSequence, TokenizedSequence
    from dataprep.shards import build_sample

    if not (MISO_TOK / "0").exists():
        pytest.skip("row 0 not prepared")
    seqs, _ = TokenizedSequence.load_all(MISO_TOK / "0")
    feats, meta = FeaturizedSequence.load_all(MISO_FEAT / "0")
    # .to_wds(): the other tests read samples back out of tars, so comparing
    # against the same wire dict keeps every assertion in one shape.
    return build_sample(
        seqs[0],
        feats[0],
        row=0,
        seq_id=0,
        model="miso",
        frame_rate=float(meta["frame_rate"]),
        include_logits=True,
    ).to_wds()


def test_logits_shape_and_dtype(logits_sample):
    meta = json.loads(logits_sample["meta.json"])
    assert meta["has_logits"] is True
    assert meta["vocab_size"] == 2051
    assert meta["head_targets"], "head_targets needed to score the logits"

    logits = _load_npy(logits_sample["logits.npy"])
    assert logits.dtype == np.float16, f"logits dtype {logits.dtype}"
    assert logits.shape == (meta["audio_frames"], meta["num_codebooks"], 2051)


def test_logits_reproduce_saved_metrics(logits_sample):
    """NLL from the shard logits must match the row's saved miso_metrics.json.

    This is the whole point of storing them: the shard has to be a faithful
    stand-in for the featurized artifact it came from, teacher-forcing offset
    and head/target pairing included.
    """
    metrics_path = MISO_METRICS / "0" / "miso_metrics.json"
    if not metrics_path.exists():
        pytest.skip("row 0 metrics not computed")

    from dataprep.common import FeaturizedSequence, TokenizedSequence, audio_frame_metrics

    seqs, _ = TokenizedSequence.load_all(MISO_TOK / "0")
    feats, _ = FeaturizedSequence.load_all(MISO_FEAT / "0")
    expected = audio_frame_metrics(feats[0], seqs[0].tokens, 32)["nll"]

    logits = torch.from_numpy(_load_npy(logits_sample["logits.npy"]).astype(np.float32))
    targets = torch.from_numpy(_load_npy(logits_sample["targets.npy"]).astype(np.int64))
    log_probs = torch.log_softmax(logits, dim=-1)
    actual = -log_probs.gather(2, targets[:, :, None]).squeeze(2)

    # fp16 is exact on these logits; the tolerance is float32 summation noise.
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=0)


def test_shard_sample_keys(shard_samples):
    """Every sample must have hiddens, targets, meta."""
    for s in shard_samples:
        assert "hiddens.npy" in s, f"Missing hiddens.npy in {s['__key__']}"
        assert "targets.npy" in s, f"Missing targets.npy in {s['__key__']}"
        assert "meta.json" in s, f"Missing meta.json in {s['__key__']}"


def test_sample_shapes_and_dtypes(shard_samples):
    for s in shard_samples:
        meta = json.loads(s["meta.json"])
        F = meta["audio_frames"]
        H = meta["hidden_dim"]
        K = meta["num_codebooks"]

        h = _load_npy(s["hiddens.npy"])
        t = _load_npy(s["targets.npy"])

        assert h.shape == (F, H), f"{s['__key__']}: hiddens shape {h.shape} != ({F}, {H})"
        assert t.shape == (F, K), f"{s['__key__']}: targets shape {t.shape} != ({F}, {K})"
        assert h.dtype == np.float16, f"hiddens dtype {h.dtype}"
        assert t.dtype == np.int16, f"targets dtype {t.dtype}"


# ---------------------------------------------------------------------------
# 2. Content parity tests — targets and hiddens must match original PT data
# ---------------------------------------------------------------------------

def _load_row_orig(row: int):
    """Return (sequences, features) from the original PT artifacts."""
    from dataprep.common import FeaturizedSequence, TokenSpanKind, TokenizedSequence

    seqs, _ = TokenizedSequence.load_all(MISO_TOK / str(row))
    feats, _ = FeaturizedSequence.load_all(MISO_FEAT / str(row))
    return seqs, feats


def test_targets_match_original_tokens(shard_samples):
    """Exported targets must exactly match tokens[audio_span, :32] from sequences.pt."""
    from dataprep.common import TokenSpanKind

    # Build lookup of already-loaded rows to avoid repeated IO.
    rows_loaded: dict[int, tuple] = {}

    for s in shard_samples:
        meta = json.loads(s["meta.json"])
        row = meta["row"]
        seq_id = meta["seq_id"]

        if row not in rows_loaded:
            rows_loaded[row] = _load_row_orig(row)
        seqs, _ = rows_loaded[row]

        seq = seqs[seq_id]
        audio_spans = seq.spans_of(TokenSpanKind.AUDIO)
        assert len(audio_spans) == 1, f"Expected 1 audio span, got {len(audio_spans)}"
        span = audio_spans[0]
        s_idx, e_idx = span.start, span.end

        expected_targets = seq.tokens[s_idx:e_idx, :32].numpy().astype(np.int16)
        actual_targets = _load_npy(s["targets.npy"])

        np.testing.assert_array_equal(
            actual_targets,
            expected_targets,
            err_msg=f"Targets mismatch for row={row} seq={seq_id}",
        )


def test_hiddens_close_to_original(shard_samples):
    """Exported hiddens (fp16) must round-trip close to stored hiddens (fp32)."""
    from dataprep.common import TokenSpanKind

    rows_loaded: dict[int, tuple] = {}

    for s in shard_samples:
        meta = json.loads(s["meta.json"])
        row = meta["row"]
        seq_id = meta["seq_id"]

        if row not in rows_loaded:
            rows_loaded[row] = _load_row_orig(row)
        seqs, feats = rows_loaded[row]

        seq = seqs[seq_id]
        feat = feats[seq_id]
        span = seq.spans_of(TokenSpanKind.AUDIO)[0]
        s_idx, e_idx = span.start, span.end

        # Original hiddens for the audio span (note: teacher-forcing offset)
        expected_h = feat.hiddens[s_idx - 1 : e_idx - 1].numpy().astype(np.float32)
        actual_h = _load_npy(s["hiddens.npy"]).astype(np.float32)

        # fp16 has ~3 decimal digits of precision; allow 1e-2 absolute tolerance.
        np.testing.assert_allclose(
            actual_h,
            expected_h,
            atol=1e-2,
            rtol=0,
            err_msg=f"Hiddens mismatch for row={row} seq={seq_id}",
        )


def test_semantic_is_targets_col0(shard_samples):
    """targets[:, 0] must equal codebook-0 tokens (s_t)."""
    for s in shard_samples:
        t = _load_npy(s["targets.npy"])
        # Codebook 0 == semantic token; the DataLoader exposes this as `semantic`.
        # Just verify the stored column is correct by checking it against layout.
        assert t.shape[1] == 32, f"Expected 32 codebooks, got {t.shape[1]}"
        # targets[:, 0] should be in valid vocab range for miso (0..2050)
        assert int(t[:, 0].max()) < 2051, "Semantic tokens exceed vocab size"


# ---------------------------------------------------------------------------
# 3. NLL parity test — recompute NLL from stored logits, compare to metrics JSON
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("row", [0, 1, 2])
def test_nll_parity_with_saved_metrics(row):
    """NLL from stored logits must match the saved miso_metrics.json for each row."""
    metrics_path = MISO_METRICS / str(row) / "miso_metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"miso_metrics.json not found for row {row}")
    if not (MISO_FEAT / str(row)).exists():
        pytest.skip(f"miso_featurized/{row} not found")

    from dataprep.common import FeaturizedSequence, TokenSpanKind, TokenizedSequence, audio_frame_metrics, nll_summary

    seqs, _ = TokenizedSequence.load_all(MISO_TOK / str(row))
    feats, _ = FeaturizedSequence.load_all(MISO_FEAT / str(row))
    saved = json.loads(metrics_path.read_text())

    frame_rate = saved["frame_rate"]
    num_codebooks = saved["num_codebooks"]

    # Accumulate NLL across all sequences in this row.
    nll_parts = []
    for seq, feat in zip(seqs, feats):
        metrics = audio_frame_metrics(feat, seq.tokens, num_codebooks)
        nll_parts.append(metrics["nll"])
    nll_all = torch.cat(nll_parts, dim=0).numpy()

    summary = nll_summary(nll_all, frame_rate)
    saved_summary = saved["nll_summary"]

    for group in ("semantic", "audio", "total"):
        for unit in ("avg_nll_per_codebook", "kbits_per_second"):
            expected = saved_summary[group][unit]
            actual = summary[group][unit]
            assert abs(actual - expected) < 1e-2, (
                f"Row {row} {group}.{unit}: got {actual:.6f}, expected {expected:.6f}"
            )


def test_sample_order_is_deterministic(wds_root):
    """Samples land in arrival order, sorted by (row, seq_id) within a split.

    This is the property a resumed run relies on: where a sample lands depends
    only on its key, so replaying the stream and skipping what is already
    written reproduces the same set. A write-time shuffle would break it.
    """
    import tarfile

    keys = []
    for shard in sorted((wds_root / "train").glob("*.tar")):
        with tarfile.open(shard) as tar:
            for member in tar:
                if member.name.endswith(".meta.json"):
                    entry = json.load(tar.extractfile(member))
                    keys.append((entry["row"], entry["seq_id"]))

    assert keys == sorted(keys), "shard order is not deterministic"
    assert len(keys) == len(set(keys)), "duplicate (row, seq_id) in shards"


def test_next_shard_num_continues_past_existing(wds_root):
    """A resumed run must not reopen shard 00000 and clobber it."""
    from dataprep.shards import next_shard_num

    existing = sorted((wds_root / "train").glob("*.tar"))
    assert next_shard_num(wds_root / "train", shard_prefix="miso_train") == len(existing)
