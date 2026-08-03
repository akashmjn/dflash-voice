"""Checks for the baseline RVQ depth decoder.

Structural tests are cheap and always run.  ``test_checkpoint_parity_segment0``
is marked expensive: it reads the published checkpoint and the featurized
row-0 artifacts, and pins the two things a val-NLL range check cannot see --
the RoPE weight permutation and the audio_head slot alignment.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

DATA = Path(__file__).resolve().parents[2] / "data"
FEATURIZED = DATA / "expresso" / "featurized" / "miso" / "0"
FIXTURE = (
    Path(__file__).resolve().parents[2]
    / "dataprep/tests/fixtures/segment0/expected/miso_featurize.json"
)


def test_forward_shapes():
    from train.model import MisoRVQDepthDecoder

    model = MisoRVQDepthDecoder().eval()
    n = 4
    hiddens = torch.randn(n, model.config.backbone_dim)
    targets = torch.randint(0, model.config.audio_vocab_size, (n, model.config.num_codebooks))
    with torch.no_grad():
        logits = model(hiddens, targets)
    assert logits.shape == (n, 31, model.config.audio_vocab_size)


def test_causal_along_codebook_axis():
    """Perturbing c_k must not change predictions for levels <= k.

    Bidirectional attention here would leak later codebooks into earlier ones
    and silently *improve* NLL -- the failure mode called out in
    dataprep/miso.py:167-181.
    """
    from train.model import MisoRVQDepthDecoder

    torch.manual_seed(0)
    model = MisoRVQDepthDecoder().eval()
    v = model.config.audio_vocab_size
    hiddens = torch.randn(1, model.config.backbone_dim)
    targets = torch.randint(0, v, (1, model.config.num_codebooks))

    with torch.no_grad():
        base = model(hiddens, targets)

    k = 10  # perturb context code c_k
    bumped = targets.clone()
    bumped[0, k] = (bumped[0, k] + 1) % v
    with torch.no_grad():
        after = model(hiddens, bumped)

    # logits index j predicts level j+1, and reads context c0..cj.
    # Changing c_k can only affect predictions for levels > k.
    unaffected = base[:, :k, :]
    torch.testing.assert_close(unaffected, after[:, :k, :], rtol=0, atol=1e-5)
    assert (base[:, k:, :] - after[:, k:, :]).abs().max() > 1e-5


def test_random_init_nll_is_near_chance():
    """An untrained head must carry no information about the targets.

    It sits slightly *above* ln(vocab), not at it: a random-but-nonzero head is
    confidently wrong on random targets, so NLL exceeds the uniform baseline
    while predictive entropy falls below it.  What matters is that it is close
    to chance and not below it.
    """
    from train.model import MisoRVQDepthDecoder, loss_fn, uniform_nll

    torch.manual_seed(0)
    model = MisoRVQDepthDecoder().eval()
    n = 64
    hiddens = torch.randn(n, model.config.backbone_dim)
    targets = torch.randint(0, model.config.audio_vocab_size, (n, model.config.num_codebooks))
    with torch.no_grad():
        loss, per_cb = loss_fn(model(hiddens, targets), targets)

    assert per_cb.shape == (31,)
    chance = uniform_nll()
    assert chance - 0.05 < loss.item() < chance + 1.0, f"got {loss.item()}, chance {chance:.3f}"


def test_converter_keys_match_model():
    from train.convert import convert_miso_decoder_state_dict, decoder_source_keys
    from train.model import MisoRVQDepthDecoder

    shapes = {
        "projection.weight": (1536, 4096),
        "audio_embeddings.weight": (65632, 4096),
        "audio_head": (31, 1536, 2051),
        "decoder.norm.scale": (1536,),
    }
    layer_shapes = {
        "attn.q_proj.weight": (1536, 1536),
        "attn.k_proj.weight": (384, 1536),
        "attn.v_proj.weight": (384, 1536),
        "attn.output_proj.weight": (1536, 1536),
        "mlp.w1.weight": (6912, 1536),
        "mlp.w2.weight": (1536, 6912),
        "mlp.w3.weight": (6912, 1536),
        "sa_norm.scale": (1536,),
        "mlp_norm.scale": (1536,),
    }
    src = {}
    for key in decoder_source_keys():
        if key in shapes:
            src[key] = torch.zeros(*shapes[key])
        else:
            suffix = key.split(".", 3)[3]
            src[key] = torch.zeros(*layer_shapes[suffix])

    converted = convert_miso_decoder_state_dict(src)
    model_keys = set(MisoRVQDepthDecoder().state_dict())
    # embed_tokens is the unused vocab_size=1 dummy; everything else must map.
    assert set(converted) == model_keys - {"decoder.embed_tokens.weight"}


def test_rope_permutation_is_an_involution_on_head_layout():
    """The permutation must be per-head and shape-preserving."""
    from train.convert import permute_for_hf_rope

    w = torch.arange(24 * 64 * 8, dtype=torch.float32).reshape(24 * 64, 8)
    out = permute_for_hf_rope(w, 24, 64)
    assert out.shape == w.shape
    assert sorted(out.flatten().tolist()) == sorted(w.flatten().tolist())
    with pytest.raises(ValueError):
        permute_for_hf_rope(w, 6, 64)  # wrong head count must not silently pass


@pytest.mark.expensive
@pytest.mark.skipif(not FEATURIZED.exists(), reason="run dataprep.prepare for miso row 0 first")
def test_checkpoint_parity_segment0():
    """Ported decoder must reproduce the golden per-codebook NLL for cb 1..31."""
    import json

    from dataprep.common import FeaturizedSequence, TokenizedSequence
    from train.convert import load_miso_decoder_state_dict
    from train.model import MisoRVQDepthDecoder, codebook_nll

    expected = json.loads(FIXTURE.read_text())["codebook_nll_nats"][1:]  # drop cb0

    sequences, _ = TokenizedSequence.load_all(DATA / "expresso" / "tokenized" / "miso" / "0")
    features, _ = FeaturizedSequence.load_all(FEATURIZED)
    seq, feat = sequences[0], features[0]

    from dataprep.common import SpanKind

    span = feat.spans_of(SpanKind.AUDIO)[0]
    pred = feat.feature_slice_for_targets(span.start, span.end)
    hiddens = torch.as_tensor(feat.hiddens[pred]).float()
    targets = torch.as_tensor(seq.tokens[span.start : span.end][:, :32]).long()

    model = MisoRVQDepthDecoder().eval()
    missing, unexpected = model.load_state_dict(load_miso_decoder_state_dict(), strict=False)
    assert not unexpected and list(missing) == ["decoder.embed_tokens.weight"]

    with torch.no_grad():
        per_cb = codebook_nll(model(hiddens, targets), targets).mean(dim=0)

    torch.testing.assert_close(
        per_cb, torch.tensor(expected, dtype=per_cb.dtype), rtol=3e-3, atol=3e-3
    )
