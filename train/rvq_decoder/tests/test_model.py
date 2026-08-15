"""Smoke checks for the baseline RVQ depth decoder."""

from __future__ import annotations

import torch


def test_forward_shapes_and_near_chance_at_init():
    """Shapes hold, and an untrained head carries no information.

    Init NLL sits slightly *above* ln(vocab): a random-but-nonzero head is
    confidently wrong on random targets. Being near chance and not below it is
    the point.
    """
    from train.rvq_decoder.model import MisoRVQDepthDecoder, loss_fn, uniform_nll

    torch.manual_seed(0)
    model = MisoRVQDepthDecoder().eval()
    n = 64
    hiddens = torch.randn(n, model.config.backbone_dim)
    targets = torch.randint(0, model.config.audio_vocab_size, (n, model.config.num_codebooks))
    with torch.no_grad():
        logits = model(hiddens, targets)
        loss, per_cb = loss_fn(logits, targets)

    assert logits.shape == (n, 31, model.config.audio_vocab_size)
    assert per_cb.shape == (31,)
    chance = uniform_nll()
    assert chance - 0.05 < loss.item() < chance + 1.0, f"got {loss.item()}, chance {chance:.3f}"


def test_causal_along_codebook_axis():
    """Perturbing c_k must not change predictions for levels <= k.

    Bidirectional attention here would leak later codebooks into earlier ones
    and silently *improve* NLL.
    """
    from train.rvq_decoder.model import MisoRVQDepthDecoder

    torch.manual_seed(0)
    model = MisoRVQDepthDecoder().eval()
    v = model.config.audio_vocab_size
    hiddens = torch.randn(1, model.config.backbone_dim)
    targets = torch.randint(0, v, (1, model.config.num_codebooks))

    with torch.no_grad():
        base = model(hiddens, targets)
    k = 10
    bumped = targets.clone()
    bumped[0, k] = (bumped[0, k] + 1) % v
    with torch.no_grad():
        after = model(hiddens, bumped)

    # logits index j predicts level j+1 and reads context c0..cj.
    torch.testing.assert_close(base[:, :k, :], after[:, :k, :], rtol=0, atol=1e-5)
    assert (base[:, k:, :] - after[:, k:, :]).abs().max() > 1e-5
