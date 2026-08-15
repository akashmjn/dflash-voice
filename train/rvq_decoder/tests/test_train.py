"""Smoke check for the depth-decoder trainer."""

from __future__ import annotations

import torch


def test_train_reduces_loss_on_overfit_batch():
    """Repeating one batch must drive the loss down: catches a detached graph,
    a mis-built optimizer, or frozen params."""
    from train.rvq_decoder.model import MisoRVQDepthDecoder, loss_fn
    from train.rvq_decoder.train import smoke_config

    cfg = smoke_config(num_codebooks=4, hidden_size=64, num_hidden_layers=1)
    torch.manual_seed(0)
    model = MisoRVQDepthDecoder(cfg).train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)

    hiddens = torch.randn(8, cfg.backbone_dim)
    targets = torch.randint(0, cfg.audio_vocab_size, (8, cfg.num_codebooks))

    losses = []
    for _ in range(60):
        loss, _ = loss_fn(model(hiddens, targets), targets)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(loss.item())

    assert losses[-1] < 0.5 * losses[0], f"{losses[0]:.3f} -> {losses[-1]:.3f}"
