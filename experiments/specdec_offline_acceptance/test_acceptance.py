"""Limit-case checks for the acceptance core.

Run directly (``python test_acceptance.py``); no pytest needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from simulate_acceptance import (
    count_accepted,
    masked_softmax,
    sample_tokens,
    speculative_step,
    tau,
)

FRAMES, VOCAB = 200_000, 64


def _dists(seed: int = 0, spread: float = 0.5):
    g = torch.Generator().manual_seed(seed)
    draft_logits = torch.randn(FRAMES, VOCAB, generator=g)
    target_logits = draft_logits + spread * torch.randn(FRAMES, VOCAB, generator=g)
    return masked_softmax(draft_logits, None), masked_softmax(target_logits, None)


def test_samplers_agree_with_the_distribution() -> None:
    """Both methods reproduce the categorical they are handed."""
    probs = masked_softmax(torch.randn(1, 8), None).repeat(FRAMES, 1)
    for method in ("inverse-cdf", "multinomial"):
        rng = torch.Generator().manual_seed(0)
        drawn = sample_tokens(probs, rng, method)
        empirical = torch.stack([(drawn == k).float().mean() for k in range(8)])
        assert torch.allclose(empirical, probs[0], atol=3e-3), method


def test_alpha_matches_the_analytic_rate() -> None:
    """alpha converges to sum_x min(draft(x), target(x))."""
    draft, target = _dists()
    analytic = torch.minimum(draft, target).sum(dim=-1).mean().item()

    rng = torch.Generator().manual_seed(0)
    _, accepted = speculative_step(draft, target, rng)
    assert abs(accepted.float().mean().item() - analytic) < 5e-3


def test_identical_models_always_accept() -> None:
    """Draft == target means the ratio is 1 everywhere."""
    draft, _ = _dists()
    rng = torch.Generator().manual_seed(0)
    _, accepted = speculative_step(draft, draft, rng)
    assert accepted.all()


def test_mask_excludes_dead_ids() -> None:
    """Masked ids get exactly zero probability, and rows still sum to 1."""
    logits = torch.randn(1000, VOCAB)
    mask = torch.zeros(VOCAB, dtype=torch.bool)
    mask[:10] = True
    probs = masked_softmax(logits, mask)
    assert probs[:, 10:].sum() == 0.0
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1000))
    assert not probs.isnan().any()


def test_count_accepted_trims_to_the_shorter_side() -> None:
    draft, target = _dists()
    rng = torch.Generator().manual_seed(0)
    _, frames = count_accepted(draft[:100], target[:80], rng)
    assert frames == 80


def test_tau_limits() -> None:
    """alpha=1 accepts the whole block; alpha=0 accepts only the bonus token."""
    assert tau(1.0, 5) == 6.0
    assert tau(0.0, 5) == 1.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"  ok  {name}")
    print("\nall checks passed")
