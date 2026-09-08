"""Coarse-grained (PCG) speculative verification over acoustic similarity groups.

Principled Coarse-Graining (arXiv 2511.13732) verifies a drafted token at the level
of its acoustic similarity group rather than the token itself: audio tokens inside a
group are perceptually interchangeable, so a draft that picks the "wrong" member of
the right group is not worth a rejection.

Groups come from the dumps ``experiments/audio_token_clustering_pcg`` writes, one file
per theta, so that experiment must run first. ``simulate_acceptance.py`` reaches this
through its ``--theta`` option.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "audio_token_clustering_pcg")
)

from cluster import cluster_path  # noqa: E402


class CoarseGroups:
    """Acoustic similarity groups, ``G(t) = {t' : cos(Emb(t), Emb(t')) > theta}``.

    One group is seeded per token. Groups overlap, so a token's mass is split equally
    across the ``owner_counts[t]`` groups holding it; those weights partition the
    distribution, leaving group masses summing to 1.

    Groups cover the trained speech codes only. Ids above them (EOS, padding) get
    singleton groups so the table spans the full head width.
    """

    def __init__(
        self,
        groups: list[list[int]],
        owner_counts: torch.Tensor,
        theta: float,
        vocab: int,
        width: int,
    ) -> None:
        # Flattened once here: `mass` scores all ~155k (group, token) pairs in a
        # single index_add_, so the hot path wants parallel arrays, not nesting.
        sizes = torch.tensor([len(g) for g in groups])
        members = torch.tensor([t for g in groups for t in g], dtype=torch.long)
        seeds = torch.repeat_interleave(torch.arange(vocab), sizes)
        weights = 1.0 / owner_counts[members].to(torch.float32)

        tail = torch.arange(vocab, width)
        self.seeds = torch.cat([seeds, tail])
        self.members = torch.cat([members, tail])
        self.weights = torch.cat([weights, torch.ones(width - vocab)])
        self.theta = theta
        self.width = width
        self.mean_size = float(members.numel() / vocab)

        # Which groups own each token, so a sampled id can draw one of its owners.
        order = torch.argsort(self.members, stable=True)
        self.owners = self.seeds[order]
        counts = torch.bincount(self.members, minlength=width)
        self.owner_indptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)])

    @classmethod
    def load(
        cls, results_dir: Path, model_slug: str, theta: float, width: int
    ) -> "CoarseGroups":
        """One theta's cluster dump, padded out to ``width``."""
        path = cluster_path(results_dir, model_slug, theta)
        if not path.exists():
            raise SystemExit(
                f"no cluster dump at {path}\n  Run: python "
                f"experiments/audio_token_clustering_pcg/cluster.py --dump-thetas {theta}"
            )
        entry = json.loads(path.read_text())
        return cls(
            entry["groups"],
            torch.as_tensor(entry["owner_counts"], dtype=torch.long),
            float(entry["theta"]),
            int(entry["vocab"]),
            width,
        )

    def mass(self, probs: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
        """``(frames, width)`` mass per group, from token probabilities.

        Chunked over frames: the gather is (frames, nnz), which at ~155k nonzeros
        would otherwise allocate gigabytes for a full row's worth of audio.
        """
        out = torch.zeros(probs.shape[0], self.width, dtype=probs.dtype)
        for lo in range(0, probs.shape[0], chunk):
            block = probs[lo : lo + chunk]
            out[lo : lo + chunk].index_add_(
                1, self.seeds, block[:, self.members] * self.weights
            )
        return out

    def draw_owner(self, tokens: torch.Tensor, rng: torch.Generator) -> torch.Tensor:
        """A group holding each token, drawn uniformly over its ``|S(t)|`` owners.

        The equal split is what makes the uniform draw correct: mass ``p(t)/|S(t)|``
        went to each owner, so routing the token back to each with probability
        ``1/|S(t)|`` keeps the group-level distribution exact.
        """
        lo = self.owner_indptr[tokens]
        size = self.owner_indptr[tokens + 1] - lo
        draws = torch.rand(tokens.shape[0], generator=rng)
        return self.owners[lo + (draws * size).to(torch.long).clamp_max_(size - 1)]


def coarse_accept(
    token: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    groups: CoarseGroups,
    rng: torch.Generator,
) -> torch.Tensor:
    """Per-frame acceptance for tokens already drawn from the draft.

    Routes each token to one of its groups and accepts with
    ``min(1, target(G)/draft(G))``, so a draft token survives whenever an
    acoustically similar token carries the target's mass instead.
    """
    group = groups.draw_owner(token, rng)
    draft_mass = groups.mass(draft_probs).gather(1, group[:, None]).squeeze(1)
    target_mass = groups.mass(target_probs).gather(1, group[:, None]).squeeze(1)

    ratio = torch.where(
        draft_mass <= 0.0,
        torch.ones_like(draft_mass),
        torch.clamp(target_mass / draft_mass.clamp_min(1e-30), max=1.0),
    )
    return (
        torch.rand(draft_probs.shape[0], generator=rng, dtype=draft_probs.dtype) < ratio
    )
