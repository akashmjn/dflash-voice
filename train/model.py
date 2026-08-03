"""Baseline RVQ depth decoder: the Miso architecture reimplemented on HF Llama.

The depth decoder runs over the *codebook* axis of a single audio frame.  Given
the backbone hidden state ``h_t`` that predicts frame ``t``, the decoder sees

    [ proj(h_t), proj(emb_0(c0)), proj(emb_1(c1)), ..., proj(emb_30(c30)) ]

a length-32 sequence, and predicts codebooks 1..31.  Codebook 0 comes from the
backbone's ``codebook0_head`` and is out of scope here -- this module models
only the residual levels.

Two off-by-ones stack, and both are load-bearing (see dataprep/miso.py:213):
  * slot ``k`` (k=1..31) holds ``emb_{k-1}(c_{k-1})``
  * slot ``k``'s output feeds ``audio_head[k-1]`` to predict ``c_k``
Slot 0's output is unused.  Level 31 is predicted but never fed back.

The teacher-forcing shift (``hiddens[i]`` predicts ``targets[i]``) is already
baked into the WDS shards by dataprep/export_wds.py -- do not re-apply it.

``forward`` runs the Llama layers in an explicit loop rather than calling
``LlamaModel.__call__``.  That costs nothing today and keeps input assembly and
masking as named seams for planned variants (extra previous hidden state,
bidirectional all-at-once prediction, cross-attention to external features).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.masking_utils import create_causal_mask
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel


@dataclass
class DepthDecoderConfig:
    """Architecture of the Miso depth decoder (``llama-300M``, ~302M params)."""

    num_codebooks: int = 32  # levels 0..31; the decoder predicts 1..31
    audio_vocab_size: int = 2051
    backbone_dim: int = 4096
    hidden_size: int = 1536
    num_hidden_layers: int = 8
    num_attention_heads: int = 24
    num_key_value_heads: int = 6
    head_dim: int = 64
    intermediate_size: int = 6912
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500_000.0
    rope_scaling_factor: float = 32.0
    rope_original_max_position: int = 8192
    attention_dropout: float = 0.0

    @property
    def num_residual_levels(self) -> int:
        """Number of codebooks this module predicts (levels 1..K-1)."""
        return self.num_codebooks - 1

    def to_llama_config(self) -> LlamaConfig:
        return LlamaConfig(
            # embed_tokens is never used -- we always feed inputs_embeds.
            vocab_size=1,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            num_hidden_layers=self.num_hidden_layers,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            rms_norm_eps=self.rms_norm_eps,
            attention_bias=False,
            mlp_bias=False,
            attention_dropout=self.attention_dropout,
            # Must be strictly greater than rope_original_max_position or
            # transformers warns; the decoder only ever reaches position 31.
            max_position_embeddings=2 * self.rope_original_max_position,
            rope_parameters={
                "rope_type": "llama3",
                "rope_theta": self.rope_theta,
                "factor": self.rope_scaling_factor,
                "low_freq_factor": 1.0,
                "high_freq_factor": 4.0,
                "original_max_position_embeddings": self.rope_original_max_position,
            },
        )


class MisoRVQDepthDecoder(nn.Module):
    """Predicts RVQ codebooks 1..K-1 for one frame from ``h_t`` and c0..c_{K-2}."""

    def __init__(self, config: DepthDecoderConfig | None = None):
        super().__init__()
        self.config = config or DepthDecoderConfig()
        cfg = self.config

        self.projection = nn.Linear(cfg.backbone_dim, cfg.hidden_size, bias=False)
        # One shared table; level k lives at offset k * audio_vocab_size.  It is
        # backbone-width -- the projection happens after the lookup.
        self.audio_embeddings = nn.Embedding(
            cfg.audio_vocab_size * cfg.num_codebooks, cfg.backbone_dim
        )
        self.decoder = LlamaModel(cfg.to_llama_config())
        # One weight matrix per predicted level: audio_head[j] reads slot j+1.
        self.audio_head = nn.Parameter(
            torch.empty(cfg.num_residual_levels, cfg.hidden_size, cfg.audio_vocab_size)
        )
        # Standard transformer output-head init.  RMSNorm pins the decoder's
        # output to unit scale, so initial logit std ~= 0.02 * sqrt(hidden_size)
        # ~= 0.78 and the untrained model sits a little above ln(vocab).
        nn.init.normal_(self.audio_head, std=0.02)

    # ---- seams for planned variants -------------------------------------

    def _assemble_input(
        self, hiddens: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """(N, backbone_dim), (N, K) -> (N, K, hidden_size) decoder input."""
        cfg = self.config
        context = targets[:, : cfg.num_residual_levels]  # c0..c_{K-2}
        offsets = cfg.audio_vocab_size * torch.arange(
            cfg.num_residual_levels, device=targets.device
        )
        stacked = torch.cat(
            [hiddens.unsqueeze(1), self.audio_embeddings(context + offsets)], dim=1
        )
        return self.projection(stacked)

    def _build_mask(self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor):
        """Causal along the codebook axis.

        Bidirectional attention here would leak ``c_{k+1}`` into the prediction
        of ``c_k`` and silently *improve* NLL, so this is covered by a test.
        """
        return create_causal_mask(
            config=self.decoder.config,
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=None,
            position_ids=position_ids,
        )

    # ---- forward ---------------------------------------------------------

    def forward(self, hiddens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """(N, backbone_dim), (N, K) -> logits (N, K-1, audio_vocab_size)."""
        x = self._assemble_input(hiddens, targets)
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        position_embeddings = self.decoder.rotary_emb(x, position_ids=position_ids)
        mask = self._build_mask(x, position_ids)

        h = x
        for layer in self.decoder.layers[: self.decoder.config.num_hidden_layers]:
            h = layer(h, attention_mask=mask, position_embeddings=position_embeddings)
            if isinstance(h, tuple):
                h = h[0]
        h = self.decoder.norm(h)

        # Slot j+1 -> audio_head[j] -> c_{j+1}.  Slot 0's output is unused.
        return torch.einsum("nid,idv->niv", h[:, 1:, :], self.audio_head)


def codebook_nll(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-frame, per-codebook NLL in nats: (N, K-1, V), (N, K) -> (N, K-1).

    Mirrors dataprep/common.py:audio_frame_metrics -- log_softmax in float32,
    gathering the ground-truth column for each head.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    gold = targets[:, 1:].unsqueeze(-1)  # levels 1..K-1
    return -log_probs.gather(-1, gold).squeeze(-1)


def loss_fn(
    logits: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean CE in nats over codebooks 1..K-1, plus the per-codebook breakdown.

    Returns ``(scalar_loss, per_codebook_nll)`` where the latter is ``(K-1,)``.
    """
    nll = codebook_nll(logits, targets)
    return nll.mean(), nll.mean(dim=0)


def uniform_nll(audio_vocab_size: int = 2051) -> float:
    """NLL of a uniform predictor -- the value a random init should sit at."""
    return math.log(audio_vocab_size)
