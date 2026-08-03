"""Map published Miso checkpoint weights onto ``train.model.MisoRVQDepthDecoder``.

The checkpoint was produced by torchtune's ``llama3_2`` builder, so it differs
from HF Llama in two ways:

1. Names.  ``attn.output_proj`` -> ``self_attn.o_proj``, ``mlp.w1/w2/w3`` ->
   ``mlp.gate_proj/down_proj/up_proj``, ``sa_norm.scale`` ->
   ``input_layernorm.weight``, and so on.

2. RoPE convention.  torchtune pairs head dims interleaved (``x.reshape(...,
   -1, 2)``); HF pairs split-half (``rotate_half``).  The two are *exactly*
   equivalent once ``q_proj``/``k_proj`` output rows are permuted per head --
   verified to 0.0 max abs difference.  This is the same permutation Meta's
   ``convert_llama_weights_to_hf`` applies.

Only the ~76 decoder-side tensors are read.  The 8B backbone in the same file
is never materialized.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import torch

# The mlx-community bf16 repo has byte-identical key names to the fp32 PyTorch
# repo at half the size (16.4 GB vs 32.8 GB), so it is the better default.
BF16_REPO = "models--mlx-community--MisoLabs-MisoTTS-bf16"
FP32_REPO = "models--MisoLabs--MisoTTS"

_LAYER_RENAMES = {
    "attn.q_proj.weight": "self_attn.q_proj.weight",
    "attn.k_proj.weight": "self_attn.k_proj.weight",
    "attn.v_proj.weight": "self_attn.v_proj.weight",
    "attn.output_proj.weight": "self_attn.o_proj.weight",
    "mlp.w1.weight": "mlp.gate_proj.weight",
    "mlp.w2.weight": "mlp.down_proj.weight",
    "mlp.w3.weight": "mlp.up_proj.weight",
    "sa_norm.scale": "input_layernorm.weight",
    "mlp_norm.scale": "post_attention_layernorm.weight",
}

_TOP_LEVEL_RENAMES = {
    "projection.weight": "projection.weight",
    "audio_embeddings.weight": "audio_embeddings.weight",
    "audio_head": "audio_head",  # bare Parameter -- no ".weight" suffix
    "decoder.norm.scale": "decoder.norm.weight",
}


def permute_for_hf_rope(w: torch.Tensor, num_heads: int, head_dim: int) -> torch.Tensor:
    """Reorder q/k output rows from torchtune's interleaved RoPE to HF's split-half."""
    out_dim, in_dim = w.shape
    expected = num_heads * head_dim
    if out_dim != expected:
        raise ValueError(
            f"expected {expected} rows for {num_heads} heads x {head_dim} dim, got {out_dim}"
        )
    return w.view(num_heads, head_dim // 2, 2, in_dim).transpose(1, 2).reshape(out_dim, in_dim)


def decoder_source_keys(num_layers: int = 8) -> list[str]:
    """Checkpoint keys this converter reads -- everything else is skipped."""
    keys = list(_TOP_LEVEL_RENAMES)
    for i in range(num_layers):
        keys.extend(f"decoder.layers.{i}.{suffix}" for suffix in _LAYER_RENAMES)
    return keys


def convert_miso_decoder_state_dict(
    src: dict[str, torch.Tensor],
    *,
    num_layers: int = 8,
    num_attention_heads: int = 24,
    num_key_value_heads: int = 6,
    head_dim: int = 64,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Remap checkpoint tensors to ``MisoRVQDepthDecoder.state_dict()`` names."""
    out: dict[str, torch.Tensor] = {}

    for src_key, dst_key in _TOP_LEVEL_RENAMES.items():
        if src_key not in src:
            raise KeyError(f"missing checkpoint key: {src_key}")
        out[dst_key] = src[src_key].to(dtype)

    for i in range(num_layers):
        for suffix, dst_suffix in _LAYER_RENAMES.items():
            src_key = f"decoder.layers.{i}.{suffix}"
            if src_key not in src:
                raise KeyError(f"missing checkpoint key: {src_key}")
            w = src[src_key].to(dtype)
            if suffix == "attn.q_proj.weight":
                w = permute_for_hf_rope(w, num_attention_heads, head_dim)
            elif suffix == "attn.k_proj.weight":
                w = permute_for_hf_rope(w, num_key_value_heads, head_dim)
            out[f"decoder.layers.{i}.{dst_suffix}"] = w

    return out


def find_checkpoint(repo: str = BF16_REPO) -> Path:
    """Locate a Miso safetensors file in the local HF cache."""
    cache = os.environ.get(
        "HF_HUB_CACHE", os.path.expanduser("~/.cache/huggingface/hub")
    )
    matches = sorted(glob.glob(os.path.join(cache, repo, "snapshots", "*", "model.safetensors")))
    if not matches:
        raise FileNotFoundError(
            f"no model.safetensors for {repo} under {cache}; download it first"
        )
    return Path(matches[-1])


def load_miso_decoder_state_dict(
    path: str | Path | None = None,
    *,
    repo: str = BF16_REPO,
    num_layers: int = 8,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """Read only the decoder tensors from the checkpoint and convert them."""
    from safetensors import safe_open

    path = Path(path) if path is not None else find_checkpoint(repo)
    wanted = decoder_source_keys(num_layers)
    with safe_open(str(path), framework="pt") as f:
        available = set(f.keys())
        missing = [k for k in wanted if k not in available]
        if missing:
            raise KeyError(f"checkpoint {path} is missing {len(missing)} keys, e.g. {missing[:3]}")
        src = {k: f.get_tensor(k) for k in wanted}
    return convert_miso_decoder_state_dict(src, num_layers=num_layers, dtype=dtype)
