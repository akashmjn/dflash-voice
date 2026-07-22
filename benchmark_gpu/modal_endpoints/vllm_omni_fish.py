"""
Fish Speech S2 Pro on Modal, served via vLLM-Omni.

Usage:
    modal run   fish_vllm_omni.py::download_model   # once: populate the weights Volume
    modal serve fish_vllm_omni.py                   # ephemeral, hot-reload -- use this to smoke test
    modal deploy fish_vllm_omni.py                  # persistent URL -- use this for the benchmark run

Exposes an OpenAI-compatible speech API:
    POST /v1/audio/speech
    POST /v1/audio/speech/batch
    GET  /v1/audio/voices
"""

import modal

MODEL_NAME = "fishaudio/s2-pro"
PORT = 8091
MINUTES = 60
GPU = "A100-80GB:1"  # need A100>45GB mem for debugging, L4/L40S insufficient, H100:1 for benchmarks

# vLLM-Omni major.minor must match vLLM.
#
# Fish DAC path: vllm-omni imports fish_speech.models.dac.{modded_dac,rvq}.
# Those need fish-speech + descript-* which are NOT transitive deps of vllm.
# Resolved via uv pip compile against a Modal-like target (see tmp/fish-deps/):
#   uv pip compile --python-version 3.12 --python-platform x86_64-manylinux_2_28
#
# Install order:
#   1) vllm stack
#   2) fish-speech / descript-* with --no-deps (conflict pins; see below)
#   3) eager runtime deps those packages need, filtered against the vllm tree
#
# --no-deps conflict pins:
#   fish-speech: pydantic==2.9.2 / numpy<=1.26.4
#   descript-audiotools: protobuf>=3.9.2,<3.20  (vllm needs protobuf>=5.29)
#   descript-audio-codec: depends on descript-audiotools (avoid re-resolve)
#
# Already covered by vllm==0.24.0 (do not re-add): omegaconf, soundfile, einops,
# numpy, scipy, torch, torchaudio, tqdm, rich, protobuf, pydantic,
# antlr4-python3-runtime.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.24.0",
        "vllm-omni==0.24.0",
        "huggingface_hub",
    )
    # --no-deps: install package layouts only. Exact conflicts vs vllm==0.24.0:
    #   fish-speech==0.1.0        → pydantic==2.9.2 (vllm needs pydantic>=2.12.0)
    #                             → numpy<=1.26.4  (vllm pulls numpy 2.x)
    #   descript-audiotools==0.7.2 → protobuf>=3.9.2,<3.20 (vllm needs protobuf>=5.29)
    #   descript-audio-codec==1.0.0 → depends on descript-audiotools (would re-trigger
    #                                 the protobuf pin if resolved normally)
    .uv_pip_install(
        "descript-audiotools==0.7.2",
        "descript-audio-codec==1.0.0",
        "fish-speech==0.1.0",
        extra_options="--no-deps",
    )
    # Manually add back eager runtime deps of fish → dac → audiotools, not already in vllm.
    .uv_pip_install(
        "hydra-core>=1.3.2",
        "librosa>=0.10.1",
        "argbind",
        "julius",
        "ffmpy",
        "flatten-dict",
        "importlib-resources",
        "randomname",
        "tensorboard",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # successor to HF_HUB_ENABLE_HF_TRANSFER
)

app = modal.App("0626-fish-s2-tts")
hf_cache = modal.Volume.from_name("0626-hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# 1. One-time weight download (~11 GB). Keeps it out of the serving cold start.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    volumes={"/root/.cache/huggingface": hf_cache},
    timeout=30 * MINUTES,
)
def download_model():
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_NAME)
    print(f"downloaded {MODEL_NAME}")


# ---------------------------------------------------------------------------
# 2. The server.
# ---------------------------------------------------------------------------
@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache},
    scaledown_window=2 * MINUTES,  # short while debugging; restore 15 for benchmarks
    timeout=60 * MINUTES,
)
# max_inputs MUST be >= your target concurrency, or Modal will scale out to a
# second GPU instead of letting vLLM batch on one -- which would mean you're
# benchmarking the autoscaler, not the engine.
@modal.concurrent(max_inputs=16)
@modal.web_server(
    port=PORT,
    startup_timeout=15 * MINUTES,  # weight load + CUDA graph capture
    requires_proxy_auth=True,      # don't leave a GPU open to the internet
)
def serve():
    import subprocess

    cmd = [
        "vllm", "serve", MODEL_NAME,
        "--omni",
        "--host", "0.0.0.0",
        "--port", str(PORT),
        # Deploy config auto-loads from vllm_omni/deploy/fish_qwen3_omni.yaml
        # (HF model_type == fish_qwen3_omni), so no --deploy-config needed.
        #
        # Do NOT add --enforce-eager: it disables CUDA graphs and caps throughput.
        #
        # Stage-level batching. Stage 0 = Slow AR + Fast AR, stage 1 = DAC codec.
        # Both need max_num_seqs > 1 to batch across in-flight requests.
        "--stage-overrides",
        '{"0":{"max_num_seqs":16,"gpu_memory_utilization":0.7},'
        ' "1":{"max_num_seqs":16,"gpu_memory_utilization":0.1}}',
    ]
    subprocess.Popen(cmd)
