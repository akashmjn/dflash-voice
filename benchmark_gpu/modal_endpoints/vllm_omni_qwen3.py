"""
Qwen3-TTS on Modal, served via vLLM-Omni.

Based on the vLLM-Omni recipe:
  https://github.com/vllm-project/vllm-omni/blob/main/recipes/Qwen/Qwen3-TTS.md
  https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/

Usage:
    modal run   qwen3_vllm_omni.py::download_model   # once: populate the weights Volume
    modal serve qwen3_vllm_omni.py                   # ephemeral, hot-reload -- use this to smoke test
    modal deploy qwen3_vllm_omni.py                  # persistent URL -- use this for the benchmark run

Exposes an OpenAI-compatible speech API:
    POST /v1/audio/speech
    POST /v1/audio/speech/batch
    GET  /v1/audio/voices

Task/model must match (one checkpoint per server):
    CustomVoice  -> Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-CustomVoice
    VoiceDesign  -> Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
    Base (clone) -> Qwen/Qwen3-TTS-12Hz-{0.6B,1.7B}-Base
"""

import modal

# Cookbook default; swap to 0.6B-CustomVoice (or Base/VoiceDesign) if needed.
MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
PORT = 8091
MINUTES = 60
GPU = "L40S"  # L40S for debugging, H100:1 for benchmarks

# vLLM-Omni major.minor must match vLLM.
# Qwen3-TTS needs onnxruntime + sox for the speech VQ / audio path (not always
# pulled in transitively; see vllm-omni#945 / #981).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.2-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("ffmpeg", "sox")
    .uv_pip_install(
        "vllm==0.24.0",
        "vllm-omni==0.24.0",
        "huggingface_hub",
        "onnxruntime",
        "sox>=1.5.0",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})  # successor to HF_HUB_ENABLE_HF_TRANSFER
)

app = modal.App("0626-qwen3-tts")
hf_cache = modal.Volume.from_name("0626-hf-cache", create_if_missing=True)


# ---------------------------------------------------------------------------
# 1. One-time weight download. Keeps it out of the serving cold start.
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
@modal.concurrent(max_inputs=10)
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
        "--trust-remote-code",
        # Deploy config auto-loads from vllm_omni/deploy/qwen3_tts.yaml
        # (HF model_type == qwen3_tts), so no --deploy-config needed.
        #
        # Stage 0 = talker, stage 1 = code2wav. Cookbook throughput tip:
        # keep both max_num_seqs at 10 (don't force stage 1 back to 1).
        "--stage-overrides",
        '{"0":{"max_num_seqs":10,"gpu_memory_utilization":0.6},'
        ' "1":{"max_num_seqs":10,"gpu_memory_utilization":0.1}}',
    ]
    subprocess.Popen(cmd)
