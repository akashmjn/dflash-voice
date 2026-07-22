import json
import math
import os
from pathlib import Path

import pytest


REFERENCE_ROOT = Path(
    os.environ.get(
        "MISO_ENTROPY_REFERENCE",
        Path(__file__).resolve().parents[3] / "MisoTTS/mimi_entropy_work/expresso_row_0",
    )
)


def _entropy(logits):
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@pytest.mark.integration
def test_prepared_single_segment_matches_saved_miso_entropy():
    if os.environ.get("DFLASH_RUN_MISO_FORWARD") != "1":
        pytest.skip("set DFLASH_RUN_MISO_FORWARD=1 to load Miso 8B")

    import torch
    from generator import load_miso_8b, resolve_inference_config
    from torchtune.modules.common_utils import disable_kv_cache

    from dataprep.miso import (
        MisoTextTokenizer,
        MisoTokenizer,
        build_teacher_forcing_batch,
    )
    from dataprep.tokenizer import Segment

    reference = json.loads((REFERENCE_ROOT / "miso_mimi_entropy.json").read_text())
    transcript = json.loads((REFERENCE_ROOT / "transcript.json").read_text())
    expected = reference["turns"][0]
    turn = transcript["turns"][expected["turn_index"]]
    payload = torch.load(
        REFERENCE_ROOT / f"channel_{expected['channel']}_mimi_codes.pt",
        map_location="cpu",
        weights_only=True,
    )
    full_codes = payload["codes"]
    frame_rate = float(payload.get("frame_rate", 12.5))
    start = math.floor(float(turn["start_time_ms"]) * frame_rate / 1000)
    end = math.ceil(float(turn["end_time_ms"]) * frame_rate / 1000)
    codes = full_codes[:, start:end].long()

    class CodecInfo:
        num_codebooks = 32
        sample_rate = 24_000
        frame_rate = 12.5

    tokenizer = MisoTokenizer(audio_codec=CodecInfo(), text_tokenizer=MisoTextTokenizer())
    prepared = tokenizer.apply_chat_template(
        [Segment(text=turn["text"], speaker=expected["speaker_id"], audio_codes=codes)]
    )
    batch = build_teacher_forcing_batch(prepared)
    tokens, token_mask, targets, target_mask, decoder_idx = batch
    assert tokens.shape == (1, prepared.spans[0].end + codes.shape[1] - 1, 33)
    assert decoder_idx.shape[1] == expected["num_frames"] == codes.shape[1]

    config = resolve_inference_config(device=os.environ.get("MISO_FORWARD_DEVICE"))
    generator = load_miso_8b(device=config.model_device, dtype=config.dtype)
    device_batch = [value.to(generator.model_device) for value in batch]
    with torch.inference_mode(), disable_kv_cache(
        generator._model.backbone
    ), disable_kv_cache(generator._model.decoder):
        c0_logits, c1_logits, *_ = generator._model.forward(*device_batch)

    target_positions = device_batch[-1][0]
    c0 = _entropy(c0_logits)[0, target_positions]
    c1 = _entropy(c1_logits)[0]
    actual = torch.cat([c0[:, None], c1], dim=1).cpu()
    saved = torch.tensor(expected["codebook_entropy_per_frame"], dtype=torch.float32)
    assert actual.shape == saved.shape == (expected["num_frames"], 32)
    torch.testing.assert_close(actual, saved, atol=3e-3, rtol=3e-3)
    torch.testing.assert_close(
        actual.mean(dim=0),
        torch.tensor(expected["codebook_entropy"]),
        atol=3e-3,
        rtol=3e-3,
    )
