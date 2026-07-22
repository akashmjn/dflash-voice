import json
import os
from pathlib import Path

import pytest

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "miso_forward"


def _entropy(logits):
    import torch

    log_probs = torch.log_softmax(logits.float(), dim=-1)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


@pytest.fixture(scope="module")
def miso_forward_reference():
    """Segment-0 Expresso codes and saved Miso predictive entropy."""
    import torch

    meta = json.loads((FIXTURE_DIR / "segment0.json").read_text(encoding="utf-8"))
    payload = torch.load(FIXTURE_DIR / "codes.pt", map_location="cpu", weights_only=True)
    codes = payload["codes"].long()
    segment = meta["segment"]
    assert codes.ndim == 2 and codes.shape == (
        meta["num_codebooks"],
        segment["num_frames"],
    )
    return {
        "meta": meta,
        "segment": segment,
        # Dataprep uses seq-major (F, C).
        "codes": codes.transpose(0, 1).contiguous(),
        "frame_rate": float(meta["frame_rate"]),
        "sample_rate": int(meta["sample_rate"]),
        "num_codebooks": int(meta["num_codebooks"]),
    }


@pytest.mark.integration
def test_prepared_single_segment_matches_saved_miso_entropy(miso_forward_reference):
    if os.environ.get("DFLASH_RUN_MISO_FORWARD") != "1":
        pytest.skip("set DFLASH_RUN_MISO_FORWARD=1 to load Miso 8B")

    import torch
    from generator import load_llama3_tokenizer, load_miso_8b, resolve_inference_config
    from torchtune.modules.common_utils import disable_kv_cache

    from dataprep.miso import MisoTokenizer, build_teacher_forcing_batch
    from dataprep.tokenizer import Segment

    expected = miso_forward_reference["segment"]
    codes = miso_forward_reference["codes"]

    class CodecInfo:
        num_codebooks = miso_forward_reference["num_codebooks"]
        sample_rate = miso_forward_reference["sample_rate"]
        frame_rate = miso_forward_reference["frame_rate"]

    tokenizer = MisoTokenizer(
        audio_codec=CodecInfo(), text_tokenizer=load_llama3_tokenizer()
    )
    prepared = tokenizer.apply_chat_template(
        [
            Segment(
                text=expected["text"],
                speaker=expected["speaker_id"],
                audio_codes=codes,
            )
        ]
    )
    batch = build_teacher_forcing_batch(prepared)
    tokens, token_mask, targets, target_mask, decoder_idx = batch
    assert tokens.shape == (1, prepared.spans[0].end + codes.shape[0] - 1, 33)
    assert decoder_idx.shape[1] == expected["num_frames"] == codes.shape[0]

    config = resolve_inference_config(device=os.environ.get("MISO_FORWARD_DEVICE"))
    generator = load_miso_8b(device=config.model_device, dtype=config.dtype)
    device_batch = [value.to(generator.model_device) for value in batch]
    with (
        torch.inference_mode(),
        disable_kv_cache(generator._model.backbone),
        disable_kv_cache(generator._model.decoder),
    ):
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
