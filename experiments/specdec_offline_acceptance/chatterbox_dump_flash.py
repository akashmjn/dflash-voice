"""Teacher-force Chatterbox Flash over the tokenized dumps, for draft logits.

The dataprep pipeline featurizes the AR checkpoint only, so the Flash side of the
draft/target pair has to be produced separately. Reads the tokenized sequences
already on disk and writes `features_flash.pt` / `metadata_flash.json` beside the
AR dump, in the same `FeaturizedSequence` layout.

Flash shares Chatterbox AR's tokenizer, ids and conditioning, so the same tokenized
sequence feeds both. It differs only by an extra input-only `[MASK]` row in
`speech_emb` (8195 vs 8194); `speech_head` keeps 8194 outputs, so the logits are
directly comparable. Its masked block-diffusion forward is not used here -- this is
the plain causal AR pass, which is what an offline acceptance comparison needs.

Usage:
    python experiments/specdec_offline_acceptance/chatterbox_dump_flash.py --rows 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataprep.chatterbox import ChatterboxFeaturizer  # noqa: E402
from dataprep.types import (  # noqa: E402
    SequenceEmbeddingContext,
    TokenizedSequence,
)

FLASH_REPO = "ResembleAI/chatterbox-flash"
FLASH_T3_FILENAME = "t3_flash.safetensors"

#: Flash's speech_emb carries one extra input-only [MASK] row past the AR vocabulary.
NUM_MASK_ROWS = 1


def load_flash_t3(device: str, dtype: torch.dtype = torch.float32):
    """Upstream ``T3`` with ``speech_emb`` widened for Flash's ``[MASK]`` row.

    That row count is the only structural difference; every other tensor loads
    from the Flash checkpoint by name (verified: no missing or unexpected keys).
    """
    import torch.nn as nn
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file

    from chatterbox.models.t3.modules.t3_config import T3Config
    from chatterbox.models.t3.t3 import T3

    hp = T3Config.english_only()
    model = T3(hp)
    model.speech_emb = nn.Embedding(hp.speech_tokens_dict_size + NUM_MASK_ROWS, model.dim)

    state = load_file(str(hf_hub_download(repo_id=FLASH_REPO, filename=FLASH_T3_FILENAME)))
    if "model" in state:
        state = state["model"][0]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Flash T3 weights did not match: {len(missing)} missing "
            f"{sorted(missing)[:3]}, {len(unexpected)} unexpected {sorted(unexpected)[:3]}"
        )
    return model.to(device=device, dtype=dtype).eval()


def dump_row(row_dir: Path, out_dir: Path, featurizer: ChatterboxFeaturizer) -> int:
    sequences, metadata = TokenizedSequence.load_all(row_dir)
    contexts = SequenceEmbeddingContext.load_all(row_dir, count=len(sequences))

    featurized = [
        featurizer.featurize(sequence, context=context)
        for sequence, context in zip(sequences, contexts)
    ]

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        [{"logits": f.logits, "hiddens": f.hiddens} for f in featurized],
        out_dir / "features_flash.pt",
    )
    payload = dict(metadata)
    payload["model"] = "chatterbox-flash"
    (out_dir / "metadata_flash.json").write_text(json.dumps(payload, indent=2) + "\n")
    return len(featurized)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10, help="dump rows 0..N-1")
    parser.add_argument("--dataset", default="expresso")
    parser.add_argument("--artifact", default="chatterbox", help="tokenized/featurized subdir")
    parser.add_argument("--device", default=None, help="defaults to mps/cuda/cpu")
    args = parser.parse_args()

    from dataprep.chatterbox import _default_device

    device = _default_device(args.device)
    print(f"loading Chatterbox Flash on {device}")
    featurizer = ChatterboxFeaturizer(model=load_flash_t3(device), device=device)

    data = REPO_ROOT / "data" / args.dataset
    for row in range(args.rows):
        row_dir = data / "tokenized" / args.artifact / str(row)
        if not row_dir.exists():
            raise SystemExit(f"no tokenized dump at {row_dir}")
        out_dir = data / "featurized" / args.artifact / str(row)
        count = dump_row(row_dir, out_dir, featurizer)
        print(f"  row {row}: {count} sequence(s) -> {out_dir}/features_flash.pt")


if __name__ == "__main__":
    main()
