"""Speculative decoding over Chatterbox speech tokens: cbox-ar target, cbox-nano draft.

Model ids and sampling settings come from ``models.yaml``.
Only token generation is speculated; detokenization is one S3Gen call at the end.
Text tokenizers and speaker conditioning differ, so each model prepares prefix/prefill state separately.

```bash
python mlx_decode/specdec.py --gamma 2 --baseline -n 2
python mlx_decode/specdec.py --prompts-file data/seedtts/prompts/seedtts_test_en_100.jsonl \
    --ref-audio-dir data/seedtts/ref_audio --save-audio --output-dir agent-workspace/seedtts-100
```
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import typer
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
from mlx_lm.sample_utils import apply_min_p, apply_top_p, make_logits_processors
from rich import print

from mlx_decode import cbox_ar, cbox_turbo
from mlx_decode.bench import (
    DEFAULT_PROMPTS_FILE,
    MODELS,
    REF_AUDIO_DIR,
    WARMUP_PROMPT,
    _load_prompts,
)

# Shared speech ids: 0-6560 codes, 6561 BOS, 6562 EOS. This is why the target/draft pair works.
# However, the AR model has an ~8k vocab embedding with untrained padding past ``LIVE_HEAD``
# So both models' logits are truncated there and compared directly.
LIVE_HEAD = 6563

DEFAULT_OUTPUT_DIR = Path("mlx_decode/output/specdec")


# ---------------------------------------------------------------------------
# Steppers -- one `prefill / step / rollback / probs` surface over two T3 variants
# ---------------------------------------------------------------------------


class _Stepper:
    """Shared KV-cache bookkeeping. ``offset`` counts speech positions consumed,
    BOS included, so ``offset - 1`` speech tokens have been fed since prefill."""

    cache: list
    offset: int

    def rollback(self, keep: int) -> None:
        """Drop cache entries past ``keep`` fed speech tokens."""
        # KVCache.trim with a negative count advances the offset instead.
        assert keep <= self.offset - 1, (keep, self.offset)
        trim_prompt_cache(self.cache, self.offset - 1 - keep)
        self.offset = 1 + keep


class ARTarget(_Stepper):
    """The AR T3 at batch 2 for CFG, sampling under ``cbox_ar``'s warpers."""

    def __init__(self, wrapper: cbox_ar.ChatterboxAR, settings: dict):
        self.model = wrapper._model
        self.t3 = self.model.t3
        self.settings = settings
        self.cfg_weight = settings["cfg_weight"]
        self.processors = make_logits_processors(
            logit_bias=None,
            repetition_penalty=settings["repetition_penalty"],
            repetition_context_size=settings["max_tokens"],
        )

    def prefill(self, text: str) -> mx.array:
        self.cache = make_prompt_cache(self.t3.tfmr)
        hidden = cbox_ar._prepare_prompt(self.model, text, self.cfg_weight, self.cache)
        self.offset = 1
        return self._logits(hidden[:, -1:, :])

    def step(self, tokens: List[int]) -> mx.array:
        """Feed ``tokens``, return CFG-combined logits (1, n, LIVE_HEAD)."""
        t3 = self.t3
        ids = mx.array([tokens], dtype=mx.int32)
        pos = mx.arange(self.offset, self.offset + len(tokens))
        embed = t3.speech_emb(ids) + t3.speech_pos_emb.get_fixed_embedding(pos)
        if self.cfg_weight > 0.0:
            embed = mx.concatenate([embed, embed], axis=0)
        hidden = t3.tfmr.model(inputs=None, input_embeddings=embed, cache=self.cache)
        self.offset += len(tokens)
        return self._logits(hidden)

    def _logits(self, hidden: mx.array) -> mx.array:
        logits = self.t3.speech_head(hidden)[..., :LIVE_HEAD].astype(mx.float32)
        if self.cfg_weight > 0.0:
            cond, uncond = logits[0:1], logits[1:2]
            return cond + self.cfg_weight * (cond - uncond)
        return logits[0:1]

    def probs(self, logits: mx.array, history: List[int]) -> mx.array:
        """``cbox_ar``'s sampling distribution: repetition penalty, then mlx-lm's
        sampler chain (top_p, min_p on the untempered logits, temperature last)."""
        s = self.settings
        for processor in self.processors:
            logits = processor(mx.array([history], dtype=mx.int32), logits)
        if 0.0 < s["top_p"] < 1.0:
            logits = apply_top_p(logits, s["top_p"])
        if s["min_p"] > 0.0:
            logits = apply_min_p(logits, s["min_p"])
        return mx.softmax(logits / s["temperature"], axis=-1)[0]


class NanoDraft(_Stepper):
    """The Nano T3 at batch 1, proposing under ``cbox_turbo``'s warpers."""

    def __init__(self, wrapper: cbox_turbo.ChatterboxTurbo, settings: dict):
        self.model = wrapper._model
        self.t3 = self.model.t3
        self.settings = settings

    def prefill(self, text: str) -> mx.array:
        hidden, self.cache = cbox_turbo._prepare_prompt(self.model, text)
        self.offset = 1
        return self._logits(hidden[:, -1:, :])

    def step(self, tokens: List[int]) -> mx.array:
        ids = mx.array([tokens], dtype=mx.int32)
        hidden, self.cache = self.t3.tfmr(inputs_embeds=self.t3.speech_emb(ids), cache=self.cache)
        self.offset += len(tokens)
        return self._logits(hidden)

    def _logits(self, hidden: mx.array) -> mx.array:
        return self.t3.speech_head(hidden)[..., :LIVE_HEAD].astype(mx.float32)

    def probs(self, logits: mx.array, history: List[int]) -> mx.array:
        """``T3._sample_token``'s chain: repetition penalty, temperature, top_k, top_p."""
        s, t3 = self.settings, self.t3
        logits = t3._apply_repetition_penalty(
            logits, mx.array([history], dtype=mx.int32), s["repetition_penalty"]
        )
        logits = logits / s["temperature"]
        if s["top_k"] > 0:
            logits = t3._top_k_filtering(logits, s["top_k"])
        if s["top_p"] < 1.0:
            logits = t3._top_p_filtering(logits, s["top_p"])
        return mx.softmax(logits, axis=-1)[0]


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------


def _sample(probs: mx.array) -> int:
    return int(mx.random.categorical(mx.log(probs)).item())


def speculative_tokens(
    target: ARTarget, draft: NanoDraft, text: str, *, gamma: int, max_tokens: int,
    tolerance: float = 0.0,
) -> tuple[List[int], dict]:
    """Draft gamma tokens, verify them in one target forward, keep the accepted prefix.

    The correction token that ends a round is not fed to the target on its own: it is
    carried in ``pending`` and prepended to the next round's block, so one forward both
    scores the new block and yields the distribution the correction conditions.

    Invariant at the top of each round: the draft's cache holds every emitted token,
    the target's holds all but ``pending``, and ``next_probs`` is the target's
    distribution for the next position.

    ``tolerance`` relaxes acceptance to ``min(1, p/q + tolerance)``; 0 is exact
    speculative sampling, anything above trades fidelity to p for longer runs.

    ``history`` (BOS + everything emitted so far) feeds both models' repetition
    penalties, so p is exactly what ``cbox_ar`` would have sampled from.

    Returns the emitted tokens and ``stats``: ``steps_per_output_token[k]`` counts
    target steps (verify forwards, prefill excluded) that output ``k + 1`` tokens,
    i.e. accepted ``k`` drafts, counted before EOS truncation; plus ``draft_s`` and
    ``verify_s``.
    """
    eos, bos = target.t3.hp.stop_speech_token, target.t3.hp.start_speech_token
    t_draft = t_verify = 0.0
    steps_per_output_token = [0] * (gamma + 1)

    target_logits = target.prefill(text)
    draft_logits = draft.prefill(text)

    history = [bos]
    emitted: List[int] = []
    pending: List[int] = []
    next_probs = target.probs(target_logits[:, -1, :], history)

    while len(emitted) < max_tokens:
        # --- draft gamma tokens autoregressively ---
        t0 = time.perf_counter()
        block, block_probs = [], []
        for i in range(gamma):
            if i:
                draft_logits = draft.step([block[-1]])
            q = draft.probs(draft_logits[:, -1, :], history + block)
            block.append(_sample(q))
            block_probs.append(q)
        t_draft += time.perf_counter() - t0

        # --- verify the block in one forward, pending token first ---
        t0 = time.perf_counter()
        skip = len(pending)
        t_logits = target.step(pending + block)
        mx.eval(t_logits)
        t_verify += time.perf_counter() - t0

        # Row `skip + i - 1` scores block position i; for i = 0 that is the pending
        # row when there is one, else the distribution carried in from last round.
        if skip:
            next_probs = target.probs(t_logits[:, skip - 1, :], history)

        n_accepted, correction = 0, None
        for i, tok in enumerate(block):
            p = next_probs if i == 0 else target.probs(t_logits[:, skip + i - 1, :], history + block[:i])
            q = block_probs[i]
            ratio = min(1.0, p[tok].item() / max(q[tok].item(), 1e-12) + tolerance)
            if mx.random.uniform().item() < ratio:
                n_accepted += 1
                continue
            residual = mx.maximum(p - q, 0.0)
            total = residual.sum().item()
            correction = _sample(residual / total if total > 0 else p)
            break

        steps_per_output_token[n_accepted] += 1
        keep = block[:n_accepted]
        if correction is None:
            # Whole block accepted; the target's last row gives a bonus token for free.
            correction = _sample(target.probs(t_logits[:, skip + gamma - 1, :], history + keep))

        emit = keep + [correction]
        stop = eos in emit
        if stop:
            emit = emit[: emit.index(eos)]
        emitted.extend(emit)
        history.extend(emit)
        if stop or len(emitted) >= max_tokens:
            break

        # The target ingested pending + block but only `keep` survived: roll it back
        # to the emitted prefix and defer the correction to the next round. The draft
        # holds at most block[:-1]; rewind it to the emitted prefix and feed what it
        # is missing -- the correction, plus block[-1] when the whole block was kept.
        target.rollback(len(emitted) - 1)
        pending = [correction]
        fed = min(len(emitted) - 1, draft.offset - 1)
        draft.rollback(fed)
        t0 = time.perf_counter()
        draft_logits = draft.step(emitted[fed:])
        mx.eval(draft_logits)
        t_draft += time.perf_counter() - t0

    stats = dict(steps_per_output_token=steps_per_output_token, draft_s=t_draft, verify_s=t_verify)
    return emitted, stats


def baseline_tokens(target: ARTarget, text: str) -> List[int]:
    """Plain ``cbox_ar`` decode, BOS/EOS stripped, for a same-settings comparison."""
    s = target.settings
    cache = make_prompt_cache(target.t3.tfmr)
    hidden = cbox_ar._prepare_prompt(target.model, text, s["cfg_weight"], cache)
    ids = cbox_ar._generate_tokens(
        target.model, hidden, cache,
        max_tokens=s["max_tokens"], temperature=s["temperature"], top_p=s["top_p"],
        min_p=s["min_p"], repetition_penalty=s["repetition_penalty"], cfg_weight=s["cfg_weight"],
    )
    hp = target.t3.hp
    return [t for t in ids if t not in (hp.start_speech_token, hp.stop_speech_token)]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _rounded(d: dict) -> dict:
    """Floats to 4 decimals, so rows and summaries stay readable."""
    r = lambda v: round(v, 4) if isinstance(v, float) else v  # noqa: E731
    return {k: [r(x) for x in v] if isinstance(v, list) else r(v) for k, v in d.items()}


def acceptance(steps_per_output_token: List[int]) -> dict:
    """Acceptance stats from a step histogram; index k counts steps that accepted k drafts.

    ``alpha`` is accepted / drafted, with every block position counted as drafted.
    ``alpha_by_pos[i]`` scores position i over the steps that reached it.
    """
    gamma = len(steps_per_output_token) - 1
    reached = [sum(steps_per_output_token[i:]) for i in range(gamma + 1)]
    accepted = sum(k * n for k, n in enumerate(steps_per_output_token))
    return {
        "alpha": accepted / max(gamma * reached[0], 1),
        "alpha_by_pos": [reached[i + 1] / max(reached[i], 1) for i in range(gamma)],
    }


def summarize(rows: List[dict]) -> dict:
    """One mode's totals, pooled over prompts so long prompts count more."""
    total = lambda key: sum(r[key] for r in rows)  # noqa: E731
    hist = [sum(col) for col in zip(*(r["steps_per_output_token"] for r in rows))]
    tokens, steps = total("tokens"), max(sum(hist), 1)
    summary = {
        "prompts": len(rows), "tokens": tokens,
        "audio_s": total("audio_s"), "total_s": total("total_s"),
        "gen_ms_per_token": 1000 * total("gen_s") / max(tokens, 1),
        # Baseline has no draft: every generation step is a target step.
        "target_ms_per_step": 1000 * total("gen_s") / steps,
        "tau": tokens / steps,
        "steps_per_output_token": hist,
    }
    if "draft_s" in rows[0]:
        summary["target_ms_per_step"] = 1000 * total("verify_s") / steps
        summary["draft_ms_per_step"] = 1000 * total("draft_s") / steps
        summary.update(acceptance(hist))
    return summary


def write_metrics(out_dir: Path, config: dict, rows: List[dict]) -> dict:
    """Write ``metrics.json`` (config + pooled summary) and ``row_metrics.json`` (one row per line)."""
    summary = _rounded(summarize(rows))
    (out_dir / "metrics.json").write_text(json.dumps({"config": config, "summary": summary}, indent=2))
    lines = ",\n".join(json.dumps(_rounded(row)) for row in rows)
    (out_dir / "row_metrics.json").write_text(f"[\n{lines}\n]\n")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help="Speculative decoding: cbox-nano drafts, cbox-ar verifies.")


def _slug(model_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", model_id.rsplit("/", 1)[-1].lower())


@app.command()
def run(
    target: str = typer.Option("cbox-ar-fp16", help="models.yaml entry for the target (a cbox_ar model)."),
    draft: str = typer.Option("cbox-nano-fp16", help="models.yaml entry for the draft (a cbox_turbo model)."),
    prompts_file: Path = typer.Option(DEFAULT_PROMPTS_FILE, exists=True, dir_okay=False,
                                      help="JSONL with a 'text' per line and an optional 'id'."),
    ref_audio_dir: Path = typer.Option(
        REF_AUDIO_DIR, exists=True, file_okay=False,
        help="Where to find the speaker reference each prompt names in `ref_audio`. "
        "Both models are conditioned on it, per prompt.",
    ),
    gamma: str = typer.Option("2", help="Comma-separated draft block sizes."),
    tolerance: str = typer.Option(
        "0", help="Comma-separated acceptance tolerances added to p/q; each nonzero one "
        "runs as its own mode, e.g. gamma2-tol0.1.",
    ),
    max_samples: Optional[int] = typer.Option(None, "--max-samples", "-n", help="First N prompts only."),
    baseline: bool = typer.Option(False, help="Also run the target's plain AR decode per prompt."),
    save_audio: bool = typer.Option(False, help="Write generated wav files."),
    seed: int = typer.Option(0, help="Reseeded before every decode."),
    warmup: bool = typer.Option(True, help="Untimed short decode before measuring."),
    slug: Optional[str] = typer.Option(
        None,
        help="Name this run's output directory instead of the target/draft checkpoint "
        "pair, so reruns and tweaks of one pair are kept apart.",
    ),
    output_dir: Path = typer.Option(DEFAULT_OUTPUT_DIR, help="Root for wavs, tokens and metrics.json."),
) -> None:
    """Decode a prompt set with speculative decoding and report acceptance statistics.

    Target and draft are conditioned on the same clip per prompt, so the pair is
    compared on one voice. Each mode writes into its own subdirectory --
    ``<target>__<draft>/{baseline,gamma2,...}/`` -- holding ``<id>.wav`` and
    ``<id>.tokens.json`` named by prompt id, as ``bench.py`` names its wavs, plus
    ``row_metrics.json`` (raw measurements per prompt) and ``metrics.json`` (run
    config plus stats pooled from those rows).
    """
    gammas = [int(g) for g in gamma.split(",") if g.strip()]
    tolerances = [float(t) for t in tolerance.split(",") if t.strip()]
    # mode name -> its decode settings, also written into that mode's metrics.json
    modes = {f"gamma{g}" + (f"-tol{t:g}" if t else ""): {"gamma": g, "tolerance": t}
             for g in gammas for t in tolerances}
    target_spec, draft_spec = MODELS[target], MODELS[draft]
    prompts = _load_prompts(prompts_file)[:max_samples]
    missing = [p["id"] for p in prompts if not p.get("ref_audio")]
    if missing:
        raise typer.BadParameter(
            f"every prompt needs a 'ref_audio' naming its speaker clip; {len(missing)} "
            f"lack one (first: {missing[0]})."
        )
    pair = f"{_slug(target_spec['model_id'])}__{_slug(draft_spec['model_id'])}"
    out_root = output_dir / (slug or pair)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Loading target {target_spec['model_id']} and draft {draft_spec['model_id']}")
    ar = cbox_ar.load_model(target_spec["model_id"], **target_spec.get("load_kwargs", {}))
    nano = cbox_turbo.load_model(draft_spec["model_id"])
    tgt = ARTarget(ar, target_spec["generate"])
    drf = NanoDraft(nano, draft_spec["generate"])
    max_tokens = target_spec["generate"]["max_tokens"]

    def set_reference(prompt: dict) -> float:
        """Condition both models on this prompt's clip; returns seconds taken."""
        path = str(ref_audio_dir / Path(prompt["ref_audio"]).name)
        return ar.set_reference(path) + nano.set_reference(path)

    def render(tokens: List[int], mode: str, pid: str) -> tuple[float, float]:
        """Detokenize with the target's S3Gen, save tokens and optionally the wav.

        Detokenization runs either way: its wall time is part of the reported RTF,
        so skipping it without ``--save-audio`` would not measure the same thing.
        """
        t0 = time.perf_counter()
        wav = cbox_ar._decode_tokens(ar._model, tokens)
        detok_s = time.perf_counter() - t0
        out_dir = out_root / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = re.sub(r"[^A-Za-z0-9._-]+", "-", pid)
        if save_audio:
            from mlx_audio.audio_io import write as audio_write

            audio_write(out_dir / f"{stem}.wav", wav, ar.sample_rate, format="wav")
        (out_dir / f"{stem}.tokens.json").write_text(json.dumps(tokens))
        return detok_s, wav.shape[0] / ar.sample_rate

    if warmup:
        # Neither checkpoint can generate before a reference is set, and the AR
        # checkpoint ships no conds at all, so warm up on the first prompt's clip.
        set_reference(prompts[0])
        tokens, _ = speculative_tokens(tgt, drf, WARMUP_PROMPT, gamma=gammas[0], max_tokens=64)
        cbox_ar._decode_tokens(ar._model, tokens)

    rows: dict[str, List[dict]] = {}

    def record(mode: str, prompt: dict, tokens: List[int], gen_s: float, cond_s: float,
               steps_per_output_token: List[int], **timings) -> None:
        detok_s, audio_s = render(tokens, mode, prompt["id"])
        row = {
            "id": prompt["id"], "ref_audio": prompt["ref_audio"],
            "tokens": len(tokens), "steps_per_output_token": steps_per_output_token,
            "audio_s": audio_s, "total_s": gen_s + detok_s + cond_s,
            "gen_s": gen_s, "detok_s": detok_s, "cond_s": cond_s, **timings,
        }
        rows.setdefault(mode, []).append(row)
        line = (f"  {mode:9s} {len(tokens):>4} tok  {1000 * gen_s / max(len(tokens), 1):5.1f} ms/tok"
                f"  RTF {row['total_s'] / max(audio_s, 1e-9):.2f}")
        if timings:
            line += f"  alpha {acceptance(steps_per_output_token)['alpha']:.2f}"
        print(line)

    for prompt in prompts:
        text = prompt["text"]
        print(f"\n[bold]{prompt['id']}[/bold]: {text[:60]!r}")
        cond_s = set_reference(prompt)

        if baseline:
            mx.random.seed(seed)
            mx.clear_cache()
            t0 = time.perf_counter()
            tokens = baseline_tokens(tgt, text)
            record("baseline", prompt, tokens, time.perf_counter() - t0, cond_s, [len(tokens)])

        for mode, settings in modes.items():
            mx.random.seed(seed)
            mx.clear_cache()
            t0 = time.perf_counter()
            tokens, stats = speculative_tokens(tgt, drf, text, max_tokens=max_tokens, **settings)
            record(mode, prompt, tokens, time.perf_counter() - t0, cond_s, **stats)

    config = {
        "target": target, "target_id": target_spec["model_id"],
        "target_load_kwargs": target_spec.get("load_kwargs", {}),
        "draft": draft, "draft_id": draft_spec["model_id"],
        "target_settings": target_spec["generate"], "draft_settings": draft_spec["generate"],
        "seed": seed, "prompts_file": str(prompts_file), "ref_audio_dir": str(ref_audio_dir),
    }
    print(f"\n{'=' * 50}")
    for mode, mode_rows in rows.items():
        mode_cfg = modes.get(mode, {"gamma": None, "tolerance": None})
        agg = write_metrics(out_root / mode, {**mode_cfg, **config}, mode_rows)
        line = (f"[bold]{mode:9s}[/bold] {agg['tokens']:>5} tok  {agg['gen_ms_per_token']:5.1f} ms/tok"
                f"  RTF {agg['total_s'] / max(agg['audio_s'], 1e-9):.2f}  tau {agg['tau']:.2f}")
        if "alpha" in agg:
            line += f"  alpha {agg['alpha']:.3f}"
        print(line)

    print(f"\nSaved metrics.json and row_metrics.json under {out_root}/<mode>/")

if __name__ == "__main__":
    app()
