# Demo: two-speaker podcast

Renders a ~1min scripted two-speaker conversation with MLX TTS, using mlx-audio's native model APIs.

```bash
uv pip install -e ".[benchmark_mlx]"
python demo/demo_tts_podcast.py render --model miso --max-segments 6
```

Writes default `demo/demo_script.yaml` to`demo/output/podcast_miso.wav`, plus one wav per turn if `--save-turns` is enabled.

```yaml
segments:
  - speaker: 0
    text: So I finally got the whole thing running on my laptop last night.
  - speaker: 1
    text: Wait, the full model? On a laptop?
```

Result is generated at 1.8-2.1x RTF and ~11 GB peak memory (on M1 Apple Silicon for Miso 8-bit). As seen in `benchmark_mlx` this is counter-intuitively dominated by the 300M depth decoder generating RVQ audio tokens.

## Notes

- **Context.** This helps generate a two-speaker conversation with natural delivery, hence we use the Miso checkpoint of the Sesame CSM TTS model to generate with interleaved context controlled by `--context-seconds`.  
- **Miso/CSM context instability.** I've noticed the sampling with `--no-warmup` is very prone to collapsing off-script, even though validation NLL is good, likely due to the model being a raw base model. Two clips from another TTS model are used as context (discarded from output) that can be re-rolled with `demo_tts_podcast.py warmup` if needed. Speaker voices still don't track consistently across turns, but fixes content and delivery.
- **Fish backend.** `--model fish` is not wired up yet and exits with a message.  
Fish has no `context`/`Segment` equivalent, so cross-turn continuity there  
needs a different approach.

Two key flags in MLX .generate() usage:

- `voice_match=False` — when on, mlx-audio collapses context down to `context[0]` and folds its text into the prompt. That is meant for single-reference voice cloning and would silently destroy multi-turn context.
- `split_pattern=None` — otherwise a turn gets re-split on newlines into  
several generations.

