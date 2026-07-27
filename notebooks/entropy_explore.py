import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import math
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import numpy as np

    return Path, alt, json, math, mo, np


@app.cell
def _(mo):
    model = mo.ui.dropdown(
        options={"Fish": "fish", "Qwen3": "qwen3", "Miso": "miso"},
        value="Fish",
        label="Model",
    )
    row = mo.ui.number(start=0, step=1, value=0, label="Expresso row")
    data_root = mo.ui.text(value="data", label="Data root")
    mo.hstack([model, row, data_root], justify="start")
    return data_root, model, row


@app.cell(hide_code=True)
def _(Path, data_root, json, model, np, row):
    source = (
        Path(data_root.value)
        / "expresso"
        / f"{model.value}_entropy"
        / str(int(row.value))
    )
    payload = json.loads((source / "entropy.json").read_text(encoding="utf-8"))
    arrays = np.load(source / payload["arrays"])
    audio_positions = arrays["audio_positions"]
    entropy = arrays["entropy"]
    frame_offsets = arrays["frame_offsets"]
    return audio_positions, entropy, frame_offsets, payload, source


@app.cell(hide_code=True)
def _(mo, payload):
    sequence_options = {
        f"Sequence {item['sequence_id']} ({item['num_frames']} frames)": index
        for index, item in enumerate(payload["sequences"])
    }
    sequence = mo.ui.dropdown(
        options=sequence_options,
        value=next(iter(sequence_options)),
        label="Sequence",
    )
    sequence
    return (sequence,)


@app.cell(hide_code=True)
def _(audio_positions, entropy, frame_offsets, math, payload, sequence):
    selected = payload["sequences"][sequence.value]
    frame_rate = float(payload["frame_rate"])
    nats_to_bits = 1.0 / math.log(2.0)
    start = int(frame_offsets[sequence.value])
    end = int(frame_offsets[sequence.value + 1])
    frames = entropy[start:end] * nats_to_bits
    positions = audio_positions[start:end]

    means = [
        {"codebook": f"{index:02d}", "entropy_bits": value * nats_to_bits}
        for index, value in enumerate(selected["codebook_entropy"])
    ]

    timeline_buckets = {}
    spans = selected.get("spans", [])
    for frame_index, position in enumerate(positions):
        span = next(
            (item for item in spans if item["start"] <= position < item["end"]),
            {},
        )
        speaker = str(span.get("speaker", span.get("speaker_id", "unknown")))
        key = (int(frame_index / frame_rate), speaker)
        timeline_buckets.setdefault(key, []).append(frame_index)
    timeline = [
        {
            "second": second,
            "speaker": speaker,
            "entropy_bits": float(frames[indices].mean()),
        }
        for (second, speaker), indices in sorted(timeline_buckets.items())
    ]
    return frame_rate, means, selected, timeline


@app.cell(hide_code=True)
def _(alt, means, timeline):
    mean_chart = (
        alt.Chart(alt.Data(values=means))
        .mark_bar()
        .encode(
            x=alt.X("codebook:N", title="Codebook"),
            y=alt.Y("entropy_bits:Q", title="Mean predictive entropy (bits)"),
            tooltip=["codebook:N", alt.Tooltip("entropy_bits:Q", format=".3f")],
        )
        .properties(title="Mean entropy by codebook", height=280)
    )
    timeline_chart = (
        alt.Chart(alt.Data(values=timeline))
        .mark_line(point=True)
        .encode(
            x=alt.X("second:Q", title="Time (seconds, 1 Hz pooled)"),
            y=alt.Y("entropy_bits:Q", title="Mean entropy (bits)"),
            color=alt.Color("speaker:N", title="Turn speaker"),
            tooltip=[
                "second:Q",
                "speaker:N",
                alt.Tooltip("entropy_bits:Q", format=".3f"),
            ],
        )
        .properties(title="Entropy vs time", height=280)
    )
    return mean_chart, timeline_chart


@app.cell(hide_code=True)
def _(
    frame_rate,
    mean_chart,
    mo,
    payload,
    selected,
    sequence,
    source,
    timeline_chart,
):
    mo.vstack(
        [
            mo.md("# TTS codebook entropy"),
            mo.md(
                f"**Source:** `{source}`  \n"
                f"**Sequence:** `{sequence.value}`  \n"
                f"**Frames:** {selected['num_frames']:,} at {frame_rate:g} Hz  \n"
                f"**Saved unit:** {payload['entropy_unit']}; charts display bits"
            ),
            mean_chart,
            timeline_chart,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
