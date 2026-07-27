import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from entropy_utils import (
        DEFAULT_DATA_ROOT,
        NATS_TO_BITS,
        codebook_mean_records,
        group_frame_records,
        load_entropy,
    )

    group_color_scale = alt.Scale(
        domain=["semantic", "audio_half_1", "audio_half_2"],
        range=["#1f77b4", "#ff7f0e", "#2ca02c"],
    )
    return (
        alt,
        codebook_mean_records,
        DEFAULT_DATA_ROOT,
        group_color_scale,
        group_frame_records,
        load_entropy,
        mo,
        NATS_TO_BITS,
        Path,
    )


@app.cell
def _(DEFAULT_DATA_ROOT, mo):
    model = mo.ui.dropdown(
        options={"Fish": "fish", "Qwen3": "qwen3", "Miso": "miso"},
        value="Miso",
        label="Model",
    )
    row = mo.ui.number(start=0, step=1, value=0, label="Expresso row")
    data_root = mo.ui.text(value=str(DEFAULT_DATA_ROOT), label="Data root")
    mo.hstack([model, row, data_root], justify="start")
    return data_root, model, row


@app.cell(hide_code=True)
def _(NATS_TO_BITS, data_root, load_entropy, model, Path, row):
    source = Path(data_root.value) / "entropy" / str(int(row.value))
    payload, arrays = load_entropy(source, model.value)
    entropy_bits = arrays["entropy"] * NATS_TO_BITS  # raw per-frame (frames, codebooks)
    frame_offsets = arrays["frame_offsets"]
    frame_rate = float(payload["frame_rate"])
    mid = int(payload["num_codebooks"]) // 2
    return entropy_bits, frame_offsets, frame_rate, mid, payload, source


@app.cell(hide_code=True)
def _(alt, group_color_scale):
    def codebook_chart(records, title):
        return (
            alt.Chart(alt.Data(values=records))
            .mark_bar()
            .encode(
                x=alt.X("codebook:N", title="Codebook"),
                y=alt.Y("entropy_bits:Q", title="Mean predictive entropy (bits)"),
                tooltip=["codebook:N", alt.Tooltip("entropy_bits:Q", format=".3f")],
            )
            .properties(title=title, height=280)
        )

    def group_chart(records, title, x_field, x_title, frame_rate=None):
        # Raw frames drawn faint; a per-group loess trend line on top does the smoothing.
        tooltip = [
            alt.Tooltip(f"{x_field}:Q", title=x_title),
            "group:N",
            alt.Tooltip("entropy_bits:Q", format=".3f"),
        ]
        chart = alt.Chart(alt.Data(values=records))
        if frame_rate:  # frame-indexed segment view: add seconds for cross-referencing
            chart = chart.transform_calculate(
                second=alt.datum[x_field] / frame_rate
            )
            tooltip.append(alt.Tooltip("second:Q", title="Time (s)", format=".2f"))
        base = chart.encode(
            x=alt.X(f"{x_field}:Q", title=x_title),
            y=alt.Y("entropy_bits:Q", title="Mean entropy (bits)"),
            color=alt.Color("group:N", title="Codebook group", scale=group_color_scale),
            tooltip=tooltip,
        )
        raw = base.mark_line(opacity=0.2, strokeWidth=1)
        trend = base.transform_loess(
            x_field, "entropy_bits", groupby=["group"], bandwidth=0.05
        ).mark_line(strokeWidth=2.5)
        return (
            (raw + trend).properties(title=title, width=680, height=240).interactive()
        )

    return codebook_chart, group_chart


@app.cell(hide_code=True)
def _(
    codebook_chart,
    codebook_mean_records,
    entropy_bits,
    frame_offsets,
    frame_rate,
    group_chart,
    group_frame_records,
    mid,
    mo,
    payload,
    source,
):
    mo.vstack(
        [
            mo.md("# TTS codebook entropy"),
            mo.md(
                f"**Source:** `{source}`  \n"
                f"**Model:** {payload['model']} row {payload['row']}  \n"
                f"**Sequences:** {len(payload['sequences'])}  \n"
                f"**Frames:** {int(frame_offsets[-1]):,} at {frame_rate:g} Hz  \n"
                f"NPZ entropy in nats; charts display bits"
            ),
            mo.md("## Global row view"),
            codebook_chart(
                codebook_mean_records(entropy_bits),
                "Row mean entropy by codebook (all sequences)",
            ),
            group_chart(
                group_frame_records(entropy_bits, mid, "second", frame_rate),
                "Row entropy vs time (semantic / audio halves)",
                x_field="second",
                x_title="Time (seconds)",
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(frame_offsets, mo, payload):
    sequence_options = {
        f"Sequence {item['sequence_id']} "
        f"({int(frame_offsets[index + 1]) - int(frame_offsets[index])} frames)": index
        for index, item in enumerate(payload["sequences"])
    }
    sequence_pick = mo.ui.dropdown(
        options=sequence_options,
        value=next(iter(sequence_options)),
        label="Per-segment / sequence",
    )
    sequence_pick
    return (sequence_pick,)


@app.cell(hide_code=True)
def _(
    codebook_chart,
    codebook_mean_records,
    entropy_bits,
    frame_offsets,
    frame_rate,
    group_chart,
    group_frame_records,
    mid,
    mo,
    sequence_pick,
):
    index = sequence_pick.value
    start = int(frame_offsets[index])
    end = int(frame_offsets[index + 1])
    segment_bits = entropy_bits[start:end]  # raw per-frame slice for this sequence
    mo.vstack(
        [
            mo.md("## Per-segment / sequence"),
            mo.md(
                f"**Sequence index:** {index}  \n"
                f"**Frames:** {end - start:,} at {frame_rate:g} Hz  \n"
                f"**Time in file:** {start / frame_rate:.2f}–{end / frame_rate:.2f} s "
                f"(local frame *i* = {start / frame_rate:.2f} + *i*/{frame_rate:g} s)"
            ),
            codebook_chart(
                codebook_mean_records(segment_bits),
                "Mean entropy by codebook",
            ),
            group_chart(
                group_frame_records(segment_bits, mid, "frame"),
                "Entropy by frame (semantic / audio halves)",
                x_field="frame",
                x_title="Audio frame index",
                frame_rate=frame_rate,
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
