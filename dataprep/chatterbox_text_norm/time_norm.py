"""Time-of-day expansion (12:30, 3.45 pm, ...) ported from
``transforms.text.time_norm`` without the SliceMap return value."""

from __future__ import annotations

import re

from .number_norm import standard_number_to_words


_time_pattern = re.compile(
    r'(?P<hour>\d{1,2})(?P<delim>[\.\:])?\s*(?P<min>\d{2})?\s*(?P<am_pm>[AaPp]\.?[Mm]\.?)?'
)


def _time_to_words(match: re.Match) -> str:
    gd = match.groupdict()
    hours = int(gd['hour'])
    delim = gd.get("delim")
    minutes = gd.get("min")
    am_pm = gd.get("am_pm")

    has_minutes = minutes is not None
    has_am_pm = am_pm is not None
    is_time = has_am_pm or (delim == ":" and has_minutes) or (delim == "." and has_am_pm)
    if not is_time:
        return match.group()

    minutes_i = int(minutes) if has_minutes else 0
    am_pm = (' ' + am_pm.replace('.', '').strip().upper()) if has_am_pm else ''
    hour_word = standard_number_to_words(hours, 0)

    if minutes_i < 10 and minutes_i != 0:
        minute_word = standard_number_to_words(minutes_i, 0)
        minute_word = f"oh {minute_word}"
    else:
        minute_word = standard_number_to_words(minutes_i, 0) if minutes_i > 0 else ""

    if minutes_i > 0:
        minute_word = f" {minute_word}"

    return f"{hour_word}{minute_word}{am_pm}"


def convert_time_to_words(text: str) -> str:
    matches = sorted(_time_pattern.finditer(text), key=lambda m: m.start(), reverse=True)
    for m in matches:
        text = text[: m.start()] + _time_to_words(m) + text[m.end():]
    return text
