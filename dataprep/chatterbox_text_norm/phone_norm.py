"""Phone number expansion (e.g. "555-1234" -> "five five five, one two three four")
ported from ``transforms.text.phone_num_norm`` without SliceMap."""

from __future__ import annotations

import re


_NUM_WORDS = {
    0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four',
    5: 'five', 6: 'six', 7: 'seven', 8: 'eight', 9: 'nine',
}

_NON_ALPHA = re.compile(r"(\s?[^a-zA-Z]+\s?)")
_PHONE_NUM_CHARS = set(list("0123456789.+-()[] "))


def is_phone_num(text: str) -> bool:
    text = text.rstrip('.!?,;')
    if text.count(".") == 1:
        return False
    if any(c.isalpha() for c in text):
        return False
    if any(c not in _PHONE_NUM_CHARS for c in text):
        return False
    nums = [c for c in text if c.isdigit()]
    if not (5 < len(nums) < 14):
        return False
    if "+" not in text and "." not in text and list(text).count("-") == 1:
        nums = [[c for c in subtext if c.isdigit()] for subtext in text.split("-")]
        if len(nums) == 2 and all(
            subnums[0] in ("1", "2") and len(subnums) == 4 for subnums in nums
        ):
            return False
    return True


def _group_num(text):
    prev_was_digit = False
    group = []
    num_groups = []
    for c in text:
        if c.isdigit():
            if len(group) == 0 or prev_was_digit:
                group.append(c)
            else:
                num_groups.append(group)
                group = [c]
            prev_was_digit = True
        else:
            prev_was_digit = False
    if group:
        num_groups.append(group)
    return num_groups


def _natural_phone_num_grouping(nums, max_natural_group=4):
    if len(nums) == 0:
        return []
    if len(nums) <= max_natural_group:
        return [nums]
    if len(nums) == 5:
        return [nums[:2], nums[2:]]
    if len(nums) == 8:
        return [nums[:2], nums[2:5], nums[5:]]
    return _natural_phone_num_grouping(nums[:-max_natural_group], 3) + [nums[-max_natural_group:]]


def _convert_phone_num_to_words(text: str) -> str:
    original_punctuation = ''
    while text and text[-1] in '.!?,;':
        original_punctuation = text[-1] + original_punctuation
        text = text[:-1]

    num_groups = _group_num(text)
    num_word_groups = [
        [" " + _NUM_WORDS[int(c)] for c in nums if c.isdigit()] for nums in num_groups
    ]

    output = []
    for num_word_group in reversed(num_word_groups):
        grouplen = len(num_word_group)
        regroupped = [num_word_group]
        if grouplen > 4:
            regroupped = reversed(_natural_phone_num_grouping(num_word_group))
        for new_group in regroupped:
            output = ["".join(new_group)] + output

    output = ",".join(output).strip()
    parts = re.split(r"\s", text)
    output = (" " if parts[0] == "" else "") + output + (" " if parts[-1] == "" else "")
    output += original_punctuation
    return output


def _convert_phone_num_match_to_words(match: re.Match) -> str:
    text = match.group()
    if is_phone_num(text):
        return _convert_phone_num_to_words(text)
    return text


def convert_phone_numbers_to_words(text: str) -> str:
    matches = sorted(_NON_ALPHA.finditer(text), key=lambda m: m.start(), reverse=True)
    for m in matches:
        text = text[: m.start()] + _convert_phone_num_match_to_words(m) + text[m.end():]
    return text
