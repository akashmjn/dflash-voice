"""Number / currency / unit / year / ordinal expansion.

Verbatim logic ported from Resemble's training-time ``transforms.text.number_norm``
with the ``SliceMap`` alignment machinery stripped. The resulting cleaned text
is byte-identical to the original pipeline for any normal English input.
"""

from __future__ import annotations

import re


_comma_number_re = re.compile(r'(\(?[A-Z]{2,3})?([\$|£|¥|€|#|\(]*[0-9][0-9\,\.]+[0-9])([^\s]+)?')
_decimal_number_re = re.compile(r'(number\s)?([0-9]+\.[0-9]+)(\.|,|\?|!)?')
_hash_number_re = re.compile(r'(#)([0-9]+(?:\.[0-9]+)?)(\.|,|\?|!)?')

_pounds_re = re.compile(r'(\(?£)([0-9\.]*[0-9]+)(\.|,|\?|\!)?')
_yen_re = re.compile(r'(\(?¥)([0-9]+)(\.|,|\?|\!)?')
_euro_re = re.compile(r'(\(?€)([0-9\.]*[0-9]+)(\.|,|\?|\!)?')
_dollars_re = re.compile(
    r'(?P<curr>\(?\$)'
    r'(?P<val>[0-9,]*\.?[0-9]+)'
    r'(?P<punc>[\.|,|\?|\!|\)]+)?'
)

_curr_abbrev_re = re.compile(
    r'(?P<curr>\(?[$£¥€])'
    r'(?P<val>[0-9]*\.?[0-9]+)'
    r'(?:(?P<unit1>[BKMT])|(?:\s(?P<unit2>[BMbmTtr]+illion)))'
    r'(?P<punc>[.,?!)]+)?'
)

_ml_re = re.compile(r'([0-9\.]*[0-9]+)(ml)(\.|,|\?|!)?')
_cl_re = re.compile(r'([0-9\.]*[0-9]+)(cl)(\.|,|\?|!)?')
_g_re = re.compile(r'([0-9\.]*[0-9]+)(g)(\.|,|\?|!)?')
_l_re = re.compile(r'([0-9\.]*[0-9]+)(l)(\.|,|\?|!)?')
_m_re = re.compile(r'([0-9\.]*[0-9]+)(m)(\.|,|\?|!)?')
_kg_re = re.compile(r'([0-9\.]*[0-9]+)(kg)(\.|,|\?|!)?')
_mm_re = re.compile(r'([0-9\.]*[0-9]+)(mm)(\.|,|\?|!)?')
_cm_re = re.compile(r'([0-9\.]*[0-9]+)(cm)(\.|,|\?|!)?')
_km_re = re.compile(r'([0-9\.]*[0-9]+)(km)(\.|,|\?|!)?')
_in_re = re.compile(r'([0-9\.]*[0-9]+)(in)(\.|,|\?|!)?')
_ft_re = re.compile(r'([0-9\.]*[0-9]+)(ft)(\.|,|\?|!)?')
_yd_re = re.compile(r'([0-9\.]*[0-9]+)(yd[s]?)(\.|,|\?|!)?')
_s_re = re.compile(r'([0-9\.]*[0-9]+)(s[ecs]*)(\.|,|\?|!)?')
_percent = re.compile(r'([0-9]{1,3}(?:,[0-9]{3})*(\.[0-9]+)?)(%)([.,?!])?')

_ordinal_re = re.compile(r'([0-9]+)(st|nd|rd|th)')
_number_re = re.compile(r'([0-9]+)(\.|,|\?|!)?')
_year_re = re.compile(
    r'(to|years|year|from|after|before|by|until|in|since|around|circa|during)'
    r'(\s)?'
    r'(?<!\$|£|¥|€|₩|₹|₽|฿|₫|₦|₪|₱|₴)'
    r'(1[1-9]|20)'
    r'([0-9]{2})'
    r'($|\-|\s\-|\D)',
    re.IGNORECASE,
)

_units = [
    '', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
    'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
    'seventeen', 'eighteen', 'nineteen',
]
_tens = [
    '', 'ten', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy',
    'eighty', 'ninety',
]
_digit_groups = ['', 'thousand', 'million', 'billion', 'trillion', 'quadrillion']

_ordinal_suffixes = [
    ('one', 'first'),
    ('two', 'second'),
    ('three', 'third'),
    ('five', 'fifth'),
    ('eight', 'eighth'),
    ('nine', 'ninth'),
    ('twelve', 'twelfth'),
    ('ty', 'tieth'),
]

_sub_ten_nums = {"00", "01", "02", "03", "04", "05", "06", "07", "08", "09"}

_curr_dict = {"$": "dollars", "£": "pounds", "¥": "yen", "€": "euros"}
_unit_dict = {"B": "billion", "K": "thousand", "M": "million", "T": "trillion"}


def standard_number_to_words(n: int, digit_group: int) -> str:
    parts = []
    if n >= 1000:
        parts.append(standard_number_to_words(n // 1000, digit_group + 1))
        n = n % 1000
    if n >= 100:
        parts.append('%s hundred' % _units[n // 100])
    if n % 100 >= len(_units):
        parts.append(_tens[(n % 100) // 10])
        parts.append(_units[(n % 100) % 10])
    else:
        parts.append(_units[n % 100])
    if n > 0:
        parts.append(_digit_groups[digit_group])
    return ' '.join(x for x in parts if x)


def _number_to_words(n: int) -> str:
    if n >= 1000000000000000000:
        return str(n)
    if n == 0:
        return 'zero'
    if n % 100 == 0 and n % 1000 != 0 and n < 3000:
        return standard_number_to_words(n // 100, 0) + ' hundred'
    return standard_number_to_words(n, 0)


def _remove_commas(match):
    _, m1, m2 = match.groups(default="")
    return ''.join([m1.replace(",", ""), m2])


def _expand_year(match):
    prep, _, mill_cent, dec_year, post = match.groups(default="")
    if mill_cent == "20" and dec_year in _sub_ten_nums:
        year_out = mill_cent + dec_year
    elif dec_year in _sub_ten_nums:
        year = dec_year[-1]
        year_out = mill_cent + " " + "oh" + " " + year
    else:
        year_out = _number_to_words(int(mill_cent)) + " " + _number_to_words(int(dec_year))

    if "-" in post:
        year_out += " to"
    else:
        year_out += post
    return prep + " " + year_out


def _re_sub(pattern, repl, text: str) -> str:
    """Backwards (end-to-beginning) regex substitution, identical match order
    to the training-time ``aligned_re_sub`` so non-overlapping match decisions
    line up exactly."""
    matches = sorted(re.finditer(pattern, text), key=lambda m: m.start(), reverse=True)
    for m in matches:
        middle = repl(m) if callable(repl) else repl
        text = text[:m.start()] + middle + text[m.end():]
    return text


# --- helpers operating on (word, _) mapping list, but with the SliceMap dropped ---


def _to_word_list(text: str):
    words = re.split(r"(\s+)", text)
    return [[w] for w in words]


def _from_word_list(mapping) -> str:
    return ''.join(t[0] for t in mapping)


def _convert_hash(mapping):
    text = _from_word_list(mapping)
    matches = re.findall(_hash_number_re, text)
    for result in matches:
        m = ''.join(result)
        r_clean = ' '.join(["number", result[1]])
        if len(result) == 3:
            r_clean = r_clean + result[2]
        for i, t in enumerate(mapping):
            if m == t[0]:
                mapping[i] = [r_clean]
    return mapping


def _expand_decimal_point(mapping):
    text = _from_word_list(mapping)
    matches = re.findall(_decimal_number_re, text)
    for m in matches:
        integral, fractional = m[1].split('.')
        expanded_frac = ""
        frac_it = iter(list(fractional))
        for digit in frac_it:
            if digit == "0":
                expanded_frac += "zero "
            else:
                expanded_frac += digit + " "
                for rest in frac_it:
                    expanded_frac += rest + " "
        out = m[0] + integral + ' point ' + expanded_frac.strip()
        if len(m) == 3:
            out = out + m[2]
        raw = ''.join(m)
        for i, t in enumerate(mapping):
            if t[0] == raw:
                mapping[i] = [out]
    return mapping


def _expand_dollars(mapping):
    text = _from_word_list(mapping)
    matches = re.findall(_dollars_re, text)
    for m in matches:
        val = m[1]
        punc = m[2] if m[2] else ''
        parts = val.split('.')
        whole_num = int(parts[0]) if parts[0] else 0
        fraction = int(parts[1]) if len(parts) > 1 and parts[1] else 0
        dollar_unit = 'dollar' if whole_num == 1 else 'dollars'
        cent_unit = 'cent' if fraction == 1 else 'cents'
        if whole_num and fraction:
            out = '%s %s, %s %s' % (whole_num, dollar_unit, fraction, cent_unit)
        elif whole_num:
            out = '%s %s' % (whole_num, dollar_unit)
        elif fraction:
            out = '%s %s' % (fraction, cent_unit)
        else:
            out = 'zero dollars'
        if punc:
            out = out + punc
        raw = ''.join(m)
        for i, t in enumerate(mapping):
            if t[0] == raw:
                mapping[i] = [out]
    return mapping


def _expand_other_currency(mapping, regex, one, many):
    text = _from_word_list(mapping)
    matches = re.findall(regex, text)
    for m in matches:
        parts = m[1].split(".")
        curr = one if int(parts[0]) == 1 else many
        try:
            out = parts[0] + " " + curr + " " + parts[1]
        except IndexError:
            out = parts[0] + " " + curr
        raw = ''.join(m)
        for i, t in enumerate(mapping):
            if t[0] == raw:
                mapping[i] = [out]
    return mapping


def _expand_abbreviated_currency_unit(mapping):
    text = _from_word_list(mapping)
    matches = re.findall(_curr_abbrev_re, text)
    to_remove = []
    for m in matches:
        curr = m[0]
        val = m[1]
        unit = m[2] if m[2] else m[3]
        punc = m[4] if m[4] else ''
        curr = curr.strip("(")
        val_parts = val.split(".")
        if len(val_parts) > 1:
            val_out = val_parts[0] + ' ' + 'point ' + ' '.join(val_parts[1])
        else:
            val_out = val_parts[0]
        try:
            out = ' '.join([val_out, _unit_dict[unit], _curr_dict[curr]])
            out = out + punc
            raw = ''.join(m)
        except KeyError:
            out = ' '.join([val_out, unit, _curr_dict[curr]])
            out = out + punc
            raw = ''.join(m[:-2]) + " " + unit + punc
        for i, t in enumerate(mapping):
            if t[0] == raw:
                mapping[i] = [out]
            try:
                join_text = ''.join([mapping[i][0], mapping[i + 1][0], mapping[i + 2][0]])
                if join_text == raw:
                    mapping[i] = [out]
                    to_remove.append(i + 1)
                    to_remove.append(i + 2)
            except IndexError:
                continue
    return [v for i, v in enumerate(mapping) if i not in to_remove]


def _expand_other_unit(mapping, regex, one, many):
    text = _from_word_list(mapping)
    matches = re.findall(regex, text)
    for m in matches:
        parts = re.split(r"\.", m[0])
        unit = one if parts[0] == "1" else many
        if len(parts) > 1:
            dec = ''.join([i + " " for i in parts[1]])
            parts = parts[0] + " " + "point" + " " + dec
            unit = many
            out = parts + unit
        else:
            out = parts[0] + " " + unit
        out = out + m[-1]
        raw = ''.join(m)
        for i, t in enumerate(mapping):
            if t[0] == raw:
                mapping[i] = [out]
    return mapping


def _expand_ordinal(mapping):
    text = _from_word_list(mapping)
    matches = re.findall(_ordinal_re, text)
    for m in matches:
        num = _number_to_words(int(m[0]))
        out = num + 'th'
        for suffix, replacement in _ordinal_suffixes:
            if num.endswith(suffix):
                out = num[:-len(suffix)] + replacement
                break
        raw = ''.join(m)
        for i, t in enumerate(mapping):
            if t[0] == raw:
                mapping[i] = [out]
    return mapping


def _expand_number(mapping):
    text = _from_word_list(mapping)
    matches = re.findall(_number_re, text)
    for m in matches:
        out = _number_to_words(int(m[0])) + m[1]
        raw = ''.join(m)
        for i, t in enumerate(mapping):
            text_re_nums = re.search(r"\d+", t[0])
            if text_re_nums is None:
                continue
            if m[0] == text_re_nums.group():
                rep = re.escape(raw)
                j = re.sub(rep, out, t[0])
                mapping[i] = [j]
    return mapping


def normalize_numbers(text: str) -> str:
    text = _re_sub(_comma_number_re, _remove_commas, text)
    # year norm is applied twice on purpose (matches training pipeline)
    text = _re_sub(_year_re, _expand_year, text)
    text = _re_sub(_year_re, _expand_year, text)

    mapping = _to_word_list(text)
    mapping = _expand_abbreviated_currency_unit(mapping)
    mapping = _expand_other_currency(mapping, _pounds_re, "pound", "pounds")
    mapping = _expand_other_currency(mapping, _yen_re, "yen", "yen")
    mapping = _expand_other_currency(mapping, _euro_re, "euro", "euros")
    mapping = _expand_other_unit(mapping, _ml_re, "milliliter", "milliliters")
    mapping = _expand_other_unit(mapping, _cl_re, "centiliter", "centiliters")
    mapping = _expand_other_unit(mapping, _g_re, "gram", "grams")
    mapping = _expand_other_unit(mapping, _kg_re, "kilogram", "kilograms")
    mapping = _expand_other_unit(mapping, _mm_re, "millimeter", "millimeters")
    mapping = _expand_other_unit(mapping, _cm_re, "centimeter", "centimeters")
    mapping = _expand_other_unit(mapping, _km_re, "kilometer", "kilometers")
    mapping = _expand_other_unit(mapping, _in_re, "inch", "inches")
    mapping = _expand_other_unit(mapping, _ft_re, "foot", "feet")
    mapping = _expand_other_unit(mapping, _l_re, "liter", "liters")
    mapping = _expand_other_unit(mapping, _m_re, "meter", "meters")
    mapping = _expand_other_unit(mapping, _yd_re, "yard", "yards")
    mapping = _expand_other_unit(mapping, _s_re, "second", "seconds")
    mapping = _expand_other_unit(mapping, _percent, "%", "percent")
    mapping = _expand_dollars(mapping)
    mapping = _convert_hash(mapping)
    mapping = _expand_decimal_point(mapping)
    mapping = _expand_ordinal(mapping)
    mapping = _expand_number(mapping)

    return _from_word_list(mapping)
