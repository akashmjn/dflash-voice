"""``en_us_cleaner`` — exact-output port of Resemble's training-time text
normalization pipeline (number / time / phone / abbreviation expansion) with
the ``SliceMap`` alignment plumbing removed.

This pipeline runs at inference time before the speech tokenizer; matching the
training-time form is critical for WER on LibriSpeech-PC / Seed-TTS test-en.
"""

from __future__ import annotations

import re
import unicodedata

from .number_norm import normalize_numbers
from .phone_norm import convert_phone_numbers_to_words
from .time_norm import convert_time_to_words


_whitespace_re = re.compile(r"\s+")

_abbreviations = [
    (re.compile(r"\b%s\b\.?" % abbrev, re.IGNORECASE), expanded)
    for abbrev, expanded in [
        ('mrs', 'misess'),
        ('mr', 'mister'),
        ('dr', 'doctor'),
        ('st', 'saint'),
        ('jr', 'junior'),
        ('maj', 'major'),
        ('gen', 'general'),
        ('drs', 'doctors'),
        ('rev', 'reverend'),
        ('lt', 'lieutenant'),
        ('hon', 'honorable'),
        ('sgt', 'sergeant'),
        ('capt', 'captain'),
        ('esq', 'esquire'),
        ('ltd', 'limited'),
        ('col', 'colonel'),
        ('ft', 'feet'),
        ('abbrev', 'abbreviation'),
        ('ave', 'avenue'),
        ('abstr', 'abstract'),
        ('addr', 'address'),
        ('jan', 'january'),
        ('feb', 'february'),
        ('mar', 'march'),
        ('apr', 'april'),
        ('jul', 'july'),
        ('aug', 'august'),
        ('sep', 'september'),
        ('sept', 'september'),
        ('oct', 'october'),
        ('nov', 'november'),
        ('dec', 'december'),
        ('mon', 'monday'),
        ('tue', 'tuesday'),
        ('wed', 'wednesday'),
        ('thur', 'thursday'),
        ('fri', 'friday'),
        ('sec', 'second'),
        ('min', 'minute'),
        ('mo', 'month'),
        ('yr', 'year'),
        ('cal', 'calorie'),
        ('dept', 'department'),
        ('gal', 'gallon'),
        ('kg', 'kilogram'),
        ('km', 'kilometer'),
        ('mt', 'mount'),
        ('oz', 'ounce'),
        ('vol', 'volume'),
        ('vs', 'versus'),
        ('yd', 'yard'),
        (r'e\.g', 'eg'),
        (r'i\.e', 'ie'),
        ('etc', 'etc'),
        ('ai', 'a i'),
        ('cms', 'c m s'),
        ('cdn', 'c d n'),
        ('ceo', 'c e o'),
        ('tts', 't t s'),
    ]
]

_custom_symbol_replacements = [
    (re.compile('–'), '-'),
    (re.compile('—'), '-'),
    (re.compile(r'(?<=\d\s)>(?=\s\d)', re.IGNORECASE), 'is greater than'),
    (re.compile(r'(?<=\d\s)<(?=\s\d)', re.IGNORECASE), 'is less than'),
    (re.compile(r'(?<=\D)\s*-\s+(?=\D)'), ', '),
    (re.compile(r'(?<=\D)\s+-\s*(?=\D)'), ', '),
    (re.compile(r'(?<=[a-z])-(?=[a-z])', re.IGNORECASE), ' '),
    (re.compile(r'(?<=[a-z])-(?=\d)', re.IGNORECASE), ' '),
    (re.compile(r'(?<=\d)-(?=[a-z])', re.IGNORECASE), ' '),
    (re.compile(r'(?<=\d)-(?=\d)', re.IGNORECASE), ' - '),
]

_abbreviations += _custom_symbol_replacements


def _expand_abbreviations(text: str) -> str:
    for regex, replacement in _abbreviations:
        text = regex.sub(replacement, text)
    return text


def _standardize_characters(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def _collapse_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text)


def en_us_cleaner(text: str) -> str:
    """English text normalization pipeline used for both training and the
    paper's evaluations.

    Matches the cleaned-text output of Resemble's ``transforms.text.cleaners.en_us_cleaner``
    for any normal English input.
    """
    text = _standardize_characters(text)
    text = _collapse_whitespace(text)
    text = _expand_abbreviations(text)
    text = convert_time_to_words(text)
    text = convert_phone_numbers_to_words(text)
    text = normalize_numbers(text)
    return text
