"""Vendored English text normalization for Chatterbox dataprep.

Copyright (c) 2026 Resemble AI, MIT License. ``cleaners.py`` / ``number_norm.py``
/ ``phone_norm.py`` / ``time_norm.py`` are copied verbatim from
https://github.com/resemble-ai/chatterbox-flash (``chatterbox_flash/text_norm/``),
which describes itself as an exact-output port of Resemble's *training-time*
pipeline. Matching that form matters: the checkpoint was trained on normalized
text, so tokenizing raw text shifts the distribution.

Vendored because ``en_us_cleaner`` lives only in the ``chatterbox-flash`` repo,
which is not a declared dependency -- the ``chatterbox-tts`` package ships only
the lighter ``punc_norm``. The two are not interchangeable: on the ``segment0``
fixture they give 161 vs 157 BPE ids, diverging on "2013". The four modules are
pure ``re`` + ``unicodedata``, so vendoring adds no install-time dependency.
"""

from dataprep.chatterbox_text_norm.cleaners import en_us_cleaner

__all__ = ["en_us_cleaner"]
