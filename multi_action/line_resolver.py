"""
Line-target resolver (rule-based).

DocEditBERT was trained only on anchor targets ("from X to Y") -- DocEdit-1000
contains no line-number commands at all. So for "second line delete pannu" the
model gets the action right (delete, 0.998) but returns from=None, to=None, and
sometimes a junk fragment such as from="second".

This module reads the line target straight off the text instead. It is
deliberately conservative: an ordinal or digit must sit next to the unit word,
so the anchor phrase "line to paragraph" (10 examples in DocEdit-1000, meaning
*from* the word "line" *to* the word "paragraph") is not mistaken for a line
number.

Returns {"kind": "line", "n": int, "unit": str, "matched": str} or None.
n is 1-based; n == -1 means the last line.
"""
import re

LAST = -1

# ── Ordinals ─────────────────────────────────────────────────────────────────
ENGLISH_ORDINALS = {
    "first": 1, "1st": 1,
    "second": 2, "2nd": 2,
    "third": 3, "3rd": 3,
    "fourth": 4, "4th": 4,
    "fifth": 5, "5th": 5,
    "sixth": 6, "6th": 6,
    "seventh": 7, "7th": 7,
    "eighth": 8, "8th": 8,
    "ninth": 9, "9th": 9,
    "tenth": 10, "10th": 10,
    "last": LAST, "final": LAST,
}

# Romanised Tamil ordinals, with the spelling variants people actually say.
TANGLISH_ORDINALS = {
    "mudhal": 1, "mudal": 1, "mudhalaavathu": 1, "mudhalavathu": 1,
    "modhal": 1, "mothal": 1,
    "rendavathu": 2, "rendaavathu": 2, "randavathu": 2, "rendaam": 2,
    "rendam": 2, "irandavathu": 2, "irandaavathu": 2,
    "moonavathu": 3, "moonaavathu": 3, "moondravathu": 3, "moondraavathu": 3,
    "moonaam": 3, "munnavathu": 3,
    "naalavathu": 4, "naangaavathu": 4, "naalaavathu": 4, "naalaam": 4,
    "anjavathu": 5, "ainthavathu": 5, "ainthaavathu": 5, "anjaam": 5,
    "aaravathu": 6, "aarvathu": 6,
    "ezhavathu": 7, "ettavathu": 8, "onbathavathu": 9, "pathavathu": 10,
    "kadaisi": LAST, "kadaisee": LAST, "kadesi": LAST,
}

ORDINALS = {**ENGLISH_ORDINALS, **TANGLISH_ORDINALS}

# English number words, for "line two".
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# ── Unit words ───────────────────────────────────────────────────────────────
LINE_WORDS = {"line", "lines", "vari", "varigal", "varila", "variya"}
PARA_WORDS = {"paragraph", "paragraphs", "para", "paras", "pathi", "pathiya"}
UNIT_WORDS = LINE_WORDS | PARA_WORDS

_PUNCT = ".,?!;:"

# How many tokens may sit between the ordinal and the unit word.
MAX_GAP = 1


def _unit_of(word):
    if word in LINE_WORDS:
        return "line"
    if word in PARA_WORDS:
        return "paragraph"
    return None


def _clean(token):
    return token.lower().strip(_PUNCT + "\"'")


def resolve(text):
    """
    Find a line/paragraph target in the text.

    Matches, in order of preference:
        "second line", "rendavathu vari", "last line", "kadaisi vari"
        "line 2", "line two", "para 3"
        "2nd line", "3rd paragraph"
    """
    tokens = [_clean(t) for t in text.split()]
    if not tokens:
        return None

    n_tokens = len(tokens)

    # Pass 1 — ordinal before the unit word: "second line", "kadaisi vari"
    for i, token in enumerate(tokens):
        if token not in ORDINALS:
            continue
        for j in range(i + 1, min(i + 2 + MAX_GAP, n_tokens)):
            unit = _unit_of(tokens[j])
            if unit:
                return {
                    "kind": "line",
                    "n": ORDINALS[token],
                    "unit": unit,
                    "matched": " ".join(tokens[i:j + 1]),
                }

    # Pass 2 — number after the unit word: "line 2", "line two"
    for i, token in enumerate(tokens):
        unit = _unit_of(token)
        if not unit:
            continue
        for j in range(i + 1, min(i + 2 + MAX_GAP, n_tokens)):
            nxt = tokens[j]
            if nxt.isdigit():
                return {
                    "kind": "line",
                    "n": int(nxt),
                    "unit": unit,
                    "matched": " ".join(tokens[i:j + 1]),
                }
            if nxt in NUMBER_WORDS:
                return {
                    "kind": "line",
                    "n": NUMBER_WORDS[nxt],
                    "unit": unit,
                    "matched": " ".join(tokens[i:j + 1]),
                }
            # Only look past a filler word, never past a content word.
            if nxt not in {"number", "no", "na", "la"}:
                break

    # Pass 3 — digit directly before the unit word: "2 line", "3 vari"
    for i, token in enumerate(tokens):
        if not token.isdigit():
            continue
        for j in range(i + 1, min(i + 2 + MAX_GAP, n_tokens)):
            unit = _unit_of(tokens[j])
            if unit:
                return {
                    "kind": "line",
                    "n": int(token),
                    "unit": unit,
                    "matched": " ".join(tokens[i:j + 1]),
                }

    return None


def describe(target):
    """Human-readable label for a resolved target, for the demo UI."""
    if not target:
        return "no target"
    if target.get("kind") != "line":
        return "anchor target"
    n = target["n"]
    unit = target.get("unit", "line")
    return f"last {unit}" if n == LAST else f"{unit} {n}"
