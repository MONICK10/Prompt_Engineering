"""
Action-word corrector.

Whisper is the untrained model in VDOC, and its mistakes cluster on our short
command words: "bold" comes back as "bowl", "bolt" or "bull", "italic" as
"italik". DocEditBERT then parses the wrong word confidently enough to look
fine, so the mistake has to be caught before parsing.

This module only ever *suggests*. Nothing is rewritten until the user says
yes, and a confirmed fix is remembered in data/user_corrections.json and
applied automatically after that.

Two guards keep it from "fixing" words that were right all along:

  safe words   every token that appears in DocEdit-1000 is legitimate, so
               anchors like "conclusion" or "footer" are never touched even
               when they sit close to a command word.
  distance     edit distance 1 always qualifies; distance 2 only for tokens
               long enough that two edits are not most of the word.

It is deliberately not part of pipeline.parse_multi, so the 1000-command
regression runs on untouched text.
"""
import json
import os
import re
from functools import lru_cache

from .line_resolver import LINE_WORDS, ORDINALS, PARA_WORDS
from .splitter import ACTION_WORDS, JOIN_WORDS

_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET = os.path.join(_ROOT, "data", "docEdit1000.json")
CORRECTIONS_FILE = os.path.join(_ROOT, "data", "user_corrections.json")

# The words worth protecting: everything the pipeline reads meaning from.
TARGET_WORDS = frozenset(
    set(ACTION_WORDS) | set(JOIN_WORDS) | set(ORDINALS)
    | set(LINE_WORDS) | set(PARA_WORDS)
)

# Legitimate Tanglish command verbs and particles. They name no action and no
# target, so they are not in TARGET_WORDS, but they are correct words and must
# never be "fixed". The dataset alone is not enough here: DocEdit-1000 has
# "panu" but not "pannu", the spelling most people actually say, so without
# this list the corrector offers "pannu -> pannitu" on a perfect command.
VERB_PARTICLES = frozenset({
    "pannu", "panu", "pannunga", "pannuga", "panuka", "pannungal",
    "sei", "seiyu", "seiya", "seiyunga", "podu", "pottu",
    "aakku", "aaku", "maathu", "maathi", "kodu", "kudu",
    "iru", "aa", "ah", "na", "um", "uh",
})

# Distance 2 is only allowed from this length up; below it, two edits change
# too much of the word to be a safe guess. Four keeps the mishearings that
# actually happen -- "bowl" and "bull" are both two edits from "bold".
MIN_LENGTH_FOR_DISTANCE_2 = 4

_PUNCT = ".,?!;:\"'"


@lru_cache(maxsize=1)
def safe_words():
    """Every token in DocEdit-1000 — words we know are legitimate."""
    try:
        with open(DATASET, encoding="utf-8") as handle:
            examples = json.load(handle)
    except (OSError, ValueError):
        return frozenset()
    words = set()
    for example in examples:
        for token in example.get("tokens", []):
            words.add(token.lower().strip(_PUNCT))
    return frozenset(words)


def levenshtein(a, b):
    if a == b:
        return 0
    if abs(len(a) - len(b)) > 2:
        return 3           # far enough; the caller only cares about <= 2
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + (ca != cb),
            ))
        previous = current
    return previous[-1]


# ── Saved corrections ────────────────────────────────────────────────────────

def load_corrections():
    """{"bowl": "bold", ...} — confirmed by the user, applied automatically."""
    try:
        with open(CORRECTIONS_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {}
    pairs = data.get("corrections", data) if isinstance(data, dict) else {}
    return {str(k).lower(): str(v).lower() for k, v in pairs.items()}


def save_correction(wrong, right):
    """Remember a confirmed fix. Returns the full map afterwards."""
    wrong = (wrong or "").lower().strip(_PUNCT)
    right = (right or "").lower().strip(_PUNCT)
    if not wrong or not right or wrong == right:
        raise ValueError("a correction needs two different words")
    if right not in TARGET_WORDS:
        raise ValueError(f'"{right}" is not one of the command words')

    corrections = load_corrections()
    corrections[wrong] = right

    os.makedirs(os.path.dirname(CORRECTIONS_FILE), exist_ok=True)
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as handle:
        json.dump({"corrections": corrections}, handle, indent=2, sort_keys=True)
    return corrections


def forget_correction(wrong):
    corrections = load_corrections()
    corrections.pop((wrong or "").lower().strip(_PUNCT), None)
    with open(CORRECTIONS_FILE, "w", encoding="utf-8") as handle:
        json.dump({"corrections": corrections}, handle, indent=2, sort_keys=True)
    return corrections


# ── Suggesting ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _ordered_targets():
    """Sorted, so a tie between two equally close words resolves the same way
    on every run -- a frozenset iterates in arbitrary order."""
    return tuple(sorted(TARGET_WORDS))


def _best_match(word):
    """Closest command word within edit distance 2, or None."""
    best, best_distance = None, 3
    for candidate in _ordered_targets():
        distance = levenshtein(word, candidate)
        if distance < best_distance:
            best, best_distance = candidate, distance
        if best_distance == 1:
            break
    if best is None or best_distance > 2:
        return None
    if best_distance == 2 and len(word) < MIN_LENGTH_FOR_DISTANCE_2:
        return None
    return {"suggestion": best, "distance": best_distance}


def review(text):
    """
    Look at an utterance and report what should change.

    Returns:
        original          the text as given
        corrected         the text with saved corrections already applied
        auto_applied      [{from, to}] fixes the user confirmed earlier
        suggestions       [{word, suggestion, distance}] awaiting a yes/no
    """
    corrections = load_corrections()
    known = safe_words()

    tokens = re.findall(r"\S+", text)
    auto_applied = []
    rebuilt = []

    for token in tokens:
        bare = token.lower().strip(_PUNCT)
        trailing = token[len(token.rstrip(_PUNCT)):]
        if bare in corrections:
            fixed = corrections[bare]
            auto_applied.append({"from": bare, "to": fixed})
            rebuilt.append(fixed + trailing)
        else:
            rebuilt.append(token)

    corrected = " ".join(rebuilt)

    suggestions = []
    seen = set()
    for token in re.findall(r"\S+", corrected):
        bare = token.lower().strip(_PUNCT)
        if not bare or bare in seen:
            continue
        seen.add(bare)
        if (bare in TARGET_WORDS or bare in known
                or bare in VERB_PARTICLES or bare.isdigit()):
            continue
        match = _best_match(bare)
        if match:
            suggestions.append({"word": bare, **match})

    return {
        "original": text,
        "corrected": corrected,
        "changed": corrected != text,
        "auto_applied": auto_applied,
        "suggestions": suggestions,
    }


def apply_choice(text, word, replacement):
    """Replace every whole-word occurrence of `word` with `replacement`."""
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return pattern.sub(replacement, text)
