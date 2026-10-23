"""
Snap Whisper output onto the vocabulary we actually use.

Whisper decodes with an English language model, so Tanglish comes back as
English-sounding words: "varikum" becomes "vari-crum", "la irunthu" becomes
"line-up is", "conclusion" becomes "convolution". The words are wrong but the
*sounds* are close, so we can pull them back by matching on sound rather than
spelling alone.

Three passes, in order:

  1. phrases   multi-word fixes from data/phrase_fixes.json ("line-up is" ->
               "la irunthu"). Editable without touching code.
  2. words     single-word fixes from the same file, for pairs too far apart
               for the phonetic matcher to reach safely.
  3. snapping  anything still outside the vocabulary is matched against it by
               Double Metaphone code plus edit distance, and replaced only
               when the match is clearly the best and clearly close.

The vocabulary is every word in DocEdit-1000, a list of Tanglish particles,
and every word in the document on screen. A word already in the vocabulary is
never touched -- that is what keeps a correct command safe.

Set ENABLE_SNAPPING = False (or VDOC_SNAPPING=0) to turn the whole thing off
without removing it.
"""
import json
import os
import re
from functools import lru_cache

from metaphone import doublemetaphone

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(_ROOT, "data", "docEdit1000.json")
FIXES_FILE = os.path.join(_ROOT, "data", "phrase_fixes.json")

ENABLE_SNAPPING = os.environ.get("VDOC_SNAPPING", "1") != "0"

# Tanglish particles. DocEdit-1000 has some of these but not all spellings
# people actually say, so they are listed explicitly.
PARTICLES = frozenset({
    "la", "irunthu", "irundhu", "irundu", "varikum", "varaikum", "vara",
    "vare", "varai", "pannu", "panu", "paniru", "panitu", "pannidu",
    "pannitu", "panuka", "pannuga", "apram", "aprom", "appuram", "piragu",
    "sei", "seiyu", "podu", "pottu", "maathu", "maathi", "vari", "varigal",
    "mudhal", "mudal", "rendavathu", "moonavathu", "naalavathu",
    "anjavathu", "kadaisi", "ku", "aa", "ah", "um",
})

# A word this short is mostly noise under edit distance, so it is only
# snapped on an exact phonetic match.
SHORT_WORD = 4

# How close a candidate must be before we replace anything.
MIN_SCORE = 0.72
# ...and how far clear of the runner-up, so an ambiguous match is left alone.
MIN_MARGIN = 0.04

# Sound counts for more than spelling here: Whisper's errors are acoustic.
PHONETIC_WEIGHT = 0.55
SPELLING_WEIGHT = 0.45

_PUNCT = ".,?!;:\"'"
_WORD_RE = re.compile(r"[A-Za-z]+")


# ── Vocabulary ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def dataset_words():
    try:
        with open(DATASET, encoding="utf-8") as handle:
            examples = json.load(handle)
    except (OSError, ValueError):
        return frozenset()
    words = set()
    for example in examples:
        for token in example.get("tokens", []):
            cleaned = token.lower().strip(_PUNCT)
            if cleaned:
                words.add(cleaned)
    return frozenset(words)


def build_vocabulary(document=""):
    """Dataset words + Tanglish particles + whatever is on screen."""
    vocab = set(dataset_words()) | set(PARTICLES)
    for word in _WORD_RE.findall((document or "").lower()):
        vocab.add(word)
    return vocab


@lru_cache(maxsize=512)
def _code(word):
    primary, secondary = doublemetaphone(word)
    return primary or "", secondary or ""


def _levenshtein(a, b):
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def _ratio(a, b):
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - _levenshtein(a, b) / longest


def similarity(word, candidate):
    """0..1. Sound is weighted above spelling."""
    spelling = _ratio(word, candidate)

    word_codes = [c for c in _code(word) if c]
    cand_codes = [c for c in _code(candidate) if c]
    if word_codes and cand_codes:
        phonetic = max(_ratio(w, c) for w in word_codes for c in cand_codes)
    else:
        phonetic = spelling

    return PHONETIC_WEIGHT * phonetic + SPELLING_WEIGHT * spelling


def best_match(word, vocabulary):
    """
    Closest vocabulary word, or None when nothing is close enough or two
    candidates are too similar to choose between.
    """
    scored = []
    for candidate in vocabulary:
        # Skip candidates of wildly different length; they cannot win anyway.
        if abs(len(candidate) - len(word)) > max(3, len(word) // 2):
            continue
        scored.append((similarity(word, candidate), candidate))

    if not scored:
        return None
    scored.sort(key=lambda pair: (-pair[0], pair[1]))

    score, winner = scored[0]
    if score < MIN_SCORE:
        return None
    if len(word) <= SHORT_WORD and _code(word)[0] != _code(winner)[0]:
        return None
    if len(scored) > 1 and score - scored[1][0] < MIN_MARGIN:
        return None
    return {"word": winner, "score": round(score, 3)}


# ── Editable fixes ───────────────────────────────────────────────────────────

def load_fixes():
    """Reloaded per request so the file can be edited during a demo."""
    try:
        with open(FIXES_FILE, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return {"phrases": {}, "words": {}}
    return {
        "phrases": {k.lower(): v for k, v in (data.get("phrases") or {}).items()},
        "words": {k.lower(): v for k, v in (data.get("words") or {}).items()},
    }


def _apply_phrases(text, phrases):
    changes = []
    result = text
    # Longest first, so "line-up is" wins over "line up".
    for phrase in sorted(phrases, key=len, reverse=True):
        pattern = re.compile(
            r"\b" + r"[\s\-]+".join(re.escape(p) for p in phrase.split()) + r"\b",
            re.IGNORECASE,
        )
        if pattern.search(result):
            result = pattern.sub(phrases[phrase], result)
            changes.append({"from": phrase, "to": phrases[phrase], "reason": "phrase"})
    return result, changes


# ── Main entry point ─────────────────────────────────────────────────────────

def snap(text, document="", enabled=None):
    """
    Pull a transcription onto our vocabulary.

    Returns {original, snapped, changed, changes[]}, where each change is
    {from, to, reason} and reason is "phrase", "word" or "sound".
    """
    if enabled is None:
        enabled = ENABLE_SNAPPING
    if not enabled or not text.strip():
        return {"original": text, "snapped": text, "changed": False,
                "changes": [], "enabled": bool(enabled)}

    fixes = load_fixes()
    vocabulary = build_vocabulary(document)

    working, changes = _apply_phrases(text, fixes["phrases"])

    rebuilt = []
    for token in re.findall(r"\S+", working):
        rebuilt.append(_snap_token(token, fixes["words"], vocabulary, changes))

    snapped = " ".join(rebuilt)
    return {
        "original": text,
        "snapped": snapped,
        "changed": snapped != text,
        "changes": changes,
        "enabled": True,
    }


def _snap_token(token, word_fixes, vocabulary, changes):
    """
    Snap one whitespace-delimited token, keeping its punctuation.

    Whisper joins Tanglish words with hyphens ("convolution-vari-crum"), so a
    hyphenated token is snapped piece by piece -- except "bold-um" style
    coordination, which is already meaningful and is left whole.
    """
    leading = token[:len(token) - len(token.lstrip(_PUNCT))]
    trailing = token[len(token.rstrip(_PUNCT)):]
    core = token[len(leading):len(token) - len(trailing) or None]

    if not core:
        return token

    if "-" in core and not _is_coordination(core, vocabulary):
        pieces = [p for p in core.split("-") if p]
        snapped_pieces = [
            _snap_word(piece, word_fixes, vocabulary, changes) for piece in pieces
        ]
        return leading + " ".join(snapped_pieces) + trailing

    return leading + _snap_word(core, word_fixes, vocabulary, changes) + trailing


def _is_coordination(core, vocabulary):
    """True for "bold-um" / "italic-um": a known word plus the -um particle."""
    if not core.lower().endswith("-um"):
        return False
    return core[:-3].lower() in vocabulary


def _snap_word(word, word_fixes, vocabulary, changes):
    lowered = word.lower()

    if lowered in word_fixes:
        replacement = word_fixes[lowered]
        changes.append({"from": lowered, "to": replacement, "reason": "word"})
        return replacement

    # Already ours — never touch it. This is the safety guarantee.
    if lowered in vocabulary or lowered.isdigit():
        return word

    # "bold-um" where bold is known: snap the stem, keep the particle.
    if lowered.endswith("-um"):
        stem = _snap_word(word[:-3], word_fixes, vocabulary, changes)
        return stem + "-um"

    match = best_match(lowered, vocabulary)
    if not match:
        return word

    changes.append({
        "from": lowered, "to": match["word"],
        "reason": "sound", "score": match["score"],
    })
    return match["word"]
