"""
Splitter v0: rule-based.

Cuts one spoken utterance into action chunks at joining words.

The split is *safe*: a candidate boundary is only accepted when both sides
carry an action word of their own. This matters because joining words already
occur inside ordinary single-action commands in DocEdit-1000 --

    "summary ku aprom conclusion insert panu"   (insert-after, 20 examples)
    "select header to footer and make it bold"  (bold)
    "apply bold formatting between name and address"  (bold)

-- and splitting those would break VDOC 1.0. In each case one side has no
action word, so the boundary is rejected and the chunk stays whole.

Chunks are character spans into the original string, so a command that is not
split yields chunk text identical to the input, joining words included.
"""
import re
from dataclasses import dataclass

# ── Action vocabulary ────────────────────────────────────────────────────────
# Mined from the B-ACTION / I-ACTION spans in data/docEdit1000.json, not
# hand-invented. See scripts/mine_action_words.py for the derivation.
#
# "sub" and "title" are mined but deliberately excluded: they are labelled
# B-ACTION only inside "sub heading" / "title aa maathu", where a second
# action word is always present anyway. On their own they are ordinary nouns
# ("title ku aprom ... insert panu", "...between main and sub") and treating
# them as actions caused 7 false splits in the DocEdit-1000 regression.
ACTION_WORDS = frozenset({
    "add", "bold", "change", "delete", "edutha", "font", "heading",
    "heading2", "heading3", "highlight", "increase", "insert", "italic",
    "keezha", "keezhey", "maathi", "maathu", "mark", "perusu", "replace",
    "underline",
})

# Words that signal an anchor target ("from X to Y", "X la irunthu Y varikum").
ANCHOR_CUES = frozenset({
    "from", "to", "between", "through", "till", "until", "upto",
    "irunthu", "irundhu", "irundu", "iruntu",
    "varikum", "varaikum", "varaikkum", "vara", "vare", "varai",
    "ku", "la", "la:", "lerunthu",
})

# Pronouns that point back at the previous chunk's target. A fragment like
# "and heading2 it" is not a new action with a missing target -- it is the
# same action restated, so the boundary before it is rejected.
BACK_REFERENCES = frozenset({"it", "them", "adha", "atha", "adhe", "idha", "ithu"})

# ── Joining words ────────────────────────────────────────────────────────────
# Single tokens that may separate two actions.
JOIN_WORDS = frozenset({
    "apram", "aprom", "appuram", "appram", "piragu", "pannitu", "panitu",
    "pannittu", "then", "and", "next",
})

# Multi-token joining phrases, longest first so they match before their parts.
JOIN_PHRASES = (
    ("adhuku", "apram"),
    ("adhuku", "aprom"),
    ("after", "that"),
)

_PUNCT = ".,?!;:"

# Punctuation that can end one instruction and start the next on its own.
_SENTENCE_BREAKS = ",.?!;"

_UM_SUFFIX = re.compile(r"-?um$")


def _normalise(token):
    """Lowercase a token, drop edge punctuation, and strip a trailing -um."""
    word = token.lower().strip(_PUNCT + "\"'")
    stripped = _UM_SUFFIX.sub("", word)
    # Only treat -um as a suffix when something is left over ("um" alone is
    # not an action word with its ending removed).
    return stripped if stripped else word


def _is_um_action(token):
    """True for an '-um' coordination form of an action word (bold-um)."""
    word = token.lower().strip(_PUNCT)
    if not _UM_SUFFIX.search(word):
        return False
    return _UM_SUFFIX.sub("", word) in ACTION_WORDS


def has_action(text):
    """True when the text carries at least one action word of its own."""
    return any(_normalise(t) in ACTION_WORDS for t in text.split())


def has_target(text):
    """
    True when the text names something to act on -- a line number or an
    anchor cue. Used to reject a boundary whose left side is not a complete
    instruction, such as the self-correction "bold is wrong, apply bold from
    header to footer" (6 of these are in DocEdit-1000).
    """
    from . import line_resolver   # local import keeps the module cycle-free
    if line_resolver.resolve(text):
        return True
    return any(_normalise(t) in ANCHOR_CUES for t in text.split())


def is_back_reference(text):
    """True for a fragment that points back instead of naming a new target."""
    words = [_normalise(t) for t in text.split()]
    return any(w in BACK_REFERENCES for w in words)


def _can_open_new_action(left, right):
    """
    Decide whether a candidate boundary is a real split.

    left   must be a complete instruction: an action AND something to act on.
    right  must carry an action of its own, and must not merely refer back
           to the left side's target.

    The right side is allowed to have no target -- that is the shared-target
    case ("first line bold-um | italic-um pannu"), which the sharer fills in.
    """
    if not has_action(left) or not has_target(left):
        return False
    if not has_action(right):
        return False
    if is_back_reference(right) and not has_target(right):
        return False
    return True


@dataclass
class Chunk:
    """One action chunk: a character span of the original utterance."""
    text: str
    start: int
    end: int
    join_before: str | None = None   # the joining word that opened this chunk

    def to_dict(self):
        return {
            "text": self.text,
            "start": self.start,
            "end": self.end,
            "join_before": self.join_before,
        }


def _tokens_with_spans(text):
    return [(m.group(0), m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _candidate_boundaries(tokens):
    """
    Find every place the utterance *could* be cut.

    Returns a list of (left_end, right_start, join_label) character offsets.
    The joining word itself falls in the gap, so merging two segments back
    together restores the original substring exactly.
    """
    bounds = []
    i = 0
    n = len(tokens)
    while i < n:
        token, start, end = tokens[i]
        word = token.lower().strip(_PUNCT)

        # Multi-token joining phrase
        matched_phrase = None
        for phrase in JOIN_PHRASES:
            if i + len(phrase) <= n:
                window = tuple(tokens[i + k][0].lower().strip(_PUNCT)
                               for k in range(len(phrase)))
                if window == phrase:
                    matched_phrase = phrase
                    break
        if matched_phrase:
            last = tokens[i + len(matched_phrase) - 1]
            bounds.append((start, last[2], " ".join(matched_phrase)))
            i += len(matched_phrase)
            continue

        # Single joining word
        if word in JOIN_WORDS:
            bounds.append((start, end, word))
            i += 1
            continue

        # "-um" coordination: boundary opens *before* the second and later
        # action-um tokens, e.g. "bold-um | italic-um pannu".
        if _is_um_action(token) and i > 0:
            earlier = any(_is_um_action(tokens[k][0]) for k in range(i))
            if earlier:
                bounds.append((start, start, "-um"))
        i += 1

    # Punctuation can separate two actions on its own, with no joining word.
    # Whisper puts a full stop between spoken sentences -- "first line bold
    # pannu. second line font size 20 vei." -- so a sentence break is just as
    # much a boundary as "apram" is. The safe rule still applies afterwards,
    # so a full stop inside a single instruction costs nothing.
    for token, start, end in tokens:
        stripped = token.rstrip(_SENTENCE_BREAKS)
        if stripped == token:
            continue
        cut = start + len(stripped)
        if not any(b[0] <= cut <= b[1] for b in bounds):
            bounds.append((cut, end, token[len(stripped):]))

    bounds.sort(key=lambda b: b[0])
    return bounds


def split(text):
    """
    Split an utterance into action chunks.

    Always returns at least one chunk. When no boundary is safe the single
    chunk's text equals the input, so downstream parsing is unchanged.
    """
    original = text
    tokens = _tokens_with_spans(original)
    if not tokens:
        return [Chunk(text=original.strip(), start=0, end=len(original))]

    bounds = _candidate_boundaries(tokens)
    if not bounds:
        return [Chunk(text=original.strip(), start=0, end=len(original))]

    # Cut at every candidate, then merge back the unsafe ones.
    # Segment i is opened by the label of boundary i-1.
    spans = []
    cursor = tokens[0][1]
    prev_label = None
    for left_end, right_start, label in bounds:
        spans.append([cursor, left_end, prev_label])
        cursor = right_start
        prev_label = label
    spans.append([cursor, tokens[-1][2], prev_label])

    # Drop empties produced by back-to-back boundaries (", apram").
    spans = [s for s in spans if original[s[0]:s[1]].strip()]
    if not spans:
        return [Chunk(text=original.strip(), start=0, end=len(original))]

    # Walk left to right and keep only the boundaries that are safe.
    # Merging unions the spans, so a rejected joining word comes back with it.
    kept = [spans[0]]
    for span in spans[1:]:
        left = original[kept[-1][0]:kept[-1][1]]
        right = original[span[0]:span[1]]
        if _can_open_new_action(left, right):
            kept.append(span)
        else:
            kept[-1][1] = span[1]
    spans = kept

    return [
        Chunk(text=original[s[0]:s[1]].strip().strip(_SENTENCE_BREAKS).strip(),
              start=s[0], end=s[1], join_before=s[2])
        for s in spans
    ]
