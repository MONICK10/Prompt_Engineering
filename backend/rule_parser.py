"""
Rule-based parser for document editing commands.
Uses 10 regex patterns to extract {action, from_anchor, to_anchor}.

Covers common phrasings; returns None when no pattern matches.
Expected accuracy: ~35-45% on DocEdit-300.
"""
import re

# ── Shared building blocks ────────────────────────────────────────────────────
_A  = r"(bold|italic|delete|underline|heading2|heading3|highlight|font-increase)"
_F  = r"(.+?)"   # from anchor (lazy)
_T  = r"(.+?)"   # to anchor   (lazy)
_TE = r"(.+)"    # to anchor at end of string (greedy to end)

VALID_ACTIONS = {
    "bold", "italic", "delete", "underline", "heading2", "heading3",
    "replace", "insert-after", "font-increase", "highlight",
}

FLAGS = re.IGNORECASE


def _c(s):
    """Strip surrounding whitespace and trailing punctuation from a capture."""
    return s.strip().rstrip(".,!?;:") if s else s


# ── 10 patterns: (compiled regex, capture order, human label) ────────────────
#
# Capture order is a tuple saying which regex group is (action, from, to).
# The same regex group numbering applies regardless of which group comes first
# in the pattern, so we record (a_idx, f_idx, t_idx) as 1-based group numbers.
#
PATTERNS = [
    # ── P1 ──────────────────────────────────────────────────────────────────
    # ACTION [the section/part/text] from F to/until T
    # Works with an arbitrary preamble before the action keyword.
    # Examples:
    #   "bold from question to answer"
    #   "please bold the section from start to finish"
    #   "i need you to bold from introduction to conclusion"
    #   "can you please bold from car to accident"
    #   "bold text from start until finish"
    #   "apply bold from first to last"
    #   "(junk,) apply bold from introduction to conclusion"
    (
        re.compile(
            rf"(?:.*\b)?{_A}\b(?:\s+the\s+(?:section|part|text(?:\s+starting)?))?"
            rf"\s+from\s+{_F}\s+(?:to|until|ending\s+at)\s+{_TE}",
            FLAGS,
        ),
        (1, 2, 3),
        "P1: ACTION from F to T",
    ),

    # ── P2 ──────────────────────────────────────────────────────────────────
    # apply [the] ACTION [formatting|style] [from|between] F [to|and|through] T
    # Examples:
    #   "apply bold formatting between name and address"
    #   "apply bold from first to last"
    #   "apply the bold style to header through footer"
    #   "apply bold to the range first to last"
    (
        re.compile(
            rf"apply\s+(?:the\s+)?{_A}(?:\s+(?:formatting|style))?"
            rf"\s+(?:from\s+|between\s+|to\s+(?:the\s+range\s+)?){_F}"
            rf"\s+(?:to|and|through)\s+{_TE}",
            FLAGS,
        ),
        (1, 2, 3),
        "P2: apply ACTION [formatting] [from|between] F [to|and|through] T",
    ),

    # ── P3 ──────────────────────────────────────────────────────────────────
    # make [it|the text|words|section|everything] [from|between] F [to|through|and] T ACTION
    # Action keyword is at the END of the sentence.
    # Examples:
    #   "make the text topic to result bold"
    #   "make chapter through end bold"
    #   "make the words from first to last bold"
    #   "make the section between question and answer bold"
    #   "please make everything from introduction to conclusion bold"
    (
        re.compile(
            rf"(?:please\s+)?make\s+(?:it\s+)?(?:the\s+)?"
            rf"(?:text|words|section|everything)?\s*"
            rf"(?:from\s+|between\s+)?{_F}"
            rf"\s+(?:to|through|and)\s+{_F}\s+{_A}$",
            FLAGS,
        ),
        (3, 1, 2),
        "P3: make [...] F [to|through|and] T ACTION",
    ),

    # ── P4 ──────────────────────────────────────────────────────────────────
    # F to T should be / needs to be / to be / is  ACTION
    # Optional "from" preamble to skip leading filler.
    # Examples:
    #   "the text from car to accident should be bold"
    #   "the region from title to summary needs to be bold"
    #   "i need question to answer to be bold"
    #   "i want the part from question to answer to be bold"
    #   "ensure question to answer is bold"
    (
        re.compile(
            rf"(?:.*?from\s+|.*?\s+|^){_F}"
            rf"\s+to\s+{_F}"
            rf"\s+(?:should\s+be|needs\s+to\s+be|to\s+be|is)\s+{_A}$",
            FLAGS,
        ),
        (3, 1, 2),
        "P4: F to T should be/to be/is ACTION",
    ),

    # ── P5 ──────────────────────────────────────────────────────────────────
    # F to/through T [formatted] as  ACTION
    # Examples:
    #   "mark car to accident as bold"
    #   "style title through summary as bold"
    #   "set name to address as bold"
    #   "i would like header to footer formatted as bold"
    #   "format the text between car and accident as bold"  (P7 fallback)
    (
        re.compile(
            rf"(?:.*?\s)?{_F}\s+(?:to|through)\s+{_F}"
            rf"\s+(?:formatted\s+)?as\s+{_A}$",
            FLAGS,
        ),
        (3, 1, 2),
        "P5: F to/through T [formatted] as ACTION",
    ),

    # ── P6 ──────────────────────────────────────────────────────────────────
    # turn/convert [the text] F to T into  ACTION
    # Examples:
    #   "turn topic to result into bold"
    #   "convert the text first to last into bold"
    (
        re.compile(
            rf"(?:turn|convert)\s+(?:the\s+text\s+)?{_F}"
            rf"\s+to\s+{_F}\s+into\s+{_A}$",
            FLAGS,
        ),
        (3, 1, 2),
        "P6: turn/convert F to T into ACTION",
    ),

    # ── P7 ──────────────────────────────────────────────────────────────────
    # ACTION [everything] between F and T
    # Optional junk prefix before the action keyword.
    # Examples:
    #   "bold everything between introduction and conclusion"
    #   "make it bold between topic and result"
    #   "format the text between car and accident as bold"  (if P5 misses)
    (
        re.compile(
            rf"(?:.*\b)?{_A}\b\s+(?:everything\s+)?between\s+{_F}\s+and\s+{_TE}",
            FLAGS,
        ),
        (1, 2, 3),
        "P7: ACTION [everything] between F and T",
    ),

    # ── P8 ──────────────────────────────────────────────────────────────────
    # from F [up] to T make [it] ACTION
    # Examples:
    #   "from title to summary make it bold"
    #   "from topic up to result make bold"
    (
        re.compile(
            rf"from\s+{_F}\s+(?:up\s+to|to)\s+{_F}\s+make(?:\s+it)?\s+{_A}$",
            FLAGS,
        ),
        (3, 1, 2),
        "P8: from F [up] to T make [it] ACTION",
    ),

    # ── P9 ──────────────────────────────────────────────────────────────────
    # i need/want ACTION formatting on / applied between  F to/and T
    # Examples:
    #   "i need bold formatting on chapter to end"
    #   "i want bold applied between first and last"
    (
        re.compile(
            rf"i\s+(?:need|want)\s+{_A}\s+"
            rf"(?:formatting\s+on\s+|applied\s+between\s+){_F}\s+(?:to|and)\s+{_TE}",
            FLAGS,
        ),
        (1, 2, 3),
        "P9: i need/want ACTION formatting on/applied between F to/and T",
    ),

    # ── P10 ─────────────────────────────────────────────────────────────────
    # put/give ACTION [on|style to] F [through|to] T
    (
        re.compile(
            rf"(?:put|give)\s+{_A}\s+(?:on\s+|style\s+to\s+){_F}"
            rf"\s+(?:through|to)\s+{_TE}",
            FLAGS,
        ),
        (1, 2, 3),
        "P10: put/give ACTION on/style-to F through/to T",
    ),

    # ── P11: REPLACE ─────────────────────────────────────────────────────────
    # replace/swap/substitute F with/by/for T
    # "replace car with vehicle"
    # "please replace car with vehicle"
    # "find car and replace with vehicle"
    (
        re.compile(
            r"(?:.*\b)?(?:replace|substitute|swap)\s+(?:the\s+(?:word\s+|text\s+|term\s+))?(.+?)"
            r"\s+(?:with|by|using)\s+(?:the\s+(?:word\s+|text\s+))?(.+)",
            FLAGS,
        ),
        None,   # handled separately — action is always "replace"
        "P11: replace F with T",
    ),

    # ── P12: INSERT-AFTER ────────────────────────────────────────────────────
    # insert/add F after T
    # "insert conclusion after summary"
    # "add the word hello after accident"
    (
        re.compile(
            r"(?:.*\b)?(?:insert|add|place|put)\s+(?:the\s+(?:word\s+|text\s+))?(.+?)"
            r"\s+(?:after|following|right after|immediately after)\s+(?:the\s+(?:word\s+))?(.+)",
            FLAGS,
        ),
        None,   # handled separately — action is always "insert-after"
        "P12: insert/add F after T",
    ),

    # ── P13: FONT-INCREASE ───────────────────────────────────────────────────
    # increase [the] font [size] from F to T
    # "increase font size from car to accident"
    (
        re.compile(
            r"(?:.*\b)?increase\s+(?:the\s+)?(?:font|text)\s+(?:size\s+)?(?:of\s+)?(?:the\s+(?:text\s+)?)?"
            rf"(?:from\s+|between\s+){_F}\s+(?:to|through|and|ending\s+at)\s+{_TE}",
            FLAGS,
        ),
        None,   # handled separately — action is always "font-increase"
        "P13: increase font [size] from F to T",
    ),
]


# ── Public API ────────────────────────────────────────────────────────────────
def parse(command: str) -> dict | None:
    """
    Parse a free-form editing command with regex patterns.

    Returns:
        dict with keys: action, from, to, confidence, method, pattern
        or None if no pattern matches.
    """
    text = command.strip()

    for pattern, capture, label in PATTERNS:
        m = pattern.search(text)
        if not m:
            continue

        groups = m.groups()

        # Patterns P11–P13 have capture=None and a fixed action
        if capture is None:
            if label.startswith("P11"):
                action, from_a, to_a = "replace", _c(groups[0]), _c(groups[1])
            elif label.startswith("P12"):
                action, from_a, to_a = "insert-after", _c(groups[0]), _c(groups[1])
            elif label.startswith("P13"):
                action, from_a, to_a = "font-increase", _c(groups[0]), _c(groups[1])
            else:
                continue
        else:
            a_idx, f_idx, t_idx = capture
            action = _c(groups[a_idx - 1])
            from_a = _c(groups[f_idx - 1])
            to_a   = _c(groups[t_idx - 1])
            if action.lower() not in VALID_ACTIONS:
                continue

        return {
            "action":     action.lower(),
            "from":       from_a,
            "to":         to_a,
            "confidence": None,
            "method":     "rule",
            "pattern":    label,
        }

    return None


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    TEST_CASES = [
        # (command, expected_action, expected_from, expected_to, should_match)
        ("bold from question to answer",
         "bold", "question", "answer", True),

        ("apply italic formatting between note and end",
         "italic", "note", "end", True),

        ("make the text topic to result bold",
         "bold", "topic", "result", True),

        ("the text from car to accident should be bold",
         "bold", "car", "accident", True),

        ("mark car to accident as bold",
         "bold", "car", "accident", True),

        ("turn topic to result into bold",
         "bold", "topic", "result", True),

        ("bold everything between introduction and conclusion",
         "bold", "introduction", "conclusion", True),

        ("from title to summary make it bold",
         "bold", "title", "summary", True),

        # ── Expected failures (no pattern covers these) ──────────────────────
        ("can you bold chapter to end",
         None, None, None, False),

        ("go ahead and delete start to finish",
         None, None, None, False),
    ]

    PASS = FAIL = 0
    print(f"\n{'#':>3}  {'Command':<50}  {'Result'}")
    print("-" * 80)

    for i, (cmd, exp_a, exp_f, exp_t, should_match) in enumerate(TEST_CASES, 1):
        result = parse(cmd)

        if should_match:
            if (result and
                    result["action"] == exp_a and
                    result["from"]   == exp_f and
                    result["to"]     == exp_t):
                status = "PASS"
                PASS += 1
            else:
                status = "FAIL"
                FAIL += 1
                got = result if result else "None"
            print(f"{i:>3}  {cmd:<50}  {status}", end="")
            if status == "FAIL" and result:
                print(f"  [got a={result['action']} f={result['from']} t={result['to']}]", end="")
            elif status == "FAIL":
                print(f"  [no match]", end="")
            print()
        else:
            if result is None:
                status = "PASS (correctly returned None)"
                PASS += 1
            else:
                status = f"FAIL (should be None; got {result})"
                FAIL += 1
            print(f"{i:>3}  {cmd:<50}  {status}")

    print("-" * 80)
    print(f"\nResults: {PASS}/10 passed")

    print("\n-- Pattern hit on all 10 cases --")
    for i, (cmd, *_) in enumerate(TEST_CASES, 1):
        result = parse(cmd)
        pat = result["pattern"] if result else "(no match)"
        print(f"  {i:>2}. {pat}")
