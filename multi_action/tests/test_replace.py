r"""
Line-scoped replace: "third line-la reported-a car-nu maathu".

Before this, a replace command that named a line could not run at all. The
line resolver won, the anchor slots were thrown away, and the executor was
handed a line target with nothing to swap. These tests pin the two halves of
the fix: the pipeline keeps both, and only for replace.

Run from the project root:

    .\.venv\Scripts\python.exe -m pytest multi_action\tests -v
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from multi_action import line_resolver, pipeline   # noqa: E402

DOC_LINES = 5


def stub(action, source, dest, confidence=0.99):
    """A fixed DocEditBERT reply, so these tests do not depend on the model."""
    def predict(_text):
        return {"action": action, "from": source, "to": dest,
                "confidence": confidence, "method": "stub"}
    return predict


def only_action(command, predict_fn):
    result = pipeline.parse_multi(command, predict_fn, line_count=DOC_LINES)
    assert result["count"] == 1
    return result["actions"][0]


# ── The anchor survives a line target, for replace only ──────────────────────

def test_line_scoped_replace_keeps_both_halves():
    entry = only_action("3rd line-la reported-a car-nu maathu",
                        stub("replace", "reported-a", "car-nu"))
    target = entry["target"]
    assert target["kind"] == "line"
    assert target["n"] == 3
    assert target["from"] == "reported"
    assert target["to"] == "car"


def test_formatting_still_targets_the_line_alone():
    """A line is the target for bold, not a scope -- no anchor is carried."""
    entry = only_action("second line bold pannu", stub("bold", "second", None))
    assert entry["target"] == {"kind": "line", "n": 2, "unit": "line",
                               "matched": "second line"}


def test_replace_without_a_line_is_unchanged():
    entry = only_action("change reported to car",
                        stub("replace", "reported", "car"))
    assert entry["target"]["kind"] == "anchor"
    assert entry["target"]["from"] == "reported"


def test_replace_with_a_line_but_no_slots_stays_a_plain_line_target():
    entry = only_action("third line change something",
                        stub("replace", None, None))
    assert entry["target"]["kind"] == "line"
    assert "from" not in entry["target"]


# ── Tanglish case markers ────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("reported-a", "reported"),
    ("address-nu", "address"),
    ("method-ku", "method"),
    ("car-aa", "car"),
    ("title-ah", "title"),
    ("conclusion", "conclusion"),
])
def test_case_marker_is_stripped(raw, expected):
    assert pipeline._clean_anchor(raw) == expected


def test_a_real_hyphenated_word_is_left_alone():
    assert pipeline._clean_anchor("rule-based") == "rule-based"


# ── The resolver reads a line through a glued-on particle ────────────────────

@pytest.mark.parametrize("command,number", [
    ("3rd line-la reported-a car-nu maathu", 3),
    ("moonavathu line-la result-ku badhila outcome podu", 3),
    ("second line-la name-a address-nu maathu", 2),
    ("kadaisi line-la study-ku badhila research podu", -1),
])
def test_hyphenated_line_phrase_resolves(command, number):
    assert line_resolver.resolve(command)["n"] == number


def test_hyphenated_content_word_is_not_a_line():
    assert line_resolver.resolve("rule-based system bold pannu") is None


# ── The label the demo card shows ────────────────────────────────────────────

def test_label_names_both_words():
    entry = only_action("3rd line-la reported-a car-nu maathu",
                        stub("replace", "reported-a", "car-nu"))
    assert pipeline.describe(entry) == 'replace "reported" with "car" on line 3'
