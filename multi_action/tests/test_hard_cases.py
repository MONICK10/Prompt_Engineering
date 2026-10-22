r"""
The four hard cases from the VDOC 2.0 spec, in English and in Tanglish.

These run against the real DocEditBERT model -- no stubbed predictions.
Run from the project root:

    .\.venv\Scripts\python.exe -m pytest multi_action\tests -v
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from multi_action import pipeline          # noqa: E402

DOC_LINES = 5


@pytest.fixture(scope="session")
def predict():
    """The real DocEditBERT parser, loaded once for the whole session."""
    import doceditbert_parser
    return doceditbert_parser.predict


def run(command, predict_fn, line_count=DOC_LINES):
    return pipeline.parse_multi(command, predict_fn, line_count=line_count)


def line_numbers(result):
    """The current line number each action landed on, in order."""
    return [
        entry.get("resolved_line", {}).get("current")
        for entry in result["actions"]
    ]


def actions_of(result):
    return [entry["action"] for entry in result["actions"]]


# ── Case 1: two actions, two targets ─────────────────────────────────────────

def test_case1_english(predict):
    result = run("first line bold, then delete the third line", predict)
    assert result["count"] == 2
    assert actions_of(result) == ["bold", "delete"]
    assert line_numbers(result) == [1, 3]


def test_case1_tanglish(predict):
    result = run("first line bold pannu, apram third line delete pannu", predict)
    assert result["count"] == 2
    assert actions_of(result) == ["bold", "delete"]
    assert line_numbers(result) == [1, 3]


# ── Case 2: one target, two actions ──────────────────────────────────────────

def test_case2_english(predict):
    result = run("make the first line bold and italic", predict)
    assert result["count"] == 2
    assert actions_of(result) == ["bold", "italic"]
    # The second chunk has no target of its own and borrows the first's.
    assert result["actions"][1]["shared_target"] is True
    assert line_numbers(result) == [1, 1]


def test_case2_tanglish(predict):
    result = run("first line bold-um italic-um pannu", predict)
    assert result["count"] == 2
    assert actions_of(result) == ["bold", "italic"]
    assert result["actions"][1]["shared_target"] is True
    assert line_numbers(result) == [1, 1]


# ── Case 3: lines shift after a delete ───────────────────────────────────────

def test_case3_english(predict):
    result = run("delete the second line, then bold the second line", predict)
    assert result["count"] == 2
    assert actions_of(result) == ["delete", "bold"]
    # Both say "second line", but the second one lands on what was line 3.
    assert line_numbers(result) == [2, 2]
    assert result["actions"][0]["resolved_line"]["original"] == 2
    assert result["actions"][1]["resolved_line"]["original"] == 3
    assert result["actions"][1]["resolved_line"]["moved"] is True


def test_case3_tanglish(predict):
    result = run("second line delete pannu, apram second line bold pannu", predict)
    assert result["count"] == 2
    assert actions_of(result) == ["delete", "bold"]
    assert line_numbers(result) == [2, 2]
    assert result["actions"][0]["resolved_line"]["original"] == 2
    assert result["actions"][1]["resolved_line"]["original"] == 3
    assert result["actions"][1]["resolved_line"]["moved"] is True


# ── Case 4: joining words all split correctly ────────────────────────────────

ENGLISH_JOINS = [
    "bold the first line and delete the third line",
    "bold the first line then delete the third line",
    "bold the first line, delete the third line",
    "bold the first line after that delete the third line",
]

TANGLISH_JOINS = [
    "first line bold pannu apram third line delete pannu",
    "first line bold pannu aprom third line delete pannu",
    "first line bold pannu pannitu third line delete pannu",
    "first line bold pannu piragu third line delete pannu",
]


@pytest.mark.parametrize("command", ENGLISH_JOINS)
def test_case4_english(command, predict):
    result = run(command, predict)
    assert result["count"] == 2, f"{command!r} did not split into 2"
    assert actions_of(result) == ["bold", "delete"]
    assert line_numbers(result) == [1, 3]


@pytest.mark.parametrize("command", TANGLISH_JOINS)
def test_case4_tanglish(command, predict):
    result = run(command, predict)
    assert result["count"] == 2, f"{command!r} did not split into 2"
    assert actions_of(result) == ["bold", "delete"]
    assert line_numbers(result) == [1, 3]


# ── Rule 1: single-action commands must not be split ─────────────────────────

NEVER_SPLIT = [
    # Joining words that live inside one action in DocEdit-1000.
    "summary ku aprom conclusion insert panu",
    "introduction ku piragu hello add panu",
    "select header to footer and make it bold",
    "apply bold formatting between name and address",
    "make the section between question and answer bold",
    "summary ku next uh conclusion insert panu",
]


@pytest.mark.parametrize("command", NEVER_SPLIT)
def test_single_action_commands_are_not_split(command, predict):
    result = run(command, predict, line_count=None)
    assert result["count"] == 1, f"{command!r} was wrongly split"
    assert result["chunks"][0]["text"] == command
