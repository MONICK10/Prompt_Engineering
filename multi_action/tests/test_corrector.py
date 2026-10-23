r"""
Action-word corrector tests.

No model is loaded here -- the corrector is pure text work, so these run in
well under a second.

    .\.venv\Scripts\python.exe -m pytest multi_action\tests\test_corrector.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from multi_action import corrector          # noqa: E402


def suggestions_for(text):
    return {s["word"]: s["suggestion"] for s in corrector.review(text)["suggestions"]}


# ── Mishearings the corrector must catch ─────────────────────────────────────

@pytest.mark.parametrize("heard,expected", [
    ("first line bowl panu", ("bowl", "bold")),
    ("first line bolt pannu", ("bolt", "bold")),
    ("first line bull pannu", ("bull", "bold")),
    ("first line italik pannu", ("italik", "italic")),
    ("second line delet pannu", ("delet", "delete")),
    ("first line underlign pannu", ("underlign", "underline")),
])
def test_catches_misheard_action_words(heard, expected):
    word, suggestion = expected
    assert suggestions_for(heard).get(word) == suggestion


# ── Correct commands must be left alone ──────────────────────────────────────

CLEAN = [
    "first line bold pannu",
    "first line bold panu",
    "second line delete pannu, apram second line bold pannu",
    "first line bold-um italic-um pannu",
    "introduction la irunthu conclusion varikum bold panu",
    "select header to footer and make it bold",
    "summary ku aprom conclusion insert panu",
    "bold the first line and delete the third line",
]


@pytest.mark.parametrize("command", CLEAN)
def test_correct_commands_get_no_suggestions(command):
    assert suggestions_for(command) == {}, f"{command!r} was wrongly flagged"


def test_pannu_is_never_corrected():
    # "pannu" is not in DocEdit-1000 (which has "panu"), so without the
    # verb-particle list the corrector offers "pannu -> pannitu" on a
    # perfectly good command.
    assert "pannu" not in suggestions_for("first line bold pannu")


def test_dataset_anchor_words_are_safe():
    # Anchor words sit close to command words but are legitimate.
    for word in ["conclusion", "footer", "header", "summary", "title"]:
        assert word not in suggestions_for(f"bold from {word} to end")


# ── Saved corrections ────────────────────────────────────────────────────────

def test_saving_and_applying_a_correction(tmp_path, monkeypatch):
    store = tmp_path / "user_corrections.json"
    monkeypatch.setattr(corrector, "CORRECTIONS_FILE", str(store))

    assert corrector.load_corrections() == {}

    corrector.save_correction("bowl", "bold")
    assert corrector.load_corrections() == {"bowl": "bold"}

    review = corrector.review("first line bowl panu")
    assert review["corrected"] == "first line bold panu"
    assert review["auto_applied"] == [{"from": "bowl", "to": "bold"}]
    assert review["suggestions"] == []

    corrector.forget_correction("bowl")
    assert corrector.load_corrections() == {}


def test_a_correction_must_point_at_a_command_word(tmp_path, monkeypatch):
    monkeypatch.setattr(corrector, "CORRECTIONS_FILE", str(tmp_path / "c.json"))
    with pytest.raises(ValueError):
        corrector.save_correction("bowl", "banana")


def test_suggestions_are_deterministic():
    # TARGET_WORDS is a frozenset; iterating it unordered would make a tie
    # resolve differently between runs.
    first = suggestions_for("first line bowl panu")
    for _ in range(5):
        assert suggestions_for("first line bowl panu") == first
