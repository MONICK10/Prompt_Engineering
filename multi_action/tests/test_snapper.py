r"""
Vocabulary snapper tests.

The important half of these is the second half: a correct command must come
out of the snapper byte-identical. Snapping sits in front of the parser, so
anything it rewrites by mistake becomes a parse error the user cannot explain.

    .\.venv\Scripts\python.exe -m pytest multi_action\tests\test_snapper.py -q
"""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from multi_action import snapper          # noqa: E402

DOCUMENT = """VDOC 2.0 Multi-Action Voice Commands
This is the introduction to the VDOC system.
The car failed to brake and the accident was reported.
Researchers compared a rule-based system against DocEditBERT.
The conclusion summarises the key findings of the study."""


def snapped(text):
    return snapper.snap(text, document=DOCUMENT)["snapped"]


# ── Real Whisper transcripts ─────────────────────────────────────────────────

def test_hyphen_blob_is_split_and_snapped():
    assert snapped("next line is, convolution-vari-crum, bold-um") == \
        "next line is, conclusion varikum, bold-um"


def test_phrase_fix_like_this():
    assert "la irunthu" in snapped("commands like this, system-paric, bold-um")


def test_phrase_fixes_line_up_is_and_to_have():
    result = snapped("car line-up is reported to have bold pannu")
    assert "car la irunthu" in result
    assert "vara bold pannu" in result


def test_unrelated_speech_is_left_alone():
    assert snapped("I just want a bold-um") == "I just want a bold-um"


# ── Correct commands must survive untouched ──────────────────────────────────

UNTOUCHED = [
    "first line bold pannu",
    "first line bold panu",
    "second line delete pannu, apram second line bold pannu",
    "first line bold-um italic-um pannu",
    "introduction la irunthu conclusion varikum bold panu",
    "select header to footer and make it bold",
    "summary ku aprom conclusion insert panu",
    "bold the first line and delete the third line",
    "last line bold pannu",
    "line 2 highlight pannu",
]


@pytest.mark.parametrize("command", UNTOUCHED)
def test_good_commands_are_not_snapped(command):
    result = snapper.snap(command, document=DOCUMENT)
    assert result["snapped"] == command, f"snapper changed {command!r}"
    assert result["changes"] == []


def test_words_in_the_document_are_safe():
    # "DocEditBERT" and "Researchers" are only in the document, not the
    # dataset, so without the document in the vocabulary they would snap.
    text = "bold from Researchers to DocEditBERT"
    assert snapper.snap(text, document=DOCUMENT)["snapped"] == text


def test_digits_are_never_snapped():
    assert snapped("line 2 bold pannu") == "line 2 bold pannu"


# ── The off switch ───────────────────────────────────────────────────────────

def test_snapping_can_be_turned_off():
    text = "next line is, convolution-vari-crum, bold-um"
    result = snapper.snap(text, document=DOCUMENT, enabled=False)
    assert result["snapped"] == text
    assert result["changed"] is False
    assert result["enabled"] is False


# ── Matching behaviour ───────────────────────────────────────────────────────

def test_vocabulary_includes_dataset_particles_and_document():
    vocab = snapper.build_vocabulary(DOCUMENT)
    assert "conclusion" in vocab      # dataset
    assert "varikum" in vocab         # particle list
    assert "doceditbert" in vocab     # document
    assert len(vocab) > 200


def test_a_clear_winner_is_accepted():
    # "bolld" is one edit from "bold" and two from "bolt" — not a coin flip.
    assert snapper.best_match("bolld", {"bold", "bolt"})["word"] == "bold"


def test_ambiguous_matches_are_refused():
    # "bole" is one edit from both, and sounds identical to both (PL / PLT),
    # so there is no honest winner and nothing should be changed.
    assert snapper.best_match("bole", {"bold", "bolt"}) is None
