"""Load prompt templates and run the self-consistency technique."""

import json
import time
from collections import Counter
from pathlib import Path

from src.models import EXHAUSTED, call_model, pause_for

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompts"

# p6 has no file of its own: it runs p5 three times and votes.
TECHNIQUES = [
    "p1_zero_shot",
    "p2_role",
    "p3_structured",
    "p4_few_shot",
    "p5_cot",
    "p6_self_consistency",
]

# The prompt-tuning track: one technique, improved by hand, three times.
# v1 is today's p4_few_shot; v2 adds eduthudu and maathu->replace examples;
# v3 adds the rule that tells formatting apart from replacing.
#
# These are kept out of TECHNIQUES on purpose, so a plain `python -m
# src.evaluate` still means "the six techniques" and does not quietly triple
# into nine. Ask for them by name with --techniques.
TUNING_VERSIONS = [
    "p4_few_shot_v1",
    "p4_few_shot_v2",
    "p4_few_shot_v3",
]

ALL_TECHNIQUES = TECHNIQUES + TUNING_VERSIONS

SELF_CONSISTENCY = "p6_self_consistency"


def load_template(name):
    """Read prompts/<name>.txt as one string."""
    return (PROMPT_DIR / f"{name}.txt").read_text(encoding="utf-8")


def build(name, command):
    """Fill {command} into a template. We use replace, not format,
    because the templates contain literal { } braces of their own."""
    return load_template(name).replace("{command}", command)


def self_consistency(model, command, runs=3, temperature=0.7):
    """Run p5 a few times at a warmer temperature and keep the most
    common JSON answer. A tie keeps the first run.

    Returns (text, total_latency_seconds) just like call_model.
    """
    from src.evaluate import parse_json  # imported here to avoid a circular import

    prompt = build("p5_cot", command)
    answers = []  # (vote_key, raw_text)
    total = 0.0

    for i in range(runs):
        text, latency = call_model(model, prompt, temperature)
        total += latency
        parsed = parse_json(text)
        # Vote on the parsed JSON so spacing differences don't split the vote.
        key = json.dumps(parsed, sort_keys=True) if parsed else f"unparsed-{i}"
        answers.append((key, text))
        if model in EXHAUSTED:
            break  # no quota left, don't spend the other runs on 429s
        if i < runs - 1:
            time.sleep(pause_for(model))  # p6 fires 3 calls, so it needs the pause too

    counts = Counter(key for key, _ in answers)
    # most_common is stable, so equal counts stay in run order -> first run wins.
    winner = counts.most_common(1)[0][0]
    for key, text in answers:
        if key == winner:
            return text, total
    return "", total


def run_technique(model, technique, command):
    """Run any technique, including p6, and return (text, latency)."""
    if technique == SELF_CONSISTENCY:
        return self_consistency(model, command)
    return call_model(model, build(technique, command), temperature=0)
