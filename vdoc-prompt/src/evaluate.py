"""Parse model output, score it, and run the whole experiment grid."""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from src.models import EXHAUSTED, MODELS, is_rate_limit, pause_for
from src.prompts import ALL_TECHNIQUES, TECHNIQUES, run_technique

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "commands.csv"
RUNS_FILE = ROOT / "results" / "runs.csv"
COLUMNS = ["model", "technique", "id", "category", "output",
           "valid_json", "action_ok", "exact_ok", "latency_s"]


def parse_json(text):
    """Pull the first JSON object out of a model reply. None if there isn't one."""
    if "</think>" in (text or ""):
        text = text.split("</think>")[-1]  # p5 writes its reasoning first
    start = text.find("{") if text else -1
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        depth += (text[i] == "{") - (text[i] == "}")
        if depth == 0:
            try:
                found = json.loads(text[start:i + 1])
            except json.JSONDecodeError:
                return None
            return found if isinstance(found, dict) else None
    return None


def _index(value):
    """Treat 2, "2" and 2.0 as the same index."""
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _text(value):
    """Loose compare for value: ignore case, quotes and a trailing dot."""
    if value is None:
        return None
    return str(value).strip().strip('"\'').rstrip(".").lower()


def _value_ok(got, expected, want_action):
    """
    Does the value match? Only insert, rewrite and replace carry one.

    replace's value is a pair, and both halves have to be right: swapping the
    wrong word out, or the wrong word in, is a different edit from the one
    that was asked for. A model that returns a bare string here ("reported ->
    car") has not followed the schema, so it does not count as exact.
    """
    if want_action == "replace":
        want = expected.get("value")
        have = got.get("value")
        if not isinstance(have, dict) or not isinstance(want, dict):
            return False
        return (_text(have.get("from")) == _text(want.get("from"))
                and _text(have.get("to")) == _text(want.get("to")))
    return _text(got.get("value")) == _text(expected.get("value"))


def _target_ok(got, expected):
    want = expected.get("target") or {}
    have = got.get("target") or {}
    if not isinstance(have, dict):
        return False
    if not want:
        return not have  # expected no target (e.g. undo)
    return (have.get("type") == want.get("type")
            and _index(have.get("index")) == _index(want.get("index")))


def score(output, expected):
    """Return (valid_json, action_ok, exact_ok) for one model reply."""
    got = parse_json(output)
    if got is None:
        return False, False, False

    want_action = expected.get("action")
    action_ok = got.get("action") == want_action

    # Unclear command: asking a question is the only right answer.
    if want_action == "clarify":
        return True, action_ok, action_ok
    # Clear command: a question means the model gave up, so it is wrong.
    if got.get("action") == "clarify":
        return True, False, False

    exact_ok = action_ok and _target_ok(got, expected)
    if want_action in ("insert", "rewrite", "replace"):  # these three carry a value
        exact_ok = exact_ok and _value_ok(got, expected, want_action)
    return True, action_ok, exact_ok


def load_commands(split="test"):
    with open(DATA_FILE, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


def already_done():
    """(model, technique, id) triples that are already in runs.csv."""
    if not RUNS_FILE.exists():
        return set()
    with open(RUNS_FILE, encoding="utf-8") as f:
        return {(r["model"], r["technique"], r["id"]) for r in csv.DictReader(f)}


def append_result(row):
    """Write one result immediately so a crash loses at most one call."""
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not RUNS_FILE.exists()
    with open(RUNS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def drop_failed_rows():
    """Remove rows that came back empty or rate limited so a rerun retries
    them. A failed call is saved as "", so that is what a 429 looks like."""
    if not RUNS_FILE.exists():
        return 0
    with open(RUNS_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    keep = [r for r in rows
            if r["output"].strip() and not is_rate_limit(r["output"])]
    removed = len(rows) - len(keep)
    if removed:
        with open(RUNS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(keep)
    return removed


def mirror_technique(source, dest):
    """
    Copy one technique's saved rows under another technique's name.

    p4_few_shot and p4_few_shot_v1 are the same file: v1 *is* the baseline,
    kept under its own name so the tuning chart has a first point. At
    temperature 0 both would send the same prompt and get the same reply, so
    running them both spends 130 requests to learn nothing -- and on a free
    key those requests are the scarce part.

    Refuses unless the two prompt files really are byte-identical, so this can
    never quietly stand in for a run that would have said something different.
    """
    from src.prompts import load_template

    if load_template(source) != load_template(dest):
        raise SystemExit(f"{source} and {dest} are different prompts — "
                         f"run {dest} properly instead of mirroring it.")
    if not RUNS_FILE.exists():
        raise SystemExit(f"{RUNS_FILE} not found — run {source} first.")

    with open(RUNS_FILE, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    done = {(r["model"], r["technique"], r["id"]) for r in rows}
    copied = 0
    for row in [r for r in rows if r["technique"] == source]:
        if (row["model"], dest, row["id"]) in done:
            continue
        append_result({**row, "technique": dest})
        copied += 1
    print(f"Copied {copied} row(s) from {source} to {dest}.")
    return copied


def run_all(models=None, techniques=None, split="test"):
    models = models or MODELS
    techniques = techniques or TECHNIQUES
    rows = load_commands(split)
    done = already_done()

    todo = [(m, t, r) for m in models for t in techniques for r in rows
            if (m, t, r["id"]) not in done]

    skipped = {}
    for model, technique, row in todo:
        # Once a model is out of daily quota, every further call just buys
        # another 429, so leave the rest of its rows for another day.
        if model in EXHAUSTED:
            skipped[model] = skipped.get(model, 0) + 1
            continue
        print(f"{model} | {technique} | #{row['id']} {row['command']}")
        output, latency = run_technique(model, technique, row["command"])
        try:
            expected = json.loads(row["expected_json"])
        except json.JSONDecodeError:
            print(f"  bad expected_json on row {row['id']}, skipping")
            continue
        valid, action_ok, exact_ok = score(output, expected)
        append_result({
            "model": model, "technique": technique, "id": row["id"],
            "category": row["category"], "output": output,
            "valid_json": valid, "action_ok": action_ok,
            "exact_ok": exact_ok, "latency_s": round(latency, 3),
        })
        time.sleep(pause_for(model))  # stay under the free-tier limits

    for model, count in sorted(skipped.items()):
        print(f"\n{model}: out of quota, {count} row(s) left for next time.")
    print(f"\nDone. Results in {RUNS_FILE}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Run every model x technique on the test split.")
    ap.add_argument("--models", nargs="+", choices=MODELS, help="default: all")
    ap.add_argument("--techniques", nargs="+", choices=ALL_TECHNIQUES,
                    metavar="NAME",
                    help="default: the six techniques. The tuning versions "
                         "(p4_few_shot_v1/v2/v3) run only when named here. "
                         f"Choices: {', '.join(ALL_TECHNIQUES)}")
    ap.add_argument("--split", default="test", choices=["test", "tune"])
    ap.add_argument("--retry-failed", action="store_true",
                    help="first delete empty/rate-limited rows so they run again")
    ap.add_argument("--mirror", nargs=2, metavar=("FROM", "TO"),
                    help="copy saved rows between two identical prompts "
                         "instead of paying for the same calls twice, e.g. "
                         "--mirror p4_few_shot p4_few_shot_v1")
    args = ap.parse_args()
    if args.mirror:
        mirror_technique(*args.mirror)
        sys.exit(0)
    if args.retry_failed:
        print(f"Removed {drop_failed_rows()} failed row(s) from {RUNS_FILE.name}")
    sys.exit(run_all(args.models, args.techniques, args.split))
