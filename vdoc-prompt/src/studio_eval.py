"""
Score studio_actions_v1 against studio_actions_zero_shot on Groq.

Same shape as src.evaluate, but the answer is a JSON *array*, so a reply is
exact only when it has the right actions in the right order with the right
fields. Getting two of three actions right is not the same edit.

    python -m src.studio_eval
    python -m src.studio_eval --techniques studio_actions_v1
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

from src.models import EXHAUSTED, call_model, pause_for
from src.prompts import build

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "studio_commands.csv"
RUNS_FILE = ROOT / "results" / "studio_runs.csv"
TECHNIQUES = ["studio_actions_v1", "studio_actions_zero_shot"]
COLUMNS = ["technique", "id", "category", "output", "valid_json",
           "count_ok", "actions_ok", "exact_ok", "latency_s"]

# Fields that have to match when they are set in the expected answer. Anything
# the label leaves as null is not checked, so a model is free to spell out
# defaults without being marked wrong for it.
CHECKED = ("topic", "heading", "style", "font", "size", "value", "format",
           "from", "to", "to_index", "text", "bold", "italic", "underline",
           "align", "section_index")


def parse_array(text):
    """Pull the first JSON array out of a reply; a lone object counts as one."""
    if not text:
        return None
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            found = json.loads(text[start : end + 1])
            if isinstance(found, list):
                return found
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            found = json.loads(text[start : end + 1])
            return [found] if isinstance(found, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _same_target(got, want):
    if want is None:
        return True
    got = got if isinstance(got, dict) else {}
    if str(got.get("type") or "").lower() != str(want.get("type") or "").lower():
        return False
    return _num(got.get("index")) == _num(want.get("index"))


def _num(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return str(value).strip().lower()


def _same_value(got, want):
    if isinstance(want, bool) or isinstance(got, bool):
        return bool(got) == bool(want)
    if isinstance(want, (int, float)) or isinstance(got, (int, float)):
        return _num(got) == _num(want)
    return str(got or "").strip().lower() == str(want or "").strip().lower()


def same_action(got, want):
    if not isinstance(got, dict):
        return False
    if str(got.get("action") or "").lower() != str(want.get("action") or "").lower():
        return False
    if "target" in want and not _same_target(got.get("target"), want.get("target")):
        return False
    for key in CHECKED:
        if key not in want or want[key] is None:
            continue
        if not _same_value(got.get(key), want[key]):
            return False
    return True


def score(output, expected):
    """(valid_json, count_ok, actions_ok, exact_ok) for one reply."""
    got = parse_array(output)
    if got is None:
        return False, False, False, False
    count_ok = len(got) == len(expected)
    verbs = [str(a.get("action", "")).lower() for a in got if isinstance(a, dict)]
    want_verbs = [str(a.get("action", "")).lower() for a in expected]
    actions_ok = verbs == want_verbs
    exact_ok = count_ok and all(
        same_action(g, w) for g, w in zip(got, expected))
    return True, count_ok, actions_ok, exact_ok


def load_commands():
    with open(DATA_FILE, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def run(techniques=None, model="groq"):
    techniques = techniques or TECHNIQUES
    rows = load_commands()
    RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if RUNS_FILE.exists():
        with open(RUNS_FILE, encoding="utf-8") as handle:
            done = {(r["technique"], r["id"]) for r in csv.DictReader(handle)}

    new_file = not RUNS_FILE.exists()
    with open(RUNS_FILE, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        for technique in techniques:
            for row in rows:
                if (technique, row["id"]) in done or model in EXHAUSTED:
                    continue
                print(f"{technique} | #{row['id']} {row['command']}")
                output, latency = call_model(model, build(technique, row["command"]),
                                             temperature=0)
                expected = json.loads(row["expected_json"])
                valid, count_ok, actions_ok, exact_ok = score(output, expected)
                writer.writerow({
                    "technique": technique, "id": row["id"],
                    "category": row["category"], "output": output,
                    "valid_json": valid, "count_ok": count_ok,
                    "actions_ok": actions_ok, "exact_ok": exact_ok,
                    "latency_s": round(latency, 3)})
                handle.flush()
                time.sleep(pause_for(model))   # respect the token-per-minute budget
    return RUNS_FILE


def report():
    with open(RUNS_FILE, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    truthy = lambda v: str(v).strip().lower() in ("true", "1")   # noqa: E731

    print(f"\n{'technique':<28}{'valid':>7}{'verbs':>8}{'exact':>8}{'median s':>10}  n")
    print("-" * 70)
    for technique in TECHNIQUES:
        mine = [r for r in rows if r["technique"] == technique]
        if not mine:
            continue
        n = len(mine)
        pct = lambda k: 100 * sum(truthy(r[k]) for r in mine) / n   # noqa: E731
        times = sorted(float(r["latency_s"]) for r in mine)
        print(f"{technique:<28}{pct('valid_json'):>6.1f}%{pct('actions_ok'):>7.1f}%"
              f"{pct('exact_ok'):>7.1f}%{times[n // 2]:>9.2f}s  {n}")

    print(f"\n{'category':<12}" + "".join(f"{t.replace('studio_actions_',''):>20}"
                                          for t in TECHNIQUES))
    for category in ("single", "multi", "unclear"):
        line = f"{category:<12}"
        for technique in TECHNIQUES:
            mine = [r for r in rows
                    if r["technique"] == technique and r["category"] == category]
            line += (f"{100 * sum(truthy(r['exact_ok']) for r in mine) / len(mine):>19.1f}%"
                     if mine else f"{'-':>20}")
        print(line)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--techniques", nargs="+", choices=TECHNIQUES)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    if not args.report_only:
        run(args.techniques)
    report()
    sys.exit(0)
