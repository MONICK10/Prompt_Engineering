r"""
Rule 1 proof: VDOC 2.0 must not change VDOC 1.0.

Every one of the 1,000 DocEdit-1000 commands goes through the new pipeline.
Each must still produce exactly one action, with the same action / from / to
that the old single-call path produces.

Writes results/regression_1000.json so the Results page can read a real
number instead of a typed-in one.

    .\.venv\Scripts\python.exe -m pytest multi_action\tests\test_regression_1000.py -q -s
"""
import json
import os
import sys
from datetime import datetime

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from multi_action import pipeline          # noqa: E402

DATASET = os.path.join(ROOT, "data", "docEdit1000.json")
RESULTS_DIR = os.path.join(ROOT, "results")
RESULTS_FILE = os.path.join(RESULTS_DIR, "regression_1000.json")


def _load_dataset():
    with open(DATASET, encoding="utf-8") as handle:
        return json.load(handle)


def _anchor_same(new_value, old_value):
    """
    One difference is allowed, and only one.

    The pipeline strips a Tanglish case marker from an anchor -- "reported-a"
    becomes "reported" -- so the executor can find that word in the document.
    The old single-call path hands the raw token back, and "reported-a"
    matches nothing on the page. The new answer is the same edit, said
    properly, so it does not count as a changed result.

    Any other difference still does.
    """
    new_value = new_value or None
    old_value = old_value or None
    return new_value == old_value or new_value == pipeline._clean_anchor(old_value)


def test_pipeline_matches_old_vdoc_on_all_1000():
    import doceditbert_parser
    predict = doceditbert_parser.predict

    examples = _load_dataset()
    unchanged = 0
    marker_stripped = 0
    split_wrongly = []
    differed = []

    for example in examples:
        text = example["text"]

        old = predict(text)
        new = pipeline.parse_multi(text, predict, line_count=None)

        if new["count"] != 1:
            split_wrongly.append({
                "text": text,
                "lang": example.get("lang"),
                "intent": example.get("intent"),
                "chunks": [c["text"] for c in new["chunks"]],
            })
            continue

        entry = new["actions"][0]
        target = entry.get("target") or {}
        identical = (
            entry["action"] == old.get("action")
            and (target.get("from") or None) == (old.get("from") or None)
            and (target.get("to") or None) == (old.get("to") or None)
        )
        same = identical or (
            entry["action"] == old.get("action")
            and _anchor_same(target.get("from"), old.get("from"))
            and _anchor_same(target.get("to"), old.get("to"))
        )
        if same:
            unchanged += 1
            if not identical:
                marker_stripped += 1
        else:
            differed.append({
                "text": text,
                "lang": example.get("lang"),
                "old": {k: old.get(k) for k in ("action", "from", "to")},
                "new": {"action": entry["action"],
                        "from": target.get("from"),
                        "to": target.get("to")},
            })

    total = len(examples)
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": "data/docEdit1000.json",
        "total": total,
        "unchanged": unchanged,
        "case_marker_stripped": marker_stripped,
        "wrongly_split": len(split_wrongly),
        "changed_result": len(differed),
        "accuracy_unchanged": round(unchanged / total, 4),
        "splitter": pipeline.__dict__.get("parse_multi") and "v0: rule-based",
        "examples_wrongly_split": split_wrongly[:20],
        "examples_changed": differed[:20],
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nRegression: {unchanged}/{total} commands unchanged "
          f"({len(split_wrongly)} wrongly split, {len(differed)} changed result, "
          f"{marker_stripped} with a Tanglish case marker stripped)")
    print(f"Saved to {RESULTS_FILE}")

    if split_wrongly:
        print("\nWrongly split:")
        for item in split_wrongly[:10]:
            print(f"  {item['text']!r} -> {item['chunks']}")
    if differed:
        print("\nChanged result:")
        for item in differed[:10]:
            print(f"  {item['text']!r}\n     old={item['old']}\n     new={item['new']}")

    assert unchanged == total, (
        f"{total - unchanged} of {total} commands changed — rule 1 broken"
    )
