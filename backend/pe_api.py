"""
Prompt-engineering endpoints: the vdoc-prompt experiment, served inside VDOC.

VDOC 2.0 answers a command with DocEditBERT, a model we fine-tuned on our own
data. The vdoc-prompt project answers the same command with a general LLM and
a written prompt, and asks which prompt wins. This module puts that project
behind HTTP so the web app can show both on the same document.

Everything is imported from vdoc-prompt/, never copied, so the prompts, the
model list and the scoring rules stay defined in one place.

    POST /pe/run      one command, one model, one technique
    POST /pe/compare  one command, three techniques, plus DocEditBERT
    GET  /pe/results  the tables, read from vdoc-prompt/results/runs.csv
"""
import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PE_ROOT = os.path.join(_ROOT, "vdoc-prompt")

# vdoc-prompt's modules import each other as "src.something", so its own
# folder has to be on the path for those imports to resolve.
if PE_ROOT not in sys.path:
    sys.path.insert(0, PE_ROOT)

# src.models loads every .env from vdoc-prompt upward on import, which is how
# the keys arrive. Nothing else here needs to touch the environment.
from src import report as pe_report              # noqa: E402
from src.evaluate import load_commands, parse_json, score   # noqa: E402
from src.models import GROQ_MODEL, MODELS, call_model       # noqa: E402
from src.prompts import ALL_TECHNIQUES, SELF_CONSISTENCY    # noqa: E402
from src.prompts import build, self_consistency             # noqa: E402

# Shown wherever a model is named, so the reader knows which Groq model ran.
MODEL_LABELS = {
    "groq": f"Groq ({GROQ_MODEL.split('/')[-1]})",
    "gemini": "Gemini",
    "ollama-local": "Ollama (local)",
}

# The three the compare view puts side by side.
COMPARE_TECHNIQUES = ["p1_zero_shot", "p4_few_shot_v1", "p4_few_shot_v3"]

TECHNIQUE_LABELS = {
    "p1_zero_shot": "Zero-shot",
    "p2_role": "Role",
    "p3_structured": "Structured",
    "p4_few_shot": "Few-shot",
    "p5_cot": "Chain of thought",
    "p6_self_consistency": "Self-consistency",
    "p4_few_shot_v1": "Few-shot v1",
    "p4_few_shot_v2": "Few-shot v2",
    "p4_few_shot_v3": "Few-shot v3",
}

# What changed at each step of the tuning track, for the caption under the chart.
TUNING_VERSIONS = ["p4_few_shot_v1", "p4_few_shot_v2", "p4_few_shot_v3"]
TUNING_NOTES = {
    "p4_few_shot_v1": "The starting prompt. Six examples, no replace example.",
    "p4_few_shot_v2": "Added 'eduthudu' = delete, and three 'maathu' = replace examples.",
    "p4_few_shot_v3": "Added the rule: change X to bold is formatting, change X to a normal word is replace.",
}

# ── Turning the LLM's JSON into something VDOC can run ───────────────────────

# The LLM writes an action name from its own schema; the demo executor knows
# these names. The two lists are close but not identical.
ACTION_MAP = {
    "bold": "bold", "italic": "italic", "underline": "underline",
    "highlight": "highlight", "font-increase": "font-increase",
    "delete": "delete", "replace": "replace", "insert": "insert-after",
}

# Actions the demo document genuinely cannot carry out. Saying so plainly
# beats applying something else and letting the reader think it worked.
UNSUPPORTED = {
    "rewrite": "the demo document has no rewriting engine",
    "move": "moving lines is not in the demo executor",
    "undo": "the demo keeps no history to undo",
    "clarify": "the model asked a question instead of editing",
}

LINE_LIKE = {"line", "paragraph", "title"}


def _index(value):
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def to_vdoc_action(parsed, command):
    """
    Convert one LLM JSON object into the entry shape applyAction expects.

    Returns (entry, note). entry is None when the reply cannot be run, and
    note then says why in plain words.
    """
    if not isinstance(parsed, dict):
        return None, "AI reply was not valid JSON"

    raw_action = parsed.get("action")
    if raw_action is None:
        return None, 'the AI reply had no "action" field'
    if raw_action in UNSUPPORTED:
        return None, f'"{raw_action}" cannot be applied here — {UNSUPPORTED[raw_action]}'
    action = ACTION_MAP.get(raw_action)
    if not action:
        return None, f'the AI used an action VDOC does not know: "{raw_action}"'

    # A model that ignores the schema can put anything here -- a bare string,
    # a list, a number. Only a dict carries a type and an index.
    target_in = parsed.get("target")
    if not isinstance(target_in, dict):
        target_in = {}
    kind = str(target_in.get("type") or "").lower()
    index = _index(target_in.get("index"))
    if kind == "title" and index is None:
        index = 1               # the title is line 1 of the demo document

    value = parsed.get("value")
    swap = value if (action == "replace" and isinstance(value, dict)) else None
    if action == "replace" and not swap:
        return None, 'replace needs value {"from": "...", "to": "..."}'

    note = None
    if kind in LINE_LIKE and index is not None:
        target = {"kind": "line", "n": index,
                  "unit": "paragraph" if kind == "paragraph" else "line"}
        if swap:
            target["from"] = swap.get("from")
            target["to"] = swap.get("to")
    elif swap:
        # No line was named, so swap the first match anywhere in the document.
        target = {"kind": "anchor", "from": swap.get("from"),
                  "to": swap.get("to"), "complete": True}
    else:
        return None, (f'the AI pointed "{raw_action}" at '
                      f'"{kind or "nothing"}", which the demo cannot locate')

    if action == "insert-after":
        note = "The demo inserts a blank line here; it does not type the text in."

    entry = {
        "index": 0, "chunk_text": command, "join_before": None,
        "action": action,
        # No confidence gate on this path: the gate belongs to DocEditBERT,
        # which reports one. An LLM does not, and a missing number must not
        # be read as a low one.
        "confidence": None, "low_confidence": False,
        "method": "llm", "target": target,
        "model_from": target.get("from"), "model_to": target.get("to"),
        "shared_target": False,
        "label": _label(action, target),
    }
    return entry, note


def _label(action, target):
    if target["kind"] == "anchor":
        return f'{action} "{target["from"]}" → "{target["to"]}"'
    unit = target.get("unit", "line")
    where = f"last {unit}" if target["n"] == -1 else f"{unit} {target['n']}"
    if target.get("from") and target.get("to"):
        return f'{action} "{target["from"]}" → "{target["to"]}" on {where}'
    return f"{action} {where}"


# ── Running one command ──────────────────────────────────────────────────────

def run_one(command, model, technique):
    """One command, one model, one technique. Never raises."""
    if model not in MODELS:
        return {"error": f"unknown model '{model}'"}
    if technique not in ALL_TECHNIQUES:
        return {"error": f"unknown technique '{technique}'"}

    if technique == SELF_CONSISTENCY:
        prompt = build("p5_cot", command)
        output, latency = self_consistency(model, command)
    else:
        prompt = build(technique, command)
        output, latency = call_model(model, prompt, temperature=0)

    parsed = parse_json(output)
    # A model is free to reply with any shape at all. Whatever it sends, this
    # endpoint answers; it never fails the request.
    try:
        entry, note = to_vdoc_action(parsed, command)
    except Exception as exc:                  # noqa: BLE001 - reported, not raised
        entry, note = None, f"could not read the AI reply ({type(exc).__name__})"
    return {
        "command": command,
        "model": model, "model_label": MODEL_LABELS.get(model, model),
        "technique": technique,
        "technique_label": TECHNIQUE_LABELS.get(technique, technique),
        "prompt_sent": prompt,
        "raw_output": output,
        "parsed_json": parsed,
        "valid_json": parsed is not None,
        "latency_s": round(latency, 3),
        "action": entry, "note": note,
        "ok": _looks_complete(parsed),
    }


def _looks_complete(parsed):
    """
    A green tick: valid JSON, an action VDOC knows, and the slots that action
    needs actually filled in. This is not accuracy -- nothing here knows the
    right answer -- it only says the reply is usable.
    """
    if not isinstance(parsed, dict):
        return False
    action = parsed.get("action")
    if action == "clarify":
        return bool(parsed.get("question"))
    if action in UNSUPPORTED:
        # Known to the schema, just not runnable on this demo document. The
        # tick judges the reply, not what VDOC can do with it afterwards.
        return action != "rewrite" or bool(parsed.get("value"))
    if action not in ACTION_MAP:
        return False
    try:
        entry, _ = to_vdoc_action(parsed, "")
    except Exception:                         # noqa: BLE001 - a bad shape is a cross
        return False
    return entry is not None


def compare(command, model):
    """The same command under three prompts, for the side-by-side cards."""
    return [run_one(command, model, t) for t in COMPARE_TECHNIQUES]


# ── The saved results ────────────────────────────────────────────────────────

def _table(frame):
    """A pandas table as {row: {column: value}}, with NaN as None."""
    import pandas as pd
    return {
        str(row): {str(col): (None if pd.isna(v) else round(float(v), 2))
                   for col, v in frame.loc[row].items()}
        for row in frame.index
    }


def results():
    """Every number comes from runs.csv; nothing here is written down."""
    test_commands = len(load_commands("test"))
    try:
        df = pe_report.load()
    except FileNotFoundError as err:
        return {"ready": False, "reason": str(err), "test_commands": test_commands}

    rows = {int(r["id"]): r for r in load_commands("test")}
    accuracy = pe_report.accuracy_table(df)
    latency = pe_report.latency_table(df)

    mistakes = []
    for _, row in df[~df["exact_ok"]].head(10).iterrows():
        source = rows.get(int(row["id"]), {})
        mistakes.append({
            "command": source.get("command", f"#{row['id']}"),
            "expected": source.get("expected_json"),
            "answer": str(row["output"])[:400],
            "model": MODEL_LABELS.get(row["model"], row["model"]),
            "technique": TECHNIQUE_LABELS.get(row["technique"], row["technique"]),
        })

    every = sorted(df["technique"].unique())
    missing = {m: [t for t in every
                   if t not in set(df[df["model"] == m]["technique"])]
               for m in sorted(df["model"].unique())}

    return {
        "ready": True,
        "test_commands": test_commands,
        "rows": int(len(df)),
        "models": [{"name": m, "label": MODEL_LABELS.get(m, m)}
                   for m in sorted(df["model"].unique())],
        "techniques": [{"name": t, "label": TECHNIQUE_LABELS.get(t, t)}
                       for t in every],
        "accuracy": _table(accuracy),
        "latency": _table(latency),
        "category": _table(pe_report.category_table(df)),
        "tuning": {
            "versions": [{"name": v, "label": TECHNIQUE_LABELS[v],
                          "note": TUNING_NOTES[v]} for v in TUNING_VERSIONS],
            "accuracy": {str(m): [
                (None if v not in accuracy.columns or _nan(accuracy.loc[m, v])
                 else round(float(accuracy.loc[m, v]), 1))
                for v in TUNING_VERSIONS]
                for m in accuracy.index},
        },
        "mistakes": mistakes,
        "missing": missing,
        "voice_limit_s": pe_report.VOICE_LIMIT_S,
        "findings": _findings(accuracy, latency),
    }


def _nan(value):
    import pandas as pd
    return bool(pd.isna(value))


def _findings(accuracy, latency):
    """Three sentences, every number read off the tables."""
    notes = []
    zero, few = "p1_zero_shot", "p4_few_shot"
    for model in accuracy.index:
        row = accuracy.loc[model]
        if zero in row.index and few in row.index and not (_nan(row[zero]) or _nan(row[few])):
            notes.append(f"{MODEL_LABELS.get(model, model)}: zero-shot "
                         f"{row[zero]:.1f}% rose to {row[few]:.1f}% with few-shot "
                         f"({row[few] - row[zero]:+.1f} points).")
            break

    if accuracy.notna().any().any():
        best_model = accuracy.max(axis=1).idxmax()
        best_technique = accuracy.loc[best_model].idxmax()
        seconds = (latency.loc[best_model, best_technique]
                   if best_technique in latency.columns else None)
        tail = "" if seconds is None or _nan(seconds) else f", about {seconds:.2f} s per command"
        notes.append(f"Best so far: {TECHNIQUE_LABELS.get(best_technique, best_technique)} "
                     f"on {MODEL_LABELS.get(best_model, best_model)}, "
                     f"{accuracy.loc[best_model, best_technique]:.1f}%{tail}.")

    if few in accuracy.columns and "p5_cot" in accuracy.columns:
        drop = (accuracy[few] - accuracy["p5_cot"]).dropna()
        if not drop.empty and drop.max() > 0:
            notes.append(f"Chain of thought hurt most on "
                         f"{MODEL_LABELS.get(drop.idxmax(), drop.idxmax())}: "
                         f"{drop.max():.1f} points below few-shot.")
        elif not drop.empty:
            notes.append("Chain of thought did not lower accuracy on any model.")
    return notes


def meta():
    """What the page needs to draw its dropdowns."""
    return {
        "models": [{"name": m, "label": MODEL_LABELS.get(m, m)} for m in MODELS],
        "techniques": [{"name": t, "label": TECHNIQUE_LABELS.get(t, t)}
                       for t in ALL_TECHNIQUES],
        "default_model": "groq",
        "default_technique": "p4_few_shot_v3",
        "test_commands": len(load_commands("test")),
    }
