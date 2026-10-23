"""
VDOC 2.0 pipeline: one utterance in, a list of resolved actions out.

    text -> splitter -> [per chunk: DocEditBERT + line resolver]
         -> sharer -> position tracker -> executor

DocEditBERT still decides every action. The new parts only decide *what the
action points at*, and only for line targets -- anchor targets take exactly
the path VDOC 1.0 took.

Resolver order is line-first, and that is deliberate: on a line command the
model sometimes emits a junk anchor fragment ("second line bold pannu" gives
from="second"), so trusting the model's slots first would produce a target
that does not exist in the document. The line resolver is conservative enough
that it stays silent on real anchor commands.
"""
from . import splitter, sharer
from . import line_resolver
from .position_tracker import PositionTracker, LineNotFound

# The dead Groq model (llama-3.1-8b-instant, retired) made the low-confidence
# path return a 404. VDOC 2.0 runs offline on DocEditBERT alone.
ENABLE_GROQ_FALLBACK = False
GROQ_FALLBACK_NOTE = "disabled (model retired)"

SPLITTER_VERSION = "v0: rule-based"

# Below this, the model is guessing. A guess is worse than a refusal here:
# acting on it edits the wrong thing and the user cannot tell why, so the
# action is flagged and the editor asks for a repeat instead.
CONFIDENCE_THRESHOLD = 0.75

# Actions that change how many lines the document has.
SHIFTING_ACTIONS = {"delete", "insert-after"}


def _clean_anchor(value):
    return (value or "").strip().strip(".,?!;:") or None


def _anchor_target(prediction):
    """
    Build an anchor target from DocEditBERT slots, VDOC 1.0 style.

    A half-filled anchor (one slot only) is still passed through, because
    that is exactly what VDOC 1.0 did -- the editor then reports "could not
    extract all fields" and applies nothing. Dropping it here would make the
    pipeline disagree with the old path on 2 DocEdit-1000 commands.
    """
    source = _clean_anchor(prediction.get("from"))
    dest = _clean_anchor(prediction.get("to"))
    if source or dest:
        return {"kind": "anchor", "from": source, "to": dest,
                "complete": bool(source and dest)}
    return None


def parse_multi(text, predict_fn, line_count=None):
    """
    Parse an utterance into an ordered list of actions.

    predict_fn   the existing DocEditBERT parser, e.g.
                 doceditbert_parser.predict -- injected so tests and the
                 regression run can supply their own.
    line_count   number of lines in the document. When given, line targets
                 are resolved against the tracker and re-mapped after each
                 shifting edit. When None, line targets stay unresolved.

    Returns a dict with the chunks, the actions, and the trace the Live Demo
    page renders.
    """
    chunks = splitter.split(text)

    actions = []
    for index, chunk in enumerate(chunks):
        prediction = predict_fn(chunk.text)
        line_target = line_resolver.resolve(chunk.text)
        target = line_target if line_target else _anchor_target(prediction)

        confidence = prediction.get("confidence")
        actions.append({
            "index": index,
            "chunk_text": chunk.text,
            "join_before": chunk.join_before,
            "action": prediction.get("action"),
            "confidence": confidence,
            "low_confidence": (confidence is not None
                               and confidence < CONFIDENCE_THRESHOLD),
            "method": prediction.get("method"),
            "target": target,
            # Kept so the demo can show what the model said about slots even
            # when the line resolver overrode it.
            "model_from": _clean_anchor(prediction.get("from")),
            "model_to": _clean_anchor(prediction.get("to")),
        })

    actions = sharer.share_targets(actions)

    tracker = None
    if line_count is not None:
        tracker = PositionTracker(line_count)
        for entry in actions:
            target = entry.get("target")
            if not target or target.get("kind") != "line":
                continue
            try:
                resolved = tracker.resolve(target["n"])
            except LineNotFound as exc:
                entry["error"] = str(exc)
                continue

            entry["resolved_line"] = resolved
            if entry["action"] in SHIFTING_ACTIONS:
                tracker.apply(entry["action"], resolved["line_id"])

    return {
        "text": text,
        "chunks": [c.to_dict() for c in chunks],
        "actions": actions,
        "count": len(actions),
        "split": len(chunks) > 1,
        "splitter": SPLITTER_VERSION,
        "groq_fallback": GROQ_FALLBACK_NOTE,
        "final_lines": tracker.snapshot() if tracker else None,
    }


def describe(entry):
    """One-line label for an action card on the Live Demo page."""
    action = entry.get("action") or "unknown"
    target = entry.get("target")

    if not target:
        return f"{action} (no target found)"

    if target["kind"] == "anchor":
        return f"{action} from {target['from']} to {target['to']}"

    resolved = entry.get("resolved_line")
    if not resolved:
        return f"{action} {line_resolver.describe(target)}"

    unit = target.get("unit", "line")
    label = f"{action} {unit} {resolved['current']}"
    if resolved["moved"]:
        label += f" (was {unit} {resolved['original']})"
    return label
