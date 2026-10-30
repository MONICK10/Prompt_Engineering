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

# TODO: "set font size to N" is not an action yet.
#
# "second line font size 20 vei" parses as font-increase, which is the
# closest of the ten intents DocEditBERT knows, and the size (20) is thrown
# away. Supporting it properly means a new intent plus a numeric slot, so it
# needs a dataset change and a retrain -- not a parser tweak. Until then the
# command does something reasonable rather than nothing, and the number is
# silently ignored.


# Tanglish case markers that glue onto an anchor word: "reported-a",
# "address-nu". The snapper splits these off on the voice path, but a typed
# command reaches us with the hyphen still attached.
CASE_MARKERS = {"a", "aa", "ah", "nu", "nnu", "ku", "kku", "ai", "ya"}

# Actions where a line is a *scope* rather than the thing being acted on.
# "third line reported-a car-nu maathu" edits a word inside line 3; the line
# only says where to look. Formatting actions are the other way round -- there
# the line itself is what changes -- so only replace is listed here.
SCOPED_ACTIONS = {"replace"}


def _clean_anchor(value):
    cleaned = (value or "").strip().strip(".,?!;:")
    # Drop a trailing case marker, but only when it really is one: "rule-based"
    # keeps its hyphen because "based" is not a marker.
    if "-" in cleaned:
        stem, _, suffix = cleaned.rpartition("-")
        if stem and suffix.lower() in CASE_MARKERS:
            cleaned = stem
    return cleaned or None


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
        anchor_target = _anchor_target(prediction)
        target = line_target if line_target else anchor_target

        # A scoped action needs both halves: without the anchor the executor
        # knows which line to edit but not which word, and the edit cannot run.
        if (line_target and anchor_target
                and prediction.get("action") in SCOPED_ACTIONS):
            target = {**line_target,
                      "from": anchor_target["from"],
                      "to": anchor_target["to"]}

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
    unit = target.get("unit", "line")
    where = (f"{unit} {resolved['current']}" if resolved
             else line_resolver.describe(target))

    # A scoped action names the words it swaps, not just the line.
    if action in SCOPED_ACTIONS and target.get("from") and target.get("to"):
        label = f"{action} \"{target['from']}\" with \"{target['to']}\" on {where}"
    else:
        label = f"{action} {where}"

    if resolved and resolved["moved"]:
        label += f" (was {unit} {resolved['original']})"
    return label
