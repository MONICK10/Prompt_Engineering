"""
VDOC Studio: speak a command, get a Word and a PDF document.

Two prompts, chained. The first turns one spoken sentence into a JSON array of
actions. The second is called only when an action needs prose written -- a new
report, or a paragraph with no text -- so a formatting command never pays for
a content call.

One document per server, with an undo stack, which is right for a single-user
demo and wrong for anything shared. Say so before putting it on a network.
"""
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PE_ROOT = os.path.join(_ROOT, "vdoc-prompt")
if PE_ROOT not in sys.path:
    sys.path.insert(0, PE_ROOT)

from src.evaluate import parse_json                       # noqa: E402
from src.models import call_model                         # noqa: E402
from src.prompts import build                             # noqa: E402
from src.studio_tools import apply_actions, new_doc, to_docx, to_pdf   # noqa: E402

ACTIONS_PROMPT = "studio_actions_v1"
CONTENT_PROMPT = "studio_content_v1"
DEFAULT_MODEL = "groq"
MAX_UNDO = 20

EXPORT_DIR = os.path.join(_ROOT, "results", "studio")
os.makedirs(EXPORT_DIR, exist_ok=True)

_STATE = {"doc": new_doc(), "history": [], "log": ["1. new document"],
          "pending": None}

# How the user may pick an option after a clarify, without the LLM being asked
# again: a number, an ordinal in either language, or the name of the target.
ORDINALS = {"1": 1, "first": 1, "mudhal": 1, "one": 1, "yes": 1,
            "2": 2, "second": 2, "rendavathu": 2, "two": 2,
            "3": 3, "third": 3, "moonavathu": 3, "three": 3,
            "4": 4, "fourth": 4, "naalavathu": 4, "four": 4}
CANCEL = {"cancel", "vendam", "venam", "no", "illa", "stop"}
MAX_OPTIONS = 4


def _first_words(text, count=5):
    words = str(text or "").split()
    return " ".join(words[:count]) + ("…" if len(words) > count else "")


def build_options(doc, pending):
    """
    Turn "which one?" into two to four buttons, read off the document itself.

    The model cannot list these: it never sees the document. So it says what
    it would do (`pending`) and this fills in the targets that exist.
    """
    if not isinstance(pending, dict) or not pending.get("action"):
        return []
    options = []

    def add(label, target):
        options.append({"label": label,
                        "actions": [{**pending, "target": target}]})

    title = (doc.get("title") or {}).get("text")
    if title:
        add(f"the title — “{_first_words(title)}”", {"type": "title", "index": None})
    for i, section in enumerate(doc.get("sections", []), start=1):
        if len(options) >= MAX_OPTIONS:
            break
        heading = (section.get("heading") or {}).get("text")
        if heading:
            add(f"heading {i} — “{_first_words(heading)}”",
                {"type": "heading", "index": i})
    for i, paragraph in enumerate(
            [p for s in doc.get("sections", []) for p in s.get("paragraphs", [])],
            start=1):
        if len(options) >= MAX_OPTIONS:
            break
        add(f"paragraph {i} — “{_first_words(paragraph.get('text'))}”",
            {"type": "paragraph", "index": i})
    return options[:MAX_OPTIONS]


def _chosen_option(text, options):
    """Match a short reply against the pending options. None if it is not one."""
    cleaned = str(text or "").strip().lower().strip(".!?")
    if not cleaned or not options:
        return None
    if cleaned in CANCEL:
        return "cancel"
    for word in cleaned.replace("-", " ").split():
        if word in ORDINALS:
            number = ORDINALS[word]
            return options[number - 1] if 1 <= number <= len(options) else None
    # "title", "heading 2", "second one" also name an option by its label.
    for option in options:
        head = option["label"].split("—")[0].strip().lower()
        if head and head in cleaned:
            return option
    return None


def _parse_array(text):
    """
    The actions prompt asks for a JSON array. Models sometimes send one object,
    or wrap the array in prose. Both are recoverable; anything else is not.
    """
    if not text:
        return None
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        try:
            found = json.loads(text[start:end + 1])
            if isinstance(found, list):
                return found
        except json.JSONDecodeError:
            pass
    single = parse_json(text)
    return [single] if isinstance(single, dict) else None


def _write_content(topic, model=DEFAULT_MODEL):
    """Prompt 2: ask for the prose of a short report on one topic."""
    prompt = build(CONTENT_PROMPT, topic).replace("{topic}", topic)
    text, _ = call_model(model, prompt, temperature=0)
    found = parse_json(text)
    return found if isinstance(found, dict) else None


def apply_option(option):
    """Run the actions behind a clarify button. No LLM call is needed."""
    doc, messages = _apply(option["actions"])
    _STATE["pending"] = None
    return {"heard": option["label"], "actions": option["actions"],
            "messages": messages, "valid_json": True, "raw_output": "",
            "doc": doc, "log": _STATE["log"], "options": [], "latency_s": 0.0}


def run_command(text, model=DEFAULT_MODEL, technique=ACTIONS_PROMPT):
    """One spoken sentence in; the new document and what was done out."""
    started = time.time()

    # A reply to a question we just asked is answered here, not by the model:
    # "rendavathu" on its own means nothing without the options beside it.
    pending = _STATE.get("pending")
    if pending and pending.get("options"):
        picked = _chosen_option(text, pending["options"])
        if picked == "cancel":
            _STATE["pending"] = None
            return {"heard": text, "actions": [], "messages": [],
                    "valid_json": True, "raw_output": "", "cancelled": True,
                    "doc": _STATE["doc"], "log": _STATE["log"], "options": [],
                    "latency_s": round(time.time() - started, 3)}
        if picked:
            return apply_option(picked)

    raw, latency = call_model(model, build(technique, text), temperature=0)
    actions = _parse_array(raw)

    if actions is None:
        return {"heard": text, "actions": [], "messages": [],
                "raw_output": raw, "valid_json": False,
                "error": "AI reply was not valid JSON",
                "doc": _STATE["doc"], "log": _STATE["log"],
                "latency_s": round(time.time() - started, 3)}

    asked = next((a for a in actions
                  if isinstance(a, dict) and a.get("action") == "clarify"), None)

    # undo is a session action, not a document one, so it is handled here.
    undo_count = sum(1 for a in actions
                     if isinstance(a, dict) and a.get("action") == "undo")
    for _ in range(undo_count):
        undo()

    doc, messages = _apply(actions, model)

    options = []
    if asked:
        options = build_options(doc, asked.get("pending"))
        _STATE["pending"] = {"question": asked.get("question"), "options": options}
    else:
        _STATE["pending"] = None

    exports = {}
    for action in actions:
        if isinstance(action, dict) and action.get("action") == "export":
            exports = write_files(action.get("format") or "both")

    return {"heard": text, "actions": actions, "messages": messages,
            "raw_output": raw, "valid_json": True,
            "question": asked.get("question") if asked else None,
            "options": options,
            "doc": _STATE["doc"], "log": _STATE["log"], "exports": exports,
            "latency_s": round(latency, 3)}


def _apply(actions, model=DEFAULT_MODEL):
    """Run actions against the session document, keeping the undo stack."""
    doc, messages = apply_actions(_STATE["doc"], actions,
                                  content_fn=lambda t: _write_content(t, model))
    if doc != _STATE["doc"]:
        _STATE["history"].append(_STATE["doc"])
        del _STATE["history"][:-MAX_UNDO]
        _STATE["doc"] = doc
    for message in messages:
        if message.get("ok"):
            _STATE["log"].append(f"{len(_STATE['log']) + 1}. {message['text']}")
    return _STATE["doc"], messages


def write_files(which="both"):
    """Write the current document to disk and say where it landed."""
    out = {}
    if which in ("docx", "both"):
        out["docx"] = to_docx(_STATE["doc"], os.path.join(EXPORT_DIR, "vdoc_studio.docx"))
    if which in ("pdf", "both"):
        out["pdf"] = to_pdf(_STATE["doc"], os.path.join(EXPORT_DIR, "vdoc_studio.pdf"))
    return out


def download_path(fmt):
    """Always rebuild before serving, so the file matches what is on screen."""
    fmt = (fmt or "docx").lower()
    if fmt not in ("docx", "pdf"):
        return None
    return write_files(fmt)[fmt]


def undo():
    if not _STATE["history"]:
        return False
    _STATE["doc"] = _STATE["history"].pop()
    _STATE["log"].append(f"{len(_STATE['log']) + 1}. undo")
    return True


def reset():
    _STATE.update({"doc": new_doc(), "history": [], "log": ["1. new document"],
                   "pending": None})
    return _STATE["doc"]


def pick(number):
    """Apply the nth pending option, counting from 1."""
    pending = _STATE.get("pending") or {}
    options = pending.get("options") or []
    if not 1 <= int(number) <= len(options):
        return None
    return apply_option(options[int(number) - 1])


def state():
    pending = _STATE.get("pending") or {}
    return {"doc": _STATE["doc"], "log": _STATE["log"],
            "can_undo": bool(_STATE["history"]),
            "question": pending.get("question"),
            "options": pending.get("options") or []}
