"""
VDOC Studio: the document model and the actions that change it.

One document per session. Every action returns a new document, so the caller
can keep the old one on a stack and undo by popping it. Nothing here calls an
LLM: `create_report` and a text-less `add_paragraph` take a `content_fn` that
the caller supplies, which keeps this module testable without a network.

Writing the document out as .docx and .pdf lives in studio_export.py and is
re-exported here, so callers only import this one name.
"""
from copy import deepcopy

from src.studio_export import to_docx, to_pdf   # noqa: F401  (public API)

ALIGNMENTS = {"left", "center", "right", "justify"}
BORDERS = {"none", "single", "double"}
RUN_KEYS = ("bold", "italic", "underline", "size", "font", "align",
            "highlight", "color")

# Word only has a fixed highlight palette, so the preview and the PDF stay
# inside it too -- a colour the .docx cannot carry would make the preview lie.
COLORS = {
    "yellow": "#ffff00", "green": "#00ff00", "cyan": "#00ffff",
    "magenta": "#ff00ff", "blue": "#0000ff", "red": "#ff0000",
    "darkblue": "#000080", "darkred": "#800000", "darkgreen": "#008000",
    "gray": "#808080", "grey": "#808080", "black": "#000000",
    "white": "#ffffff", "orange": "#ffa500", "pink": "#ffc0cb",
}


def normalise_color(value, default="yellow"):
    """true -> the default, a known name or #rrggbb -> itself, else None."""
    if value is True:
        return default
    if value in (False, None, ""):
        return None
    text = str(value).strip().lower()
    if text in COLORS:
        return text
    if text.startswith("#") and len(text) in (4, 7):
        return text
    return default


def new_run(text="", **over):
    run = {"text": text, "bold": False, "italic": False, "underline": False,
           "size": None, "font": None, "align": "left",
           "highlight": None, "color": None}
    run.update({k: v for k, v in over.items() if k in run})
    return run


def new_doc():
    return {
        "title": new_run("Untitled document", size=16, bold=True, align="center"),
        "sections": [],
        "page": {"border": "none", "font": "Times New Roman", "size": 12,
                 "margins_cm": 2.5, "line_spacing": 1.5},
    }


def _index(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _sections(doc):
    return doc.get("sections") or []


def _all_paragraphs(doc):
    """Every paragraph in reading order, so "paragraph 3" means the third one."""
    return [p for s in _sections(doc) for p in s.get("paragraphs", [])]


def _resolve(doc, target):
    """The runs an action points at. Unknown targets resolve to nothing."""
    target = target if isinstance(target, dict) else {}
    kind = str(target.get("type") or "").lower()
    index = _index(target.get("index"))
    sections = _sections(doc)

    if kind == "title":
        return [doc["title"]]
    if kind == "document":
        return [doc["title"]] + [s["heading"] for s in sections] + _all_paragraphs(doc)
    if kind == "heading":
        picked = _pick(["" for _ in sections] and sections, index)
        return [s["heading"] for s in picked]
    if kind == "section":
        runs = []
        for s in _pick(sections, index):
            runs.append(s["heading"])
            runs.extend(s.get("paragraphs", []))
        return runs
    if kind == "paragraph":
        return _pick(_all_paragraphs(doc), index)
    return []


def _pick(items, index):
    """1-based index, -1 for the last one, None for all of them."""
    if not items:
        return []
    if index is None:
        return list(items)
    if index == -1:
        return [items[-1]]
    return [items[index - 1]] if 1 <= index <= len(items) else []


def _describe(target):
    target = target if isinstance(target, dict) else {}
    kind = target.get("type") or "document"
    index = _index(target.get("index"))
    if index is None:
        return kind
    return f"last {kind}" if index == -1 else f"{kind} {index}"


# ── The actions ──────────────────────────────────────────────────────────────

def apply_actions(doc, actions, content_fn=None):
    """
    Run a list of actions against a document.

    Returns (new_doc, messages). A message is one short line per action, for
    the chips above the preview. An action that cannot run says so rather than
    failing silently, and never raises.
    """
    doc = deepcopy(doc)
    messages = []
    for raw in actions or []:
        if not isinstance(raw, dict):
            messages.append({"ok": False, "text": "ignored an action that was not an object"})
            continue
        name = str(raw.get("action") or "").lower()
        try:
            doc, message = _one(doc, name, raw, content_fn)
        except Exception as exc:                 # noqa: BLE001 - shown, not raised
            message = {"ok": False, "text": f"{name or 'action'} failed ({type(exc).__name__})"}
        messages.append(message)
    return doc, messages


def _one(doc, name, raw, content_fn):
    if name == "create_report":
        topic = (raw.get("topic") or "").strip() or "a general topic"
        if content_fn is None:
            return doc, {"ok": False, "text": "no content writer available"}
        content = content_fn(topic)
        if not isinstance(content, dict) or not content.get("sections"):
            return doc, {"ok": False, "text": f"could not write a report on {topic}"}
        doc["title"] = new_run(content.get("title") or topic.title(),
                               size=16, bold=True, align="center")
        doc["sections"] = [
            {"heading": new_run(s.get("heading", ""), bold=True, size=13),
             "paragraphs": [new_run(t, align="justify")
                            for t in s.get("paragraphs", []) if str(t).strip()]}
            for s in content["sections"]
        ]
        # The topic is quoted so a mis-transcribed one is obvious on screen:
        # “Karunya College” and “caroniya college bathi aur a” look nothing
        # alike, and the chip is where that gets caught before the download.
        return doc, {"ok": True,
                     "text": f"report on “{topic}” ({len(doc['sections'])} sections)"}

    if name == "add_section":
        heading = (raw.get("heading") or "New section").strip()
        doc["sections"].append({"heading": new_run(heading, bold=True, size=13),
                                "paragraphs": []})
        return doc, {"ok": True, "text": f"added section “{heading}”"}

    if name == "add_paragraph":
        text = raw.get("text")
        if not text and content_fn is not None:
            written = content_fn(raw.get("topic") or "this section")
            paragraphs = (written or {}).get("sections", [{}])[0].get("paragraphs", [])
            text = paragraphs[0] if paragraphs else ""
        if not text:
            return doc, {"ok": False, "text": "no text to add"}
        sections = _sections(doc)
        if not sections:
            doc["sections"].append({"heading": new_run("Section", bold=True, size=13),
                                    "paragraphs": []})
            sections = doc["sections"]
        chosen = _pick(sections, _index(raw.get("section_index"), -1))
        target = chosen[0] if chosen else sections[-1]
        target["paragraphs"].append(new_run(str(text), align="justify"))
        return doc, {"ok": True, "text": "added a paragraph"}

    if name == "format":
        runs = _resolve(doc, raw.get("target"))
        if not runs:
            return doc, {"ok": False, "text": f"nothing found at {_describe(raw.get('target'))}"}
        changed = []
        for key in RUN_KEYS:
            value = raw.get(key)
            if value is None:
                continue
            if key == "align" and str(value).lower() not in ALIGNMENTS:
                continue
            if key == "size":
                value = _index(value)
                if not value:
                    continue
            if key in ("highlight", "color"):
                value = normalise_color(value, "yellow" if key == "highlight" else "red")
                if value is None:
                    for run in runs:
                        run[key] = None
                    changed.append(f"no {key}")
                    continue
            for run in runs:
                run[key] = str(value).lower() if key == "align" else value
            changed.append(f"{key} {value}"
                           if key in ("size", "font", "align", "highlight", "color")
                           else key)
        if not changed:
            return doc, {"ok": False, "text": "no formatting given"}
        return doc, {"ok": True, "text": f"{', '.join(changed)} on {_describe(raw.get('target'))}"}

    if name == "page_border":
        style = str(raw.get("style") or "single").lower()
        style = style if style in BORDERS else "single"
        doc["page"]["border"] = style
        return doc, {"ok": True, "text": f"page border: {style}"}

    if name == "set_font":
        font, size = raw.get("font"), _index(raw.get("size"))
        if font:
            doc["page"]["font"] = str(font)
        if size:
            doc["page"]["size"] = size
        for run in _resolve(doc, {"type": "document"}):
            if font:
                run["font"] = str(font)
        if not font and not size:
            return doc, {"ok": False, "text": "no font or size given"}
        return doc, {"ok": True, "text": f"document font {font or doc['page']['font']}"
                                         f" {size or doc['page']['size']}"}

    if name == "line_spacing":
        try:
            doc["page"]["line_spacing"] = float(raw.get("value"))
        except (TypeError, ValueError):
            return doc, {"ok": False, "text": "line spacing needs a number"}
        return doc, {"ok": True, "text": f"line spacing {doc['page']['line_spacing']}"}

    if name == "delete":
        return _delete(doc, raw.get("target"))

    if name == "replace":
        source, dest = raw.get("from"), raw.get("to")
        if not source or dest is None:
            return doc, {"ok": False, "text": "replace needs a from and a to"}
        runs = _resolve(doc, raw.get("target")) or _resolve(doc, {"type": "document"})
        hits = 0
        for run in runs:
            if source.lower() in run["text"].lower():
                run["text"] = _swap(run["text"], source, str(dest))
                hits += 1
        if not hits:
            return doc, {"ok": False, "text": f"“{source}” not found"}
        return doc, {"ok": True, "text": f"replaced “{source}” with “{dest}” ({hits})"}

    if name == "move":
        sections = _sections(doc)
        chosen = _pick(sections, _index((raw.get("target") or {}).get("index")))
        to_index = _index(raw.get("to_index"))
        if not chosen or to_index is None:
            return doc, {"ok": False, "text": "move needs a section and a position"}
        section = chosen[0]
        sections.remove(section)
        sections.insert(max(0, min(len(sections), to_index - 1)), section)
        return doc, {"ok": True, "text": f"moved a section to {to_index}"}

    if name == "clear":
        return new_doc(), {"ok": True, "text": "cleared the document"}

    # undo, export and clarify are decided by the session, not the document.
    if name in ("undo", "export", "clarify"):
        return doc, {"ok": True, "text": {"undo": "undo", "export": "export",
                                          "clarify": "asked a question"}[name]}

    return doc, {"ok": False, "text": f"unknown action “{name}”"}


def _swap(text, source, dest):
    lowered, needle = text.lower(), source.lower()
    out, at = [], 0
    while True:
        found = lowered.find(needle, at)
        if found == -1:
            out.append(text[at:])
            return "".join(out)
        out.append(text[at:found])
        out.append(dest)
        at = found + len(needle)


def _delete(doc, target):
    target = target if isinstance(target, dict) else {}
    kind = str(target.get("type") or "").lower()
    index = _index(target.get("index"))
    sections = _sections(doc)

    if kind == "section" or kind == "heading":
        chosen = _pick(sections, index)
        if not chosen:
            return doc, {"ok": False, "text": f"no {_describe(target)}"}
        for section in chosen:
            sections.remove(section)
        return doc, {"ok": True, "text": f"deleted {_describe(target)}"}
    if kind == "paragraph":
        chosen = _pick(_all_paragraphs(doc), index)
        if not chosen:
            return doc, {"ok": False, "text": f"no {_describe(target)}"}
        for section in sections:
            section["paragraphs"] = [p for p in section["paragraphs"] if p not in chosen]
        return doc, {"ok": True, "text": f"deleted {_describe(target)}"}
    if kind == "title":
        doc["title"]["text"] = ""
        return doc, {"ok": True, "text": "cleared the title"}
    if kind == "document":
        return new_doc(), {"ok": True, "text": "cleared the document"}
    return doc, {"ok": False, "text": f"cannot delete {_describe(target)}"}
