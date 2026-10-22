"""
Groq LLM parser — zero-shot command parsing via llama3-8b-8192.
Lazy-loads the Groq client on first call.
"""
import os
import json
import re

_client = None

VALID_ACTIONS = {
    "bold", "italic", "delete", "underline", "heading2", "heading3",
    "replace", "insert-after", "font-increase", "highlight",
}

_SYSTEM = (
    "You are a document editing command parser. "
    "Return ONLY valid JSON, no explanation."
)
_USER_TMPL = (
    'Command: {cmd}\n'
    'Return JSON with exactly these keys: {{"action": "...", "from_anchor": "...", "to_anchor": "..."}}.\n'
    'action must be one of: bold, italic, delete, underline, heading2, heading3, '
    'replace, insert-after, font-increase, highlight.\n'
    'from_anchor and to_anchor are the exact words from the command.'
)


def _get_client():
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set — add it to .env or the environment")
        _client = Groq(api_key=api_key)
    return _client


def predict(command: str) -> dict:
    """
    Zero-shot parse via Groq llama3-8b-8192.
    Returns dict: {action, from, to, confidence: None, method: "groq"}
    On error returns the same shape with all None values plus an "error" key.
    """
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user",   "content": _USER_TMPL.format(cmd=command)},
            ],
            temperature=0,
            max_tokens=120,
        )
        raw = response.choices[0].message.content.strip()

        # Pull the first {...} block out, even if model adds prose around it
        m = re.search(r'\{[^}]+\}', raw, re.DOTALL)
        if not m:
            return _err("no JSON block in response")

        parsed = json.loads(m.group())

        action  = (parsed.get("action")       or "").lower().strip()
        from_a  = (parsed.get("from_anchor")  or parsed.get("from") or "").strip()
        to_a    = (parsed.get("to_anchor")    or parsed.get("to")   or "").strip()

        # Strip trailing punctuation (same defence as the editor)
        from_a = re.sub(r'[.,?!;:]+$', '', from_a)
        to_a   = re.sub(r'[.,?!;:]+$', '', to_a)

        if action not in VALID_ACTIONS:
            action = None

        return {
            "action":     action or None,
            "from":       from_a or None,
            "to":         to_a   or None,
            "confidence": None,
            "method":     "groq",
        }

    except json.JSONDecodeError as e:
        return _err(f"JSON parse error: {e}")
    except Exception as e:
        return _err(str(e))


def _err(msg: str) -> dict:
    return {"action": None, "from": None, "to": None,
            "confidence": None, "method": "groq", "error": msg}
