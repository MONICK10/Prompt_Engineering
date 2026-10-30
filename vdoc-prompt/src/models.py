"""One function to call any of the three models we compare."""

import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

# Load every .env from the project folder upward. load_dotenv never overwrites
# a value that is already set, so the closest file wins and the ones above it
# fill in whatever is still missing (e.g. keys kept in a parent folder).
PROJECT_DIR = Path(__file__).resolve().parent.parent
for _folder in [PROJECT_DIR, *PROJECT_DIR.parents]:
    load_dotenv(_folder / ".env")

# Change these in .env if you want different model versions.
# Not sure a name still works? Run:  python -m src.models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")


def _gemini(prompt, temperature):
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(GEMINI_MODEL)
    reply = model.generate_content(
        prompt, generation_config={"temperature": temperature}
    )
    return reply.text


# Groq's free tier caps tokens per minute, not only requests, and that is the
# limit the long prompts hit first: p4_few_shot_v3 costs about 1,070 tokens a
# call against a budget of 8,000 a minute, so a flat one-second pause runs
# roughly eight times over and everything after the first few calls comes back
# 429. A new API key does not help -- the budget belongs to the account.
#
# The server reports what is left after every call, so the pause is read off
# the reply instead of guessed, and it retunes itself when a prompt changes.
_GROQ_BUDGET = {"remaining": None, "reset_s": None, "cost": None}


def _duration(text):
    """Groq writes these as '585ms', '7.5s' or '1m26.4s'."""
    if not text:
        return None
    match = re.fullmatch(r"(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", text.strip())
    if match and any(match.groups()):
        return float(match.group(1) or 0) * 60 + float(match.group(2) or 0)
    match = re.fullmatch(r"(\d+(?:\.\d+)?)ms", text.strip())
    return float(match.group(1)) / 1000 if match else None


def _groq(prompt, temperature):
    from groq import Groq

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    raw = client.chat.completions.with_raw_response.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
    )
    headers = raw.headers
    reply = raw.parse()

    remaining = headers.get("x-ratelimit-remaining-tokens")
    _GROQ_BUDGET["remaining"] = int(remaining) if remaining else None
    _GROQ_BUDGET["reset_s"] = _duration(headers.get("x-ratelimit-reset-tokens"))
    usage = getattr(reply, "usage", None)
    if usage is not None:
        _GROQ_BUDGET["cost"] = usage.total_tokens
    return reply.choices[0].message.content


def _ollama(prompt, temperature):
    import ollama

    reply = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": temperature},
    )
    return reply["message"]["content"]


# model_name -> the function that talks to it
BACKENDS = {
    "gemini": _gemini,
    "groq": _groq,
    "ollama-local": _ollama,
}

MODELS = list(BACKENDS)

# Seconds to wait between calls. Gemini's free tier allows only about
# 5 requests a minute, so it needs a much longer pause than the others.
PAUSE_SECONDS = {"gemini": 13.0, "groq": 1.0, "ollama-local": 1.0}
DEFAULT_PAUSE = 1.0
# A token-per-minute limit refills within the minute, so a 429 here is worth
# sitting out a few times rather than giving up and saving an empty row.
MAX_RATE_LIMIT_RETRIES = 5
# A server that asks for longer than this is out of daily quota, not just
# busy for a moment, so there is no point sleeping through it.
MAX_RETRY_WAIT = 120.0

# Models that ran out of daily quota during this run. Once a model is in
# here, evaluate.run_all stops calling it instead of collecting more 429s.
EXHAUSTED = set()


# Leave this much of the token budget unspent before pausing, so a reply that
# runs longer than the last one does not tip us over.
TOKEN_SAFETY = 1.5
MAX_TOKEN_WAIT = 70.0


def _groq_pause():
    """
    Spend the minute's token budget, then wait for it to refill.

    Bursting and then waiting beats a flat pause: the cheap prompts
    (p1 at ~570 tokens) get through about twice as fast as the dear ones
    (v3 at ~1,070), without either of them going over.
    """
    budget = _GROQ_BUDGET
    if budget["remaining"] is None or budget["cost"] is None:
        return PAUSE_SECONDS["groq"]          # first call, nothing measured yet
    if budget["remaining"] > budget["cost"] * TOKEN_SAFETY:
        return PAUSE_SECONDS["groq"]          # room for another, carry on
    wait = budget["reset_s"]
    return min(wait, MAX_TOKEN_WAIT) if wait else MAX_TOKEN_WAIT


def pause_for(model_name):
    """How long to wait after a call to this model before the next one."""
    if model_name == "groq":
        return _groq_pause()
    return PAUSE_SECONDS.get(model_name, DEFAULT_PAUSE)


def is_rate_limit(text):
    """True if an error message (or a saved output) looks like a 429."""
    low = str(text).lower()
    return any(w in low for w in ("429", "quota", "rate limit", "resource_exhausted"))


def retry_seconds(err, fallback=25.0):
    """Read the wait the server asks for, e.g. 'retry in 7.5s' or
    'retry_delay { seconds: 7 }'. Falls back to a safe default."""
    text = str(err)
    for pattern in (r"retry in ([\d.]+)\s*s", r"seconds:\s*(\d+)",
                    r"retry[-_]after[\"']?\s*[:=]\s*([\d.]+)"):
        found = re.search(pattern, text, re.IGNORECASE)
        if found:
            return float(found.group(1)) + 1  # +1s of slack
    return fallback


def call_model(model_name, prompt, temperature=0):
    """Send one prompt to one model.

    Returns (text, latency_seconds). A 429 waits as long as the server asks
    and tries again (up to MAX_RATE_LIMIT_RETRIES times); any other error is
    retried once. After that it returns an empty string, so a bad call never
    stops the experiment. Waiting time is not counted in the latency.
    """
    if model_name not in BACKENDS:
        raise ValueError(f"unknown model '{model_name}', pick one of {MODELS}")

    backend = BACKENDS[model_name]
    rate_tries, plain_tries = 0, 0
    while True:
        start = time.time()  # time the call itself, not the waiting
        try:
            text = backend(prompt, temperature)
            return (text or "").strip(), time.time() - start
        except Exception as err:  # network, quota, bad key, model not pulled...
            if is_rate_limit(err):
                wait = retry_seconds(err)
                if wait > MAX_RETRY_WAIT:
                    if model_name not in EXHAUSTED:
                        print(f"  [{model_name}] out of quota - the server asks "
                              f"for {wait / 3600:.1f}h. Stopping this model; "
                              f"rerun later with --retry-failed.")
                    EXHAUSTED.add(model_name)
                    return "", time.time() - start
                if rate_tries >= MAX_RATE_LIMIT_RETRIES:
                    print(f"  [{model_name}] still rate limited after "
                          f"{MAX_RATE_LIMIT_RETRIES} retries, giving up")
                    return "", time.time() - start
                rate_tries += 1
                print(f"  [{model_name}] rate limited, waiting {wait:.0f}s "
                      f"(retry {rate_tries}/{MAX_RATE_LIMIT_RETRIES})")
                time.sleep(wait)
                continue
            plain_tries += 1
            print(f"  [{model_name}] attempt {plain_tries} failed: {err}")
            if plain_tries >= 2:
                return "", time.time() - start
            time.sleep(2)  # short pause, then one retry


if __name__ == "__main__":
    # Ping each backend so you can see which ones are ready before a full run.
    names = {"gemini": GEMINI_MODEL, "groq": GROQ_MODEL, "ollama-local": OLLAMA_MODEL}
    for backend_name in MODELS:
        reply, seconds = call_model(backend_name, "Reply with the single word: ok")
        status = "OK    " if reply else "FAILED"
        print(f"{backend_name:<13} {names[backend_name]:<22} {status} {seconds:5.2f}s")
