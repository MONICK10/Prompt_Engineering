"""
VDOC Flask backend.

Endpoints
---------
POST /transcribe     audio file  -> {"text": "..."}
POST /parse          command str -> DocEditBERT result (one action)
POST /parse-multi    command str -> VDOC 2.0 list of actions
POST /parse-rule     command str -> rule-based result
POST /parse-bilstm   command str -> BiLSTM result
POST /parse-all      command str -> all three results combined
POST /batch-evaluate commands[]  -> per-system accuracy over a labelled batch
GET  /multi-action/config -> splitter version, fallback status
"""
import os
import sys
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS

# Load .env from project root so GROQ_API_KEY is available
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

# Allow imports from backend/ itself regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import doceditbert_parser              # DocEditBERT
import bilstm_parser                  # BiLSTM
import groq_parser                    # Groq LLM zero-shot
import xlmr_parser                    # XLM-RoBERTa zero-shot baseline
import mdistilbert_parser              # mDistilBERT Tanglish baseline
from rule_parser import parse as rule_parse  # regex rules
import threading

# VDOC 2.0 multi-action pipeline (sits on top of DocEditBERT, does not
# replace it). Project root is already on sys.path via the parent of backend/.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from multi_action import pipeline as multi_pipeline
from multi_action import corrector
from multi_action import snapper

app = Flask(__name__)
CORS(app)   # allow all origins (frontend runs on a different port)

# ── Pre-warm local models in background so first batch request is fast ────────
def _warmup():
    try:
        doceditbert_parser.predict("warmup")
        bilstm_parser.predict("warmup")
        print("[warmup] DocEditBERT + BiLSTM models ready", flush=True)
    except Exception as e:
        print(f"[warmup] {e}", flush=True)

threading.Thread(target=_warmup, daemon=True).start()

# ── Whisper (lazy-loaded on first /transcribe call) ───────────────────────────
import whisper_asr


# ── Helpers ───────────────────────────────────────────────────────────────────
def _cmd_from_request():
    data = request.get_json(silent=True) or {}
    return (data.get("command") or "").strip()


def _rule_result(command):
    result = rule_parse(command)
    if result is None:
        return {"action": None, "from": None, "to": None,
                "confidence": None, "method": "rule"}
    return result


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/transcribe", methods=["POST"])
def transcribe():
    """
    Accepts a multipart audio file (wav/webm/mp3).
    Runs Whisper locally and returns the transcription.
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file in request (field name: 'audio')"}), 400

    audio_file = request.files["audio"]
    suffix     = os.path.splitext(audio_file.filename)[-1] or ".wav"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name)
        tmp_path = tmp.name

    try:
        # whisper_asr forces English decoding (so code-switched Tanglish comes
        # back romanised, matching DocEditBERT's training script), prepends
        # 300 ms of silence so the first word is not clipped, conditions on
        # our vocabulary, and writes debug/last_audio.wav.
        result = whisper_asr.transcribe_file(tmp_path)
        return jsonify({
            "text":        result["text"],
            "diagnostics": result["diagnostics"],
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)


BERT_CONFIDENCE_THRESHOLD = 0.75

# The confidence router's Groq fallback is switched off: the model it called,
# llama-3.1-8b-instant, has been retired and now returns HTTP 404. With this
# False the whole app runs offline on DocEditBERT alone. The routing code is
# left intact so a working model can be dropped in later.
ENABLE_GROQ_FALLBACK = multi_pipeline.ENABLE_GROQ_FALLBACK
GROQ_FALLBACK_NOTE   = multi_pipeline.GROQ_FALLBACK_NOTE


@app.route("/parse", methods=["POST"])
def parse_doceditbert():
    """
    DocEditBERT inference with confidence-based Groq fallback.
    If DocEditBERT confidence < 0.75, Groq LLM is called instead and the
    response carries method='llm-fallback' and routed_to='groq'.
    All responses include a 'routed_to' field.

    With ENABLE_GROQ_FALLBACK off, low-confidence commands stay on
    DocEditBERT and carry fallback='disabled (model retired)'.
    """
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400

    result     = doceditbert_parser.predict(command)
    confidence = result.get("confidence") or 0.0

    if confidence >= BERT_CONFIDENCE_THRESHOLD:
        result["routed_to"] = "doceditbert"
        return jsonify(result)

    if not ENABLE_GROQ_FALLBACK:
        result["routed_to"] = "doceditbert"
        result["fallback"]  = GROQ_FALLBACK_NOTE
        return jsonify(result)

    # Low confidence — fall back to Groq LLM
    try:
        fallback                   = groq_parser.predict(command)
        fallback["routed_to"]      = "groq"
        fallback["method"]         = "llm-fallback"
        fallback["bert_confidence"] = round(confidence, 4)
        return jsonify(fallback)
    except Exception as e:
        # Groq unavailable — return BERT result with a note
        result["routed_to"]     = "doceditbert"
        result["fallback_error"] = str(e)
        return jsonify(result)


@app.route("/parse-multi", methods=["POST"])
def parse_multi():
    """
    VDOC 2.0: split one utterance into a list of actions.

    Request:  {"command": "...", "line_count": 5}
              line_count is optional; without it, line targets are found but
              not resolved to a position.

    Response: {text, chunks[], actions[], count, split, splitter,
               groq_fallback, final_lines}

    A single-action command returns a list of one, and that one action is
    identical to what POST /parse returns.
    """
    data    = request.get_json(silent=True) or {}
    command = (data.get("command") or "").strip()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400

    line_count = data.get("line_count")
    if line_count is not None:
        try:
            line_count = int(line_count)
        except (TypeError, ValueError):
            return jsonify({"error": "'line_count' must be a whole number"}), 400
        if line_count < 0:
            return jsonify({"error": "'line_count' must not be negative"}), 400

    result = multi_pipeline.parse_multi(
        command, doceditbert_parser.predict, line_count=line_count
    )
    # Pre-render the label each action card shows.
    for entry in result["actions"]:
        entry["label"] = multi_pipeline.describe(entry)
    return jsonify(result)


@app.route("/multi-action/config", methods=["GET"])
def multi_action_config():
    """What the dashboard needs to label the pipeline honestly."""
    return jsonify({
        "splitter":            multi_pipeline.SPLITTER_VERSION,
        "line_resolver":       "rule-based",
        "corrector":           "rule-based (edit distance 1-2)",
        "snapper":             ("rule-based (Double Metaphone + edit distance)"
                                if snapper.ENABLE_SNAPPING else "disabled"),
        "snapping_enabled":    snapper.ENABLE_SNAPPING,
        "groq_fallback":       GROQ_FALLBACK_NOTE,
        "groq_fallback_on":    ENABLE_GROQ_FALLBACK,
        "confidence_threshold": BERT_CONFIDENCE_THRESHOLD,
        "model":               "DocEditBERT (models/doceditbert_1000)",
        "whisper":             whisper_asr.config(),
    })


# ── Prompt engineering (the vdoc-prompt project, served inside VDOC) ─────────
#
# Imported lazily: vdoc-prompt pulls in pandas and the LLM clients, and a
# missing key or a missing package there must not stop the rest of VDOC from
# starting. If the import fails, these four routes report why and the other
# pages carry on working.
try:
    import pe_api
    PE_IMPORT_ERROR = None
except Exception as exc:                      # noqa: BLE001 - reported, not raised
    pe_api, PE_IMPORT_ERROR = None, f"{type(exc).__name__}: {exc}"


def _pe_ready():
    if pe_api is None:
        return jsonify({"error": f"prompt-engineering module unavailable — "
                                 f"{PE_IMPORT_ERROR}"}), 503
    return None


@app.route("/pe/meta", methods=["GET"])
def pe_meta():
    """Model and technique lists for the page's dropdowns."""
    return _pe_ready() or jsonify(pe_api.meta())


@app.route("/pe/run", methods=["POST"])
def pe_run():
    """One command, one model, one technique -> prompt, reply, JSON, time."""
    not_ready = _pe_ready()
    if not_ready:
        return not_ready
    data = request.get_json(silent=True) or {}
    command = (data.get("command") or "").strip()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400
    result = pe_api.run_one(command,
                            data.get("model") or "groq",
                            data.get("technique") or "p4_few_shot_v3")
    return (jsonify(result), 400) if result.get("error") else jsonify(result)


@app.route("/pe/compare", methods=["POST"])
def pe_compare():
    """The same command under zero-shot, few-shot v1 and few-shot v3."""
    not_ready = _pe_ready()
    if not_ready:
        return not_ready
    data = request.get_json(silent=True) or {}
    command = (data.get("command") or "").strip()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400
    return jsonify({"command": command,
                    "results": pe_api.compare(command, data.get("model") or "groq")})


@app.route("/pe/results", methods=["GET"])
def pe_results():
    """Tables and charts data, all read from vdoc-prompt/results/runs.csv."""
    return _pe_ready() or jsonify(pe_api.results())


# ── VDOC Studio (speak a command, get Word and PDF) ──────────────────────────
try:
    import studio_api
    STUDIO_IMPORT_ERROR = None
except Exception as exc:                      # noqa: BLE001 - reported, not raised
    studio_api, STUDIO_IMPORT_ERROR = None, f"{type(exc).__name__}: {exc}"


def _studio_ready():
    if studio_api is None:
        return jsonify({"error": f"VDOC Studio unavailable — {STUDIO_IMPORT_ERROR}"}), 503
    return None


@app.route("/studio/state", methods=["GET"])
def studio_state():
    return _studio_ready() or jsonify(studio_api.state())


@app.route("/studio/command", methods=["POST"])
def studio_command():
    """One spoken or typed sentence -> actions -> a changed document."""
    not_ready = _studio_ready()
    if not_ready:
        return not_ready
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or data.get("command") or "").strip()
    if not text:
        return jsonify({"error": "Missing 'text' field"}), 400
    try:
        return jsonify(studio_api.run_command(
            text, data.get("model") or "groq",
            data.get("technique") or studio_api.ACTIONS_PROMPT))
    except Exception as exc:                  # noqa: BLE001 - never crash the tab
        return jsonify({"error": f"{type(exc).__name__}: {exc}",
                        **studio_api.state()}), 500


@app.route("/studio/pick", methods=["POST"])
def studio_pick():
    """Apply one of the buttons offered after a clarify. No LLM call."""
    not_ready = _studio_ready()
    if not_ready:
        return not_ready
    data = request.get_json(silent=True) or {}
    picked = studio_api.pick(data.get("index") or 0)
    if picked is None:
        return jsonify({"error": "no such option", **studio_api.state()}), 400
    return jsonify(picked)


@app.route("/studio/undo", methods=["POST"])
def studio_undo():
    not_ready = _studio_ready()
    if not_ready:
        return not_ready
    undone = studio_api.undo()
    return jsonify({"undone": undone, **studio_api.state()})


@app.route("/studio/reset", methods=["POST"])
def studio_reset():
    not_ready = _studio_ready()
    if not_ready:
        return not_ready
    studio_api.reset()
    return jsonify(studio_api.state())


@app.route("/studio/download", methods=["GET"])
def studio_download():
    """Rebuilt on every request, so the file always matches the preview."""
    not_ready = _studio_ready()
    if not_ready:
        return not_ready
    fmt = (request.args.get("format") or "docx").lower()
    path = studio_api.download_path(fmt)
    if not path:
        return jsonify({"error": "format must be docx or pdf"}), 400
    # inline=1 is the PDF-view iframe: it needs to display the file, not
    # download it. Everything else is a download.
    inline = request.args.get("inline") in ("1", "true", "yes")
    return send_file(path, as_attachment=not inline,
                     download_name=f"vdoc_studio.{fmt}",
                     mimetype="application/pdf" if fmt == "pdf" else None)


@app.route("/suggest-corrections", methods=["POST"])
def suggest_corrections():
    """
    Clean up an utterance before it is parsed.

    Request: {"command": "...", "document": "optional text on screen"}

    Three stages, all reported separately so the UI can show what happened:
      1. snapping  — pull Whisper's English-sounding output back onto our
                     Tanglish vocabulary (phrases, word fixes, then sound)
      2. saved     — corrections the user confirmed earlier, applied now
      3. suggested — anything else, offered as a yes/no question

    Only stages 1 and 2 change the text. Stage 3 never does.
    """
    data     = request.get_json(silent=True) or {}
    command  = (data.get("command") or "").strip()
    document = data.get("document") or ""
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400

    snapped = snapper.snap(command, document=document)
    review  = corrector.review(snapped["snapped"])

    # `original` stays the raw transcription so the UI can show the journey.
    review["original"] = command
    review["snapped"] = snapped["snapped"]
    review["snap_changes"] = snapped["changes"]
    review["snapping_enabled"] = snapped["enabled"]
    review["changed"] = review["corrected"] != command
    return jsonify(review)


@app.route("/confirm-correction", methods=["POST"])
def confirm_correction():
    """Remember a correction the user accepted, and return the new text."""
    data    = request.get_json(silent=True) or {}
    wrong   = (data.get("word") or "").strip()
    right   = (data.get("suggestion") or "").strip()
    command = (data.get("command") or "").strip()

    if not wrong or not right:
        return jsonify({"error": "Missing 'word' or 'suggestion'"}), 400

    try:
        corrections = corrector.save_correction(wrong, right)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "saved":       {"from": wrong.lower(), "to": right.lower()},
        "command":     corrector.apply_choice(command, wrong, right) if command else "",
        "corrections": corrections,
    })


@app.route("/corrections", methods=["GET"])
def list_corrections():
    """Every correction the user has confirmed so far."""
    return jsonify({"corrections": corrector.load_corrections(),
                    "file": "data/user_corrections.json"})


@app.route("/parse-rule", methods=["POST"])
def parse_rule():
    """Rule-based regex parser."""
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400
    return jsonify(_rule_result(command))


@app.route("/parse-bilstm", methods=["POST"])
def parse_bilstm():
    """BiLSTM baseline inference."""
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400
    return jsonify(bilstm_parser.predict(command))


@app.route("/parse-groq", methods=["POST"])
def parse_groq():
    """Groq LLM zero-shot parser."""
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400
    return jsonify(groq_parser.predict(command))


@app.route("/parse-xlmr-zero", methods=["POST"])
def parse_xlmr_zero():
    """XLM-RoBERTa zero-shot (random heads, no DocEdit training)."""
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400
    return jsonify(xlmr_parser.predict_zero(command))



@app.route("/parse-mdistilbert", methods=["POST"])
def parse_mdistilbert():
    """mDistilBERT fine-tuned on Tanglish-500."""
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400
    return jsonify(mdistilbert_parser.predict(command))


@app.route("/parse-all", methods=["POST"])
def parse_all():
    """
    Run all four parsers and return their results side-by-side.
    Response shape:
    {
        "rule":        { action, from, to, confidence, method },
        "bilstm":      { action, from, to, confidence, method },
        "doceditbert": { action, from, to, confidence, method },
        "groq":        { action, from, to, confidence, method }
    }
    """
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400

    return jsonify({
        "rule":        _rule_result(command),
        "bilstm":      bilstm_parser.predict(command),
        "doceditbert": doceditbert_parser.predict(command),
        "groq":        groq_parser.predict(command),
    })


@app.route("/batch-evaluate", methods=["POST"])
def batch_evaluate():
    """
    Evaluate all parsers on a labelled batch.

    Request body:
    {
        "commands": [
            {"text": "...", "intent": "bold", "from_anchor": "X", "to_anchor": "Y"},
            ...
        ]
    }

    Response:
    {
        "rule_accuracy":        0.387,
        "bilstm_accuracy":      0.641,
        "doceditbert_accuracy": 0.913,
        "total":                300,
        "details": [
            { "text": "...", "ground_truth": {...},
              "rule": {...}, "bilstm": {...}, "doceditbert": {...} }
        ]
    }
    """
    data     = request.get_json(silent=True) or {}
    commands = data.get("commands", [])
    if not commands:
        return jsonify({"error": "Missing or empty 'commands' list"}), 400

    def _correct(pred, gt):
        if pred is None:
            return False
        return (pred.get("action") == gt["intent"] and
                (pred.get("from") or "").lower() == gt["from_anchor"].lower() and
                (pred.get("to")   or "").lower() == gt["to_anchor"].lower())

    include_groq        = data.get("include_groq",        False)
    include_xlmr_zero   = data.get("include_xlmr_zero",   False)
    include_mdistilbert = data.get("include_mdistilbert", False)

    # Run local parsers sequentially; optional baselines on request
    local_results = []
    for item in commands:
        text = item.get("text", "")
        gt   = {"intent":      item.get("intent", ""),
                "from_anchor": item.get("from_anchor", ""),
                "to_anchor":   item.get("to_anchor", "")}
        row  = {
            "text": text,
            "gt":   gt,
            "rule":        _rule_result(text),
            "bilstm":      bilstm_parser.predict(text),
            "doceditbert": doceditbert_parser.predict(text),
        }
        if include_xlmr_zero:
            row["xlmr_zero"]    = xlmr_parser.predict_zero(text)
        if include_mdistilbert:
            row["mdistilbert"]  = mdistilbert_parser.predict(text)
        local_results.append(row)

    # Run Groq calls in sub-batches of 5 to stay within rate limits.
    # 5 parallel threads per sub-batch, 300 ms pause between sub-batches.
    groq_preds = {}
    if include_groq:
        texts = [r["text"] for r in local_results]
        SUB = 5
        with ThreadPoolExecutor(max_workers=SUB) as pool:
            for sub_start in range(0, len(texts), SUB):
                sub = texts[sub_start:sub_start + SUB]
                futures = {pool.submit(groq_parser.predict, t): sub_start + i
                           for i, t in enumerate(sub)}
                for fut in as_completed(futures):
                    groq_preds[futures[fut]] = fut.result()
                if sub_start + SUB < len(texts):
                    time.sleep(0.3)

    rule_ok = bilstm_ok = bert_ok = groq_ok = xlmr_zero_ok = mdistilbert_ok = 0
    details = []

    for i, row in enumerate(local_results):
        gt     = row["gt"]
        g_pred = groq_preds.get(i) if include_groq else None

        rule_ok   += int(_correct(row["rule"],        gt))
        bilstm_ok += int(_correct(row["bilstm"],      gt))
        bert_ok   += int(_correct(row["doceditbert"], gt))
        if include_groq:
            groq_ok += int(_correct(g_pred, gt))
        if include_xlmr_zero:
            xlmr_zero_ok   += int(_correct(row.get("xlmr_zero"),   gt))
        if include_mdistilbert:
            mdistilbert_ok += int(_correct(row.get("mdistilbert"), gt))

        d = {
            "text":         row["text"],
            "ground_truth": gt,
            "rule":         row["rule"],
            "bilstm":       row["bilstm"],
            "doceditbert":  row["doceditbert"],
            "groq":         g_pred,
        }
        if include_xlmr_zero:
            d["xlmr_zero"]   = row.get("xlmr_zero")
        if include_mdistilbert:
            d["mdistilbert"] = row.get("mdistilbert")
        details.append(d)

    n = len(commands)
    result = {
        "rule_accuracy":        round(rule_ok   / n, 4),
        "bilstm_accuracy":      round(bilstm_ok / n, 4),
        "doceditbert_accuracy": round(bert_ok   / n, 4),
        "total":                n,
        "details":              details,
    }
    if include_groq:
        result["groq_accuracy"]         = round(groq_ok        / n, 4)
    if include_xlmr_zero:
        result["xlmr_zero_accuracy"]    = round(xlmr_zero_ok   / n, 4)
    if include_mdistilbert:
        result["mdistilbert_accuracy"]  = round(mdistilbert_ok / n, 4)
    return jsonify(result)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "message": "VDOC backend running"})


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False)
