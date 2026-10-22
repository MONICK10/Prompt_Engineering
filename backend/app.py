"""
VDOC Flask backend.

Endpoints
---------
POST /transcribe     audio file  -> {"text": "..."}
POST /parse          command str -> DocEditBERT result
POST /parse-rule     command str -> rule-based result
POST /parse-bilstm   command str -> BiLSTM result
POST /parse-all      command str -> all three results combined
POST /batch-evaluate commands[]  -> per-system accuracy over a labelled batch
"""
import os
import sys
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import Flask, request, jsonify
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
_whisper_model = None

def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        _whisper_model = whisper.load_model("base")
    return _whisper_model


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
        model  = _get_whisper()
        # Force English decoding so Tamil-accented / code-switched (Tanglish)
        # speech comes back romanized (Latin script), matching the script
        # DocEditBERT/mDistilBERT were trained on, instead of Whisper
        # auto-detecting Tamil and returning Tamil-script text.
        result = model.transcribe(tmp_path, language="en")
        return jsonify({"text": result["text"].strip()})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        os.unlink(tmp_path)


BERT_CONFIDENCE_THRESHOLD = 0.75

@app.route("/parse", methods=["POST"])
def parse_doceditbert():
    """
    DocEditBERT inference with confidence-based Groq fallback.
    If DocEditBERT confidence < 0.75, Groq LLM is called instead and the
    response carries method='llm-fallback' and routed_to='groq'.
    All responses include a 'routed_to' field.
    """
    command = _cmd_from_request()
    if not command:
        return jsonify({"error": "Missing 'command' field"}), 400

    result     = doceditbert_parser.predict(command)
    confidence = result.get("confidence") or 0.0

    if confidence >= BERT_CONFIDENCE_THRESHOLD:
        result["routed_to"] = "doceditbert"
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
