"""
Whisper speech-to-text for VDOC, with diagnostics.

Whisper is the one model in VDOC that was never trained on our data -- it is
OpenAI's ready-made model. DocEditBERT only ever sees Whisper's output, so a
mishearing here ("first line bold panu" -> "like, word, or name") reaches the
parser as a genuine-looking command and gets parsed with low confidence.

Three things make it behave better on our vocabulary:

  model size     "base" is small and fast but weak on accented and
                 code-switched speech. WHISPER_MODEL picks the size.
  initial_prompt a short list of the words we expect. Whisper conditions on
                 it, which pulls "bowl"/"bolt" towards "bold" and keeps the
                 Tanglish particles ("pannu", "apram") intact.
  leading pad    browsers drop the first fraction of a second while the
                 recorder spins up, which clips the first word. We prepend
                 300 ms of silence so the first word is never at sample 0.

Every call writes the audio it actually decoded to debug/last_audio.wav, so
a bad transcription can be listened to afterwards.
"""
import os
import time
import wave

import numpy as np

SAMPLE_RATE = 16000              # Whisper always resamples to this

# Model size. "small" is cached locally; "base" was the original setting.
WHISPER_MODEL = os.environ.get("VDOC_WHISPER_MODEL", "small")

# Forced to English so Tanglish comes back romanised (Latin script), matching
# what DocEditBERT was trained on, instead of Whisper switching to Tamil
# script when it detects Tamil words.
WHISPER_LANGUAGE = "en"

# Vocabulary hint. Whisper treats this as the text preceding the audio, so it
# biases decoding towards these spellings.
INITIAL_PROMPT = (
    "first line, second line, third line, last line, "
    "bold, italic, underline, highlight, delete, heading, "
    "pannu, panu, apram, pannitu, bold-um, italic-um, "
    "vari, mudhal, rendavathu"
)

LEADING_PAD_MS = 300

_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DEBUG_DIR = os.path.join(_ROOT, "debug")
DEBUG_WAV = os.path.join(DEBUG_DIR, "last_audio.wav")

_model = None
_model_name = None


def get_model(name=None):
    """Load (and cache) the Whisper model. Downloads once, then offline."""
    global _model, _model_name
    wanted = name or WHISPER_MODEL
    if _model is None or _model_name != wanted:
        import whisper
        _model = whisper.load_model(wanted)
        _model_name = wanted
    return _model


def _save_wav(audio, path):
    """Write a float32 mono array to a 16-bit PCM wav."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    with wave.open(path, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())


def _probe(path):
    """Original container's sample rate / channels, via ffprobe if present."""
    import json
    import subprocess
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "a:0", path],
            capture_output=True, text=True, timeout=20, check=True,
        ).stdout
        stream = (json.loads(out).get("streams") or [{}])[0]
        return {
            "source_sample_rate": int(stream["sample_rate"]) if stream.get("sample_rate") else None,
            "source_channels": stream.get("channels"),
            "source_codec": stream.get("codec_name"),
        }
    except Exception:
        return {"source_sample_rate": None, "source_channels": None,
                "source_codec": None}


def transcribe_file(path, model_name=None, save_debug=True):
    """
    Transcribe an audio file and report how it was decoded.

    Returns {text, diagnostics}. diagnostics carries the model size, the
    language setting, the clip length, the sample rate, and how long
    inference took -- everything needed to tell a mishearing apart from a
    recording problem.
    """
    import whisper

    source = _probe(path)

    # load_audio runs the file through ffmpeg: mono, 16 kHz, float32.
    audio = whisper.load_audio(path)
    raw_seconds = len(audio) / SAMPLE_RATE

    pad_samples = int(SAMPLE_RATE * LEADING_PAD_MS / 1000)
    padded = np.concatenate([np.zeros(pad_samples, dtype=audio.dtype), audio])

    if save_debug:
        _save_wav(padded, DEBUG_WAV)

    model = get_model(model_name)
    started = time.perf_counter()
    result = model.transcribe(
        padded,
        language=WHISPER_LANGUAGE,
        initial_prompt=INITIAL_PROMPT,
        fp16=False,                       # CPU only; avoids the FP16 warning
        temperature=0.0,                  # no random retries: same clip, same
        condition_on_previous_text=False, # text every time, which a demo needs
    )
    elapsed = time.perf_counter() - started

    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0

    return {
        "text": (result.get("text") or "").strip(),
        "diagnostics": {
            "model": _model_name,
            "language": WHISPER_LANGUAGE,
            "initial_prompt_used": True,
            "sample_rate": SAMPLE_RATE,
            "clip_seconds": round(raw_seconds, 3),
            "leading_pad_ms": LEADING_PAD_MS,
            "padded_seconds": round(len(padded) / SAMPLE_RATE, 3),
            "peak_amplitude": round(peak, 4),
            "silent": peak < 0.01,
            "inference_seconds": round(elapsed, 2),
            "debug_wav": os.path.relpath(DEBUG_WAV, _ROOT).replace("\\", "/"),
            **source,
        },
    }


def config():
    """What the dashboard shows about the ASR stage."""
    return {
        "model": WHISPER_MODEL,
        "loaded": _model_name,
        "language": WHISPER_LANGUAGE,
        "sample_rate": SAMPLE_RATE,
        "leading_pad_ms": LEADING_PAD_MS,
        "initial_prompt": INITIAL_PROMPT,
    }
