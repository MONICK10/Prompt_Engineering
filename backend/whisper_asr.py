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

# Vocabulary hint. Whisper treats this as the text that came just before the
# audio, so it biases decoding towards these spellings and, more importantly,
# towards this *style* of sentence.
#
# Full Tanglish sentences work much better here than a word list: a bare list
# gives Whisper no grammar to follow, so it still falls back on English and
# turns "varikum" into "vari-crum". Sentences taken from DocEdit-1000 show it
# what a whole command looks like. Kept well under the 224-token limit --
# anything longer is silently truncated from the front, which would drop the
# examples we care about most.
_EXAMPLE_COMMANDS = (
    "car la irunthu accident varikum bold panu. "
    "introduction la irunthu conclusion vara italic sei. "
    "header irunthu footer varikum underline panu. "
    "second line delete pannu, apram second line bold pannu. "
    "first line bold-um italic-um pannu. "
    "mudhal vari highlight pannu, apram kadaisi vari delete pannu. "
    "summary ku aprom conclusion insert panu. "
    "rendavathu line heading2 aa maathu."
)

VOCAB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "..", "data", "vocabulary.txt")


def load_vocabulary():
    """
    Names and command words from data/vocabulary.txt, read every time.

    A proper noun Whisper has never met comes back as whatever it sounds
    like -- "Karunya" turns into "caroniya" -- and no amount of grammar in
    the examples fixes a word the decoder does not know. Listing the word is
    what fixes it, and keeping the list in a file means adding one needs no
    code change and no restart.
    """
    try:
        with open(VOCAB_FILE, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
    except OSError:
        return ""
    words = [line for line in lines if line and not line.startswith("#")]
    return ", ".join(words)


def build_initial_prompt():
    """
    Vocabulary first, example sentences last.

    Both matter: the word list fixes spellings, the sentences show Whisper
    what a whole Tanglish command sounds like so it does not fall back on
    English grammar. The sentences go last because an over-long prompt is
    truncated from the front, and losing a name costs less than losing the
    shape of a command.
    """
    vocabulary = load_vocabulary()
    return f"{vocabulary}. {_EXAMPLE_COMMANDS}" if vocabulary else _EXAMPLE_COMMANDS


INITIAL_PROMPT = build_initial_prompt()

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
        initial_prompt=build_initial_prompt(),   # re-read, so edits need no restart
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
        "initial_prompt": build_initial_prompt(),
    }
