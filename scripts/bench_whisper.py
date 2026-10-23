r"""
Compare Whisper model sizes on VDOC commands.

Runs every .wav in a folder through each model, with and without the
vocabulary hint, and prints what was heard plus how long it took. This is how
the WHISPER_MODEL choice in backend/whisper_asr.py was decided -- it is not a
guess about which size is fast enough.

    .\.venv\Scripts\python.exe scripts\bench_whisper.py debug\mine
    .\.venv\Scripts\python.exe scripts\bench_whisper.py debug\mine --models base small
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "backend"))

import whisper                                   # noqa: E402
from whisper_asr import INITIAL_PROMPT, LEADING_PAD_MS, SAMPLE_RATE   # noqa: E402


def load_padded(path):
    audio = whisper.load_audio(path)
    pad = np.zeros(int(SAMPLE_RATE * LEADING_PAD_MS / 1000), dtype=audio.dtype)
    return np.concatenate([pad, audio]), len(audio) / SAMPLE_RATE


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="folder of .wav clips")
    parser.add_argument("--models", nargs="+", default=["base", "small"])
    parser.add_argument("--no-prompt", action="store_true",
                        help="also run without the vocabulary hint")
    args = parser.parse_args()

    folder = args.folder if os.path.isabs(args.folder) else os.path.join(ROOT, args.folder)
    clips = sorted(f for f in os.listdir(folder) if f.lower().endswith((".wav", ".webm", ".mp3", ".m4a")))
    if not clips:
        print(f"No audio files in {folder}")
        return

    print(f"{len(clips)} clip(s) from {folder}\n")

    for name in args.models:
        print("=" * 76)
        print(f"MODEL: {name}")
        loaded = time.perf_counter()
        model = whisper.load_model(name)
        print(f"  (load took {time.perf_counter() - loaded:.1f}s)\n")

        times = []
        for clip in clips:
            audio, seconds = load_padded(os.path.join(folder, clip))

            started = time.perf_counter()
            with_hint = model.transcribe(audio, language="en", fp16=False,
                                         initial_prompt=INITIAL_PROMPT)
            elapsed = time.perf_counter() - started
            times.append(elapsed)

            print(f"  {clip}  ({seconds:.1f}s audio, {elapsed:.2f}s decode)")
            print(f"     with hint : {with_hint['text'].strip()!r}")

            if args.no_prompt:
                plain = model.transcribe(audio, language="en", fp16=False)
                print(f"     no hint   : {plain['text'].strip()!r}")
            print()

        if times:
            print(f"  {name}: mean {sum(times)/len(times):.2f}s, max {max(times):.2f}s "
                  f"over {len(times)} clips\n")


if __name__ == "__main__":
    main()
