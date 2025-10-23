#!/usr/bin/env python3

import os
import glob
import shlex

from multiprocessing import Pool

PIANOTEQ_EXEC = f"/home/loubb/pianoteq/x86-64bit/Pianoteq 8 STAGE"
# Directories
MIDI_DIR = "./mid"
AUDIO_DIR = "./audio"

# Pianoteq Settings (from your example)
PRESET = "NY Steinway D Classical Recording"
SAMPLE_RATE = 44100
OUTPUT_FORMAT = "mp3"  # Can be "wav", "mp3", "flac"
# -----------------------------------


def worker_fn(mid_path: str):
    base_filename = os.path.splitext(os.path.basename(mid_path))[0]
    audio_path = os.path.join(AUDIO_DIR, f"{base_filename}.{OUTPUT_FORMAT}")

    cmd = (
        f"{shlex.quote(PIANOTEQ_EXEC)} "
        f"--preset {shlex.quote(PRESET)} "
        f"--rate {SAMPLE_RATE} "
        f"--midi {shlex.quote(mid_path)} "
        f"--{OUTPUT_FORMAT} {shlex.quote(audio_path)}"
    )

    print(
        f"Converting: {os.path.basename(mid_path)} -> {os.path.basename(audio_path)}"
    )
    os.system(cmd)


def convert_files():
    # 1. Ensure the output audio directory exists
    os.makedirs(AUDIO_DIR, exist_ok=True)

    # 2. Find all MIDI files in the input directory
    # Using "*.mid*" catches both .mid and .midi extensions
    search_pattern = os.path.join(MIDI_DIR, "*.mid*")
    midi_files = glob.glob(search_pattern)

    if not os.path.exists(PIANOTEQ_EXEC):
        print(f"Error: Pianoteq executable not found at:")
        print(f"{PIANOTEQ_EXEC}")
        print("Please edit the PIANOTEQ_EXEC variable in the script.")
        return

    if not midi_files:
        print(f"No MIDI files found in {MIDI_DIR}")
        return

    print(f"Found {len(midi_files)} files. Starting conversion...")

    with Pool() as pool:
        res = pool.imap_unordered(worker_fn, midi_files)
        res = list(res)

    print("\nBatch conversion complete.")


if __name__ == "__main__":
    convert_files()
