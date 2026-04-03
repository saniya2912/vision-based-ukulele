"""
generate_sounds.py
------------------
Generates synthetic ukulele chord .wav files using additive synthesis.
Run this ONCE before starting the app if you don't have real recordings.

Usage:
    python generate_sounds.py
"""

import os
import struct
import wave

import numpy as np

SOUNDS_DIR  = os.path.join(os.path.dirname(__file__), "sounds")
SAMPLE_RATE = 44100
DURATION    = 2.0   # seconds per chord

# Approximate frequencies for each ukulele chord (GCEA tuning)
# Using the 4 open strings + a few chord tones for a richer sound
CHORD_FREQUENCIES: dict[str, list[float]] = {
    "C":  [261.63, 329.63, 392.00, 523.25, 659.25],   # C E G C E
    "G":  [196.00, 246.94, 293.66, 392.00, 493.88],   # G B D G B
    "Am": [220.00, 261.63, 329.63, 440.00, 523.25],   # A C E A C
    "F":  [174.61, 220.00, 261.63, 349.23, 440.00],   # F A C F A
}


def _make_chord_wave(freqs: list[float]) -> np.ndarray:
    """
    Sum sine waves for each frequency in `freqs`.
    Apply an exponential decay envelope to simulate pluck.
    """
    t = np.linspace(0, DURATION, int(SAMPLE_RATE * DURATION), endpoint=False)
    wave_data = np.zeros_like(t)

    for freq in freqs:
        harmonic  = np.sin(2 * np.pi * freq * t)
        # Add slight overtone (2nd harmonic) for warmth
        harmonic += 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        envelope  = np.exp(-t * 1.8)   # decay constant
        wave_data += harmonic * envelope

    # Normalise to [-1, 1]
    peak = np.max(np.abs(wave_data))
    if peak > 0:
        wave_data /= peak
    wave_data *= 0.85   # slight headroom

    return (wave_data * 32767).astype(np.int16)


def _save_wav(path: str, data: np.ndarray) -> None:
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)          # mono
        wf.setsampwidth(2)          # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(data.tobytes())


def main() -> None:
    os.makedirs(SOUNDS_DIR, exist_ok=True)

    for chord, freqs in CHORD_FREQUENCIES.items():
        out = os.path.join(SOUNDS_DIR, f"{chord}.wav")
        data = _make_chord_wave(freqs)
        _save_wav(out, data)
        print(f"  Generated → {out}")

    print(
        f"\nDone. {len(CHORD_FREQUENCIES)} chord files in  sounds/\n"
        "Tip: replace these synthetic files with real ukulele recordings\n"
        "     for a more authentic sound."
    )


if __name__ == "__main__":
    main()
