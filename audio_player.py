"""
audio_player.py
---------------
Plays ukulele chord audio via ElevenLabs Sound Effects API.

On first use, each chord is generated and cached in sounds/ as a .mp3.
Subsequent plays use the cached file — no repeat API calls.
Falls back to a local .wav file if the API is unavailable.

Environment variable required:
    ELEVENLABS_API_KEY  — set in .env (never commit this file)
"""

import os
import threading
import time
from typing import Optional

import numpy as np
import requests

try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; key can be set directly in env

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SOUNDS_DIR   = os.path.join(os.path.dirname(__file__), "sounds")
API_ENDPOINT = "https://api.elevenlabs.io/v1/sound-generation"
API_KEY      = os.environ.get("ELEVENLABS_API_KEY", "")

# Text prompts sent to ElevenLabs for each chord
CHORD_PROMPTS: dict[str, str] = {
    "C":  "ukulele strumming a bright C major chord, clean acoustic",
    "G":  "ukulele strumming a warm G major chord, clean acoustic",
    "Am": "single clean downstroke strum on ukulele A minor chord, one hit, acoustic",
    "F":  "ukulele strumming an F major chord, clean acoustic",
}

GENERATION_PARAMS = {
    "duration_seconds":  2.0,
    "prompt_influence":  0.4,   # 0–1; higher = closer to the text prompt
}


# ---------------------------------------------------------------------------
# AudioPlayer
# ---------------------------------------------------------------------------

class AudioPlayer:
    """
    Generates chord audio via ElevenLabs (cached), plays with pygame.

    Parameters
    ----------
    debounce_time : float
        Minimum seconds between plays of the same chord.
    """

    def __init__(self, debounce_time: float = 1.5):
        self.debounce_time   = debounce_time
        self._last_chord: Optional[str] = None
        self._last_play_time: float     = 0.0
        self._sounds: dict              = {}
        self._lock = threading.Lock()

        os.makedirs(SOUNDS_DIR, exist_ok=True)

        if not _PYGAME_AVAILABLE:
            print("[AudioPlayer] pygame not installed — audio disabled.")
            return

        if not API_KEY:
            print("[AudioPlayer] ELEVENLABS_API_KEY not set — audio disabled.")
            return

        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self._load_all_chords()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def play(self, chord: str) -> None:
        """Play chord with debounce. Only triggers when chord changes or
        enough time has passed."""
        if not _PYGAME_AVAILABLE or chord not in self._sounds:
            return

        now = time.monotonic()
        with self._lock:
            if chord == self._last_chord and (now - self._last_play_time) < self.debounce_time:
                return
            self._last_chord     = chord
            self._last_play_time = now

        threading.Thread(target=self._play_sound, args=(chord,), daemon=True).start()

    def close(self) -> None:
        if _PYGAME_AVAILABLE:
            try:
                pygame.mixer.quit()
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Sound loading — fetch from ElevenLabs or use cache
    # ------------------------------------------------------------------

    def _load_all_chords(self) -> None:
        for chord in CHORD_PROMPTS:
            cache_path = os.path.join(SOUNDS_DIR, f"{chord}.mp3")

            if os.path.exists(cache_path):
                self._load_cached(chord, cache_path)
            else:
                self._fetch_and_cache(chord, cache_path)

    def _fetch_and_cache(self, chord: str, cache_path: str) -> None:
        """Call ElevenLabs API, save result, then load into pygame."""
        print(f"[AudioPlayer] Generating '{chord}' via ElevenLabs...")
        try:
            payload = {
                "text": CHORD_PROMPTS[chord],
                **GENERATION_PARAMS,
            }
            headers = {
                "xi-api-key":   API_KEY,
                "Content-Type": "application/json",
            }
            response = requests.post(
                API_ENDPOINT,
                json=payload,
                headers=headers,
                timeout=30,
            )
            response.raise_for_status()

            with open(cache_path, "wb") as f:
                f.write(response.content)

            print(f"[AudioPlayer] Cached  → {cache_path}")
            self._load_cached(chord, cache_path)

        except requests.exceptions.HTTPError as e:
            print(f"[AudioPlayer] API error for '{chord}': {e.response.status_code} {e.response.text}")
            self._try_fallback(chord)
        except Exception as exc:
            print(f"[AudioPlayer] Failed to generate '{chord}': {exc}")
            self._try_fallback(chord)

    def _load_cached(self, chord: str, path: str) -> None:
        try:
            sound = pygame.mixer.Sound(path)
            self._sounds[chord] = self._normalize(sound)
            print(f"[AudioPlayer] Loaded  {os.path.basename(path)}")
        except Exception as exc:
            print(f"[AudioPlayer] Could not load {path}: {exc}")
            self._try_fallback(chord)

    @staticmethod
    def _normalize(sound: "pygame.mixer.Sound", target_rms: float = 5000.0) -> "pygame.mixer.Sound":
        """Scale sound amplitude so all chords play at the same perceived volume."""
        samples = pygame.sndarray.array(sound).astype(np.float32)
        rms = np.sqrt(np.mean(samples ** 2))
        if rms > 0:
            samples = samples * (target_rms / rms)
            samples = np.clip(samples, -32768, 32767).astype(np.int16)
            sound = pygame.sndarray.make_sound(samples)
        return sound

    def _try_fallback(self, chord: str) -> None:
        """Use a local .wav file if API/cache fails."""
        wav_path = os.path.join(SOUNDS_DIR, f"{chord}.wav")
        if os.path.exists(wav_path):
            try:
                self._sounds[chord] = pygame.mixer.Sound(wav_path)
                print(f"[AudioPlayer] Fallback loaded  {chord}.wav")
            except Exception as exc:
                print(f"[AudioPlayer] Fallback also failed for '{chord}': {exc}")

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------

    def _play_sound(self, chord: str) -> None:
        try:
            self._sounds[chord].play()
        except Exception as exc:
            print(f"[AudioPlayer] Playback error for '{chord}': {exc}")
