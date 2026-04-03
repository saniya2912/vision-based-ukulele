# Vision-Based Ukulele Chord Detector

Real-time hand gesture recognition that maps hand poses to ukulele chords and plays the corresponding sound via ElevenLabs AI audio.

---

## Setup

```bash
# 1. Activate your virtual environment
source ukulele_venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your ElevenLabs API key to .env
echo "ELEVENLABS_API_KEY=your_key_here" > .env

# 4. Run — chord sounds are auto-generated on first launch
python main.py
```

> **First run** downloads the MediaPipe hand model (~5 MB) and generates chord audio via ElevenLabs. Both are cached locally — subsequent runs start instantly.

---

## Controls

| Key | Action |
|-----|--------|
| `Q` or `Esc` | Quit |

---

## How Chord Detection Works

The camera sees your hand and extracts 21 landmarks (fingertip, knuckle, and joint positions). Each of the 5 fingers is classified as **UP** or **DOWN**:

- **Fingers (Index → Pinky):** UP if the fingertip is higher than the middle knuckle (PIP joint)
- **Thumb:** UP if the tip extends to the left (right hand in mirror view)

This gives a 5-element pattern like `[Thumb, Index, Middle, Ring, Pinky]` which is matched against the chord table below.

---

## Finger Patterns for Each Chord

Hold your **right hand** up in front of the camera, palm facing you (like a mirror).

### G Major — Open Hand

```
Thumb  Index  Middle  Ring  Pinky
  UP     UP     UP     UP    UP
  ✅     ✅     ✅     ✅    ✅
```

> Fully open hand, all five fingers extended and pointing up.

```
   | | | | |
   | | | | |
   |_|_|_|_|
       👍
```

---

### C Major — Middle + Ring Down

```
Thumb  Index  Middle  Ring  Pinky
  UP     UP    DOWN   DOWN   UP
  ✅     ✅     ❌     ❌    ✅
```

> Keep thumb, index, and pinky up. Curl your middle and ring fingers down into your palm.

```
   |   |   |
   |   |   |
   | ✊ ✊ |
       👍
```

---

### A Minor — Ring + Pinky Down

```
Thumb  Index  Middle  Ring  Pinky
  UP     UP     UP    DOWN   DOWN
  ✅     ✅     ✅     ❌     ❌
```

> Extend thumb, index, and middle finger. Curl ring finger and pinky down.

```
   | | |
   | | |
   | | | ✊ ✊
       👍
```

---

### F Major — Thumb Tucked

```
Thumb  Index  Middle  Ring  Pinky
 DOWN    UP     UP     UP    UP
  ❌     ✅     ✅     ✅    ✅
```

> Tuck or fold your thumb across your palm. Keep all four fingers (index to pinky) pointing straight up.

```
     | | | |
     | | | |
  👍 |_|_|_|
  (tucked)
```

---

## Tips for Accurate Detection

- **Use your right hand**, palm facing the camera
- **Good lighting** makes a big difference — avoid strong backlight
- **Hold each pose steady** for ~0.5 seconds (smoothing window = 7 frames)
- Keep your hand **centered** in the frame
- The **bounding box** turning green confirms the hand is detected

---

## Adding New Chords

Open `gesture_classifier.py` and add an entry to `CHORD_MAP`:

```python
CHORD_MAP: dict[str, List[bool]] = {
    "G":  [True,  True,  True,  True,  True],
    "C":  [True,  True,  False, False, True],
    "Am": [True,  True,  True,  False, False],
    "F":  [False, True,  True,  True,  True],
    # Add your chord here:
    "D":  [False, True,  True,  False, False],   # example
}
```

Pattern order: `[Thumb, Index, Middle, Ring, Pinky]` — `True` = UP, `False` = DOWN.

Then add a matching entry in `audio_player.py` under `CHORD_PROMPTS` and delete the old cached `.mp3` so it regenerates.

---

## Project Structure

```
main.py               → application loop (camera, integrates all modules)
hand_tracker.py       → MediaPipe Tasks hand detection + bounding box drawing
gesture_classifier.py → finger state extraction + chord pattern matching
audio_player.py       → ElevenLabs API audio generation + cached playback
utils.py              → landmark normalization, FPS counter, HUD overlay
generate_sounds.py    → generates offline fallback .wav files (no API needed)
sounds/               → cached chord audio files (auto-created on first run)
hand_landmarker.task  → MediaPipe model (auto-downloaded on first run)
.env                  → your ElevenLabs API key (never commit this)
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| No sound | Check that `sounds/` contains `.mp3` files; verify `ELEVENLABS_API_KEY` in `.env` |
| Wrong chord detected | Run with debug prints active in `main.py` — check terminal output for live finger states |
| Hand not detected | Improve lighting; move hand to center of frame |
| Low FPS | Reduce resolution in `main.py` (`FRAME_WIDTH`, `FRAME_HEIGHT`) |
| Camera not opening | Change `CAMERA_INDEX = 0` to `1` or `2` in `main.py` |
