"""
main.py
-------
Real-time ukulele chord detection via webcam.

Pipeline:
  Webcam → Flip → HandTracker → GestureClassifier → AudioPlayer
                                                   → UI Overlay
"""

import sys

import cv2

from audio_player import AudioPlayer
from gesture_classifier import GestureClassifier
from hand_tracker import HandTracker
from utils import FPSCounter, draw_info_overlay

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CAMERA_INDEX        = 0
FRAME_WIDTH         = 640
FRAME_HEIGHT        = 480
DETECTION_CONF      = 0.7
TRACKING_CONF       = 0.7
SMOOTHING_WINDOW    = 7        # frames for chord smoothing
DEBOUNCE_TIME       = 1.5      # seconds between repeated chord plays
WINDOW_TITLE        = "Ukulele Chord Detector  |  Press Q to quit"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Camera setup ---
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {CAMERA_INDEX}.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # --- Module initialisation ---
    tracker    = HandTracker(detection_confidence=DETECTION_CONF,
                             tracking_confidence=TRACKING_CONF)
    classifier = GestureClassifier(smoothing_window=SMOOTHING_WINDOW)
    player     = AudioPlayer(debounce_time=DEBOUNCE_TIME)
    fps_ctr    = FPSCounter()

    print("[INFO] Starting. Show your right hand to the camera.")
    print("[INFO] Supported chords: C, G, Am, F")
    print("[INFO] Press  Q  to quit.\n")

    chord         = None
    finger_states = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Failed to grab frame — retrying...")
            continue

        # Mirror for natural interaction
        frame = cv2.flip(frame, 1)

        # --- Hand detection ---
        frame, landmarks, _bbox = tracker.process(frame)

        # --- Chord classification ---
        if landmarks:
            chord, finger_states = classifier.predict(landmarks)
            raw = classifier._classify_chord(classifier._get_finger_states(landmarks))
            print(f"\rfingers={finger_states}  raw={raw}  chord={chord}   ", end="", flush=True)
            if chord:
                player.play(chord)
        else:
            # Hand lost — reset history gradually via classifier's own deque
            chord         = None
            finger_states = None

        # --- Overlay ---
        fps = fps_ctr.tick()
        draw_info_overlay(frame, chord, fps, finger_states)

        cv2.imshow(WINDOW_TITLE, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:   # Q or Esc
            break

    # --- Cleanup ---
    cap.release()
    tracker.close()
    player.close()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
