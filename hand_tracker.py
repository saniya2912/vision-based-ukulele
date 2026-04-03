"""
hand_tracker.py
---------------
MediaPipe Tasks HandLandmarker wrapper (compatible with mediapipe >= 0.10).
Handles model download, landmark extraction, drawing, and bounding box.
"""

import os
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

# MediaPipe hand connections (pairs of landmark indices)
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),          # thumb
    (0,5),(5,6),(6,7),(7,8),          # index
    (0,9),(9,10),(10,11),(11,12),     # middle
    (0,13),(13,14),(14,15),(15,16),   # ring
    (0,17),(17,18),(18,19),(19,20),   # pinky
    (5,9),(9,13),(13,17),             # palm
]


def _ensure_model() -> None:
    """Download hand_landmarker.task if not already present."""
    if os.path.exists(MODEL_PATH):
        return
    print("[HandTracker] Downloading hand landmarker model (~5 MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("[HandTracker] Model downloaded.")


# ---------------------------------------------------------------------------
# HandTracker
# ---------------------------------------------------------------------------

class HandTracker:
    """Detects and tracks a single hand using MediaPipe Tasks API."""

    def __init__(
        self,
        max_hands: int = 1,
        detection_confidence: float = 0.7,
        tracking_confidence: float = 0.7,
    ):
        _ensure_model()

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._detector = HandLandmarker.create_from_options(options)
        self._timestamp_ms = 0

    def process(self, frame):
        """
        Detect hand in frame, draw landmarks + bounding box.

        Args:
            frame: BGR image (already flipped for mirror view)

        Returns:
            frame:     Annotated BGR frame
            landmarks: List of 21 (x_px, y_px, z) tuples, or None
            bbox:      (x1, y1, x2, y2) in pixels, or None
        """
        h, w = frame.shape[:2]

        # Convert BGR → RGB for MediaPipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Increment timestamp (VIDEO mode requires monotonically increasing ms)
        self._timestamp_ms += 33
        result = self._detector.detect_for_video(mp_image, self._timestamp_ms)

        landmarks = None
        bbox      = None

        if result.hand_landmarks:
            raw = result.hand_landmarks[0]   # first hand only

            # Convert normalised → pixel coords
            landmarks = [
                (int(lm.x * w), int(lm.y * h), lm.z)
                for lm in raw
            ]

            # Draw connections
            for start, end in HAND_CONNECTIONS:
                pt1 = landmarks[start][:2]
                pt2 = landmarks[end][:2]
                cv2.line(frame, pt1, pt2, (80, 180, 80), 2, cv2.LINE_AA)

            # Draw landmark dots
            for (cx, cy, _) in landmarks:
                cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (cx, cy), 4, (0, 150, 255),    1, cv2.LINE_AA)

            # Bounding box
            xs = [p[0] for p in landmarks]
            ys = [p[1] for p in landmarks]
            pad = 20
            x1  = max(0, min(xs) - pad)
            y1  = max(0, min(ys) - pad)
            x2  = min(w, max(xs) + pad)
            y2  = min(h, max(ys) + pad)
            bbox = (x1, y1, x2, y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return frame, landmarks, bbox

    def close(self) -> None:
        self._detector.close()
