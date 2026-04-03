"""
utils.py
--------
Shared helper utilities:
  - Landmark normalization (wrist-relative, scale-invariant)
  - FPS counter with exponential smoothing
  - On-screen overlay drawing
"""

import time
from typing import List, Optional

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Landmark normalization
# ---------------------------------------------------------------------------

def normalize_landmarks(landmarks: list) -> Optional[List[float]]:
    """
    Normalize 21 landmarks to be wrist-relative and scale-invariant.

    Steps:
      1. Translate so that wrist (landmark 0) is at the origin.
      2. Scale by the wrist → middle-finger-MCP distance (landmark 9).

    Returns a flat list of 42 floats [x0,y0, x1,y1, ...] or None.
    Useful for ML feature vectors.
    """
    if not landmarks or len(landmarks) < 21:
        return None

    wrist   = np.array(landmarks[0][:2], dtype=float)
    mid_mcp = np.array(landmarks[9][:2], dtype=float)
    scale   = np.linalg.norm(mid_mcp - wrist)

    if scale < 1e-6:
        return None

    normalized: List[float] = []
    for lm in landmarks:
        pt = (np.array(lm[:2], dtype=float) - wrist) / scale
        normalized.extend(pt.tolist())

    return normalized


# ---------------------------------------------------------------------------
# FPS counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """
    Smoothed FPS counter using exponential moving average.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor (higher = faster response to changes).
    """

    def __init__(self, alpha: float = 0.1):
        self._alpha = alpha
        self._prev_time = time.monotonic()
        self._fps: float = 0.0

    def tick(self) -> int:
        """Call once per processed frame. Returns smoothed FPS as int."""
        now = time.monotonic()
        dt = now - self._prev_time
        self._prev_time = now

        instant_fps = 1.0 / (dt + 1e-9)
        if self._fps == 0.0:
            self._fps = instant_fps
        else:
            self._fps = self._alpha * instant_fps + (1 - self._alpha) * self._fps

        return max(0, int(self._fps))


# ---------------------------------------------------------------------------
# On-screen overlay
# ---------------------------------------------------------------------------

# Colour scheme
_COL_GREEN  = (0,   255,  80)
_COL_GREY   = (120, 120, 120)
_COL_YELLOW = (0,   220, 255)
_COL_RED    = (0,    60, 255)
_COL_WHITE  = (255, 255, 255)

_FINGER_LABELS = ["T", "I", "M", "R", "P"]


def draw_info_overlay(
    frame,
    chord: Optional[str],
    fps: int,
    finger_states: Optional[List[bool]] = None,
) -> None:
    """
    Draw a HUD panel on the frame (in-place).

    Shows:
      - Detected chord name (large)
      - FPS counter
      - Per-finger UP/DOWN indicators at the bottom
    """
    h, w = frame.shape[:2]

    # --- semi-transparent background panel ---
    panel_h = 110
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # --- chord name ---
    if chord:
        cv2.putText(
            frame, f"Chord: {chord}", (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, _COL_GREEN, 3, cv2.LINE_AA,
        )
    else:
        cv2.putText(
            frame, "Chord: ---", (10, 52),
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, _COL_GREY, 2, cv2.LINE_AA,
        )

    # --- FPS ---
    cv2.putText(
        frame, f"FPS: {fps}", (10, 95),
        cv2.FONT_HERSHEY_SIMPLEX, 0.75, _COL_YELLOW, 2, cv2.LINE_AA,
    )

    # --- finger state dots ---
    if finger_states:
        dot_y = h - 15
        for i, (label, state) in enumerate(zip(_FINGER_LABELS, finger_states)):
            cx = 15 + i * 32
            color = _COL_GREEN if state else _COL_RED
            cv2.circle(frame, (cx, dot_y - 6), 7, color, -1)
            cv2.putText(
                frame, label, (cx - 5, dot_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, _COL_WHITE, 1, cv2.LINE_AA,
            )
