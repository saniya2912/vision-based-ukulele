"""
gesture_classifier.py
---------------------
Rule-based hand gesture → ukulele chord classifier.

Finger states (UP=True / DOWN=False) are determined by comparing
landmark tip positions against their PIP joints.

Chord patterns: [Thumb, Index, Middle, Ring, Pinky]
Designed to be swapped for an ML model by replacing classify_chord().
"""

from collections import Counter, deque
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Chord definitions — easy to add more entries here
# [Thumb, Index, Middle, Ring, Pinky] : True = UP, False = DOWN
# ---------------------------------------------------------------------------
CHORD_MAP: dict[str, List[bool]] = {
    "G":  [True,  True,  True,  True,  True],   # Open hand (all up)
    "C":  [True,  True,  False, False, True],    # C shape (ring+middle down)
    "Am": [True,  True,  True,  False, False],   # Am (ring+pinky down)
    "F":  [False, True,  True,  True,  True],    # F (thumb tucked)
}

# MediaPipe landmark indices
FINGER_TIP_IDS  = [8, 12, 16, 20]   # Index → Pinky tips
FINGER_PIP_IDS  = [6, 10, 14, 18]   # Index → Pinky PIPs
THUMB_TIP_IDX   = 4
THUMB_IP_IDX    = 3


class GestureClassifier:
    """
    Classifies hand pose into ukulele chords with temporal smoothing.

    Parameters
    ----------
    smoothing_window : int
        Number of recent frames used to smooth predictions.
    confidence_threshold : float
        Fraction of window that must agree on the same chord.
    """

    def __init__(self, smoothing_window: int = 7, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self._history: deque = deque(maxlen=smoothing_window)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self, landmarks: list
    ) -> Tuple[Optional[str], Optional[List[bool]]]:
        """
        Main entry point. Returns (chord_name, finger_states).
        chord_name is None when pose is unknown or confidence is too low.
        """
        finger_states = self._get_finger_states(landmarks)
        raw_chord = self._classify_chord(finger_states)
        self._history.append(raw_chord)
        smoothed = self._smooth()
        return smoothed, finger_states

    # ------------------------------------------------------------------
    # Finger state detection
    # ------------------------------------------------------------------

    def _get_finger_states(self, landmarks: list) -> Optional[List[bool]]:
        """
        Returns [thumb_up, index_up, middle_up, ring_up, pinky_up].

        Thumb uses horizontal x-comparison (works for right hand in
        mirror view where the hand appears as "left" to MediaPipe).
        All other fingers use vertical y-comparison (tip vs PIP).
        """
        if landmarks is None or len(landmarks) < 21:
            return None

        states: List[bool] = []

        # Thumb: tip.x < ip.x → extended to the left in mirror view
        thumb_tip_x = landmarks[THUMB_TIP_IDX][0]
        thumb_ip_x  = landmarks[THUMB_IP_IDX][0]
        states.append(thumb_tip_x < thumb_ip_x)

        # Index → Pinky: tip.y < pip.y → finger pointing up (y=0 is top)
        for tip_id, pip_id in zip(FINGER_TIP_IDS, FINGER_PIP_IDS):
            tip_y = landmarks[tip_id][1]
            pip_y = landmarks[pip_id][1]
            states.append(tip_y < pip_y)

        return states

    # ------------------------------------------------------------------
    # Chord lookup
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_chord(finger_states: Optional[List[bool]]) -> Optional[str]:
        """
        Rule-based lookup. Returns chord name or None.
        Swap this method with an ML inference call to upgrade the system.
        """
        if finger_states is None:
            return None
        for chord, pattern in CHORD_MAP.items():
            if finger_states == pattern:
                return chord
        return None

    @staticmethod
    def is_fist(finger_states: Optional[List[bool]]) -> bool:
        """Returns True when all fingers are down (closed fist = reset gesture)."""
        if finger_states is None:
            return False
        return not any(finger_states)

    # ------------------------------------------------------------------
    # Temporal smoothing
    # ------------------------------------------------------------------

    def _smooth(self) -> Optional[str]:
        """Return most-frequent chord in history if above confidence threshold."""
        valid = [c for c in self._history if c is not None]
        if not valid:
            return None

        most_common_chord, count = Counter(valid).most_common(1)[0]
        confidence = count / len(self._history)

        return most_common_chord if confidence >= self.confidence_threshold else None
