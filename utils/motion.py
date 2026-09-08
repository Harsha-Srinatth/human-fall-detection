"""Motion analysis and fall timer for emergency detection."""

import time

import cv2
import numpy as np

from .config import EMERGENCY_TIMEOUT_SEC, MOTION_THRESHOLD


class MotionAnalyzer:
    def __init__(self):
        self.prev_gray = None

    def is_person_moving(self, frame, bbox):
        """Return True if noticeable motion is detected inside the bounding box."""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        if self.prev_gray is None or self.prev_gray.shape != gray.shape:
            self.prev_gray = gray
            return False

        diff = cv2.absdiff(self.prev_gray, gray)
        motion_score = np.mean(diff) / 255.0
        self.prev_gray = gray
        return motion_score > MOTION_THRESHOLD

    def reset(self):
        """Reset motion history."""
        self.prev_gray = None


class FallTimer:
    """Tracks immobile fall duration and triggers emergency after timeout."""

    def __init__(self, timeout_sec: float = EMERGENCY_TIMEOUT_SEC):
        self.timeout_sec = timeout_sec
        self.timer_start = None
        self.emergency_triggered = False

    @property
    def is_active(self) -> bool:
        return self.timer_start is not None

    def start(self):
        if self.timer_start is None:
            self.timer_start = time.time()

    def reset(self):
        self.timer_start = None
        self.emergency_triggered = False

    def elapsed(self) -> float:
        if self.timer_start is None:
            return 0.0
        return time.time() - self.timer_start

    def check_emergency(self) -> bool:
        """Return True once when immobile fall exceeds the timeout."""
        if self.emergency_triggered or self.timer_start is None:
            return False
        if self.elapsed() >= self.timeout_sec:
            self.emergency_triggered = True
            return True
        return False
