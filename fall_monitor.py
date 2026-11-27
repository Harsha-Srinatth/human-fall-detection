"""Fall logic: one alert per fall event.

Old bug: the timer (and its "already alerted" flag) was reset whenever the
detection flickered for one frame or tiny motion was seen. A person lying on
the floor for 30 min therefore got a new alert every ~20-30 seconds.

Fix:
  * First alert after EMERGENCY_TIMEOUT_SEC (10s) of lying still.
  * While the person stays down, a follow-up reminder is sent every
    REMIND_EVERY_SEC (2 min) - escalation, not spam every frame.
  * The "alert sent" flag is only cleared when the person is seen standing
    again for RECOVER_SEC seconds (or nobody is seen for a long time).
  * Short detection drop-outs (< GRACE_SEC) do not reset the timer.
"""

import os
import time

import cv2
import numpy as np

EMERGENCY_TIMEOUT_SEC = float(os.getenv("EMERGENCY_TIMEOUT_SEC", "10"))  # lying still this long -> alert
RECOVER_SEC = float(os.getenv("RECOVER_SEC", "3"))   # standing this long -> new fall can alert again
REMIND_EVERY_SEC = float(os.getenv("REMIND_EVERY_SEC", "120"))  # still down -> reminder alert every 2 min
GRACE_SEC = 3.0            # ignore short detection gaps
NOBODY_RESET_SEC = 60.0    # nobody in view this long -> reset
MOTION_THRESHOLD = 0.12    # mean pixel change (0-1) inside the box that counts as "moving"


def is_fallen_box(x1, y1, x2, y2):
    """Lying body -> box wider than tall."""
    return (x2 - x1) > (y2 - y1)


class FallMonitor:
    def __init__(self):
        self.reset()

    def reset(self):
        self.still_since = None       # when the fallen person stopped moving
        self.last_fallen_seen = None
        self.last_person_seen = None
        self.upright_since = None
        self.alert_sent = False
        self.last_alert_at = None
        self.first_alert_at = None
        self.alert_count = 0
        self.prev_roi = None

    def _moving(self, frame, box):
        x1, y1, x2, y2 = box
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return False
        gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))
        prev, self.prev_roi = self.prev_roi, gray
        if prev is None:
            return False
        return float(np.mean(cv2.absdiff(prev, gray))) / 255.0 > MOTION_THRESHOLD

    def update(self, frame, detections, now=None):
        """Process one frame. Returns dict with state and `new_alert` (True only once per fall)."""
        now = now or time.time()
        fallen = [d for d in detections if is_fallen_box(*d[:4])]
        new_alert = False

        if detections:
            self.last_person_seen = now

        if fallen:
            self.last_fallen_seen = now
            self.upright_since = None
            if self._moving(frame, fallen[0][:4]) or self.still_since is None:
                self.still_since = now
            if not self.alert_sent and now - self.still_since >= EMERGENCY_TIMEOUT_SEC:
                self.alert_sent = True
                self.first_alert_at = now
                self.alert_count = 0
                new_alert = True
            elif self.alert_sent and now - self.last_alert_at >= REMIND_EVERY_SEC:
                new_alert = True  # still on the floor: escalate, but only every REMIND_EVERY_SEC
            if new_alert:
                self.last_alert_at = now
                self.alert_count += 1
        elif detections:  # someone standing
            self.prev_roi = None
            self.upright_since = self.upright_since or now
            if now - self.upright_since >= RECOVER_SEC:
                self.still_since = None
                self.alert_sent = False
        else:  # nobody detected
            if self.last_fallen_seen and now - self.last_fallen_seen > GRACE_SEC and not self.alert_sent:
                self.still_since = None
            if self.last_person_seen and now - self.last_person_seen > NOBODY_RESET_SEC:
                self.reset()

        if self.alert_sent:
            status = "EMERGENCY - alert sent"
        elif fallen or (self.still_since is not None and not detections):
            status = "Possible fall"
        elif detections:
            status = "Standing"
        else:
            status = "No person"

        timer = 0.0 if self.still_since is None else now - self.still_since
        return {
            "status": status,
            "timer": round(min(timer, EMERGENCY_TIMEOUT_SEC), 1),
            "timeout": EMERGENCY_TIMEOUT_SEC,
            "alert_sent": self.alert_sent,
            "new_alert": new_alert,
            "alert_count": self.alert_count,
            "down_for": int(now - self.first_alert_at + EMERGENCY_TIMEOUT_SEC) if self.alert_sent else 0,
        }
