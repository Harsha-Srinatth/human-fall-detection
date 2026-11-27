"""Fall logic built from rules used by existing open-source / commercial systems.

Per-frame posture ("is the person lying?"):
  * REQUIRED: torso angle from vertical > 60 deg (shoulder-mid -> hip-mid). taufeeque9/HumanFallDetection
    treats |angle| < 45 deg as upright; papers use 60-120 deg as the fall range. 60 avoids
    "leaning back in a chair" (~30-50 deg).
  * PLUS one of: box width >= 1.2 x height (cwlroda/falldetection_openpifpaf), or keypoints spread
    more horizontally than vertically (MoveNet rule-based papers).
  * If shoulders/hips are not visible we do NOT call it a fall. The old code used the box
    shape only, so a person sitting close to the webcam (wide box) triggered false alerts.

Fall motion ("did they fall, or just lie down?"):
  * hip centre drops >= 40% of standing height within 1.5 s, or
  * person was clearly upright (torso < 45 deg) less than 1.5 s before lying
  (falls take ~0.5-1 s; lying down on purpose goes through sitting and takes 2-4 s)
  (cwlroda: centroid moves >= 0.5 x box diagonal within ~1 s)

Smoothing: lying must hold for >= 50% of frames over 1 s before it counts
(taufeeque9 needs 9 consecutive frames before reporting a fall).

Timers (Apple Watch: hard fall + ~1 min no movement -> auto call; we keep it short for a demo):
  * FALL + no movement for STILL_ALERT_SEC (10 s)          -> EMERGENCY alert
  * On the floor for LONG_LIE_SEC (30 s), even if moving   -> alert ("can't get up")
  * Still down -> reminder every REMIND_EVERY_SEC (120 s)  (no spam every frame)
  * Upright for RECOVER_SEC (3 s) -> reset, next fall alerts again
  * Detection lost < GRACE_SEC (2 s) -> ignored (cwlroda drops a track after 2 x fps frames)
"""

import math
import os
import time
from collections import deque

import numpy as np

from detector import L_HIP, L_SHOULDER, R_HIP, R_SHOULDER

STILL_ALERT_SEC = float(os.getenv("STILL_ALERT_SEC", "10"))
LONG_LIE_SEC = float(os.getenv("LONG_LIE_SEC", "30"))
REMIND_EVERY_SEC = float(os.getenv("REMIND_EVERY_SEC", "120"))
RECOVER_SEC = float(os.getenv("RECOVER_SEC", "3"))
GRACE_SEC = 2.0
NOBODY_RESET_SEC = 60.0

ANGLE_LYING_DEG = 60
ANGLE_UPRIGHT_DEG = 45
BOX_RATIO_LYING = 1.2
KPT_MIN_SCORE = 0.35
FALL_WINDOW_SEC = 1.5
DROP_RATIO = 0.4
SMOOTH_SEC = 1.0
SMOOTH_FRACTION = 0.5
MOVE_CHECK_SEC = 0.5
MOVE_THRESHOLD = 0.08   # mean keypoint shift (fraction of body size) per check = "moving"


def _mid(kpts, a, b):
    pa, pb = kpts[a], kpts[b]
    if pa[2] >= KPT_MIN_SCORE and pb[2] >= KPT_MIN_SCORE:
        return (pa[:2] + pb[:2]) / 2
    if pa[2] >= KPT_MIN_SCORE:
        return pa[:2]
    if pb[2] >= KPT_MIN_SCORE:
        return pb[:2]
    return None


def analyze_posture(person):
    """Return posture features + lying vote for one detected person."""
    x1, y1, x2, y2 = person["box"]
    kpts = person["kpts"]
    w, h = max(1, x2 - x1), max(1, y2 - y1)
    ratio = w / h

    shoulder = _mid(kpts, L_SHOULDER, R_SHOULDER)
    hip = _mid(kpts, L_HIP, R_HIP)
    angle = None
    if shoulder is not None and hip is not None and np.linalg.norm(shoulder - hip) > 5:
        dx, dy = shoulder - hip
        angle = abs(math.degrees(math.atan2(dx, -dy)))  # 0 = upright, 90 = horizontal, 180 = head down

    visible = kpts[kpts[:, 2] >= KPT_MIN_SCORE][:, :2]
    spread_horizontal = None
    if len(visible) >= 5:
        spread_horizontal = bool(np.ptp(visible[:, 0]) > np.ptp(visible[:, 1]))

    votes = {
        "torso_angle": bool(angle is not None and angle > ANGLE_LYING_DEG),
        "box_ratio": bool(ratio >= BOX_RATIO_LYING),
        "body_spread": bool(spread_horizontal),
    }
    lying = bool(votes["torso_angle"] and (votes["box_ratio"] or votes["body_spread"]))
    # clearly upright: torso near vertical, or (no torso visible) a tall narrow box
    upright = bool((angle is not None and angle < ANGLE_UPRIGHT_DEG) or (angle is None and ratio < 0.8))

    center = hip if hip is not None else np.array([(x1 + x2) / 2, (y1 + y2) / 2])
    return {
        "lying": lying,
        "upright": upright,
        "votes": votes,
        "angle": None if angle is None else round(angle),
        "ratio": round(float(ratio), 2),
        "center": center,
        "conf": person["conf"],
        "height": h,
        "size": math.hypot(w, h),
    }


class FallMonitor:
    def __init__(self):
        self.last_ts = None
        self.reset()

    def reset(self):
        self.history = deque()          # (t, center_y, height, angle, lying, upright)
        self.lying_since = None         # smoothed "on the floor" start
        self.fall_motion = False        # a real fall movement was seen before lying
        self.last_drop_at = None
        self.still_since = None
        self.ref_kpts, self.ref_t, self.moving = None, None, False
        self.upright_since = None
        self.last_seen = None
        self.last_lying_seen = None
        self.alert_sent = False
        self.first_alert_at = None
        self.last_alert_at = None
        self.alert_count = 0
        self.reason = ""

    # --- helpers -------------------------------------------------------
    def _pick(self, postures):
        """Follow the main person: biggest, most confident box (lying people still have big boxes)."""
        return max(postures, key=lambda p: p["conf"] * p["size"])

    def _detect_fall_motion(self, now, cur):
        """Fast transition: upright/standing -> lying within FALL_WINDOW_SEC, or a big hip drop."""
        window = [hh for hh in self.history if now - hh[0] <= FALL_WINDOW_SEC and not hh[4]]
        # slow server (frames seconds apart): the frame right before lying also counts, up to 4 s back
        if self.history and not self.history[-1][4] and now - self.history[-1][0] <= 4.0:
            window.append(self.history[-1])
        if not window:
            return False
        if any(hh[5] for hh in window):          # was upright moments ago -> fell fast
            return True
        standing_height = max(hh[2] for hh in window)
        drop = cur["center"][1] - min(hh[1] for hh in window)
        return drop >= DROP_RATIO * standing_height

    def _update_motion(self, now, person, size):
        kpts = person["kpts"]
        if self.ref_kpts is None:
            self.ref_kpts, self.ref_t = kpts, now
            return
        if now - self.ref_t < MOVE_CHECK_SEC:
            return
        ok = (kpts[:, 2] >= KPT_MIN_SCORE) & (self.ref_kpts[:, 2] >= KPT_MIN_SCORE)
        if ok.sum() >= 3:
            shift = np.linalg.norm(kpts[ok, :2] - self.ref_kpts[ok, :2], axis=1).mean() / max(size, 1)
            self.moving = bool(shift > MOVE_THRESHOLD)
        self.ref_kpts, self.ref_t = kpts, now

    # --- main ----------------------------------------------------------
    def update(self, people, now=None):
        now = time.time() if now is None else now
        new_alert = False
        person = posture = None

        if people:
            postures = [analyze_posture(p) for p in people]
            posture = self._pick(postures)
            person = people[postures.index(posture)]
            self.last_seen = now

            if posture["lying"] and self._detect_fall_motion(now, posture):
                self.last_drop_at = now
            self.history.append((now, posture["center"][1], posture["height"], posture["angle"],
                                 posture["lying"], posture["upright"]))
            while self.history and now - self.history[0][0] > max(FALL_WINDOW_SEC, SMOOTH_SEC) + 0.5:
                self.history.popleft()

            recent = [hh[4] for hh in self.history if now - hh[0] <= SMOOTH_SEC]
            lying_now = len(recent) >= 1 and sum(recent) / len(recent) >= SMOOTH_FRACTION and posture["lying"]

            if lying_now:
                self.last_lying_seen = now
                self.upright_since = None
                if self.lying_since is None:
                    self.lying_since = now
                    self.still_since = now
                    self.fall_motion = self.last_drop_at is not None and now - self.last_drop_at <= 3
                elif not self.fall_motion and self.last_drop_at and now - self.last_drop_at <= 3:
                    self.fall_motion = True
                self._update_motion(now, person, posture["size"])
                if self.moving:
                    self.still_since = now
            elif posture["upright"]:
                self.ref_kpts, self.moving = None, False
                self.upright_since = self.upright_since or now
                if now - self.upright_since >= RECOVER_SEC:
                    self._clear_event()
        else:
            if self.lying_since and not self.alert_sent and self.last_lying_seen and now - self.last_lying_seen > GRACE_SEC:
                self._clear_event()
            if self.last_seen and now - self.last_seen > NOBODY_RESET_SEC:
                self.reset()

        # --- alert decisions ---
        on_floor = 0.0 if self.lying_since is None else now - self.lying_since
        still_for = 0.0 if self.still_since is None else now - self.still_since
        if self.lying_since is not None and not self.alert_sent:
            if self.fall_motion and still_for >= STILL_ALERT_SEC:
                self.reason = f"Fall detected, no movement for {int(still_for)} s"
                new_alert = True
            elif on_floor >= LONG_LIE_SEC:
                self.reason = f"Person on the floor for {int(on_floor)} s"
                new_alert = True
            if new_alert:
                self.alert_sent = True
                self.first_alert_at = now
                self.alert_count = 0
        elif self.alert_sent and self.lying_since is not None and now - self.last_alert_at >= REMIND_EVERY_SEC:
            new_alert = True
            mins, secs = divmod(int(on_floor), 60)
            self.reason = f"Person still down for {mins} min {secs} s"
        if new_alert:
            self.last_alert_at = now
            self.alert_count += 1

        if self.alert_sent:
            status = "EMERGENCY - alert sent"
        elif self.lying_since is not None:
            status = "Fall detected" if self.fall_motion else "Lying down"
        elif people:
            status = "Possible fall" if posture["lying"] else "Upright" if posture["upright"] else "Sitting / bending"
        else:
            status = "No person"

        return {
            "status": status,
            "new_alert": new_alert,
            "alert_sent": self.alert_sent,
            "alert_count": self.alert_count,
            "reason": self.reason,
            "fall_motion": self.fall_motion,
            "moving": self.moving if self.lying_since else None,
            "on_floor": round(on_floor, 1),
            "still_for": round(still_for, 1) if self.lying_since else 0.0,
            "still_alert_sec": STILL_ALERT_SEC,
            "long_lie_sec": LONG_LIE_SEC,
            "posture": None if posture is None else {
                "angle": posture["angle"], "ratio": posture["ratio"], "votes": posture["votes"],
                "lying": posture["lying"],
            },
        }

    def _clear_event(self):
        self.lying_since = None
        self.still_since = None
        self.fall_motion = False
        self.last_drop_at = None
        self.alert_sent = False
        self.ref_kpts, self.moving = None, False
