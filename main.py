"""Phases 5–8: Fall detection, motion timer, evidence saving, and emergency alert."""

from collections import deque

import cv2

from utils.alert import print_emergency, save_evidence
from utils.config import CAMERA_ID, CLIP_BUFFER_SECONDS, EMERGENCY_TIMEOUT_SEC
from utils.detector import HumanDetector
from utils.fall import get_status
from utils.motion import FallTimer, MotionAnalyzer


def draw_detection(frame, x1, y1, x2, y2, confidence, timer: FallTimer | None = None):
    """Draw bounding box, confidence, fall status, and optional timer."""
    status, fallen = get_status(x1, y1, x2, y2)
    color = (0, 0, 255) if fallen else (0, 255, 0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(
        frame,
        f"Person {confidence:.2f}",
        (x1, y1 - 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )
    cv2.putText(
        frame,
        status,
        (x1, y1 - 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    if timer and timer.is_active:
        elapsed = timer.elapsed()
        cv2.putText(
            frame,
            f"Timer: {elapsed:.1f}s / {EMERGENCY_TIMEOUT_SEC}s",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
        )

    return fallen


def main():
    detector = HumanDetector()
    motion = MotionAnalyzer()
    fall_timer = FallTimer()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Check that a camera is connected.")

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 20
    frame_buffer = deque(maxlen=fps * CLIP_BUFFER_SECONDS)

    print("Fall detection started. Press Q to quit.")
    print(f"Emergency triggers after {EMERGENCY_TIMEOUT_SEC}s of no movement while fallen.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame from webcam.")
            break

        frame_buffer.append(frame.copy())
        detections = detector.detect(frame)
        fallen_bbox = None
        any_fallen = False

        for x1, y1, x2, y2, confidence in detections:
            fallen = draw_detection(frame, x1, y1, x2, y2, confidence, fall_timer)
            if fallen:
                any_fallen = True
                fallen_bbox = (x1, y1, x2, y2)

        if any_fallen and fallen_bbox:
            if motion.is_person_moving(frame, fallen_bbox):
                fall_timer.reset()
            else:
                fall_timer.start()
                if fall_timer.check_emergency():
                    now = save_evidence(frame, list(frame_buffer), CAMERA_ID, fps)
                    print_emergency(CAMERA_ID)
                    print(f"Evidence saved to: {now}")
        else:
            fall_timer.reset()
            motion.reset()

        cv2.imshow("Human Fall Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed.")


if __name__ == "__main__":
    main()
