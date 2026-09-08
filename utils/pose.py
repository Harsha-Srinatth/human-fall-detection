"""Pose estimation and fall detection via body angle."""

import math
import os
import time
import urllib.request

import cv2
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import drawing_styles, drawing_utils

from .config import FALL_ANGLE_THRESHOLD, POSE_MODEL_PATH, POSE_MODEL_URL

# Key landmarks for head, shoulders, elbows, hands, hips, knees, and feet.
BODY_LANDMARKS = {
    "Head": vision.PoseLandmark.NOSE,
    "L Shoulder": vision.PoseLandmark.LEFT_SHOULDER,
    "R Shoulder": vision.PoseLandmark.RIGHT_SHOULDER,
    "L Elbow": vision.PoseLandmark.LEFT_ELBOW,
    "R Elbow": vision.PoseLandmark.RIGHT_ELBOW,
    "L Hand": vision.PoseLandmark.LEFT_WRIST,
    "R Hand": vision.PoseLandmark.RIGHT_WRIST,
    "L Hip": vision.PoseLandmark.LEFT_HIP,
    "R Hip": vision.PoseLandmark.RIGHT_HIP,
    "L Knee": vision.PoseLandmark.LEFT_KNEE,
    "R Knee": vision.PoseLandmark.RIGHT_KNEE,
    "L Foot": vision.PoseLandmark.LEFT_ANKLE,
    "R Foot": vision.PoseLandmark.RIGHT_ANKLE,
}


def get_torso_angle(landmarks):
    """Calculate torso angle from vertical in degrees, or None if landmarks missing."""
    if not landmarks:
        return None

    left_shoulder = landmarks[vision.PoseLandmark.LEFT_SHOULDER]
    right_shoulder = landmarks[vision.PoseLandmark.RIGHT_SHOULDER]
    left_hip = landmarks[vision.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[vision.PoseLandmark.RIGHT_HIP]

    shoulder = (
        (left_shoulder.x + right_shoulder.x) / 2,
        (left_shoulder.y + right_shoulder.y) / 2,
    )
    hip = (
        (left_hip.x + right_hip.x) / 2,
        (left_hip.y + right_hip.y) / 2,
    )

    dx = shoulder[0] - hip[0]
    dy = shoulder[1] - hip[1]
    return abs(math.degrees(math.atan2(dx, dy)))


def _ensure_pose_model():
    """Download pose landmarker model if not present."""
    if os.path.exists(POSE_MODEL_PATH):
        return
    os.makedirs(os.path.dirname(POSE_MODEL_PATH), exist_ok=True)
    print(f"Downloading pose model to {POSE_MODEL_PATH}...")
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)


def _create_landmarker(running_mode):
    _ensure_pose_model()
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        min_pose_detection_confidence=0.5,
    )
    return vision.PoseLandmarker.create_from_options(options)


class PoseDetector:
    """Full-frame pose detection for live webcam skeleton drawing."""

    def __init__(self):
        self.landmarker = _create_landmarker(vision.RunningMode.VIDEO)
        self._start_time = time.time()

    def detect(self, frame, timestamp_ms=None):
        """Detect pose landmarks in a frame. Returns landmark list or None."""
        if timestamp_ms is None:
            timestamp_ms = int((time.time() - self._start_time) * 1000)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.pose_landmarks:
            return None
        return result.pose_landmarks[0]

    def draw_skeleton(self, frame, landmarks):
        """Draw pose skeleton and key body-part labels on the frame."""
        drawing_utils.draw_landmarks(
            frame,
            landmarks,
            vision.PoseLandmarksConnections.POSE_LANDMARKS,
            landmark_drawing_spec=drawing_styles.get_default_pose_landmarks_style(),
            connection_drawing_spec=drawing_utils.DrawingSpec(
                color=(0, 255, 0), thickness=2
            ),
        )

        height, width = frame.shape[:2]
        for name, index in BODY_LANDMARKS.items():
            landmark = landmarks[index]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
            cv2.putText(
                frame,
                name,
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
            )

    def close(self):
        self.landmarker.close()


class PoseEstimator:
    def __init__(self):
        self.landmarker = _create_landmarker(vision.RunningMode.IMAGE)

    def get_angle(self, frame, bbox):
        """Calculate torso angle from vertical. Returns angle in degrees or None."""
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_image)
        if not result.pose_landmarks:
            return None

        landmarks = result.pose_landmarks[0]
        return get_torso_angle(landmarks)

    def is_fallen(self, angle):
        """Return True if body angle indicates a fall."""
        return angle is not None and angle > FALL_ANGLE_THRESHOLD

    def close(self):
        self.landmarker.close()
