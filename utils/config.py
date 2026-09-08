"""Configuration settings for Human Fall Detection."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
POSE_MODEL_PATH = os.path.join(BASE_DIR, "models", "pose_landmarker_lite.task")
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)
VIDEOS_DIR = os.path.join(BASE_DIR, "videos")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
ALERTS_DIR = os.path.join(BASE_DIR, "alerts")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

CONFIDENCE_THRESHOLD = 0.5
FALL_ANGLE_THRESHOLD = 60  # degrees from vertical
MOTION_THRESHOLD = 0.3
EMERGENCY_TIMEOUT_SEC = 20
CLIP_BUFFER_SECONDS = 10
CAMERA_ID = 1
