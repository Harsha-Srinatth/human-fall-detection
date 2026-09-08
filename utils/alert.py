"""Alert handling — evidence saving and emergency console output."""

import os
from datetime import datetime

import cv2
import numpy as np

from .config import ALERTS_DIR, CAMERA_ID


def print_emergency(camera_id: int = CAMERA_ID, when: datetime | None = None):
    """Print emergency alert to console."""
    now = when or datetime.now()
    print()
    print("*" * 15)
    print("EMERGENCY")
    print("*" * 15)
    print()
    print("Person Fallen")
    print()
    print(f"Camera : {camera_id}")
    print(f"Time   : {now.strftime('%H:%M:%S')}")
    print(f"Date   : {now.strftime('%d-%m-%Y')}")
    print()


def _write_video_clip(clip_path: str, frames_buffer, fps: float) -> str | None:
    """Write a video clip using a Windows-friendly codec."""
    if not frames_buffer:
        return False

    height, width = frames_buffer[0].shape[:2]
    fps = max(float(fps), 1.0)

    # MJPG inside AVI plays reliably in Windows Media Player and VS Code preview.
    codecs = [
        ("MJPG", ".avi"),
        ("XVID", ".avi"),
        ("mp4v", ".mp4"),
    ]

    for fourcc_name, extension in codecs:
        path = os.path.splitext(clip_path)[0] + extension
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*fourcc_name),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            continue

        for frame in frames_buffer:
            writer.write(np.ascontiguousarray(frame))
        writer.release()

        if os.path.exists(path) and os.path.getsize(path) > 0:
            return path

    return None


def save_evidence(frame, frames_buffer, camera_id: int = CAMERA_ID, fps: float = 20):
    """Save snapshot, video clip, and metadata inside alerts/."""
    now = datetime.now()
    folder_name = now.strftime("%Y%m%d_%H%M%S")
    alert_folder = os.path.join(ALERTS_DIR, folder_name)
    os.makedirs(alert_folder, exist_ok=True)

    image_path = os.path.join(alert_folder, "snapshot.jpg")
    clip_path = os.path.join(alert_folder, "clip.mp4")
    details_path = os.path.join(alert_folder, "details.txt")

    cv2.imwrite(image_path, frame)

    saved_clip = _write_video_clip(clip_path, frames_buffer, fps)

    with open(details_path, "w", encoding="utf-8") as file:
        file.write(f"Time: {now.strftime('%H:%M:%S')}\n")
        file.write(f"Date: {now.strftime('%d-%m-%Y')}\n")
        file.write(f"Camera: {camera_id}\n")
        file.write(f"Image: {image_path}\n")
        file.write(f"Video: {saved_clip or 'not saved'}\n")

    return alert_folder
