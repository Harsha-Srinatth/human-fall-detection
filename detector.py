"""YOLOv8n-pose person + 17 keypoint detection using OpenCV DNN (no PyTorch needed)."""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8n-pose.onnx")
INPUT_SIZE = 640
CONF_THRESHOLD = 0.3  # people lying on the floor get lower scores
NMS_THRESHOLD = 0.45

# COCO keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_HIP, R_HIP = 5, 6, 11, 12
L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 13, 14, 15, 16

SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]


class PersonDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def detect(self, frame):
        """Return list of dicts: {box: (x1,y1,x2,y2), conf, kpts: (17,3) array [x, y, score]}."""
        h, w = frame.shape[:2]
        size = max(h, w)
        square = np.zeros((size, size, 3), dtype=np.uint8)
        square[:h, :w] = frame
        scale = size / INPUT_SIZE

        blob = cv2.dnn.blobFromImage(square, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # (8400, 56): cx, cy, w, h, conf, 17 x (x, y, score)

        out = out[out[:, 4] > CONF_THRESHOLD]
        if len(out) == 0:
            return []

        boxes = [[int((cx - bw / 2) * scale), int((cy - bh / 2) * scale), int(bw * scale), int(bh * scale)]
                 for cx, cy, bw, bh in out[:, :4]]
        idxs = cv2.dnn.NMSBoxes(boxes, out[:, 4].tolist(), CONF_THRESHOLD, NMS_THRESHOLD)

        people = []
        for i in np.array(idxs).flatten():
            x, y, bw, bh = boxes[i]
            kpts = out[i, 5:].reshape(17, 3).copy()
            kpts[:, :2] *= scale
            people.append({
                "box": (max(0, x), max(0, y), min(w, x + bw), min(h, y + bh)),
                "conf": float(out[i, 4]),
                "kpts": kpts,
            })
        return people
