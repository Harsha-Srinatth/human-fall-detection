"""YOLOv8n person detection using OpenCV DNN (no PyTorch needed)."""

import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "yolov8n.onnx")
INPUT_SIZE = 640
CONF_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45


class PersonDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def detect(self, frame):
        """Return list of (x1, y1, x2, y2, confidence) for people in the frame."""
        h, w = frame.shape[:2]
        size = max(h, w)
        square = np.zeros((size, size, 3), dtype=np.uint8)
        square[:h, :w] = frame
        scale = size / INPUT_SIZE

        blob = cv2.dnn.blobFromImage(square, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True)
        self.net.setInput(blob)
        out = self.net.forward()[0].T  # (8400, 84): cx, cy, w, h, 80 class scores

        scores = out[:, 4]  # class 0 = person
        keep = scores > CONF_THRESHOLD
        out, scores = out[keep], scores[keep]
        if len(out) == 0:
            return []

        boxes = []
        for cx, cy, bw, bh in out[:, :4]:
            boxes.append([int((cx - bw / 2) * scale), int((cy - bh / 2) * scale),
                          int(bw * scale), int(bh * scale)])

        idxs = cv2.dnn.NMSBoxes(boxes, scores.tolist(), CONF_THRESHOLD, NMS_THRESHOLD)
        results = []
        for i in np.array(idxs).flatten():
            x, y, bw, bh = boxes[i]
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(w, x + bw), min(h, y + bh)
            results.append((x1, y1, x2, y2, float(scores[i])))
        return results
