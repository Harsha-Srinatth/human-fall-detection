"""YOLOv8-based human detection."""

from ultralytics import YOLO

from .config import CONFIDENCE_THRESHOLD, MODEL_PATH


class HumanDetector:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model = YOLO(model_path)

    def detect(self, frame):
        """Detect humans in a frame. Returns list of (x1, y1, x2, y2, confidence)."""
        results = self.model(frame, conf=CONFIDENCE_THRESHOLD, classes=[0], verbose=False)
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                boxes.append((x1, y1, x2, y2, confidence))
        return boxes
