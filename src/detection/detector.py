import numpy as np
from ultralytics import YOLO
from .detection import Detection


class Detector:
    """YOLOv8-based object detector."""

    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5,
                 iou_threshold=0.45, classes=None, device=None):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes
        self.device = device

    def detect(self, frame):
        """Return list of Detection objects for a single frame."""
        kwargs = dict(conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
        if self.classes is not None:
            kwargs["classes"] = self.classes
        if self.device is not None:
            kwargs["device"] = self.device

        results = self.model(frame, **kwargs)
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                tlwh = np.array([x1, y1, x2 - x1, y2 - y1])
                detections.append(
                    Detection(tlwh, float(box.conf[0]), int(box.cls[0]))
                )
        return detections
