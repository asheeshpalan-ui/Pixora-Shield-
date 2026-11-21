import os
from pathlib import Path
import cv2
import numpy as np


class PlateDetector:
    def __init__(self, weights_path=None, conf_thresh=0.25):
        self.weights_path = weights_path or os.getenv("PLATE_MODEL_PATH")
        self.conf_thresh = conf_thresh
        self.model = None
        if self.weights_path and Path(self.weights_path).is_file():
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
            except Exception:
                self.model = None

    def process_frame(self, frame, frame_id):
        if self.model is None:
            return frame_id, self._fallback_plate_regions(frame)

        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            verbose=False,
        )
        rectangles = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box
                rectangles.append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ])
        return frame_id, rectangles

    def _fallback_plate_regions(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 220)
        edges = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        contours, _ = cv2.findContours(
            edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )

        candidates = []
        for contour in contours:
            x, y, box_w, box_h = cv2.boundingRect(contour)
            if box_w < 45 or box_h < 12:
                continue
            aspect = box_w / float(box_h)
            area = box_w * box_h
            if 1.8 <= aspect <= 6.5 and area >= 800:
                edge_region = edges[y:y + box_h, x:x + box_w]
                edge_density = float(cv2.countNonZero(edge_region)) / max(1, area)
                if edge_density >= 0.12:
                    padding = max(5, int(box_h * 0.20))
                    candidates.append([
                        max(0, x - padding),
                        max(0, y - padding),
                        min(w, x + box_w + padding),
                        min(h, y + box_h + padding),
                    ])

        candidates.sort(key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)
        unique = []
        for candidate in candidates:
            if not any(self._overlap(candidate, existing) > 0.45 for existing in unique):
                unique.append(candidate)
            if len(unique) >= 8:
                break
        return unique

    @staticmethod
    def _overlap(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[2], b[2]), min(a[3], b[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - intersection
        return intersection / union if union else 0.0
