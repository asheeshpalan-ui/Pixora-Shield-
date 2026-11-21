import os
import base64
from pathlib import Path
import json
from urllib import request as urllib_request
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode


class PlateDetector:
    def __init__(self, weights_path=None, conf_thresh=0.25):
        self.weights_path = weights_path or os.getenv("PLATE_MODEL_PATH")
        self.conf_thresh = conf_thresh
        self.model = None
        self.roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")
        self.roboflow_model_id = os.getenv("ROBOFLOW_MODEL_ID")
        self.roboflow_version = os.getenv("ROBOFLOW_VERSION", "")
        if self.weights_path and Path(self.weights_path).is_file():
            try:
                from ultralytics import YOLO
                self.model = YOLO(self.weights_path)
            except Exception:
                self.model = None

    @property
    def is_available(self):
        return self.model is not None or bool(
            self.roboflow_api_key and self.roboflow_model_id
        )

    def process_frame(self, frame, frame_id):
        if self.model is not None:
            return frame_id, self._local_inference(frame)
        if self.roboflow_api_key and self.roboflow_model_id:
            return frame_id, self._roboflow_inference(frame)
        return frame_id, []

    def _local_inference(self, frame):
        rectangles = []
        results = self.model.predict(
            source=frame,
            conf=self.conf_thresh,
            verbose=False,
        )
        for result in results:
            if result.boxes is None:
                continue
            for box, confidence in zip(
                result.boxes.xyxy.cpu().numpy(),
                result.boxes.conf.cpu().numpy(),
            ):
                if float(confidence) < self.conf_thresh:
                    continue
                x1, y1, x2, y2 = box
                rectangles.append([
                    int(x1),
                    int(y1),
                    int(x2),
                    int(y2),
                ])
        return rectangles

    def _roboflow_inference(self, frame):
        import cv2

        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            return []

        model_id = self.roboflow_model_id
        version = f"/{self.roboflow_version}" if self.roboflow_version else ""
        url = f"https://detect.roboflow.com/{model_id}{version}"
        try:
            query = urlencode({
                "api_key": self.roboflow_api_key,
                "confidence": int(self.conf_thresh * 100),
            })
            http_request = urllib_request.Request(
                f"{url}?{query}",
                data=base64.b64encode(encoded.tobytes()),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            with urllib_request.urlopen(http_request, timeout=15) as response:
                predictions = json.loads(response.read().decode("utf-8")).get(
                    "predictions", []
                )
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError):
            return []

        h, w = frame.shape[:2]
        rectangles = []
        for prediction in predictions:
            confidence = float(prediction.get("confidence", 0))
            if confidence < self.conf_thresh:
                continue
            cx = float(prediction["x"])
            cy = float(prediction["y"])
            box_w = float(prediction["width"])
            box_h = float(prediction["height"])
            x1 = max(0, int(cx - box_w / 2))
            y1 = max(0, int(cy - box_h / 2))
            x2 = min(w, int(cx + box_w / 2))
            y2 = min(h, int(cy + box_h / 2))
            rectangles.append([x1, y1, x2, y2])
        return rectangles
