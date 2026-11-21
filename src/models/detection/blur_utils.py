import cv2
import numpy as np


def clamp_region(frame_shape, region):
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = (int(value) for value in region)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def apply_blur_regions(frame, rectangles, kernel_size=35):
    result = frame.copy()
    odd_kernel = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    for region in rectangles:
        box = clamp_region(frame.shape, region)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        roi = result[y1:y2, x1:x2]
        if roi.size:
            result[y1:y2, x1:x2] = cv2.GaussianBlur(
                roi, (odd_kernel, odd_kernel), 0
            )
    return result


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def box_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union else 0.0


def box_center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


class RegionTracker:
    """Keep detections visible for a few frames when a detector misses."""

    def __init__(self, max_misses=8, iou_threshold=0.25):
        self.max_misses = max_misses
        self.iou_threshold = iou_threshold
        self.tracks = []

    def update(self, detections):
        unmatched = list(range(len(detections)))
        updated = []

        for box, misses in self.tracks:
            best = None
            best_iou = self.iou_threshold
            for index in unmatched:
                iou = box_iou(box, detections[index])
                if iou > best_iou:
                    best_iou = iou
                    best = index
            if best is not None:
                current = detections[best]
                unmatched.remove(best)
                updated.append((current, 0))
            elif misses + 1 < self.max_misses:
                updated.append((box, misses + 1))

        for index in unmatched:
            updated.append((detections[index], 0))

        self.tracks = updated
        return [box for box, _ in updated]
