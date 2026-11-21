import cv2
import numpy as np

from src.models.detection.blur_utils import RegionTracker, box_iou
from src.models.detection.plate_blur.plate_detector import PlateDetector


def test_region_tracker_keeps_short_detection_gaps():
    tracker = RegionTracker(max_misses=3)
    first = tracker.update([[10, 10, 50, 50]])
    held = tracker.update([])

    assert first == [[10, 10, 50, 50]]
    assert held == [[10, 10, 50, 50]]


def test_region_tracker_updates_matching_box():
    tracker = RegionTracker(max_misses=3)
    tracker.update([[10, 10, 50, 50]])
    updated = tracker.update([[12, 12, 52, 52]])

    assert updated == [[12, 12, 52, 52]]
    assert box_iou([10, 10, 50, 50], [12, 12, 52, 52]) > 0.8


def test_fallback_plate_detector_finds_rectangular_plate():
    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    cv2.rectangle(frame, (240, 300), (390, 350), (20, 20, 20), -1)
    cv2.rectangle(frame, (250, 310), (380, 340), (220, 220, 220), 2)

    _, regions = PlateDetector().process_frame(frame, 0)

    assert regions
