from src.models.detection.blur_utils import RegionTracker, box_iou
from src.models.detection.plate_blur.plate_detector import PlateDetector
from src.web.backend.app import choose_primary_face


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


def test_unconfigured_plate_detector_blurs_nothing():
    import numpy as np

    frame = np.full((480, 640, 3), 255, dtype=np.uint8)
    _, regions = PlateDetector().process_frame(frame, 0)
    assert regions == []


def test_primary_face_is_selected_and_not_treated_as_secondary():
    primary, misses = choose_primary_face(
        [[100, 100, 200, 200], [300, 120, 370, 190]],
        None,
        0,
    )

    assert primary == [100, 100, 200, 200]
    assert misses == 0
