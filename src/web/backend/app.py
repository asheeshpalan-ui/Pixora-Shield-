import os
import time
import uuid
from pathlib import Path

import cv2
from flask import Flask, jsonify, send_from_directory
from flask import request
from flask_cors import CORS

from src.models.detection.blur_utils import (
    RegionTracker,
    apply_blur_regions,
    box_area,
    box_iou,
    box_center,
)
from src.models.detection.plate_blur.plate_detector import PlateDetector


BASE_DIR = Path(__file__).resolve().parents[3]
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(__name__)
CORS(app, origins=os.getenv("CORS_ALLOWED_ORIGINS", "http://localhost:3000").split(","))

def find_face_cascade():
    candidates = []
    candidates.append(
        str(BASE_DIR / "models" / "haarcascade_frontalface_default.xml")
    )
    if hasattr(cv2, "data"):
        candidates.append(
            str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        )
    candidates.extend([
        "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
    ])
    return next((path for path in candidates if Path(path).is_file()), "")


face_cascade_path = find_face_cascade()
try:
    face_detector = (
        cv2.CascadeClassifier(face_cascade_path)
        if face_cascade_path and hasattr(cv2, "CascadeClassifier")
        else None
    )
except cv2.error:
    face_detector = None
plate_detector = PlateDetector(conf_thresh=float(os.getenv("PLATE_CONF_THRESH", "0.25")))


def detect_faces(frame):
    if face_detector is None or face_detector.empty():
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.05,
        minNeighbors=3,
        minSize=(20, 20),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [[int(x), int(y), int(x + w), int(y + h)] for (x, y, w, h) in faces]


def choose_primary_face(face_regions, previous_primary, missing_frames):
    if not face_regions:
        return previous_primary, missing_frames + 1

    if previous_primary is None:
        return max(face_regions, key=box_area), 0

    previous_center = box_center(previous_primary)

    def tracking_score(face):
        iou = box_iou(previous_primary, face)
        center = box_center(face)
        distance = (
            (previous_center[0] - center[0]) ** 2
            + (previous_center[1] - center[1]) ** 2
        ) ** 0.5
        return iou * 100 - distance + box_area(face) / 100000

    best_face = max(face_regions, key=tracking_score)
    return best_face, 0


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "face_detection": bool(face_detector is not None and not face_detector.empty()),
        "plate_detection": plate_detector.is_available,
    })


@app.post("/api/process")
def process_video():
    video = request.files.get("video")
    if video is None or not video.filename:
        return jsonify({"error": "A video file is required"}), 400

    protect_main_face = request.form.get(
        "protect_main_face", "false"
    ).strip().lower() in {"1", "true", "yes", "on"}

    if not video.filename.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return jsonify({"error": "Unsupported video format"}), 400

    job_id = uuid.uuid4().hex
    input_path = UPLOAD_DIR / f"{job_id}_{video.filename}"
    output_name = f"{job_id}_protected.mp4"
    output_path = OUTPUT_DIR / output_name
    video.save(input_path)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "Could not open the uploaded video"}), 400

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        input_path.unlink(missing_ok=True)
        return jsonify({"error": "Could not create the output video"}), 500

    started = time.perf_counter()
    processed_frames = 0
    detected_regions = 0
    frame_id = 0
    primary_face = None
    primary_missing_frames = 0
    face_tracker = RegionTracker(max_misses=10)
    plate_tracker = RegionTracker(max_misses=12)
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        detected_faces = detect_faces(frame)
        if protect_main_face:
            primary_face, primary_missing_frames = choose_primary_face(
                detected_faces, primary_face, primary_missing_frames
            )
            secondary_faces = [
                face for face in detected_faces
                if primary_face is None or box_iou(primary_face, face) < 0.35
            ]
            face_regions = face_tracker.update(secondary_faces)
            face_regions = [
                face for face in face_regions
                if box_iou(primary_face, face) < 0.35
            ]
        else:
            primary_face = None
            face_regions = face_tracker.update(detected_faces)
        _, plate_regions = plate_detector.process_frame(frame, frame_id)
        plate_regions = plate_tracker.update(plate_regions)
        plate_regions = [
            plate for plate in plate_regions
            if primary_face is None or box_iou(primary_face, plate) < 0.35
        ]
        regions = face_regions + plate_regions
        detected_regions += len(regions)
        processed = apply_blur_regions(frame, regions)
        writer.write(processed)
        processed_frames += 1
        frame_id += 1

    capture.release()
    writer.release()

    elapsed = time.perf_counter() - started
    original_url = f"/api/original/{input_path.name}"
    processed_url = f"/api/output/{output_name}"
    return jsonify({
        "job_id": job_id,
        "original_url": original_url,
        "processed_url": processed_url,
        "metrics": {
            "frames": processed_frames,
            "reported_frames": total_frames,
            "faces_and_plates_detected": detected_regions,
            "processing_seconds": round(elapsed, 2),
            "average_fps": round(processed_frames / elapsed, 2) if elapsed else 0,
            "resolution": f"{width}x{height}",
            "plate_detection_enabled": plate_detector.model is not None,
        },
    })


@app.get("/api/original/<path:filename>")
def serve_original(filename):
    return send_from_directory(UPLOAD_DIR, filename)


@app.get("/api/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.get("/api/download/<path:filename>")
def download_output(filename):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
