# Pixora Shield

Pixora Shield is a focused privacy MVP that uploads a video, detects faces, and
optionally detects license plates, then produces an MP4 with those regions
blurred.

The largest face first detected in a video is treated as the main subject and
is kept visible. Other detected faces are blurred. Short detector misses are
covered by frame-to-frame tracking. License-plate detection is enhanced by a
YOLO model when `PLATE_MODEL_PATH` is configured; without one, a built-in
rectangular-region fallback remains active.

## Run locally

Backend:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.web.backend.run
```

Frontend:

```bash
cd src/web/frontend
npm install
npm run dev
```

Open `http://localhost:3000`. The API runs at `http://localhost:5000`.


