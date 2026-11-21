# Pixora Shield

Pixora Shield is a focused privacy MVP that uploads a video, detects faces, and
optionally detects license plates, then produces an MP4 with those regions
blurred.

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

## Optional license-plate model

Set `PLATE_MODEL_PATH` to a YOLO `.pt` model:

```bash
pip install ultralytics torch
PLATE_MODEL_PATH=/path/to/plate_best.pt python -m src.web.backend.run
```
