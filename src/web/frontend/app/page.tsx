"use client";

import { ChangeEvent, DragEvent, useMemo, useState } from "react";

type Metrics = {
  frames: number;
  reported_frames: number;
  faces_and_plates_detected: number;
  processing_seconds: number;
  average_fps: number;
  resolution: string;
  plate_detection_enabled: boolean;
  face_models?: {
    yunet: boolean;
    cascades: number;
  };
};

type Result = {
  original_url: string;
  processed_url: string;
  metrics: Metrics;
};

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:5000";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [protectMainFace, setProtectMainFace] = useState(false);

  const sourceUrl = useMemo(
    () => (file ? URL.createObjectURL(file) : ""),
    [file],
  );

  function chooseFile(event: ChangeEvent<HTMLInputElement>) {
    const selected = event.target.files?.[0] || null;
    setFile(selected);
    setResult(null);
    setError("");
  }

  function dropFile(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragging(false);
    const selected = event.dataTransfer.files?.[0] || null;
    if (selected) {
      setFile(selected);
      setResult(null);
      setError("");
    }
  }

  async function processVideo() {
    if (!file) return;
    setBusy(true);
    setError("");
    setResult(null);
    try {
      const form = new FormData();
      form.append("video", file);
      form.append("protect_main_face", protectMainFace ? "true" : "false");
      const response = await fetch(`${API_URL}/api/process`, {
        method: "POST",
        body: form,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Processing failed");
      setResult({
        original_url: `${API_URL}${data.original_url}`,
        processed_url: `${API_URL}${data.processed_url}`,
        metrics: data.metrics,
      });
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Processing failed");
    } finally {
      setBusy(false);
    }
  }

  return (
    <main>
      <nav>
        <div className="brand"><span className="brandMark">P</span> PIXORA SHIELD</div>
        <span className="status"><i /> Local privacy engine</span>
      </nav>

      <section className="hero">
        <p className="eyebrow">VIDEO PRIVACY MVP</p>
        <h1>Protect every face.<br /><em>Keep every moment.</em></h1>
        <p className="lede">
          Upload a video and Pixora Shield removes identifiable faces and, when
          configured, license plates before you share it.
        </p>
      </section>

      <section className="workspace">
        <div className="uploadPanel">
          <label
            className={`dropzone ${dragging ? "dragging" : ""}`}
            onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={dropFile}
          >
            <input type="file" accept="video/*" onChange={chooseFile} />
            <div className="uploadIcon">↑</div>
            <strong>{file ? file.name : "Choose a video"}</strong>
            <span>{file ? `${(file.size / 1024 / 1024).toFixed(1)} MB` : "MP4, MOV, AVI, MKV or WebM"}</span>
          </label>
          <label className="protectOption">
            <input
              type="checkbox"
              checked={protectMainFace}
              onChange={(event) => setProtectMainFace(event.target.checked)}
            />
            <span>
              <strong>Protect the main person</strong>
              <small>Blur other faces instead</small>
            </span>
          </label>
          <button
            className="processButton"
            disabled={!file || busy}
            onClick={processVideo}
          >
            {busy ? <><span className="spinner" /> Processing video</> : "Blur & protect video"}
          </button>
          {error && <p className="error">{error}</p>}
        </div>

        {result ? (
          <div className="results">
            <div className="resultHeading">
              <div>
                <p className="eyebrow">PROCESSING COMPLETE</p>
                <h2>Your protected video</h2>
              </div>
              <a
                className="download"
                href={`${API_URL}/api/download/${result.processed_url.split("/").pop()}`}
                download
              >
                Download MP4
              </a>
            </div>

            <div className="comparison">
              <div>
                <span>BEFORE</span>
                <video src={sourceUrl} controls muted />
              </div>
              <div>
                <span>PROTECTED</span>
                <video src={result.processed_url} controls muted />
              </div>
            </div>

            <div className="metrics">
              <Metric value={String(result.metrics.frames)} label="Frames processed" />
              <Metric
                value={String(result.metrics.faces_and_plates_detected)}
                label="Privacy regions found"
              />
              <Metric value={`${result.metrics.average_fps} fps`} label="Average processing" />
              <Metric value={result.metrics.resolution} label="Resolution" />
            </div>

            {!result.metrics.plate_detection_enabled && (
              <p className="note">
                Face protection is active. Plate protection becomes active when
                <code> PLATE_MODEL_PATH</code> is configured.
              </p>
            )}
          </div>
        ) : (
          <div className="privacyCard">
            <p className="eyebrow">PRIVACY BY DESIGN</p>
            <h2>Nothing leaves the demo environment.</h2>
            <p>
              Uploaded and processed videos are served only from the local API.
              Keep files on this machine and remove them after the demo.
            </p>
            <ol>
              <li><span>01</span><div><strong>Detect</strong><small>Faces and optional plate regions</small></div></li>
              <li><span>02</span><div><strong>Blur</strong><small>Gaussian privacy masks</small></div></li>
              <li><span>03</span><div><strong>Export</strong><small>Download a processed MP4</small></div></li>
            </ol>
          </div>
        )}
      </section>
    </main>
  );
}

function Metric({ value, label }: { value: string; label: string }) {
  return <div><strong>{value}</strong><span>{label}</span></div>;
}
