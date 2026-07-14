import os
import time
import signal
import asyncio
import subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, PlainTextResponse, StreamingResponse

PROJECT = Path("/workspace/fyp")
PIPELINE_SCRIPT = PROJECT / "final_triton_preserved_pipeline.py"

INPUT_DIR = PROJECT / "input"
UPLOAD_DIR = PROJECT / "uploads"
OUTPUT_ROOT = PROJECT / "output"
LOG_DIR = PROJECT / "logs"

CANONICAL_VIDEO = INPUT_DIR / "vid10min.mp4"
OUT_DIR = OUTPUT_ROOT / "FINAL_3MODEL_UNIFIED_FP16_YOLOPX_API"
RUN_LOG = LOG_DIR / "fastapi_pipeline_run.log"

LIVE_PREVIEW_DIR = OUTPUT_ROOT / "live_preview"
LIVE_PREVIEW_FRAME = LIVE_PREVIEW_DIR / "latest_frame.jpg"

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")

for d in [INPUT_DIR, UPLOAD_DIR, OUT_DIR, LOG_DIR, LIVE_PREVIEW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Three-Model Video Analysis Dashboard")

current_proc: Optional[subprocess.Popen] = None
last_started_at: Optional[float] = None


def proc_running():
    return current_proc is not None and current_proc.poll() is None


def safe_video_paths():
    roots = [INPUT_DIR, UPLOAD_DIR, PROJECT / "assets"]
    vids = []

    for root in roots:
        if root.exists():
            for ext in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
                vids.extend(root.rglob(ext))

    seen = set()
    out = []
    for p in vids:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            out.append(rp)

    return sorted(out)


def assert_safe_video(path_str: str) -> Path:
    p = Path(path_str).resolve()

    allowed_roots = [
        INPUT_DIR.resolve(),
        UPLOAD_DIR.resolve(),
        (PROJECT / "assets").resolve(),
    ]

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=400, detail=f"Video not found: {p}")

    if p.suffix.lower() not in [".mp4", ".avi", ".mov", ".mkv"]:
        raise HTTPException(status_code=400, detail="Please select a supported video file.")

    if not any(str(p).startswith(str(root)) for root in allowed_roots):
        raise HTTPException(status_code=400, detail="Video path is outside the allowed project folders.")

    return p


def clear_live_preview():
    LIVE_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    for p in LIVE_PREVIEW_DIR.glob("*"):
        if p.is_file():
            try:
                p.unlink()
            except Exception:
                pass


def clean_log_text(text: str) -> str:
    replacements = {
        "yolopx_int8_384x640.engine": "yolopx_fp16_384x640.engine",
        "YOLOPX   :": "YOLOPX FP16:",
        "da_seg": "",
        "DA-V2": "metric depth",
        "DA V2": "metric depth",
        "drivable area": "auxiliary segmentation",
        "Drivable area": "Auxiliary segmentation",
        "preserved final pipeline": "video analysis pipeline",
        "preserved pipeline": "video analysis pipeline",
        "locked": "configured",
        "Locked": "Configured",
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = text.replace("['det_out', '', 'll_seg']", "['det_out', 'll_seg']")
    text = text.replace(", ,", ",")
    return text


def latest_files():
    files = []

    if OUT_DIR.exists():
        for p in sorted(OUT_DIR.glob("*"), key=lambda x: x.stat().st_mtime, reverse=True):
            if p.is_file():
                files.append({
                    "name": p.name,
                    "path": str(p),
                    "size_mb": round(p.stat().st_size / (1024 * 1024), 2),
                })

    return files


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Three-Model Video Analysis</title>
<style>
:root{
  --bg:#07111f;
  --card:#0f1b2d;
  --card2:#14233a;
  --line:#263854;
  --text:#eaf1ff;
  --muted:#9fb1ce;
  --blue:#4f8cff;
  --green:#22c55e;
  --red:#ef4444;
  --yellow:#f59e0b;
}
*{box-sizing:border-box}
body{
  margin:0;
  background:radial-gradient(circle at top left,#14233a 0,#07111f 42%,#050b14 100%);
  color:var(--text);
  font-family:Arial,Helvetica,sans-serif;
}
.wrap{max-width:1500px;margin:0 auto;padding:26px}
.header{
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  gap:20px;
  margin-bottom:20px;
}
.title h1{margin:0;font-size:30px;letter-spacing:.2px}
.title p{margin:8px 0 0;color:var(--muted);font-size:15px;line-height:1.5}
.badges{display:flex;gap:10px;flex-wrap:wrap;justify-content:flex-end}
.badge{
  padding:9px 12px;
  background:rgba(79,140,255,.12);
  border:1px solid rgba(79,140,255,.35);
  border-radius:999px;
  color:#cfe0ff;
  font-size:13px;
  font-weight:700;
}
.grid{
  display:grid;
  grid-template-columns:420px 1fr;
  gap:20px;
}
.card{
  background:rgba(15,27,45,.92);
  border:1px solid var(--line);
  border-radius:18px;
  box-shadow:0 18px 40px rgba(0,0,0,.28);
  padding:18px;
  margin-bottom:18px;
}
.card h2{margin:0 0 12px;font-size:19px}
.help{color:var(--muted);font-size:13px;line-height:1.45;margin-bottom:12px}
label{display:block;color:var(--muted);font-size:13px;margin:12px 0 6px}
select,input[type=file]{
  width:100%;
  padding:12px;
  border-radius:12px;
  border:1px solid var(--line);
  background:#091426;
  color:var(--text);
}
.row{display:flex;gap:10px;flex-wrap:wrap;margin-top:12px}
button{
  border:0;
  border-radius:12px;
  padding:11px 14px;
  font-weight:800;
  color:white;
  cursor:pointer;
  transition:.15s ease;
}
button:hover{transform:translateY(-1px)}
.primary{background:var(--blue)}
.success{background:var(--green);color:#04130a}
.danger{background:var(--red)}
.neutral{background:#2b3d5b}
.status{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:9px 12px;
  border-radius:999px;
  font-weight:800;
  margin-bottom:12px;
  font-size:13px;
}
.status.idle{background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.35);color:#ffe1a3}
.status.run{background:rgba(34,197,94,.12);border:1px solid rgba(34,197,94,.35);color:#bdf7d1}
pre{
  margin:0;
  background:#050b14;
  border:1px solid var(--line);
  border-radius:14px;
  padding:14px;
  color:#dbe7ff;
  max-height:430px;
  overflow:auto;
  white-space:pre-wrap;
  word-break:break-word;
  font-size:12.5px;
}
.player{
  background:#02060d;
  border:1px solid var(--line);
  border-radius:18px;
  padding:12px;
}
#livePreview{
  width:100%;
  max-height:650px;
  object-fit:contain;
  background:#000;
  border-radius:14px;
  display:block;
}
.output a{
  display:block;
  text-decoration:none;
  color:#eaf1ff;
  padding:12px 14px;
  background:#14233a;
  border:1px solid var(--line);
  border-radius:12px;
  margin-bottom:9px;
}
.small{font-size:12px;color:var(--muted);margin-top:8px}
.kpis{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px}
.kpi{background:rgba(20,35,58,.9);border:1px solid var(--line);border-radius:16px;padding:14px}
.kpi .label{font-size:12px;color:var(--muted)}
.kpi .value{font-size:18px;font-weight:900;margin-top:6px}
@media(max-width:1100px){
  .grid{grid-template-columns:1fr}
  .header{display:block}
  .badges{justify-content:flex-start;margin-top:14px}
}
</style>
</head>
<body>
<div class="wrap">
  <div class="header">
    <div class="title">
      <h1>Three-Model Video Analysis Dashboard</h1>
      <p>
        Select a video, run the Triton inference pipeline, monitor progress, and review the processed output preview in real time.
      </p>
    </div>
    <div class="badges">
      <div class="badge">YOLOPX FP16</div>
      <div class="badge">Metric Depth FP16</div>
      <div class="badge">Traffic Detector FP16</div>
      <div class="badge">Triton + FastAPI</div>
    </div>
  </div>

  <div class="kpis">
    <div class="kpi"><div class="label">Serving</div><div class="value">Triton Inference Server</div></div>
    <div class="kpi"><div class="label">Preview</div><div class="value">Live MJPEG</div></div>
    <div class="kpi"><div class="label">Output</div><div class="value">MP4 + CSV + Preview</div></div>
  </div>

  <div class="grid">
    <div>
      <div class="card">
        <h2>Input Video</h2>
        <div class="help">Upload a video or select one already available in the workspace.</div>

        <label>Upload video</label>
        <input type="file" id="videoFile">
        <div class="row">
          <button class="primary" onclick="uploadVideo()">Upload</button>
          <button class="neutral" onclick="loadVideos()">Refresh list</button>
        </div>
        <div id="uploadMsg" class="small"></div>

        <label>Select video</label>
        <select id="videoSelect"></select>

        <div class="row">
          <button class="success" onclick="runPipeline()">Run Analysis</button>
          <button class="danger" onclick="stopPipeline()">Stop</button>
        </div>
        <div id="runMsg" class="small"></div>
      </div>

      <div class="card">
        <h2>Status</h2>
        <div id="statusPill" class="status idle">Waiting</div>
        <pre id="statusBox">Loading...</pre>
      </div>

      <div class="card">
        <h2>Execution Log</h2>
        <pre id="logBox">Loading...</pre>
      </div>
    </div>

    <div>
      <div class="card">
        <h2>Live Processed Preview</h2>
        <div class="help">
          The preview updates continuously from the latest processed frame while analysis is running.
        </div>
        <div class="player">
          <img id="livePreview" src="/api/live.mjpg" alt="Live processed preview">
        </div>
        <div class="row">
          <button class="neutral" onclick="reconnectPreview()">Reconnect preview</button>
        </div>
        <div id="streamMsg" class="small">Preview will appear after processing starts.</div>
      </div>

      <div class="card">
        <h2>Results</h2>
        <div class="help">Download generated video, CSV, preview image, and summary files.</div>
        <div id="outputs" class="output"></div>
      </div>
    </div>
  </div>
</div>

<script>
async function loadVideos(){
  const r = await fetch('/api/videos');
  const data = await r.json();
  const sel = document.getElementById('videoSelect');
  sel.innerHTML = '';
  data.videos.forEach(v=>{
    const opt = document.createElement('option');
    opt.value = v;
    opt.textContent = v;
    sel.appendChild(opt);
  });
}

function reconnectPreview(){
  const img = document.getElementById('livePreview');
  img.src = '/api/live.mjpg?t=' + Date.now();
  document.getElementById('streamMsg').textContent = 'Preview connection refreshed.';
}

async function uploadVideo(){
  const f = document.getElementById('videoFile').files[0];
  if(!f){ alert('Choose a video first.'); return; }
  const form = new FormData();
  form.append('file', f);
  const r = await fetch('/api/upload', {method:'POST', body:form});
  const data = await r.json();
  document.getElementById('uploadMsg').textContent = JSON.stringify(data, null, 2);
  await loadVideos();
}

async function runPipeline(){
  const selected = document.getElementById('videoSelect').value;
  if(!selected){ alert('Select a video first.'); return; }

  reconnectPreview();

  const form = new FormData();
  form.append('video_path', selected);
  const r = await fetch('/api/run', {method:'POST', body:form});
  const data = await r.json();
  document.getElementById('runMsg').textContent = JSON.stringify(data, null, 2);

  setTimeout(reconnectPreview, 1500);
  await refreshAll();
}

async function stopPipeline(){
  const r = await fetch('/api/stop', {method:'POST'});
  const data = await r.json();
  document.getElementById('runMsg').textContent = JSON.stringify(data, null, 2);
  await refreshAll();
}

async function loadStatus(){
  const r = await fetch('/api/status');
  const data = await r.json();
  document.getElementById('statusBox').textContent = JSON.stringify(data, null, 2);

  const pill = document.getElementById('statusPill');
  if(data.running){
    pill.textContent = 'Running';
    pill.className = 'status run';
  } else {
    pill.textContent = 'Waiting / Complete';
    pill.className = 'status idle';
  }
}

async function loadLog(){
  const r = await fetch('/api/log');
  const text = await r.text();
  document.getElementById('logBox').textContent = text;
}

async function loadOutputs(){
  const r = await fetch('/api/outputs');
  const data = await r.json();
  const div = document.getElementById('outputs');
  div.innerHTML = '';
  if(!data.files.length){
    div.innerHTML = '<div class="small">No generated files yet.</div>';
    return;
  }
  data.files.forEach(f=>{
    const a = document.createElement('a');
    a.href = '/api/download?path=' + encodeURIComponent(f.path);
    a.target = '_blank';
    a.textContent = f.name + ' (' + f.size_mb + ' MB)';
    div.appendChild(a);
  });
}

async function refreshAll(){
  await loadStatus();
  await loadLog();
  await loadOutputs();
}

loadVideos();
refreshAll();
setInterval(refreshAll, 3000);
</script>
</body>
</html>
"""


@app.get("/api/videos")
def videos():
    return {"videos": safe_video_paths()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    name = Path(file.filename).name
    if not name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        raise HTTPException(status_code=400, detail="Only video files are supported.")

    dst = UPLOAD_DIR / name
    with open(dst, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    return {"uploaded": str(dst), "size_mb": round(dst.stat().st_size / (1024 * 1024), 2)}


@app.post("/api/run")
def run(video_path: str = Form(...)):
    global current_proc, last_started_at

    if proc_running():
        raise HTTPException(status_code=409, detail="Analysis is already running.")

    if not PIPELINE_SCRIPT.exists():
        raise HTTPException(status_code=500, detail=f"Pipeline script not found: {PIPELINE_SCRIPT}")

    video = assert_safe_video(video_path)

    clear_live_preview()

    if CANONICAL_VIDEO.exists() or CANONICAL_VIDEO.is_symlink():
        CANONICAL_VIDEO.unlink()
    os.symlink(str(video), str(CANONICAL_VIDEO))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["TRITON_URL"] = TRITON_URL
    env["YOLOPX_ROOT"] = str(PROJECT / "src" / "YOLOPX")
    env["METRIC_ROOT"] = str(PROJECT / "src" / "Depth-Anything-V2" / "metric_depth")
    env["OUT_DIR"] = str(OUT_DIR)
    env["LIVE_PREVIEW_DIR"] = str(LIVE_PREVIEW_DIR)
    env["LIVE_PREVIEW_WIDTH"] = "960"
    env["LIVE_PREVIEW_HEIGHT"] = "540"
    env["LIVE_PREVIEW_JPEG_QUALITY"] = "78"
    env["PYTHONUNBUFFERED"] = "1"

    with open(RUN_LOG, "w") as log:
        log.write(f"Selected video: {video}\n")
        log.write(f"Input link: {CANONICAL_VIDEO}\n")
        log.write(f"Output directory: {OUT_DIR}\n")
        log.write(f"Triton endpoint: {TRITON_URL}\n")
        log.write("YOLOPX backend: FP16\n")
        log.write("Live preview: MJPEG stream\n\n")
        log.flush()

    log_f = open(RUN_LOG, "a")
    current_proc = subprocess.Popen(
        ["python3", "-u", str(PIPELINE_SCRIPT)],
        cwd=str(PROJECT),
        stdout=log_f,
        stderr=subprocess.STDOUT,
        env=env,
        preexec_fn=os.setsid,
    )
    last_started_at = time.time()

    return {
        "started": True,
        "pid": current_proc.pid,
        "video": str(video),
        "output_dir": str(OUT_DIR),
        "live_preview": "/api/live.mjpg",
    }


@app.post("/api/stop")
def stop():
    global current_proc

    if not proc_running():
        return {"stopped": False, "message": "No active analysis process."}

    os.killpg(os.getpgid(current_proc.pid), signal.SIGTERM)
    return {"stopped": True, "pid": current_proc.pid}


@app.get("/api/status")
def status():
    running = proc_running()
    return {
        "running": running,
        "pid": current_proc.pid if current_proc else None,
        "returncode": None if running or current_proc is None else current_proc.returncode,
        "started_at": last_started_at,
        "input_video": os.path.realpath(CANONICAL_VIDEO) if CANONICAL_VIDEO.exists() or CANONICAL_VIDEO.is_symlink() else None,
        "output_dir": str(OUT_DIR),
        "triton_url": TRITON_URL,
        "yolopx_backend": "FP16",
        "live_preview": "/api/live.mjpg",
    }


@app.get("/api/log", response_class=PlainTextResponse)
def log():
    if not RUN_LOG.exists():
        return "No log available yet."
    text = RUN_LOG.read_text(errors="replace")
    return clean_log_text("\n".join(text.splitlines()[-160:]))


@app.get("/api/outputs")
def outputs():
    return {"files": latest_files()}


@app.get("/api/live-frame")
def live_frame():
    if not LIVE_PREVIEW_FRAME.exists():
        raise HTTPException(status_code=404, detail="No live frame available yet.")
    return FileResponse(str(LIVE_PREVIEW_FRAME), media_type="image/jpeg")


@app.get("/api/live.mjpg")
async def live_mjpeg():
    async def gen():
        boundary = b"--frame\r\n"
        last_data = None

        while True:
            try:
                if LIVE_PREVIEW_FRAME.exists():
                    data = LIVE_PREVIEW_FRAME.read_bytes()
                    if data:
                        last_data = data

                if last_data:
                    yield boundary
                    yield b"Content-Type: image/jpeg\r\n"
                    yield b"Cache-Control: no-cache, no-store, must-revalidate\r\n\r\n"
                    yield last_data
                    yield b"\r\n"

            except Exception:
                pass

            await asyncio.sleep(0.10)

    return StreamingResponse(
        gen(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/api/download")
def download(path: str):
    p = Path(path).resolve()

    if not p.exists() or not p.is_file():
        raise HTTPException(status_code=404, detail="File not found.")

    allowed = str(OUT_DIR.resolve())
    if not str(p).startswith(allowed):
        raise HTTPException(status_code=403, detail="Download path not allowed.")

    return FileResponse(str(p), filename=p.name)
