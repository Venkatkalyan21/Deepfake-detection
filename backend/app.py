"""
FastAPI Server — Deepfake Detection REST API
Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import os, shutil, uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from detector import DeepfakeDetector

# ── Setup directories ───────────────────────────────────────────────
for d in ["uploads", "results", "models"]:
    Path(d).mkdir(exist_ok=True)

# ── App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Deepfake Detection API",
    description="Hybrid Spatial-Frequency CNN for deepfake video detection",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve result images (Grad-CAM heatmaps)
app.mount("/results", StaticFiles(directory="results"), name="results")

# ── Load detector once at startup ───────────────────────────────────
MODEL_PATH = os.getenv("MODEL_PATH", "models/deepfake_model.pth")
MODEL_EXISTS = Path(MODEL_PATH).exists()

detector = DeepfakeDetector(
    model_path=MODEL_PATH if MODEL_EXISTS else None,
    max_frames=32,
    # Use higher threshold in demo mode to avoid false positives on real videos
    threshold=0.5 if MODEL_EXISTS else 0.80,
)

ALLOWED_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
MAX_FILE_MB  = 200


# ── Endpoints ────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status":    "ok",
        "model":     "HybridDeepfakeDetector",
        "device":    detector.device,
        "version":   "1.0.0",
        "demo_mode": not MODEL_EXISTS,   # True = no trained weights loaded
        "threshold": detector.threshold,
        "warning":   "No trained model found — results unreliable!" if not MODEL_EXISTS else "Trained model loaded OK",
    }


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # Validate file type
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise HTTPException(400, f"Unsupported file type: {suffix}. "
                                 f"Allowed: {ALLOWED_EXTS}")

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > MAX_FILE_MB:
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_MB} MB.")

    # Save to disk
    session_id = str(uuid.uuid4())
    upload_dir = Path("uploads") / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    video_path = upload_dir / file.filename

    with open(video_path, "wb") as f:
        f.write(content)

    # Run detection
    try:
        result = detector.analyze(str(video_path), session_id=session_id)
    except Exception as e:
        shutil.rmtree(upload_dir, ignore_errors=True)
        raise HTTPException(500, f"Detection error: {str(e)}")

    return JSONResponse(content=result)


@app.get("/results/{session_id}")
def get_results(session_id: str):
    result_dir = Path("results") / session_id
    if not result_dir.exists():
        raise HTTPException(404, "Session not found.")
    files = list(result_dir.glob("*.jpg"))
    return {"session_id": session_id, "heatmaps": [str(f) for f in files]}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    for d in ["uploads", "results"]:
        shutil.rmtree(Path(d) / session_id, ignore_errors=True)
    return {"deleted": session_id}


# ── Serve Frontend UI ────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
def serve_ui():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "DeepGuard API running. Open frontend/index.html manually."}

@app.get("/style.css")
def serve_css():
    return FileResponse(str(FRONTEND_DIR / "style.css"), media_type="text/css")

@app.get("/app.js")
def serve_js():
    return FileResponse(str(FRONTEND_DIR / "app.js"), media_type="application/javascript")
