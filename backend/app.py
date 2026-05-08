"""
DeepShield — Multi-Modal Deepfake Detection API
FastAPI server handling Video, Image, and Audio deepfake detection.
Run: uvicorn app:app --host 0.0.0.0 --port 8000 --reload
"""
import os, shutil, uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse

from multimodal_detector import MultiModalDetector

# ── Setup directories ─────────────────────────────────────────────
for d in ["uploads", "results", "models"]:
    Path(d).mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title       = "DeepShield API",
    description = "Multi-Modal AI-Based Deepfake Detection — Video, Image & Audio",
    version     = "2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Serve Grad-CAM heatmaps
app.mount("/results", StaticFiles(directory="results"), name="results")

# ── Load detector once at startup ─────────────────────────────────
VIDEO_MODEL = os.getenv("VIDEO_MODEL_PATH", "models/deepfake_model.pth")
IMAGE_MODEL = os.getenv("IMAGE_MODEL_PATH", "models/image_model_best.pth")
AUDIO_MODEL = os.getenv("AUDIO_MODEL_PATH", "models/audio_model_best.pth")
THRESHOLD   = float(os.getenv("THRESHOLD", "0.5"))

detector = MultiModalDetector(
    video_model_path = VIDEO_MODEL if Path(VIDEO_MODEL).exists() else None,
    image_model_path = IMAGE_MODEL if Path(IMAGE_MODEL).exists() else None,
    audio_model_path = AUDIO_MODEL if Path(AUDIO_MODEL).exists() else None,
    threshold        = THRESHOLD,
)

# ── Allowed file types ────────────────────────────────────────────
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_MB     = {"video": 500, "image": 20, "audio": 50}


async def save_upload(file: UploadFile, session_id: str, ext: str) -> Path:
    upload_dir = Path("uploads") / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"input{ext}"
    content = await file.read()
    dest.write_bytes(content)
    return dest, len(content)


# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.get("/health")
def health():
    return {
        "status":          "ok",
        "system":          "DeepShield v2.0",
        "device":          detector.device,
        "video_model":     "Loaded" if Path(VIDEO_MODEL).exists() else "Demo mode",
        "image_model":     "Loaded" if Path(IMAGE_MODEL).exists() else "Demo mode",
        "audio_available": detector.audio_available,
        "modalities":      ["video", "image", "audio"],
    }


# ── VIDEO Detection ───────────────────────────────────────────────
@app.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(400, f"Unsupported video type: {suffix}. Allowed: {VIDEO_EXTS}")

    session_id = str(uuid.uuid4())
    dest, size = await save_upload(file, session_id, suffix)
    size_mb = size / (1024 * 1024)
    if size_mb > MAX_MB["video"]:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {MAX_MB['video']} MB.")

    try:
        result = detector.analyze_video(str(dest), session_id=session_id)
    except Exception as e:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise HTTPException(500, f"Detection error: {e}")

    return JSONResponse(content=result)


# ── IMAGE Detection ───────────────────────────────────────────────
@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise HTTPException(400, f"Unsupported image type: {suffix}. Allowed: {IMAGE_EXTS}")

    session_id = str(uuid.uuid4())
    dest, size = await save_upload(file, session_id, suffix)
    size_mb = size / (1024 * 1024)
    if size_mb > MAX_MB["image"]:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {MAX_MB['image']} MB.")

    try:
        result = detector.analyze_image(str(dest))
        result["session_id"] = session_id
    except Exception as e:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise HTTPException(500, f"Detection error: {e}")

    return JSONResponse(content=result)


# ── AUDIO Detection ───────────────────────────────────────────────
@app.post("/detect/audio")
async def detect_audio(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(400, f"Unsupported audio type: {suffix}. Allowed: {AUDIO_EXTS}")

    session_id = str(uuid.uuid4())
    dest, size = await save_upload(file, session_id, suffix)
    size_mb = size / (1024 * 1024)
    if size_mb > MAX_MB["audio"]:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise HTTPException(413, f"File too large ({size_mb:.1f} MB). Max {MAX_MB['audio']} MB.")

    try:
        result = detector.analyze_audio(str(dest))
        result["session_id"] = session_id
    except Exception as e:
        shutil.rmtree(dest.parent, ignore_errors=True)
        raise HTTPException(500, f"Detection error: {e}")

    return JSONResponse(content=result)


# ── Legacy /detect endpoint (backward compat — video) ────────────
@app.post("/detect")
async def detect_legacy(file: UploadFile = File(...)):
    """Backward-compatible video detection endpoint."""
    return await detect_video(file)


# ── Results & cleanup ─────────────────────────────────────────────
@app.get("/results/{session_id}")
def get_results(session_id: str):
    result_dir = Path("results") / session_id
    if not result_dir.exists():
        raise HTTPException(404, "Session not found.")
    files = [str(f) for f in result_dir.glob("*.jpg")]
    return {"session_id": session_id, "heatmaps": files}


@app.delete("/session/{session_id}")
def delete_session(session_id: str):
    for d in ["uploads", "results"]:
        shutil.rmtree(Path(d) / session_id, ignore_errors=True)
    return {"deleted": session_id}


# ── Serve Frontend ────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

@app.get("/")
def serve_ui():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "DeepShield API v2.0 running. Open frontend/index.html manually."}

@app.get("/style.css")
def serve_css():
    return FileResponse(str(FRONTEND_DIR / "style.css"), media_type="text/css")

@app.get("/app.js")
def serve_js():
    return FileResponse(str(FRONTEND_DIR / "app.js"), media_type="application/javascript")
