"""
DeepShield — Multi-Modal Deepfake Detection API (Flask)
Flask server handling Video, Image, and Audio deepfake detection with MongoDB integration.
"""
import os
import shutil
import uuid
from pathlib import Path
from dotenv import load_dotenv

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

from multimodal_detector import MultiModalDetector
from db import save_detection, get_recent_detections

load_dotenv()

# ── Setup directories ─────────────────────────────────────────────
for d in ["uploads", "results", "models"]:
    Path(d).mkdir(exist_ok=True)

# ── App ───────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max limit

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

def save_upload(file, session_id, ext):
    upload_dir = Path("uploads") / session_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    filename = secure_filename(file.filename)
    dest = upload_dir / f"input{ext}"
    file.save(str(dest))
    return dest, os.path.getsize(str(dest)), filename

# ═══════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":          "ok",
        "system":          "DeepShield v2.0 (Flask)",
        "device":          detector.device,
        "video_model":     "Loaded" if Path(VIDEO_MODEL).exists() else "Demo mode",
        "image_model":     "Loaded" if Path(IMAGE_MODEL).exists() else "Demo mode",
        "audio_available": detector.audio_available,
        "modalities":      ["video", "image", "audio"],
    })

@app.route("/history", methods=["GET"])
def get_history():
    limit = request.args.get('limit', default=50, type=int)
    history = get_recent_detections(limit)
    return jsonify({
        "total": len(history),
        "history": history
    })

# ── VIDEO Detection ───────────────────────────────────────────────
@app.route("/detect/video", methods=["POST"])
def detect_video():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in VIDEO_EXTS:
        return jsonify({"detail": f"Unsupported video type: {suffix}. Allowed: {VIDEO_EXTS}"}), 400

    session_id = str(uuid.uuid4())
    dest, size, orig_filename = save_upload(file, session_id, suffix)
    size_mb = size / (1024 * 1024)
    if size_mb > MAX_MB["video"]:
        shutil.rmtree(dest.parent, ignore_errors=True)
        return jsonify({"detail": f"File too large ({size_mb:.1f} MB). Max {MAX_MB['video']} MB."}), 413

    try:
        result = detector.analyze_video(str(dest), session_id=session_id)
        # Save to MongoDB
        save_detection(
            session_id=session_id,
            filename=orig_filename,
            modality="video",
            verdict=result.get("verdict"),
            confidence=result.get("confidence"),
            details={"frames_analyzed": result.get("frames_analyzed")}
        )
    except Exception as e:
        shutil.rmtree(dest.parent, ignore_errors=True)
        return jsonify({"detail": f"Detection error: {e}"}), 500

    return jsonify(result)

# ── IMAGE Detection ───────────────────────────────────────────────
@app.route("/detect/image", methods=["POST"])
def detect_image():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_EXTS:
        return jsonify({"detail": f"Unsupported image type: {suffix}. Allowed: {IMAGE_EXTS}"}), 400

    session_id = str(uuid.uuid4())
    dest, size, orig_filename = save_upload(file, session_id, suffix)
    size_mb = size / (1024 * 1024)
    if size_mb > MAX_MB["image"]:
        shutil.rmtree(dest.parent, ignore_errors=True)
        return jsonify({"detail": f"File too large ({size_mb:.1f} MB). Max {MAX_MB['image']} MB."}), 413

    try:
        result = detector.analyze_image(str(dest))
        result["session_id"] = session_id
        # Save to MongoDB
        save_detection(
            session_id=session_id,
            filename=orig_filename,
            modality="image",
            verdict=result.get("verdict"),
            confidence=result.get("confidence"),
            details={"faces_detected": result.get("faces_detected", 1)}
        )
    except Exception as e:
        shutil.rmtree(dest.parent, ignore_errors=True)
        return jsonify({"detail": f"Detection error: {e}"}), 500

    return jsonify(result)

# ── AUDIO Detection ───────────────────────────────────────────────
@app.route("/detect/audio", methods=["POST"])
def detect_audio():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in AUDIO_EXTS:
        return jsonify({"detail": f"Unsupported audio type: {suffix}. Allowed: {AUDIO_EXTS}"}), 400

    session_id = str(uuid.uuid4())
    dest, size, orig_filename = save_upload(file, session_id, suffix)
    size_mb = size / (1024 * 1024)
    if size_mb > MAX_MB["audio"]:
        shutil.rmtree(dest.parent, ignore_errors=True)
        return jsonify({"detail": f"File too large ({size_mb:.1f} MB). Max {MAX_MB['audio']} MB."}), 413

    try:
        result = detector.analyze_audio(str(dest))
        result["session_id"] = session_id
        # Save to MongoDB
        save_detection(
            session_id=session_id,
            filename=orig_filename,
            modality="audio",
            verdict=result.get("verdict"),
            confidence=result.get("confidence")
        )
    except Exception as e:
        shutil.rmtree(dest.parent, ignore_errors=True)
        return jsonify({"detail": f"Detection error: {e}"}), 500

    return jsonify(result)

# ── Results & cleanup ─────────────────────────────────────────────
@app.route("/results/<session_id>/<filename>", methods=["GET"])
def get_result_file(session_id, filename):
    result_dir = Path("results") / session_id
    return send_from_directory(str(result_dir), filename)

@app.route("/results/<session_id>", methods=["GET"])
def get_results(session_id):
    result_dir = Path("results") / session_id
    if not result_dir.exists():
        return jsonify({"detail": "Session not found."}), 404
    files = [f.name for f in result_dir.glob("*.jpg")]
    return jsonify({"session_id": session_id, "heatmaps": files})

@app.route("/session/<session_id>", methods=["DELETE"])
def delete_session(session_id):
    for d in ["uploads", "results"]:
        shutil.rmtree(Path(d) / session_id, ignore_errors=True)
    return jsonify({"deleted": session_id})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
