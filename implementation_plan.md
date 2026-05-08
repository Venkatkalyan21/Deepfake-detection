# DeepShield — Full Stack Rebuild Plan

Complete rebuild using the prescribed tech stack. Optimized for **easiest possible free deployment**.

---

## 🚀 Deployment Architecture (100% Free)

```
┌─────────────────────────────────────────────────────────┐
│   USER BROWSER                                          │
│   React App → Vercel (free, instant deploy from GitHub) │
└──────────────────────┬──────────────────────────────────┘
                       │ API calls
┌──────────────────────▼──────────────────────────────────┐
│   FLASK BACKEND → Render.com (free tier, auto-deploy)   │
│   /detect/video  /detect/image  /detect/audio  /history │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│   MONGODB ATLAS (free M0 cluster, 512MB, cloud)         │
│   Stores: session_id, verdict, confidence, timestamp     │
└─────────────────────────────────────────────────────────┘
```

| Layer | Technology | Deployment | Cost |
|-------|-----------|------------|------|
| **Frontend** | React 18 + Tailwind CSS v3 | **Vercel** | Free |
| **Backend** | Flask | **Render.com** | Free |
| **Database** | MongoDB Atlas | **Atlas M0** | Free |
| **Image AI** | EfficientNetV2 + MTCNN | Render | Free |
| **Audio AI** | Wav2Vec2 | Render | Free |
| **Video AI** | MTCNN + EfficientNetV2 frames | Render | Free |

> [!TIP]
> **Why this stack?** Vercel + Render + Atlas is the easiest free deployment combo for a Flask + React + MongoDB app. No server management, no Docker needed, direct GitHub push-to-deploy.

---

## 📦 Datasets

### 🎬 Video Dataset — Already Have It!
Your **Celeb-DF** dataset at `Deepfake-detection/Celeb-DF/` is perfect:
```
Celeb-DF/
├── Celeb-real/          ← Real celebrity videos
├── Celeb-synthesis/     ← Deepfake synthesized videos
├── YouTube-real/        ← Additional real videos
└── List_of_testing_videos.txt
```

### 🖼️ Image Datasets (Download These)

| Dataset | Size | Link | Notes |
|---------|------|------|-------|
| **140K Real & Fake Faces** | ~2.6 GB | [Kaggle →](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) | Best for training EfficientNetV2. 70K real + 70K GAN-generated |
| **DFDC Preview** | ~470 MB | [Kaggle →](https://www.kaggle.com/competitions/deepfake-detection-challenge/data) | Facebook's Deepfake Detection Challenge dataset |
| **FaceForensics++** | Request | [GitHub →](https://github.com/ondyari/FaceForensics) | Academic use, fill form for download script |

> [!IMPORTANT]
> **Recommended**: Start with **140K Real & Fake Faces** on Kaggle — no approval needed, direct download, best for EfficientNetV2 training.

### 🎵 Audio Datasets (Download These)

| Dataset | Size | Link | Notes |
|---------|------|------|-------|
| **WaveFake** | ~170 GB | [Zenodo →](https://doi.org/10.5281/zenodo.5642694) | Best for Wav2Vec2 training. 6 TTS architectures |
| **ASVspoof 2019** | ~14 GB | [Kaggle →](https://www.kaggle.com/datasets/awsaf49/asvpoof-2019-dataset) | Industry standard anti-spoofing dataset |
| **ASVspoof 2021** | ~50 GB | [Zenodo →](https://zenodo.org/records/4837263) | Newer, includes more attack types |
| **FakeAVCeleb** | Request | [GitHub →](https://github.com/DASH-Lab/FakeAVCeleb) | Audio+Video combined, requires form submission |

> [!IMPORTANT]
> **Recommended**: Start with **ASVspoof 2019** on Kaggle — direct download, no approval, ~14 GB, perfect for Wav2Vec2 fine-tuning.

---

## Proposed Code Changes

### Backend (Flask)

#### [MODIFY] `backend/app.py` — Replace FastAPI with Flask
- All routes: `POST /detect/video`, `/detect/image`, `/detect/audio`
- `GET /health`, `GET /history` (returns last 50 MongoDB records)
- Flask-CORS enabled
- MongoDB Atlas connection via `MONGO_URI` env variable

#### [MODIFY] `backend/image_detector.py` — EfficientNetV2 + MTCNN
- MTCNN extracts faces from input image
- EfficientNetV2-S classifies each face crop
- Returns: verdict, confidence, faces_detected, heatmap path

#### [MODIFY] `backend/multimodal_detector.py` — Updated orchestrator
- Video: frame extraction → MTCNN → EfficientNetV2 per frame
- Audio: Wav2Vec2 (existing, keep)
- Image: MTCNN → EfficientNetV2

#### [MODIFY] `backend/requirements.txt`
```
flask>=3.0.0
flask-cors>=4.0.0
pymongo>=4.6.0
dnspython>=2.6.0
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
timm>=0.9.16
facenet-pytorch>=2.5.2
transformers>=4.30.0
librosa>=0.10.0
soundfile>=0.12.1
opencv-python-headless>=4.9.0
numpy>=1.24.0
Pillow>=10.3.0
python-dotenv>=1.0.1
gunicorn>=21.2.0
```

#### [NEW] `backend/.env.example`
```
MONGO_URI=mongodb+srv://<user>:<pass>@cluster0.mongodb.net/deepshield
IMAGE_MODEL_PATH=models/image_model_best.pth
AUDIO_MODEL_PATH=models/audio_model_best.pth
VIDEO_MODEL_PATH=models/deepfake_model.pth
THRESHOLD=0.5
```

#### [MODIFY] `render.yaml` — Render deployment config
- Point to Flask `gunicorn app:app`

---

### Frontend (React 18 + Tailwind CSS v3 via Vite)

#### [DELETE] `frontend/index.html`, `frontend/app.js`, `frontend/style.css`
Old vanilla files replaced by React app.

#### [NEW] `frontend/` — Vite + React project
```
frontend/
├── src/
│   ├── App.jsx
│   ├── main.jsx
│   ├── index.css            ← Tailwind directives
│   ├── api/
│   │   └── deepshield.js    ← Axios API calls to Flask
│   ├── components/
│   │   ├── Navbar.jsx       ← Top nav with logo + tabs
│   │   ├── UploadZone.jsx   ← Drag-and-drop (video/image/audio)
│   │   ├── ResultPanel.jsx  ← Animated verdict + confidence gauge
│   │   ├── HistoryTable.jsx ← MongoDB scan history table
│   │   └── StatsCards.jsx   ← Total scans, fake %, accuracy
│   └── pages/
│       ├── Dashboard.jsx    ← Main detection page
│       └── History.jsx      ← Full history page
├── public/
│   └── shield-logo.svg
├── tailwind.config.js
├── vite.config.js
└── package.json
```

---

### Database (MongoDB Atlas)

#### Schema: `deepshield` → `detections` collection
```json
{
  "_id": "ObjectId",
  "session_id": "uuid-string",
  "filename": "upload.mp4",
  "modality": "video",
  "verdict": "FAKE",
  "confidence": 0.91,
  "faces_detected": 3,
  "frames_analyzed": 48,
  "timestamp": "2026-05-08T11:45:00Z"
}
```

---

## Verification Plan

### Local Testing
1. `cd backend && flask run` → `http://localhost:5000/health` returns `{"status":"ok"}`
2. `cd frontend && npm run dev` → React app at `http://localhost:5173`
3. Upload test image → result appears + MongoDB record created

### Deployment
1. Push to GitHub → Vercel auto-deploys frontend
2. Push to GitHub → Render auto-deploys Flask backend
3. Set env vars on Render dashboard (MONGO_URI, model paths)

---

## Open Questions

> [!IMPORTANT]
> **Ready to build?** Just reply **"go"** and I'll start building all the code immediately. I'll use:
> - **Tailwind CSS v3** (most stable, easiest)
> - **MongoDB Atlas** free tier (you create the free cluster at mongodb.com/atlas, takes 2 min, then share the connection string)
> - **Vercel** for React frontend
> - **Render** for Flask backend
