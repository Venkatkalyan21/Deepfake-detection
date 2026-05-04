# 🛡️ DeepGuard — Deepfake Video Detection

> **IEEE Research Project** · Hybrid Spatial-Frequency CNN · EfficientNet-B4 + FFT Branch

---

## 🏗️ Project Structure

```
DeepfakeDetection/
├── backend/
│   ├── app.py              ← FastAPI REST API
│   ├── model.py            ← HybridDeepfakeDetector architecture
│   ├── detector.py         ← Full video analysis pipeline
│   ├── face_extractor.py   ← OpenCV face extraction
│   ├── gradcam.py          ← Grad-CAM heatmap visualizer
│   ├── metrics.py          ← AUC / ACC / EER metrics
│   ├── train.py            ← Training script (FF++, Celeb-DF)
│   └── requirements.txt
├── frontend/
│   ├── index.html          ← Dashboard UI
│   ├── style.css           ← Dark glassmorphism styles
│   └── app.js              ← Dashboard logic
├── models/                 ← Place trained .pth files here
└── README.md
```

---

## ⚡ Quick Start

### Step 1 — Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 2 — Start the Backend API

```bash
cd backend
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

API will be running at: `http://localhost:8000`

If `uvicorn` is still not found, make sure the backend dependencies are installed in the active Python environment with `pip install -r requirements.txt`.

### Step 3 — Open the Frontend

Open `frontend/index.html` in your browser  
(or use Live Server extension in VS Code).

---

## 🧠 Model Architecture

```
Input (224×224 RGB)
       │
  ┌────┴─────┐
  │          │
Spatial    Frequency
Branch     Branch
(EfficientNet-B4)  (FFT → CNN)
  │          │
  └────┬─────┘
  Feature Fusion (FC layers)
       │
  Sigmoid → Fake Probability [0, 1]
```

- **Spatial Branch**: EfficientNet-B0 pretrained on ImageNet → 1280-dim features
- **Frequency Branch**: 2D FFT log-magnitude → 3-layer CNN → 128-dim features
- **Fusion**: Concatenate → FC(512) → FC(128) → FC(1)

---

## 🔬 Training for IEEE Paper

### Dataset Setup (FaceForensics++)

```
data/
  train/
    real/   ← face crops from real videos
    fake/   ← face crops from manipulated videos
  val/
    real/
    fake/
```

### Run Training

```bash
cd backend
python train.py \
  --data_dir ../data/train \
  --epochs 30 \
  --batch_size 32 \
  --lr 1e-4 \
  --save_dir ../models
```

Best model saved to `models/deepfake_model.pth`

### Load Trained Model

```bash
MODEL_PATH=models/deepfake_model.pth python -m uvicorn app:app --reload
```

---

## 📊 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | API health check |
| POST | `/detect` | Upload video for analysis |
| GET | `/results/{session_id}` | Fetch Grad-CAM heatmaps |
| DELETE | `/session/{session_id}` | Clean up session files |

### Example Request

```bash
curl -X POST http://localhost:8000/detect \
  -F "file=@test_video.mp4"
```

### Example Response

```json
{
  "session_id": "abc-123",
  "verdict": "FAKE",
  "confidence": 87.5,
  "fake_prob": 0.875,
  "frames_analyzed": 28,
  "elapsed_sec": 12.4,
  "frame_scores": [
    { "frame_idx": 0, "fake_prob": 0.91, "verdict": "FAKE", "cam_path": "/results/abc-123/frame_0000_cam.jpg" },
    ...
  ]
}
```

---

## 📄 IEEE Paper

**Title**: HybridDF: A Spatial-Frequency Dual-Branch Network for Robust Deepfake Video Detection

**Target**: IEEE Transactions on Information Forensics and Security (TIFS)

**Key Contributions**:
1. Novel hybrid architecture (spatial + frequency branches)
2. Adaptive feature fusion module
3. Cross-dataset generalization evaluation
4. Grad-CAM forensic localization

---

## 📚 Datasets for IEEE Experiments

| Dataset | Download | Size |
|---------|----------|------|
| FaceForensics++ | [GitHub](https://github.com/ondyari/FaceForensics) | ~14 GB |
| Celeb-DF v2 | [GitHub](https://github.com/yuezunli/celeb-deepfakeforensics) | ~3 GB |
| DFDC (Preview) | [Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge) | ~400 GB |

---

## 🤝 Citation

```bibtex
@article{yourname2025hybriddf,
  title   = {HybridDF: A Spatial-Frequency Dual-Branch Network for Robust Deepfake Video Detection},
  author  = {Your Name},
  journal = {IEEE Transactions on Information Forensics and Security},
  year    = {2025}
}
```
