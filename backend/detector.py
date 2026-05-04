"""
DeepfakeDetector — orchestrates the full video analysis pipeline
"""
import os, uuid, time
import numpy as np
import torch
import torchvision.transforms as T
from pathlib import Path
from typing import Optional

from model import HybridDeepfakeDetector
from face_extractor import FaceExtractor
from gradcam import GradCAM

# ── Image pre-processing ─────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


class DeepfakeDetector:
    """
    End-to-end video deepfake detector.
    Steps:
      1. Sample frames from video
      2. Crop face regions
      3. Run hybrid model inference
      4. Generate Grad-CAM heatmaps
      5. Aggregate per-frame scores into video-level verdict
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        max_frames: int = 32,
        threshold: float = 0.5,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.max_frames = max_frames

        # Load model
        self.model = HybridDeepfakeDetector(pretrained=(model_path is None))
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state)
            print(f"[Detector] Loaded weights from {model_path}")
        else:
            print("[Detector] Using ImageNet pretrained weights (demo mode). "
                  "Train on FF++ for research-quality results.")

        self.model.to(self.device).eval()

        self.extractor = FaceExtractor()
        self.gradcam   = GradCAM(self.model)

    # ── Single frame inference ───────────────────────────────────────
    def _infer_frame(self, face_rgb: np.ndarray) -> float:
        """Returns fake probability for one face crop."""
        tensor = transform(face_rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = self.model.predict_proba(tensor).item()
        return float(prob)

    # ── Main analysis ────────────────────────────────────────────────
    def analyze(self, video_path: str, session_id: Optional[str] = None) -> dict:
        t0 = time.time()
        session_id = session_id or str(uuid.uuid4())

        results_dir = Path("results") / session_id
        results_dir.mkdir(parents=True, exist_ok=True)

        # 1. Extract faces
        faces = self.extractor.extract(video_path, max_frames=self.max_frames)
        if not faces:
            return {
                "session_id": session_id,
                "verdict": "UNKNOWN",
                "confidence": 0.0,
                "error": "No faces detected in video.",
                "frames": [],
            }

        # 2. Per-frame inference + Grad-CAM
        frame_results = []
        scores = []

        for i, item in enumerate(faces):
            face_rgb = item["face"]
            fidx     = item["frame_idx"]
            prob     = self._infer_frame(face_rgb)
            scores.append(prob)

            # Generate and save Grad-CAM heatmap
            cam_path = str(results_dir / f"frame_{i:04d}_cam.jpg")
            self.gradcam.generate(face_rgb, cam_path)

            frame_results.append({
                "frame_idx": fidx,
                "fake_prob": round(prob, 4),
                "verdict":   "FAKE" if prob >= self.threshold else "REAL",
                "cam_path":  f"/results/{session_id}/frame_{i:04d}_cam.jpg",
            })

        # 3. Video-level score
        video_score = float(np.mean(scores))
        verdict = "FAKE" if video_score >= self.threshold else "REAL"

        elapsed = round(time.time() - t0, 2)

        return {
            "session_id":   session_id,
            "verdict":      verdict,
            "confidence":   round(video_score * 100, 2),
            "fake_prob":    round(video_score, 4),
            "frames_analyzed": len(faces),
            "elapsed_sec":  elapsed,
            "frame_scores": frame_results,
        }
