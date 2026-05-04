"""
Face Extractor — OpenCV-based face detection and frame sampling
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


class FaceExtractor:
    """
    Extracts and crops face regions from video frames.
    Uses OpenCV's Haar cascade (no extra dependencies).
    """

    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(cascade_path)
        if self.detector.empty():
            raise RuntimeError("Failed to load Haar cascade classifier.")

    # ── Frame sampling ──────────────────────────────────────────────
    def sample_frames(
        self,
        video_path: str,
        max_frames: int = 32,
        sample_fps: float = 2.0,
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Returns a list of (frame_index, BGR_frame) tuples.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        video_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total       = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval    = max(1, int(video_fps / sample_fps))

        frames: List[Tuple[int, np.ndarray]] = []
        idx = 0

        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if idx % interval == 0:
                frames.append((idx, frame))
            idx += 1

        cap.release()
        return frames

    # ── Face crop ───────────────────────────────────────────────────
    def crop_face(
        self,
        frame: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
        padding_ratio: float = 0.25,
    ) -> Optional[np.ndarray]:
        """
        Detects the largest face and returns a padded RGB crop.
        Returns None if no face detected (caller should decide what to do).
        """
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=4, minSize=(48, 48)
        )

        h, w = frame.shape[:2]

        if len(faces) == 0:
            # Fallback: centre-crop as square
            size = min(h, w)
            y0 = (h - size) // 2
            x0 = (w - size) // 2
            crop = frame[y0 : y0 + size, x0 : x0 + size]
        else:
            # Largest face
            fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
            pad = int(max(fw, fh) * padding_ratio)
            x1 = max(0, fx - pad)
            y1 = max(0, fy - pad)
            x2 = min(w, fx + fw + pad)
            y2 = min(h, fy + fh + pad)
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        crop = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return crop

    # ── Full pipeline ────────────────────────────────────────────────
    def extract(
        self,
        video_path: str,
        max_frames: int = 32,
        sample_fps: float = 2.0,
    ) -> List[dict]:
        """
        Returns list of dicts: {frame_idx, face_rgb (H,W,3 uint8)}
        """
        raw_frames = self.sample_frames(video_path, max_frames, sample_fps)
        results = []
        for fidx, frame in raw_frames:
            face = self.crop_face(frame)
            if face is not None:
                results.append({"frame_idx": fidx, "face": face})
        return results
