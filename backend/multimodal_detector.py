"""
MultiModalDetector — Unified orchestrator for DeepShield
Handles Video (visual + audio fusion), Image, and Audio detection.
"""
import os, uuid, time, subprocess
import numpy as np
from pathlib import Path
from typing import Optional
import torch

from detector       import DeepfakeDetector        # visual video branch
from audio_detector import AudioDeepfakeDetector
from image_detector import ImageDeepfakeDetector


# ─────────────────────────────────────────────────────────────────
# Utility: Extract audio from video using ffmpeg
# ─────────────────────────────────────────────────────────────────
def extract_audio_from_video(video_path: str, output_wav: str) -> bool:
    """
    Extracts the audio track from a video file and saves as 16 kHz mono WAV.
    Returns True on success, False if the video has no audio or ffmpeg fails.
    """
    try:
        cmd = [
            "ffmpeg", "-y",
            "-i",  video_path,
            "-vn",                     # No video
            "-ar", "16000",            # Resample to 16 kHz
            "-ac", "1",                # Mono
            "-f",  "wav",
            output_wav,
        ]
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        return result.returncode == 0 and Path(output_wav).exists() and Path(output_wav).stat().st_size > 0
    except Exception as e:
        print(f"[AudioExtract] ffmpeg failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────
# MultiModalDetector
# ─────────────────────────────────────────────────────────────────
class MultiModalDetector:
    """
    Unified deepfake detector for three modalities:
      - Video  : EfficientNet (visual) + Wav2Vec2 (audio) → fused score
      - Image  : EfficientNetV2-S + MTCNN face detection
      - Audio  : Wav2Vec2-base + attention-pooling classifier

    Fusion strategy for video:
      fused_score = 0.60 × visual_score + 0.40 × audio_score
      (if no audio track, fused_score = visual_score)
    """

    VISUAL_WEIGHT = 0.60
    AUDIO_WEIGHT  = 0.40

    def __init__(
        self,
        video_model_path: Optional[str] = None,
        image_model_path: Optional[str] = None,
        audio_model_path: Optional[str] = None,
        device:           Optional[str] = None,
        threshold:        float          = 0.5,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        print(f"[DeepShield] Initializing MultiModalDetector on {self.device}")

        # ── Video (visual) detector ───────────────────────────
        self.video_detector = DeepfakeDetector(
            model_path=video_model_path,
            device=self.device,
            max_frames=32,
            threshold=threshold,
        )

        # ── Image detector ────────────────────────────────────
        self.image_detector = ImageDeepfakeDetector(
            model_path=image_model_path,
            device=self.device,
            threshold=threshold,
        )

        # ── Audio detector ────────────────────────────────────
        self.audio_available = False
        self.audio_model     = None
        try:
            from audio_detector import AudioDeepfakeDetector, WAV2VEC_AVAILABLE
            if WAV2VEC_AVAILABLE:
                self.audio_model = AudioDeepfakeDetector(
                    pretrained=True,
                    freeze_base=True,
                )
                if audio_model_path and Path(audio_model_path).exists():
                    state = torch.load(
                        audio_model_path, map_location=self.device, weights_only=True
                    )
                    self.audio_model.load_state_dict(state)
                    print(f"[AudioDetector] Loaded weights from {audio_model_path}")
                else:
                    print("[AudioDetector] Using pretrained Wav2Vec2 features (demo mode).")
                self.audio_model.to(self.device).eval()
                self.audio_available = True
            else:
                print("[AudioDetector] transformers not installed — audio branch disabled.")
        except Exception as e:
            print(f"[AudioDetector] Init failed: {e}")

    # ── VIDEO ─────────────────────────────────────────────────────
    def analyze_video(self, video_path: str, session_id: Optional[str] = None) -> dict:
        """
        Full multimodal video analysis:
          1. Visual branch: face extraction → EfficientNet inference → Grad-CAM
          2. Audio branch : ffmpeg extract → Wav2Vec2 inference
          3. Score fusion : 60/40 weighted average
        """
        t0         = time.time()
        session_id = session_id or str(uuid.uuid4())

        # 1. Visual analysis (existing pipeline)
        visual_result = self.video_detector.analyze(video_path, session_id=session_id)
        visual_score  = visual_result.get("fake_prob", 0.5)

        # 2. Audio analysis
        audio_info = {"audio_available": False, "audio_fake_prob": None}
        if self.audio_available and self.audio_model is not None:
            audio_wav = str(Path("uploads") / session_id / "audio.wav")
            has_audio = extract_audio_from_video(video_path, audio_wav)
            if has_audio:
                try:
                    waveform   = AudioDeepfakeDetector.load_audio(audio_wav)
                    audio_prob = self.audio_model.predict_proba(waveform, self.device)
                    audio_info = {
                        "audio_available":   True,
                        "audio_fake_prob":   round(audio_prob, 4),
                        "audio_verdict":     "FAKE" if audio_prob >= self.threshold else "REAL",
                        "audio_confidence":  round(audio_prob * 100, 2),
                    }
                except Exception as e:
                    audio_info = {"audio_available": False, "audio_error": str(e)}
            else:
                audio_info = {"audio_available": False, "audio_error": "No audio track found."}

        # 3. Fusion
        if audio_info.get("audio_available") and audio_info.get("audio_fake_prob") is not None:
            fused_score = (
                self.VISUAL_WEIGHT * visual_score
                + self.AUDIO_WEIGHT * audio_info["audio_fake_prob"]
            )
        else:
            fused_score = visual_score

        fused_verdict = "FAKE" if fused_score >= self.threshold else "REAL"

        return {
            **visual_result,
            **audio_info,
            "visual_fake_prob":   round(visual_score, 4),
            "visual_confidence":  round(visual_score * 100, 2),
            "visual_verdict":     "FAKE" if visual_score >= self.threshold else "REAL",
            "fused_fake_prob":    round(fused_score, 4),
            "fused_confidence":   round(fused_score, 4),
            "fused_verdict":      fused_verdict,
            "verdict":            fused_verdict,
            "confidence":         round(fused_score, 4),
            "fake_prob":          round(fused_score, 4),
            "elapsed_sec":        round(time.time() - t0, 2),
            "modality":           "video",
        }

    # ── IMAGE ─────────────────────────────────────────────────────
    def analyze_image(self, image_path: str) -> dict:
        """EfficientNetV2-S + MTCNN image pipeline."""
        return self.image_detector.analyze(image_path)

    # ── AUDIO ─────────────────────────────────────────────────────
    def analyze_audio(self, audio_path: str) -> dict:
        """Wav2Vec2 audio-only pipeline."""
        if not self.audio_available or self.audio_model is None:
            return {
                "verdict":    "ERROR",
                "error":      "Audio detection unavailable. Install: transformers, librosa.",
                "confidence": 0.0,
                "fake_prob":  0.0,
                "modality":   "audio",
            }
        try:
            waveform = AudioDeepfakeDetector.load_audio(audio_path)
            prob     = self.audio_model.predict_proba(waveform, self.device)
            return {
                "verdict":    "FAKE" if prob >= self.threshold else "REAL",
                "confidence": round(prob, 4),
                "fake_prob":  round(prob, 4),
                "modality":   "audio",
            }
        except Exception as e:
            return {
                "verdict":    "ERROR",
                "error":      str(e),
                "confidence": 0.0,
                "fake_prob":  0.0,
                "modality":   "audio",
            }
