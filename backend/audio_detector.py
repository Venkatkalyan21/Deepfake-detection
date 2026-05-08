"""
AudioDeepfakeDetector — Wav2Vec2-based audio deepfake detection
Uses Facebook's Wav2Vec2-base for feature extraction + lightweight classification head.
Detects AI-generated or voice-cloned audio (TTS, VC spoofing).
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
    WAV2VEC_AVAILABLE = True
except ImportError:
    WAV2VEC_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Classifier Head (on top of Wav2Vec2 hidden states)
# ─────────────────────────────────────────────────────────────────
class AudioClassifierHead(nn.Module):
    """
    Pools temporal hidden states from Wav2Vec2 and classifies Real/Fake.
    Input:  (B, T, D)  — Wav2Vec2 last_hidden_state
    Output: (B, 1)     — raw logits (apply sigmoid for probability)
    """
    def __init__(self, input_dim: int = 768):
        super().__init__()
        self.attention_pool = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Attention-weighted pooling
        weights = torch.softmax(self.attention_pool(hidden_states), dim=1)  # (B, T, 1)
        pooled  = (hidden_states * weights).sum(dim=1)                       # (B, D)
        return self.classifier(pooled)                                       # (B, 1)


# ─────────────────────────────────────────────────────────────────
# Main Audio Deepfake Detector
# ─────────────────────────────────────────────────────────────────
class AudioDeepfakeDetector(nn.Module):
    """
    Wav2Vec2-base + Attention-Pooling Classifier for audio deepfake detection.

    Pipeline:
      Raw waveform (16 kHz mono)
        → Wav2Vec2 feature extractor (normalisation)
        → Wav2Vec2 transformer encoder  → hidden states (T × 768)
        → Attention-weighted pooling    → 768-dim vector
        → FC classifier head            → fake probability
    """
    SAMPLE_RATE      = 16_000
    MAX_DURATION_SEC = 10.0       # Clip to 10 s for speed

    def __init__(self, pretrained: bool = True, freeze_base: bool = True):
        super().__init__()

        if not WAV2VEC_AVAILABLE:
            raise RuntimeError(
                "transformers library not installed.\n"
                "Run: pip install transformers>=4.30.0"
            )

        model_name = "facebook/wav2vec2-base"
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.wav2vec2           = Wav2Vec2Model.from_pretrained(model_name)

        # Optionally freeze Wav2Vec2 backbone for efficient fine-tuning
        if freeze_base:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        hidden_dim = self.wav2vec2.config.hidden_size   # 768
        self.head  = AudioClassifierHead(hidden_dim)

    # ── Forward ─────────────────────────────────────────────────
    def forward(
        self,
        input_values:   torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        return self.head(outputs.last_hidden_state)   # (B, 1) logits

    # ── Inference helpers ────────────────────────────────────────
    def predict_proba(self, waveform: np.ndarray, device: str = "cpu") -> float:
        """
        Args:
            waveform: 1-D numpy float32 array at 16 kHz
            device:   'cpu' or 'cuda'
        Returns:
            Fake probability in [0, 1]
        """
        max_samples = int(self.MAX_DURATION_SEC * self.SAMPLE_RATE)
        if len(waveform) > max_samples:
            waveform = waveform[:max_samples]

        inputs = self.feature_extractor(
            waveform,
            sampling_rate=self.SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        input_values = inputs.input_values.to(device)

        self.eval()
        with torch.no_grad():
            logits = self.forward(input_values)
            prob   = torch.sigmoid(logits).squeeze().item()
        return float(prob)

    # ── Static helpers ───────────────────────────────────────────
    @staticmethod
    def load_audio(path: str, target_sr: int = 16_000) -> np.ndarray:
        """Load any audio/video file and resample to 16 kHz mono."""
        if not LIBROSA_AVAILABLE:
            raise RuntimeError("librosa not installed. Run: pip install librosa>=0.10.0")
        waveform, _ = librosa.load(path, sr=target_sr, mono=True)
        return waveform.astype(np.float32)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "AudioDeepfakeDetector":
        """Load a trained detector from a .pth file."""
        model = AudioDeepfakeDetector(pretrained=False, freeze_base=False)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    if WAV2VEC_AVAILABLE:
        model  = AudioDeepfakeDetector(pretrained=False, freeze_base=False)
        dummy  = torch.randn(2, 16_000)  # 1-second batch
        logits = model(dummy)
        total  = sum(p.numel() for p in model.parameters())
        print(f"Output shape : {logits.shape}")
        print(f"Total params : {total:,}")
        print("AudioDeepfakeDetector OK ✓")
    else:
        print("transformers not installed — skipping sanity check.")
