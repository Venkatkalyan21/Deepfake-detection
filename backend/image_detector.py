"""
ImageDeepfakeDetector — EfficientNetV2-S + MTCNN for static image deepfake detection
Detects facial inconsistencies, texture/pixel artifacts in images (JPG, PNG, WEBP).
"""
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple
import torchvision.transforms as T

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    # facenet-pytorch works functionally even with version warning vs torch>=2.4
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from facenet_pytorch import MTCNN as FacenetMTCNN
    MTCNN_AVAILABLE = True
except (ImportError, Exception):
    MTCNN_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────
# Image pre-processing
# ─────────────────────────────────────────────────────────────────
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

inference_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])

train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomCrop(224),
    T.RandomHorizontalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    T.ToTensor(),
    T.Normalize(mean=MEAN, std=STD),
])


# ─────────────────────────────────────────────────────────────────
# EfficientNetV2-S Backbone + Classifier Head
# ─────────────────────────────────────────────────────────────────
class EfficientNetV2Detector(nn.Module):
    """
    EfficientNetV2-S backbone for fake image detection.
    ImageNet-pretrained → fine-tuned on deepfake face datasets.
    Detects:
      - Facial boundary blending artifacts
      - Texture inconsistencies (blurriness, over-smoothing)
      - GAN frequency fingerprints in pixel space
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()

        if TIMM_AVAILABLE:
            # Use tf_efficientnetv2_s which has pretrained weights in all timm versions
            try:
                self.backbone = timm.create_model(
                    "tf_efficientnetv2_s",
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
            except Exception:
                self.backbone = timm.create_model(
                    "efficientnet_b4",
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg",
                )
            self.feat_dim = self.backbone.num_features   # 1280
        else:
            # Fallback: EfficientNet-B0 via torchvision
            import torchvision.models as tv
            net = tv.efficientnet_b0(pretrained=pretrained)
            self.backbone = nn.Sequential(
                net.features,
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.feat_dim = 1280

        self.classifier = nn.Sequential(
            nn.Linear(self.feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)   # (B, 1) logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)).squeeze(1)   # (B,)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "EfficientNetV2Detector":
        model = EfficientNetV2Detector(pretrained=False)
        state = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.eval()
        return model


# ─────────────────────────────────────────────────────────────────
# MTCNN Face Extractor Wrapper
# ─────────────────────────────────────────────────────────────────
class MTCNNExtractor:
    """
    MTCNN-based face extractor using facenet-pytorch.
    Localises and aligns faces before passing to the classifier.
    """
    def __init__(self, device: str = "cpu"):
        if MTCNN_AVAILABLE:
            self.mtcnn = FacenetMTCNN(
                image_size=224,
                margin=30,
                keep_all=False,
                post_process=False,
                device=device,
            )
        else:
            self.mtcnn = None
            print("[MTCNN] facenet-pytorch not installed — using full image fallback.")

    def extract(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Returns:
            face_rgb   : (H, W, 3) uint8 numpy array — face crop or full image
            detected   : bool — True if MTCNN found a face
        """
        if self.mtcnn is None:
            return image_rgb, False

        pil_img = Image.fromarray(image_rgb)
        try:
            face_tensor = self.mtcnn(pil_img)   # (C, H, W) float in [0, 255]
            if face_tensor is not None:
                face_np = face_tensor.permute(1, 2, 0).byte().numpy()
                return face_np, True
        except Exception:
            pass
        return image_rgb, False


# ─────────────────────────────────────────────────────────────────
# Full Image Detection Pipeline
# ─────────────────────────────────────────────────────────────────
class ImageDeepfakeDetector:
    """
    End-to-end pipeline:
      1. Load image from disk
      2. Detect & crop face with MTCNN
      3. Classify with EfficientNetV2-S
      4. Return verdict + confidence
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str]     = None,
        threshold: float          = 0.5,
    ):
        self.device    = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold

        # EfficientNetV2 model
        self.model = EfficientNetV2Detector(pretrained=(model_path is None))
        if model_path and Path(model_path).exists():
            state = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state)
            print(f"[ImageDetector] Loaded weights from {model_path}")
        else:
            print("[ImageDetector] Using ImageNet pretrained weights (demo mode).")
        self.model.to(self.device).eval()

        # MTCNN face extractor
        self.extractor = MTCNNExtractor(device=self.device)

    def analyze(self, image_path: str) -> dict:
        """Run deepfake detection on a single image file."""
        # ── Load ──────────────────────────────────────────────
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            return {
                "verdict":       "ERROR",
                "error":         "Cannot read image file.",
                "confidence":    0.0,
                "fake_prob":     0.0,
                "face_detected": False,
                "modality":      "image",
            }

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w    = img_rgb.shape[:2]

        # ── Face extraction ───────────────────────────────────
        face_rgb, face_detected = self.extractor.extract(img_rgb)

        # ── Pre-process & infer ───────────────────────────────
        pil_img = Image.fromarray(face_rgb)
        tensor  = inference_transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            prob = self.model.predict_proba(tensor).item()

        verdict = "FAKE" if prob >= self.threshold else "REAL"

        return {
            "verdict":       verdict,
            "confidence":    round(prob * 100, 2),
            "fake_prob":     round(prob, 4),
            "face_detected": face_detected,
            "image_size":    f"{w}×{h}",
            "modality":      "image",
        }


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == "__main__":
    model = EfficientNetV2Detector(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out   = model.predict_proba(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {out.shape}")
    print(f"Total params : {total:,}")
    print("EfficientNetV2Detector OK ✓")
