"""
HybridDeepfakeDetector — Dual-branch Spatial + Frequency CNN
Architecture for IEEE Research Paper on Deepfake Video Detection
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

import torchvision.models as tv_models


# ─────────────────────────────────────────────
# Frequency Analysis Branch
# ─────────────────────────────────────────────
class FrequencyBranch(nn.Module):
    """
    Extracts GAN fingerprint artifacts from the frequency domain.
    GAN generators leave periodic patterns in the DCT/FFT spectrum
    that are invisible to the human eye but detectable by CNNs.
    """
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 112x112

            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 56x56

            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                          # 28x28

            nn.Conv2d(128, out_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),             # 1x1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Convert to grayscale: (B, 1, H, W)
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

        # 2D FFT → log-magnitude spectrum
        fft = torch.fft.fft2(gray)
        magnitude = torch.abs(fft)
        magnitude = torch.log(magnitude + 1e-8)

        # Normalize per sample
        b = magnitude.shape[0]
        m = magnitude.view(b, -1)
        mn = m.mean(dim=1, keepdim=True).view(b, 1, 1, 1)
        std = m.std(dim=1, keepdim=True).view(b, 1, 1, 1) + 1e-8
        magnitude = (magnitude - mn) / std

        return self.conv_layers(magnitude).flatten(1)   # (B, out_dim)


# ─────────────────────────────────────────────
# Spatial Branch (EfficientNet-B0 backbone)
# ─────────────────────────────────────────────
class SpatialBranch(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        if TIMM_AVAILABLE:
            self.backbone = timm.create_model(
                "efficientnet_b0",
                pretrained=pretrained,
                num_classes=0,          # Remove classifier head
                global_pool="avg",
            )
            self.out_dim = self.backbone.num_features   # 1280
        else:
            # Fallback: MobileNetV3-Small from torchvision
            backbone = tv_models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2],
                                          nn.AdaptiveAvgPool2d(1), nn.Flatten())
            self.out_dim = 576

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)     # (B, out_dim)


# ─────────────────────────────────────────────
# Fusion Classifier
# ─────────────────────────────────────────────
class FusionClassifier(nn.Module):
    def __init__(self, spatial_dim: int, freq_dim: int):
        super().__init__()
        combined = spatial_dim + freq_dim
        self.fc = nn.Sequential(
            nn.Linear(combined, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, spatial_feat, freq_feat):
        x = torch.cat([spatial_feat, freq_feat], dim=1)
        return self.fc(x)   # (B, 1) — raw logits


# ─────────────────────────────────────────────
# HybridDeepfakeDetector (Main Model)
# ─────────────────────────────────────────────
class HybridDeepfakeDetector(nn.Module):
    """
    Novel dual-branch architecture combining:
    - Spatial branch (EfficientNet-B0): captures texture/semantic artifacts
    - Frequency branch (FFT-CNN): captures GAN frequency fingerprints
    Fused via FC layers for binary Real/Fake classification.
    """
    def __init__(self, pretrained: bool = True, freq_dim: int = 128):
        super().__init__()
        self.spatial = SpatialBranch(pretrained=pretrained)
        self.freq    = FrequencyBranch(out_dim=freq_dim)
        self.fusion  = FusionClassifier(self.spatial.out_dim, freq_dim)

    def forward(self, x: torch.Tensor):
        s = self.spatial(x)
        f = self.freq(x)
        return self.fusion(s, f)        # (B, 1) logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns fake probability in [0, 1]."""
        with torch.no_grad():
            logits = self.forward(x)
            return torch.sigmoid(logits).squeeze(1)  # (B,)

    @staticmethod
    def load(path: str, device: str = "cpu") -> "HybridDeepfakeDetector":
        model = HybridDeepfakeDetector(pretrained=False)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model


# ─────────────────────────────────────────────
# Quick sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    model = HybridDeepfakeDetector(pretrained=False)
    dummy = torch.randn(4, 3, 224, 224)
    out = model.predict_proba(dummy)
    total = sum(p.numel() for p in model.parameters())
    print(f"Output shape : {out.shape}")
    print(f"Total params : {total:,}")
    print("Model OK ✓")
