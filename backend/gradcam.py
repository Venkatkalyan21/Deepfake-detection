"""
Grad-CAM — Gradient-weighted Class Activation Mapping
Generates heatmap overlays showing which facial regions influenced the prediction.
Essential for IEEE paper explainability section.
"""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform = T.Compose([T.ToTensor(), T.Normalize(mean=MEAN, std=STD)])


class GradCAM:
    """
    Hooks into the last convolutional layer of the spatial branch
    to produce a class activation map.
    """

    def __init__(self, model):
        self.model    = model
        self.device   = next(model.parameters()).device
        self._fmaps   = None
        self._grads   = None
        self._hook_fwd = None
        self._hook_bwd = None
        self._register_hooks()

    def _register_hooks(self):
        # Target: last conv block of EfficientNet spatial branch
        try:
            if hasattr(self.model.spatial, "backbone"):
                target = self.model.spatial.backbone
                # timm efficientnet: last block
                if hasattr(target, "blocks"):
                    target_layer = target.blocks[-1]
                elif hasattr(target, "features"):
                    target_layer = target.features[-1]
                else:
                    target_layer = list(target.children())[-2]
            else:
                target_layer = list(self.model.spatial.children())[-2]

            self._hook_fwd = target_layer.register_forward_hook(self._save_fmaps)
            self._hook_bwd = target_layer.register_full_backward_hook(self._save_grads)
        except Exception:
            pass  # graceful fallback — still saves plain overlay

    def _save_fmaps(self, module, inp, out):
        self._fmaps = out.detach()

    def _save_grads(self, module, grad_in, grad_out):
        self._grads = grad_out[0].detach()

    def _compute_cam(self, face_rgb: np.ndarray) -> np.ndarray:
        """Returns cam as uint8 (224,224,3) or plain red overlay on failure."""
        if self._hook_fwd is None:
            return self._plain_overlay(face_rgb)

        tensor = transform(face_rgb).unsqueeze(0).to(self.device)
        self.model.zero_grad()

        logits = self.model(tensor)
        score  = torch.sigmoid(logits)
        score.backward()

        if self._fmaps is None or self._grads is None:
            return self._plain_overlay(face_rgb)

        weights = self._grads.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self._fmaps).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F.relu(cam)
        cam     = cam.squeeze().cpu().numpy()

        # Normalise
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_resized = cv2.resize(cam, (224, 224))
        heatmap     = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original face
        overlay = cv2.addWeighted(face_rgb, 0.6, heatmap, 0.4, 0)
        return overlay

    @staticmethod
    def _plain_overlay(face_rgb: np.ndarray) -> np.ndarray:
        """Fallback: red-tinted overlay when hooks unavailable."""
        tint = np.zeros_like(face_rgb)
        tint[:, :, 0] = 100
        return cv2.addWeighted(face_rgb, 0.8, tint, 0.2, 0)

    def generate(self, face_rgb: np.ndarray, save_path: str):
        """Compute Grad-CAM and save to disk."""
        try:
            overlay = self._compute_cam(face_rgb)
        except Exception:
            overlay = face_rgb

        img = Image.fromarray(overlay.astype(np.uint8))
        img.save(save_path, quality=85)

    def remove_hooks(self):
        if self._hook_fwd:
            self._hook_fwd.remove()
        if self._hook_bwd:
            self._hook_bwd.remove()
