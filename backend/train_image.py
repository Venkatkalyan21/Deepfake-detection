"""
train_image.py — Training script for EfficientNetV2-S image deepfake detector
Uses MTCNN-preprocessed face crops for training.

Dataset structure:
  data/images/
    train/
      real/  ← real face crops (JPG/PNG)
      fake/  ← deepfake face crops (JPG/PNG)
    val/
      real/
      fake/

Compatible datasets:
  - FaceForensics++ (face crops from real/manipulated videos)
  - Celeb-DF v2 (face crops)
  - DFDC face crops
  - Any real/fake image folder pair

Usage:
  python train_image.py \
    --data_dir   ../data/images \
    --epochs     30 \
    --batch_size 32 \
    --lr         1e-4 \
    --save_dir   ../models
"""

import argparse, os, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.metrics import roc_auc_score, accuracy_score
from PIL import Image

from image_detector import EfficientNetV2Detector, train_transform, inference_transform

# ─────────────────────────────────────────────────────────────────
SEED      = 42
IMG_EXTS  = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
class ImageFakeDataset(Dataset):
    """
    Folder-based image dataset.
    Expects: <root>/<split>/real/  and  <root>/<split>/fake/
    """

    def __init__(self, root: str, split: str, augment: bool = False):
        self.transform = train_transform if augment else inference_transform
        self.samples   = []

        for label, name in [(0, "real"), (1, "fake")]:
            folder = Path(root) / split / name
            if not folder.exists():
                print(f"[WARN] Folder not found: {folder}")
                continue
            for f in folder.rglob("*"):
                if f.suffix.lower() in IMG_EXTS:
                    self.samples.append((str(f), label))

        random.shuffle(self.samples)
        real_n = sum(1 for _, l in self.samples if l == 0)
        fake_n = sum(1 for _, l in self.samples if l == 1)
        print(f"[Dataset/{split}] real={real_n}  fake={fake_n}  total={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        tensor = self.transform(img)
        return tensor, torch.tensor(label, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────
def mixup(x, y, alpha=0.2):
    """MixUp augmentation for better generalisation."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx    = torch.randperm(x.size(0))
    mixed  = lam * x + (1 - lam) * x[idx]
    y_mix  = lam * y + (1 - lam) * y[idx]
    return mixed, y_mix


def run_epoch(model, loader, criterion, optimizer, device, training: bool, use_mixup: bool = False):
    model.train(training)
    total_loss, all_probs, all_labels = 0.0, [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)

        if training and use_mixup:
            imgs, labels = mixup(imgs, labels, alpha=0.2)

        with torch.set_grad_enabled(training):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)

        if training:
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item() * len(labels)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / max(len(loader.dataset), 1)
    preds    = [1 if p >= 0.5 else 0 for p in all_probs]
    # For mixed labels, round for metric computation
    int_labels = [round(l) for l in all_labels]
    acc        = accuracy_score(int_labels, preds)
    try:
        auc = roc_auc_score(int_labels, all_probs)
    except Exception:
        auc = 0.5
    return avg_loss, acc, auc


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main(args):
    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}  |  Data: {args.data_dir}")

    # ── Datasets ──────────────────────────────────────────────
    train_ds = ImageFakeDataset(args.data_dir, "train", augment=True)
    val_ds   = ImageFakeDataset(args.data_dir, "val",   augment=False)

    if len(train_ds) == 0:
        print("\n[WARN] No training images found!")
        print("Expected structure:")
        print("  data/images/train/real/*.jpg")
        print("  data/images/train/fake/*.jpg")
        print("  data/images/val/real/*.jpg")
        print("  data/images/val/fake/*.jpg")
        print("\nYou can generate face crops from FaceForensics++ using preprocess_celebdf.py")
        print("Running in demo mode (no actual training).")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4,
    )

    # ── Model ─────────────────────────────────────────────────
    model = EfficientNetV2Detector(pretrained=True)

    # Freeze backbone, only train classifier head initially
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.to(device)

    # ── Loss, optimiser, scheduler ────────────────────────────
    real_n     = sum(1 for _, l in train_ds.samples if l == 0)
    fake_n     = sum(1 for _, l in train_ds.samples if l == 1)
    pos_weight = torch.tensor([real_n / max(fake_n, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1)

    # ── Training loop — Phase 1 (frozen backbone) ─────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    print("\n" + "=" * 65)
    print("  DeepShield — Image Detector Training (EfficientNetV2-S)")
    print("=" * 65)
    print(f"  Phase 1: {args.epochs} epochs with frozen backbone")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, True,  use_mixup=True)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader,   criterion, None,      device, False, use_mixup=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Epoch {epoch:03d}/{args.epochs}  "
            f"| train loss={tr_loss:.4f}  acc={tr_acc:.3f}  AUC={tr_auc:.3f}"
            f"  | val loss={va_loss:.4f}  acc={va_acc:.3f}  AUC={va_auc:.3f}"
            f"  | {elapsed:.1f}s"
        )

        if va_auc > best_auc:
            best_auc = va_auc
            ckpt = save_dir / "image_model_best.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  ✔  Best model saved → {ckpt}  (AUC={best_auc:.4f})")

    # ── Phase 2: Unfreeze + fine-tune whole network ───────────
    if args.finetune_epochs > 0:
        print(f"\n  Phase 2: Unfreezing backbone for {args.finetune_epochs} epochs...")
        for param in model.backbone.parameters():
            param.requires_grad = True

        optimizer2 = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        scheduler2 = CosineAnnealingWarmRestarts(optimizer2, T_0=args.finetune_epochs)

        for epoch in range(1, args.finetune_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion, optimizer2, device, True)
            va_loss, va_acc, va_auc = run_epoch(model, val_loader,   criterion, None,       device, False)
            scheduler2.step()
            elapsed = time.time() - t0
            print(
                f"  [P2] Epoch {epoch:03d}/{args.finetune_epochs}  "
                f"| val AUC={va_auc:.3f}  acc={va_acc:.3f} | {elapsed:.1f}s"
            )
            if va_auc > best_auc:
                best_auc = va_auc
                ckpt = save_dir / "image_model_best.pth"
                torch.save(model.state_dict(), ckpt)
                print(f"  ✔  Best model updated → {ckpt}  (AUC={best_auc:.4f})")

    torch.save(model.state_dict(), save_dir / "image_model_final.pth")
    print(f"\n[Done] Final model saved → {save_dir / 'image_model_final.pth'}")
    print(f"[Done] Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Image Deepfake Detector Training")
    parser.add_argument("--data_dir",        type=str,   default="../data/images")
    parser.add_argument("--epochs",          type=int,   default=30,
                        help="Phase-1 epochs (frozen backbone)")
    parser.add_argument("--finetune_epochs", type=int,   default=10,
                        help="Phase-2 epochs (full fine-tune)")
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--save_dir",        type=str,   default="../models")
    args = parser.parse_args()
    main(args)
