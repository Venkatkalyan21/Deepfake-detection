"""
Training Script — IEEE Research Experiments
Trains HybridDeepfakeDetector on FaceForensics++ or Celeb-DF v2

Usage:
  python train.py --data_dir /path/to/dataset --epochs 30 --batch_size 32
"""
import argparse, time, os
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import roc_auc_score

from model import HybridDeepfakeDetector

# ── Dataset ─────────────────────────────────────────────────────────
class DeepfakeDataset(Dataset):
    """
    Expects directory layout:
      data_dir/
        real/   ← real face crops (PNG/JPG)
        fake/   ← fake face crops
    """
    TRAIN_TF = T.Compose([
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        T.RandomRotation(10),
        T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    VAL_TF = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, data_dir: str, split: str = "train"):
        self.tf = self.TRAIN_TF if split == "train" else self.VAL_TF
        self.samples = []
        for label, folder in [(0, "real"), (1, "fake")]:
            p = Path(data_dir) / folder
            if p.exists():
                for img in p.rglob("*.jpg"):
                    self.samples.append((str(img), label))
                for img in p.rglob("*.png"):
                    self.samples.append((str(img), label))

        if not self.samples:
            raise ValueError(f"No images found in {data_dir}. "
                             f"Ensure real/ and fake/ sub-directories exist under {data_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.tf(img), torch.tensor(label, dtype=torch.float32)


def get_sampler(dataset: DeepfakeDataset) -> WeightedRandomSampler:
    labels  = [s[1] for s in dataset.samples]
    counts  = [labels.count(0), labels.count(1)]
    weights = [1.0 / counts[l] for l in labels]
    return WeightedRandomSampler(weights, len(weights))


# ── Training loop ───────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss, n = 0.0, 0
    total_batches = len(loader)

    iterator = tqdm(loader, desc="  Training", unit="batch",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] loss={postfix}") \
               if tqdm else loader

    for batch_idx, (imgs, labels) in enumerate(iterator):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type=device, enabled=scaler is not None):
            logits = model(imgs).squeeze(1)
            loss   = criterion(logits, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
        # Update tqdm with current avg loss
        if tqdm and hasattr(iterator, 'set_postfix_str'):
            iterator.set_postfix_str(f"{total_loss/n:.4f}")
        elif not tqdm and (batch_idx % 50 == 0 or batch_idx == total_batches - 1):
            pct = (batch_idx + 1) / total_batches * 100
            print(f"    Batch {batch_idx+1}/{total_batches} ({pct:.0f}%)  avg_loss={total_loss/n:.4f}",
                  flush=True)
    return total_loss / n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        probs = torch.sigmoid(model(imgs).squeeze(1)).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy())
    auc = roc_auc_score(all_labels, all_probs)
    preds = [1 if p >= 0.5 else 0 for p in all_probs]
    acc = np.mean(np.array(preds) == np.array(all_labels))
    return {"auc": auc, "acc": acc}


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   required=True)
    parser.add_argument("--epochs",     type=int,   default=30)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_dir",   default="models")
    parser.add_argument("--device",     default=None)
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")

    # Datasets
    # data_dir should be the base data/ folder; train/val subdirs are appended internally
    train_split_dir = str(Path(args.data_dir) / "train")
    val_split_dir   = str(Path(args.data_dir) / "val")
    train_ds = DeepfakeDataset(train_split_dir, "train")
    val_ds   = DeepfakeDataset(val_split_dir,   "val")
    sampler  = get_sampler(train_ds)

    # num_workers=0 avoids Windows multiprocessing issues
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,  num_workers=0, pin_memory=False, drop_last=True)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    print(f"[Train] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    # Model
    model = HybridDeepfakeDetector(pretrained=True).to(device)

    # Loss with label smoothing
    criterion = nn.BCEWithLogitsLoss(label_smoothing=0.1) if hasattr(
        nn.BCEWithLogitsLoss, "label_smoothing"
    ) else nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler    = torch.cuda.amp.GradScaler() if device == "cuda" else None

    os.makedirs(args.save_dir, exist_ok=True)
    best_auc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0    = time.time()
        loss  = train_epoch(model, train_dl, optimizer, criterion, device, scaler)
        mets  = evaluate(model, val_dl, device)
        scheduler.step()

        print(f"Epoch {epoch:03d}/{args.epochs}  "
              f"Loss={loss:.4f}  AUC={mets['auc']*100:.2f}%  "
              f"ACC={mets['acc']*100:.2f}%  "
              f"[{time.time()-t0:.1f}s]")

        if mets["auc"] > best_auc:
            best_auc = mets["auc"]
            save_path = Path(args.save_dir) / "deepfake_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Best model saved → {save_path}  (AUC={best_auc*100:.2f}%)")

    print(f"\n[Done] Best AUC: {best_auc*100:.2f}%")


if __name__ == "__main__":
    main()
