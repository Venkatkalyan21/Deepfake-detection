"""
train_audio.py — Training script for AudioDeepfakeDetector (Wav2Vec2)

Compatible dataset formats:
  1. ASVspoof 2019 / 2021  (LA / PA protocols)
  2. Custom folders:
       data/audio/
         train/real/  ← bonafide .wav / .flac / .mp3 files
         train/fake/  ← spoofed  .wav / .flac / .mp3 files
         val/real/
         val/fake/

Usage:
  python train_audio.py \
    --data_dir  ../data/audio \
    --epochs    20 \
    --batch_size 16 \
    --lr        1e-4 \
    --save_dir  ../models
"""

import argparse, os, time, random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, accuracy_score

try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    print("[ERROR] librosa not installed. Run: pip install librosa>=0.10.0")
    LIBROSA_AVAILABLE = False

try:
    from transformers import Wav2Vec2FeatureExtractor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("[ERROR] transformers not installed. Run: pip install transformers>=4.30.0")
    TRANSFORMERS_AVAILABLE = False

from audio_detector import AudioDeepfakeDetector

# ─────────────────────────────────────────────────────────────────
SAMPLE_RATE      = 16_000
MAX_DURATION_SEC = 6.0
MAX_SAMPLES      = int(MAX_DURATION_SEC * SAMPLE_RATE)
SEED             = 42


def seed_everything(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────
AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


class AudioFakeDataset(Dataset):
    """
    Folder-based audio dataset.
    Expects: <root>/<split>/real/  and  <root>/<split>/fake/
    """

    def __init__(self, root: str, split: str, feature_extractor):
        self.feature_extractor = feature_extractor
        self.samples = []

        for label, name in [(0, "real"), (1, "fake")]:
            folder = Path(root) / split / name
            if not folder.exists():
                print(f"[WARN] Folder not found: {folder}")
                continue
            for f in folder.rglob("*"):
                if f.suffix.lower() in AUDIO_EXTS:
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
            waveform, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            waveform = np.zeros(SAMPLE_RATE, dtype=np.float32)

        # Pad / truncate
        if len(waveform) > MAX_SAMPLES:
            start    = random.randint(0, len(waveform) - MAX_SAMPLES)
            waveform = waveform[start: start + MAX_SAMPLES]
        else:
            waveform = np.pad(waveform, (0, MAX_SAMPLES - len(waveform)))

        inputs = self.feature_extractor(
            waveform.astype(np.float32),
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )
        return inputs.input_values.squeeze(0), torch.tensor(label, dtype=torch.float32)


def collate_fn(batch):
    input_values, labels = zip(*batch)
    # Pad to the longest sample in the batch
    max_len = max(x.shape[0] for x in input_values)
    padded  = torch.stack([
        torch.nn.functional.pad(x, (0, max_len - x.shape[0])) for x in input_values
    ])
    return padded, torch.stack(labels)


# ─────────────────────────────────────────────────────────────────
# Training helpers
# ─────────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    total_loss, all_probs, all_labels = 0.0, [], []

    for input_values, labels in loader:
        input_values = input_values.to(device)
        labels       = labels.to(device)

        with torch.set_grad_enabled(training):
            logits = model(input_values).squeeze(1)
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

    avg_loss = total_loss / len(loader.dataset)
    preds    = [1 if p >= 0.5 else 0 for p in all_probs]
    acc      = accuracy_score(all_labels, preds)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except Exception:
        auc = 0.5
    return avg_loss, acc, auc


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main(args):
    if not LIBROSA_AVAILABLE or not TRANSFORMERS_AVAILABLE:
        raise SystemExit("Missing dependencies. See error messages above.")

    seed_everything()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] Device: {device}  |  Data: {args.data_dir}")

    # ── Model ─────────────────────────────────────────────────
    model = AudioDeepfakeDetector(pretrained=True, freeze_base=True)
    model.to(device)

    # ── Feature extractor for dataset ─────────────────────────
    feat_ext = model.feature_extractor

    # ── Datasets & loaders ────────────────────────────────────
    train_ds = AudioFakeDataset(args.data_dir, "train", feat_ext)
    val_ds   = AudioFakeDataset(args.data_dir, "val",   feat_ext)

    if len(train_ds) == 0:
        print("\n[WARN] No training files found!")
        print("Expected structure:")
        print("  data/audio/train/real/*.wav")
        print("  data/audio/train/fake/*.wav")
        print("  data/audio/val/real/*.wav")
        print("  data/audio/val/fake/*.wav")
        print("\nRunning in demo mode (no actual training).")
        return

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device == "cuda"),
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn,
    )

    # ── Loss, optimiser, scheduler ────────────────────────────
    # Compute class weights to handle imbalance
    real_n = sum(1 for _, l in train_ds.samples if l == 0)
    fake_n = sum(1 for _, l in train_ds.samples if l == 1)
    pos_weight = torch.tensor([real_n / max(fake_n, 1)], device=device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Only fine-tune the classifier head (Wav2Vec2 frozen)
    optimizer  = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler  = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_auc = 0.0

    print("\n" + "=" * 60)
    print("  DeepShield — Audio Detector Training (Wav2Vec2)")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader,   criterion, None,      device, training=False)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:03d}/{args.epochs}  "
            f"| train loss={tr_loss:.4f}  acc={tr_acc:.3f}  AUC={tr_auc:.3f}"
            f"  | val loss={va_loss:.4f}  acc={va_acc:.3f}  AUC={va_auc:.3f}"
            f"  | {elapsed:.1f}s"
        )

        if va_auc > best_auc:
            best_auc = va_auc
            ckpt = save_dir / "audio_model_best.pth"
            torch.save(model.state_dict(), ckpt)
            print(f"  ✔  Best model saved → {ckpt}  (AUC={best_auc:.4f})")

    # Save final checkpoint
    final = save_dir / "audio_model_final.pth"
    torch.save(model.state_dict(), final)
    print(f"\n[Done] Final model saved → {final}")
    print(f"[Done] Best val AUC: {best_auc:.4f}")

    # ── Phase 2: Unfreeze Wav2Vec2 and fine-tune further ──────
    if args.unfreeze_epochs > 0:
        print(f"\n[Phase 2] Unfreezing Wav2Vec2 for {args.unfreeze_epochs} more epochs...")
        for param in model.wav2vec2.parameters():
            param.requires_grad = True
        optimizer2 = AdamW(model.parameters(), lr=args.lr * 0.1, weight_decay=1e-4)
        scheduler2 = CosineAnnealingLR(optimizer2, T_max=args.unfreeze_epochs, eta_min=1e-7)

        for epoch in range(1, args.unfreeze_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion, optimizer2, device, True)
            va_loss, va_acc, va_auc = run_epoch(model, val_loader,   criterion, None,       device, False)
            scheduler2.step()
            elapsed = time.time() - t0
            print(
                f"[P2] Epoch {epoch:03d}/{args.unfreeze_epochs}  "
                f"| val AUC={va_auc:.3f}  acc={va_acc:.3f} | {elapsed:.1f}s"
            )
            if va_auc > best_auc:
                best_auc = va_auc
                ckpt = save_dir / "audio_model_best.pth"
                torch.save(model.state_dict(), ckpt)
                print(f"  ✔  Best model updated → {ckpt}  (AUC={best_auc:.4f})")

        torch.save(model.state_dict(), save_dir / "audio_model_phase2.pth")
        print(f"\n[Done] Phase-2 model saved. Best AUC = {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepShield Audio Deepfake Detector Training")
    parser.add_argument("--data_dir",        type=str, default="../data/audio",
                        help="Root folder with train/val subfolders")
    parser.add_argument("--epochs",          type=int, default=20)
    parser.add_argument("--unfreeze_epochs", type=int, default=5,
                        help="Extra epochs after unfreezing Wav2Vec2 backbone")
    parser.add_argument("--batch_size",      type=int, default=16)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--save_dir",        type=str, default="../models")
    args = parser.parse_args()
    main(args)
