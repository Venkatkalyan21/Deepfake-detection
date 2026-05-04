"""
preprocess_celebdf.py — Extract face crops from Celeb-DF v2 for training
Reads from:
  Celeb-DF/Celeb-real/       → label 0 (REAL)
  Celeb-DF/YouTube-real/     → label 0 (REAL)
  Celeb-DF/Celeb-synthesis/  → label 1 (FAKE)

Outputs to:
  data/train/real/  and  data/train/fake/
  data/val/real/    and  data/val/fake/

Usage:
  python preprocess_celebdf.py --dataset_dir ../Celeb-DF --out_dir ../data
                               --frames_per_video 15 --val_split 0.15
"""
import argparse
import os
import random
import sys
from pathlib import Path

import cv2

# ── Face detector (OpenCV, no extra deps) ────────────────────────────
CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_det = cv2.CascadeClassifier(CASCADE)


def extract_faces(video_path: str, n_frames: int = 15, size: int = 224):
    """Sample n_frames evenly, detect face, return list of BGR crops."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = sorted(random.sample(range(total), min(n_frames, total)))
    crops = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_det.detectMultiScale(gray, 1.1, 4, minSize=(48, 48))
        h, w = frame.shape[:2]

        if len(faces) == 0:
            # centre crop fallback
            s = min(h, w)
            y0, x0 = (h - s) // 2, (w - s) // 2
            crop = frame[y0:y0+s, x0:x0+s]
        else:
            fx, fy, fw, fh = max(faces, key=lambda r: r[2]*r[3])
            pad = int(max(fw, fh) * 0.2)
            x1, y1 = max(0, fx-pad), max(0, fy-pad)
            x2, y2 = min(w, fx+fw+pad), min(h, fy+fh+pad)
            crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue
        crops.append(cv2.resize(crop, (size, size)))

    cap.release()
    return crops


def save_crops(crops, out_dir: Path, stem: str):
    saved = 0
    for i, crop in enumerate(crops):
        p = out_dir / f"{stem}_{i:03d}.jpg"
        cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        saved += 1
    return saved


def collect_videos(folders):
    videos = []
    for folder in folders:
        p = Path(folder)
        if p.exists():
            videos += list(p.glob("*.mp4")) + list(p.glob("*.avi"))
    return videos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="../Celeb-DF",
                        help="Path to Celeb-DF root (contains Celeb-real, Celeb-synthesis, YouTube-real)")
    parser.add_argument("--out_dir",     default="../data",
                        help="Output directory for face crops")
    parser.add_argument("--frames_per_video", type=int, default=15,
                        help="Face crops to extract per video")
    parser.add_argument("--val_split",   type=float, default=0.15,
                        help="Fraction of videos held out for validation")
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    ds = Path(args.dataset_dir)

    # ── Source folders ──────────────────────────────────────────────
    real_folders = [ds / "Celeb-real", ds / "YouTube-real"]
    fake_folders = [ds / "Celeb-synthesis"]

    real_videos = collect_videos(real_folders)
    fake_videos = collect_videos(fake_folders)

    print(f"Found {len(real_videos)} REAL videos | {len(fake_videos)} FAKE videos")

    def split(vids):
        random.shuffle(vids)
        n_val = max(1, int(len(vids) * args.val_split))
        return vids[n_val:], vids[:n_val]   # train, val

    real_train, real_val = split(real_videos)
    fake_train, fake_val = split(fake_videos)

    out = Path(args.out_dir)
    splits = {
        ("train", "real"): real_train,
        ("train", "fake"): fake_train,
        ("val",   "real"): real_val,
        ("val",   "fake"): fake_val,
    }

    # ── Extract ─────────────────────────────────────────────────────
    total_saved = 0
    for (split_name, label), vids in splits.items():
        out_dir = out / split_name / label
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{split_name}/{label}]  Processing {len(vids)} videos → {out_dir}")

        for i, vpath in enumerate(vids):
            crops = extract_faces(str(vpath), n_frames=args.frames_per_video)
            stem  = vpath.stem
            saved = save_crops(crops, out_dir, stem)
            total_saved += saved
            print(f"  [{i+1:04d}/{len(vids)}] {vpath.name}  →  {saved} crops", end="\r")

        print()  # newline after carriage-returns

    print(f"\n✅ Done! Total face crops saved: {total_saved}")
    print(f"   Output: {out.resolve()}")

    # Summary
    for (s, l), _ in splits.items():
        d = out / s / l
        n = len(list(d.glob("*.jpg")))
        print(f"   {s}/{l}: {n} images")


if __name__ == "__main__":
    main()
