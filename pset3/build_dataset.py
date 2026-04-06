"""
HD-EPIC Kitchens -> VLM Fine-Tuning Dataset Builder

Extracts frames from P01/P02 videos at narration midpoints,
pairs them with verb+noun action labels, and outputs a JSONL
dataset ready for Qwen2.5-VL LoRA fine-tuning.

Output structure:
  mmai-data/
  ├── images/
  │   ├── P01_00001.jpg
  │   └── ...
  ├── data.jsonl       (training split)
  └── test.jsonl       (held-out test split)

Then creates mmai-data.zip for upload to Google Drive / Colab.

Usage:
  cd pset3/
  python build_dataset.py
"""

import json
import os
import pickle
import random
import subprocess
import zipfile
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # mmaihw/
NARRATIONS_PKL = ROOT / "hdepics_annotations" / "narrations-and-action-segments" / "HD_EPIC_Narrations.pkl"
VIDEOS = {
    "P01-20240202-110250": ROOT / "hdepic_example" / "P01-20240202-110250.mp4",
    "P02-20240209-184316": ROOT / "hdepic_example" / "P02-20240209-184316.mp4",
}
OUTPUT_DIR = ROOT / "pset3" / "mmai-data"
IMAGES_DIR = OUTPUT_DIR / "images"
TRAIN_JSONL = OUTPUT_DIR / "data.jsonl"
TEST_JSONL = OUTPUT_DIR / "test.jsonl"
ZIP_PATH = ROOT / "pset3" / "mmai-data.zip"

# ── Sampling ───────────────────────────────────────────────────
# Keep dataset small: sample every Nth narration per video.
# P01 has 224 narrations, P02 has 1171.
# Target ~120 frames total -> P01 every 2nd (~112), P02 every 12th (~98)
SAMPLE_EVERY = {
    "P01-20240202-110250": 2,
    "P02-20240209-184316": 12,
}

# ── Train/test split ──────────────────────────────────────────
TEST_RATIO = 0.2  # 20% held out for testing
SEED = 42

# ── Consistent question prompt ────────────────────────────────
QUESTION = "What action is being performed?"

# ── Frame resolution (keep zip small) ─────────────────────────
FRAME_WIDTH = 512  # height scales proportionally


def load_narrations():
    """Load and filter narrations for our two videos."""
    with open(NARRATIONS_PKL, "rb") as f:
        df = pickle.load(f)

    df = df[df["video_id"].isin(VIDEOS.keys())].copy()
    # Drop rows with empty main_actions
    df = df[df["main_actions"].apply(len) > 0].copy()
    df = df.sort_values(["video_id", "start_timestamp"]).reset_index(drop=True)
    return df


def extract_frame(video_path: Path, timestamp_sec: float, output_path: Path):
    """Extract a single frame at the given timestamp using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-ss", f"{timestamp_sec:.3f}",
        "-i", str(video_path),
        "-frames:v", "1",
        "-vf", f"scale={FRAME_WIDTH}:-2",
        "-q:v", "4",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  WARNING: ffmpeg failed for {output_path.name}: {result.stderr[-200:]}")
        return False
    return True


def format_action(main_actions: list) -> str:
    """Format main_actions list -> 'verb noun' string."""
    verb, noun = main_actions[0]
    return f"{verb} {noun}"


def create_zip(output_dir: Path, zip_path: Path):
    """Create a zip with images/ and *.jsonl at the root (no parent folder)."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fpath in sorted(output_dir.rglob("*")):
            if fpath.is_file():
                arcname = fpath.relative_to(output_dir)
                zf.write(fpath, arcname)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nZip created: {zip_path} ({size_mb:.1f} MB)")


def main():
    random.seed(SEED)

    print("Loading narrations...")
    df = load_narrations()
    print(f"  {len(df)} narrations for P01+P02 (after filtering empty actions)")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # ── Extract frames ────────────────────────────────────────
    records = []
    frame_idx = 0

    for video_id, video_path in VIDEOS.items():
        if not video_path.exists():
            print(f"  SKIP: {video_path} not found")
            continue

        video_df = df[df["video_id"] == video_id]
        step = SAMPLE_EVERY.get(video_id, 5)
        sampled = video_df.iloc[::step]
        print(f"\n{video_id}: {len(video_df)} narrations -> sampling every {step}th -> {len(sampled)} frames")

        for _, row in sampled.iterrows():
            midpoint = (row["start_timestamp"] + row["end_timestamp"]) / 2.0
            frame_idx += 1
            fname = f"{row['participant_id']}_{frame_idx:05d}.jpg"
            out_path = IMAGES_DIR / fname

            ok = extract_frame(video_path, midpoint, out_path)
            if not ok:
                continue

            answer = format_action(row["main_actions"])
            records.append({
                "image": f"images/{fname}",
                "question": QUESTION,
                "answer": answer,
            })

            if frame_idx % 20 == 0:
                print(f"  Extracted {frame_idx} frames...")

    # ── Train/test split ──────────────────────────────────────
    random.shuffle(records)
    n_test = max(4, int(len(records) * TEST_RATIO))  # at least 4 for Problem 3.2
    test_records = records[:n_test]
    train_records = records[n_test:]

    # Write train JSONL
    with open(TRAIN_JSONL, "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")

    # Write test JSONL
    with open(TEST_JSONL, "w") as f:
        for rec in test_records:
            f.write(json.dumps(rec) + "\n")

    print(f"\nDone! {len(train_records)} train + {len(test_records)} test = {len(records)} total")
    print(f"  Train: {TRAIN_JSONL}")
    print(f"  Test:  {TEST_JSONL}")

    # Sample entries
    print("\nSample TRAIN entries:")
    for rec in train_records[:3]:
        print(f"  {json.dumps(rec)}")
    print("\nSample TEST entries:")
    for rec in test_records[:3]:
        print(f"  {json.dumps(rec)}")

    # ── Create zip for Google Drive upload ────────────────────
    create_zip(OUTPUT_DIR, ZIP_PATH)
    print(f"\nUpload {ZIP_PATH.name} to Google Drive, then paste the share link into the notebook.")


if __name__ == "__main__":
    main()
