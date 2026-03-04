import os
import torch
from pathlib import Path

# --- AMD/ROCm Stability Fixes ---
os.environ["MIOPEN_FIND_MODE"] = "3"
os.environ["MIOPEN_DISABLE_CACHE"] = "1"
os.environ["MIOPEN_DEBUG_CONV_IMPLICIT_GEMM"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "max_split_size_mb:128"

from ultralytics import YOLO

# --- CONFIGURATION ---
BASE_DIR = r"C:\dev\projects\PycharmProjects\tzefa"
DATASET_ROOT = os.path.join(BASE_DIR, r"Tzefa_Datasets\Line_Segmentation\Unified_Batch")
DATA_YAML_PATH = os.path.join(DATASET_ROOT, "data.yaml")

# SOTA Model (Extra Large) - Fresh Start
PRETRAINED_MODEL = "yolo11x-obb.pt"

PROJECT_DIR = os.path.join(BASE_DIR, r"Tzefa_Models\Line_Segmentation")
RUN_NAME = "yolo11x-obb-fresh"  # Unique name for this training run

EPOCHS = 50
IMG_SIZE = 640
# 'X' model is heavy. If OOM, lower to 4 or 2.
BATCH_SIZE = 4


def clear_cache():
    """Deletes .npy cache files to force YOLO to re-check the data."""
    print("🧹 Cleaning old .npy cache files...")

    # Check both root/images and split directories (train/images, val/images)
    dirs_to_check = [
        Path(DATASET_ROOT) / "images",
        Path(DATASET_ROOT) / "train" / "images",
        Path(DATASET_ROOT) / "val" / "images",
        Path(DATASET_ROOT) / "test" / "images"
    ]

    total_deleted = 0
    for images_dir in dirs_to_check:
        if not images_dir.exists():
            continue

        npy_files = list(images_dir.glob("*.npy"))
        for f in npy_files:
            try:
                os.remove(f)
                total_deleted += 1
            except:
                pass

    print(f"   Deleted {total_deleted} cache files.")


def train():
    clear_cache()

    if not os.path.exists(DATA_YAML_PATH):
        print(f"❌ Error: data.yaml not found at {DATA_YAML_PATH}")
        return

    print(f"Loading SOTA model: {PRETRAINED_MODEL}...")
    model = YOLO(PRETRAINED_MODEL)

    # --- RESTORED: CHANNELS LAST OPTIMIZATION ---
    print("⚡ Optimizing memory format to Channels Last (NHWC)...")
    try:
        model.model.to(memory_format=torch.channels_last)
    except Exception as e:
        print(f"⚠️ Warning: Channels Last failed ({e}). Continuing without it.")
    # --------------------------------------------

    print(f"Starting Fresh Training in {RUN_NAME}...")

    try:
        model.train(
            data=DATA_YAML_PATH,
            project=PROJECT_DIR,
            name=RUN_NAME,
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            batch=BATCH_SIZE,
            device=0,
            patience=15,
            save=True,
            exist_ok=True,
            cache="disk",
            # --- RESTORED: MIXED PRECISION ---
            amp=True,  # <--- Back to True (Faster, standard for SOTA)
            # Standard settings
            lr0=0.01,
            warmup_epochs=3.0,
            # Augmentations
            degrees=180.0,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.1,
            copy_paste=0.1,
        )
    except Exception as e:
        print(f"Training crashed: {e}")

    print(f"Done.")

if __name__ == "__main__":
    train()