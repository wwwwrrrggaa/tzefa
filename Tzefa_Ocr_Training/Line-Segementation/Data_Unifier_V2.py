"""
Data Unifier V2 for Line Segmentation Training

Combines all Line_Segmentation batches into a single Unified_Batch with:
  1. Binarization applied (matching inference pipeline)
  2. Squash-resize to 640x640 (matching inference behavior)
  3. Labels rescaled to 640x640 coordinate space
  4. 85/15 train/val split for meaningful mAP tracking
  5. Proper data.yaml for YOLO-OBB training

Processes:
  - Batch_1: Synthetic (640x640 already, needs binarization)
  - Batch_2: CORD receipts (needs resize + binarization)
  - Batch_6: IAM handwriting (needs resize + binarization)
  - Batch_7: Multi-scale synthetic (640x640 already, needs binarization)
  - Any future Batch_8+ datasets
"""

import os
import cv2
import torch
import shutil
import numpy as np
import yaml
import random
from pathlib import Path
from tqdm import tqdm
import segmentation_models_pytorch as smp
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Line_Segmentation")
OUTPUT_DIR = BASE_DIR / "Unified_Batch"
TARGET_SIZE = 640
VAL_RATIO = 0.15  # 15% validation split

# Binarization Model — use the same model as inference
BINARIZATION_CKPT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization\run_mitb5_highres_manet\step_3200.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Which batches to include and how to handle them
BATCH_CONFIGS = {
    "Batch_1":  {"mode": "BINARIZE_ONLY",   "is_iam": False},
    "Batch_2":  {"mode": "BINARIZE_RESIZE", "is_iam": False},
    "Batch_6":  {"mode": "BINARIZE_RESIZE", "is_iam": True},
    "Batch_7":  {"mode": "BINARIZE_ONLY",   "is_iam": False},
}


# --- BINARIZER (HighResMAnet mit_b5 — matches inference) ---
import torch.nn as nn


class HighResMAnet(nn.Module):
    """Same architecture as Tzefa_Ocr/Binarization.py"""
    def __init__(self, encoder_name="mit_b5", classes=1):
        super().__init__()
        self.base_model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=classes,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )
        self.high_res_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.final_fusion = nn.Sequential(
            nn.Conv2d(16 + 32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, classes, kernel_size=1),
        )

    def forward(self, x):
        high_res_features = self.high_res_stem(x)
        features = self.base_model.encoder(x)
        decoder_output = self.base_model.decoder(features)
        combined = torch.cat([decoder_output, high_res_features], dim=1)
        return self.final_fusion(combined)


class Binarizer:
    """Tiled binarization matching inference pipeline (Binarization.py)"""
    TILE_SIZE = 640
    BATCH_SIZE = 16

    def __init__(self, checkpoint_path):
        print(f"--- Loading Binarizer (HighResMAnet mit_b5) from {checkpoint_path} ---")
        self.model = HighResMAnet(encoder_name="mit_b5", classes=1)
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        self.model.load_state_dict(state)
        self.model.to(DEVICE)
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def process(self, img_bgr):
        if len(img_bgr.shape) == 2:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = img_rgb.shape[:2]
        pad_w = (self.TILE_SIZE - (orig_w % self.TILE_SIZE)) % self.TILE_SIZE
        pad_h = (self.TILE_SIZE - (orig_h % self.TILE_SIZE)) % self.TILE_SIZE
        if pad_w > 0 or pad_h > 0:
            img_rgb = cv2.copyMakeBorder(img_rgb, 0, pad_h, 0, pad_w,
                                         cv2.BORDER_CONSTANT, value=[255, 255, 255])

        new_h, new_w = img_rgb.shape[:2]
        result = np.full((new_h, new_w), 255, dtype=np.uint8)

        tiles = []
        coords = []

        # Simple sliding window
        for y in range(0, new_h, self.TILE_SIZE):
            for x in range(0, new_w, self.TILE_SIZE):
                tile = img_rgb[y:y + self.TILE_SIZE, x:x + self.TILE_SIZE]
                tiles.append(tile)
                coords.append((x, y))

                if len(tiles) >= self.BATCH_SIZE:
                    self._infer_and_paste(tiles, coords, result)
                    tiles, coords = [], []

        if tiles:
            self._infer_and_paste(tiles, coords, result)

        return result[0:orig_h, 0:orig_w]

    def _infer_and_paste(self, tiles, coords, canvas):
        tensor_list = []
        for tile in tiles:
            t = tile.astype(np.float32) / 255.0
            t = (t - self.mean) / self.std
            tensor_list.append(torch.from_numpy(t.transpose(2, 0, 1)).float())

        if not tensor_list:
            return

        batch = torch.stack(tensor_list).to(DEVICE)
        with torch.no_grad():
            logits = self.model(batch)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        for i, prob in enumerate(probs):
            x, y = coords[i]
            binary = np.where(prob > 0.5, 0, 255).astype(np.uint8)
            canvas[y:y + self.TILE_SIZE, x:x + self.TILE_SIZE] = binary


# --- HELPERS ---
def setup_structure():
    if OUTPUT_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_DIR)
        except Exception as e:
            print(f"Warning: Could not delete {OUTPUT_DIR}: {e}")

    for split in ["train", "val"]:
        (OUTPUT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    yaml_content = {
        "path": str(OUTPUT_DIR.absolute()),
        "train": "train/images",
        "val": "val/images",
        "names": {0: "line"},
    }
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)


def read_obb_labels(lbl_path):
    labels = []
    if not lbl_path.exists():
        return labels
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 9:
                coords = [float(x) for x in parts[1:]]
                labels.append(coords)
    return labels


def save_sample(img, labels, name, split):
    img_dir = OUTPUT_DIR / split / "images"
    lbl_dir = OUTPUT_DIR / split / "labels"

    # Save as png (lossless)
    if len(img.shape) == 2:
        img_3ch = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_3ch = img

    cv2.imwrite(str(img_dir / f"{name}.png"), img_3ch)

    with open(lbl_dir / f"{name}.txt", "w") as f:
        for coords in labels:
            coord_str = " ".join([f"{c:.6f}" for c in coords])
            f.write(f"0 {coord_str}\n")


def get_split():
    return "val" if random.random() < VAL_RATIO else "train"


def crop_iam_margin(img, labels):
    """
    Crops the top (header) and bottom (footer/name) of IAM form images.
    """
    if not labels:
        return img, labels

    h, w = img.shape[:2]

    # Gather Y coords
    all_ys = []
    for l in labels:
        poly_ys = l[1::2]
        all_ys.extend(poly_ys)

    if not all_ys:
        return img, labels

    min_y_norm = min(all_ys)
    max_y_norm = max(all_ys)

    min_y_px = int(min_y_norm * h)
    max_y_px = int(max_y_norm * h)

    # Margins (Top header ~50px above first line, Bottom footer ~50px below last line)
    margin = 50

    cut_y_top = max(0, min_y_px - margin)
    cut_y_bottom = min(h, max_y_px + margin)

    # Safety: don't crop if limits are weird
    if cut_y_top >= cut_y_bottom:
        return img, labels

    if cut_y_top == 0 and cut_y_bottom == h:
        return img, labels

    # Perform Crop
    cropped_img = img[cut_y_top:cut_y_bottom, :]
    new_h, new_w = cropped_img.shape[:2]

    # Adjust Labels
    new_labels = []
    for l in labels:
        new_coords = []
        for i, val in enumerate(l):
            if i % 2 == 0:
                # X coordinate
                new_coords.append(val)
            else:
                # Y coordinate
                y_px = val * h
                y_new_px = y_px - cut_y_top
                y_new_norm = y_new_px / new_h
                y_new_norm = max(0.0, min(1.0, y_new_norm))
                new_coords.append(y_new_norm)
        new_labels.append(new_coords)

    return cropped_img, new_labels


def process_single_image(args):
    """
    Worker function for multiprocessing.
    Main pipeline: Load (Bin) -> Crop -> Resize -> Save
    """
    img_path, lbl_path, batch_name, target_size, is_iam, forced_bin_path = args

    # 1. Read Labels
    if not lbl_path.exists(): return 0
    labels = read_obb_labels(lbl_path)
    if not labels: return 0

    # 2. Read Image (Expect Binarized Input)
    # If passed forced_bin_path, use it.
    # Otherwise read raw and assume it's already binary? No, this function expects binary flow now.
    # We will assume process_batch prepared the binary file or we load it from cache.

    src_path = forced_bin_path if (forced_bin_path and forced_bin_path.exists()) else img_path

    # Read grayscale as it should be binary/grayscale
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None: return 0

    # 3. Crop IAM Header & Footer
    if is_iam:
        img, labels = crop_iam_margin(img, labels)

    # 4. Filter empty/full
    if len(img.shape) == 2:
        ratio = np.sum(img == 255) / img.size
        if ratio > 0.999 or ratio < 0.001: return 0

    # 5. Resize
    h, w = img.shape[:2]
    if h != target_size or w != target_size:
        img = cv2.resize(img, (target_size, target_size))

    # 6. Save
    split = get_split()
    save_sample(img, labels, f"{batch_name.lower()}_{img_path.stem}", split)
    return 1


# --- BATCH PROCESSOR ---
def process_batch(batch_name, config, binarizer):
    mode = config["mode"]
    is_iam = config.get("is_iam", False)

    print(f"\n>>> Processing {batch_name} (Mode: {mode}, IAM-Crop: {is_iam})...")

    batch_dir = BASE_DIR / batch_name
    img_dir = batch_dir / "images"
    lbl_dir = batch_dir / "labels"
    bin_dir = batch_dir / "binarized_images"
    bin_dir.mkdir(exist_ok=True) # Ensure cache dir exists

    if not img_dir.exists():
        print(f"  [!] {img_dir} not found, skipping.")
        return 0

    # Collect all tasks
    images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))

    cpu_tasks = []
    gpu_tasks = []

    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        cached_path = bin_dir / f"{img_path.stem}.png"

        if cached_path.exists():
            # Already binarized -> CPU Task
            cpu_tasks.append((img_path, lbl_path, batch_name, TARGET_SIZE, is_iam, cached_path))
        else:
            # Needs Binarization -> GPU Task
            gpu_tasks.append((img_path, lbl_path, cached_path))

    # 1. Run GPU Inference for missing binaries
    if gpu_tasks and binarizer:
        print(f"  [GPU] Binarizing {len(gpu_tasks)} images...")
        for img_path, lbl_path, cached_path in tqdm(gpu_tasks, desc=f"{batch_name} [GPU]"):
            # Read Raw
            img = cv2.imread(str(img_path))
            if img is None: continue

            # Binarize FULL (Conserves geometry for subsequent crop)
            bin_img = binarizer.process(img)

            # Save to Cache
            cv2.imwrite(str(cached_path), bin_img)

            # Add to CPU queue for final formatting
            # Note: We pass cached_path as the source
            cpu_tasks.append((img_path, lbl_path, batch_name, TARGET_SIZE, is_iam, cached_path))

    elif gpu_tasks and not binarizer:
        print(f"  [!] skipped {len(gpu_tasks)} images because no binarizer loaded and no cache found.")

    # 2. Run CPU Processing (Crop -> Resize -> Save)
    if cpu_tasks:
        print(f"  [CPU] Formatting {len(cpu_tasks)} images...")
        # Reduce workers slightly to avoid system choke
        workers = max(1, multiprocessing.cpu_count() - 2)
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(tqdm(executor.map(process_single_image, cpu_tasks), total=len(cpu_tasks), desc=f"{batch_name} [CPU]"))
        return sum(results)

    return 0


# --- MAIN ---
if __name__ == "__main__":
    if multiprocessing.get_start_method(allow_none=True) != "spawn":
         # Safety for PyTorch + Multiprocessing
         pass

    setup_structure()

    # Initialize Binarizer only if needed?
    # For simplicity, we initialize it, but if all batches have pre-bin, we might not use it.
    # But checking all batches first is tedious. We load it.
    try:
        binarizer = Binarizer(BINARIZATION_CKPT)
    except Exception as e:
        print(f"Warning: Could not load Binarizer: {e}")
        binarizer = None

    total = 0

    for batch_name, config in BATCH_CONFIGS.items():
        batch_dir = BASE_DIR / batch_name
        if not batch_dir.exists():
            continue

        total += process_batch(batch_name, config, binarizer)

    print(f"\n{'='*60}")
    print(f"✅ Unification Complete! Total samples: {total}")
    print(f"   Output: {OUTPUT_DIR}")

