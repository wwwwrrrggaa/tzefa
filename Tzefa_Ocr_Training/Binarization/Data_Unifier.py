import os
import cv2
import numpy as np
import random
import shutil
from math import ceil
import multiprocessing
from functools import partial

# --- MASTER CONFIGURATION ---
BASE_PROJECT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization"

PATHS = {
    # 1. Small Images (256x256) -> STREAM (Native tiling)
    "BATCH_1": {
        "mode": "STREAM",
        "active": True,
        "images": os.path.join(BASE_PROJECT_DIR, "Batch_1", "images"),
        "masks":  os.path.join(BASE_PROJECT_DIR, "Batch_1", "masks")
    },

    # 2. Variable/Large Images -> STREAM (Cuts into tiles, wraps remainders)
    "BATCH_2": {
        "mode": "STREAM",
        "active": True,
        "images": os.path.join(BASE_PROJECT_DIR, "Batch_2", "images"),
        "masks":  os.path.join(BASE_PROJECT_DIR, "Batch_2", "masks")
    },

    # 3. Tiny Patches (48x48) -> STREAM (Tiles natively)
    "BATCH_3": {
        "mode": "STREAM",
        "active": True,
        "images": os.path.join(BASE_PROJECT_DIR, "Batch_3", "images"),
        "masks":  os.path.join(BASE_PROJECT_DIR, "Batch_3", "masks")
    },

    # 4. Generated (640x640) -> COPY (Already target size)
    "BATCH_4": {
        "mode": "COPY",
        "active": True,
        "images": os.path.join(BASE_PROJECT_DIR, "Batch_4", "images"),
        "masks":  os.path.join(BASE_PROJECT_DIR, "Batch_4", "masks")
    },

    # 5. Palm Leaf (Long/Short) -> STREAM (Sliding window logic handles long strips naturally)
    "BATCH_5": {
        "mode": "STREAM",
        "active": True,
        "images": os.path.join(BASE_PROJECT_DIR, "Batch_5", "images"),
        "masks":  os.path.join(BASE_PROJECT_DIR, "Batch_5", "masks")
    },

    # 6. Fixed 400x400 -> PAD_CENTER (Center in 640x640 frame)
    "BATCH_6": {
        "mode": "PAD_CENTER",
        "active": True,
        "images": os.path.join(BASE_PROJECT_DIR, "Batch_6", "images"),
        "masks":  os.path.join(BASE_PROJECT_DIR, "Batch_6", "masks")
    },

    # 7. NoisyOffice (Variable < 640) -> PAD_CENTER (Center with padding)
    "BATCH_7": {
        "mode": "PAD_CENTER",
        "active": True,
        "images": os.path.join(BASE_PROJECT_DIR, "Batch_7", "images"),
        "masks":  os.path.join(BASE_PROJECT_DIR, "Batch_7", "masks")
    }
}

# Final Output Directory
FINAL_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "Unified_Batch")
TARGET_SIZE = 640

# Global output dirs (will be created in main)
OUT_IMG_DIR = os.path.join(FINAL_OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(FINAL_OUTPUT_DIR, "masks")

# --- Helper: Validation & Normalization ---
def normalize_pair(img, msk):
    """
    Ensures:
    1. Image is 3-channel (BGR)
    2. Mask is 1-channel (Grayscale)
    3. Mask is strictly binary (0 or 255)
    """
    # 1. Image Check
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4: # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # 2. Mask Check
    if len(msk.shape) == 3:
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

    # 3. Strict Binarization (Threshold at 127)
    # Output: 0 (Black/Ink) or 255 (White/Background)
    _, msk = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)

    return img, msk

def save_pair(img, msk, batch_name, idx):
    """Saves standardized pair to disk."""
    # Force format: JPG for Image, PNG for Mask
    img_name = f"{batch_name}_{idx:06d}.jpg"
    msk_name = f"{batch_name}_{idx:06d}.png"

    cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.imwrite(os.path.join(OUT_MSK_DIR, msk_name), msk)

# --- Helper: Find Pairs ---
def get_image_mask_pairs(img_dir, msk_dir):
    if not os.path.exists(img_dir) or not os.path.exists(msk_dir):
        print(f"  ! Warning: Directory not found: {img_dir}")
        return []

    pairs = []
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')

    # Simple matching by filename stem
    # Pre-caching mask filenames for speed
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(msk_dir) if f.lower().endswith(valid_exts)}

    for img_f in os.listdir(img_dir):
        if not img_f.lower().endswith(valid_exts):
            continue

        base_name = os.path.splitext(img_f)[0]
        if base_name in mask_files:
            pairs.append((img_f, mask_files[base_name]))

    return pairs

# --- Mode 1: Stream Stitcher ---
class StreamBuffer:
    def __init__(self, batch_name):
        self.name = batch_name
        self.canvas_img = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
        self.canvas_msk = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255
        self.cx = 0
        self.cy = 0
        self.count = 0

    def save(self):
        save_pair(self.canvas_img, self.canvas_msk, f"{self.name}_stream", self.count)
        self.count += 1
        self.canvas_img.fill(255)
        self.canvas_msk.fill(255)
        self.cx = 0
        self.cy = 0

    def push(self, img, msk):
        h, w = img.shape[:2]
        space_x = TARGET_SIZE - self.cx
        space_y = TARGET_SIZE - self.cy

        copy_w = min(w, space_x)
        copy_h = min(h, space_y)

        self.canvas_img[self.cy:self.cy+copy_h, self.cx:self.cx+copy_w] = img[0:copy_h, 0:copy_w]
        self.canvas_msk[self.cy:self.cy+copy_h, self.cx:self.cx+copy_w] = msk[0:copy_h, 0:copy_w]

        if copy_w < space_x:
            self.cx += copy_w
            if h > copy_h:
                self.push(img[copy_h:, :], msk[copy_h:, :])
        else:
            self.cx = 0
            self.cy += copy_h
            if self.cy >= TARGET_SIZE:
                self.save()
            if w > copy_w:
                self.push(img[0:copy_h, copy_w:], msk[0:copy_h, copy_w:])
            if h > copy_h:
                self.push(img[copy_h:, :], msk[copy_h:, :])

def process_stream_batch(batch_name, config):
    print(f"[{batch_name}] Starting Stream Mode...")
    pairs = get_image_mask_pairs(config["images"], config["masks"])
    if not pairs: return

    random.shuffle(pairs)
    buffer = StreamBuffer(batch_name)

    for img_f, msk_f in pairs:
        img = cv2.imread(os.path.join(config["images"], img_f), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(config["masks"], msk_f), cv2.IMREAD_GRAYSCALE)

        if img is None or msk is None: continue

        img, msk = normalize_pair(img, msk)
        buffer.push(img, msk)

    if buffer.cx > 0 or buffer.cy > 0:
        buffer.save()
    print(f"[{batch_name}] Finished.")

# --- Mode 2: Pad Center ---
def process_pad_center_batch(batch_name, config):
    print(f"[{batch_name}] Starting Pad-Center Mode...")
    pairs = get_image_mask_pairs(config["images"], config["masks"])
    if not pairs: return

    count = 0
    for img_f, msk_f in pairs:
        img = cv2.imread(os.path.join(config["images"], img_f), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(config["masks"], msk_f), cv2.IMREAD_GRAYSCALE)

        if img is None or msk is None: continue

        img, msk = normalize_pair(img, msk)
        h, w = img.shape[:2]

        # Create Canvas (White)
        canvas_img = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
        canvas_msk = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255

        # Calculate centering offsets
        # If image > target, we crop the center of the image
        # If image < target, we paste in center of canvas

        src_x, src_y = 0, 0
        dst_x, dst_y = 0, 0
        copy_w, copy_h = w, h

        if w > TARGET_SIZE:
            src_x = (w - TARGET_SIZE) // 2
            copy_w = TARGET_SIZE
        else:
            dst_x = (TARGET_SIZE - w) // 2

        if h > TARGET_SIZE:
            src_y = (h - TARGET_SIZE) // 2
            copy_h = TARGET_SIZE
        else:
            dst_y = (TARGET_SIZE - h) // 2

        canvas_img[dst_y:dst_y+copy_h, dst_x:dst_x+copy_w] = img[src_y:src_y+copy_h, src_x:src_x+copy_w]
        canvas_msk[dst_y:dst_y+copy_h, dst_x:dst_x+copy_w] = msk[src_y:src_y+copy_h, src_x:src_x+copy_w]

        save_pair(canvas_img, canvas_msk, f"{batch_name}_pad", count)
        count += 1
    print(f"[{batch_name}] Finished.")

# --- Mode 3: Copy ---
def process_copy_batch(batch_name, config):
    print(f"[{batch_name}] Starting Copy Mode...")
    pairs = get_image_mask_pairs(config["images"], config["masks"])
    if not pairs: return

    count = 0
    for img_f, msk_f in pairs:
        img = cv2.imread(os.path.join(config["images"], img_f), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(config["masks"], msk_f), cv2.IMREAD_GRAYSCALE)

        if img is None or msk is None: continue

        img, msk = normalize_pair(img, msk)

        save_pair(img, msk, f"{batch_name}_copy", count)
        count += 1
    print(f"[{batch_name}] Finished.")

# --- Multiprocessing Wrapper ---
def process_batch_wrapper(batch_item):
    """Wrapper to unpack the tuple arguments for pool.map"""
    name, cfg = batch_item
    if not cfg["active"]: return

    try:
        if cfg["mode"] == "STREAM":
            process_stream_batch(name, cfg)
        elif cfg["mode"] == "PAD_CENTER":
            process_pad_center_batch(name, cfg)
        elif cfg["mode"] == "COPY":
            process_copy_batch(name, cfg)
        else:
            print(f"Unknown mode for {name}")
    except Exception as e:
        print(f"Error processing {name}: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Setup Directories
    if os.path.exists(FINAL_OUTPUT_DIR):
        print("Cleaning previous Unified Output...")
        try:
            shutil.rmtree(FINAL_OUTPUT_DIR)
        except OSError:
            print("Warning: Could not fully clean output dir (files in use?)")

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    print(f"Starting Unification into: {FINAL_OUTPUT_DIR}")

    # 2. Prepare Task List
    tasks = list(PATHS.items())

    # 3. Run Multiprocessing Pool
    # One process per batch
    num_procs = min(len(tasks), multiprocessing.cpu_count())
    print(f"Spawning {num_procs} processes...")

    with multiprocessing.Pool(processes=num_procs) as pool:
        pool.map(process_batch_wrapper, tasks)

    print("\nAll Batches Processed.")
    count = len(os.listdir(OUT_IMG_DIR))
    print(f"Total Unified Images: {count}")