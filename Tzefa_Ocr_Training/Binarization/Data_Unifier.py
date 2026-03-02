import os
import cv2
import numpy as np
import shutil
import multiprocessing

# --- MASTER CONFIGURATION ---
BASE_PROJECT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization"
FINAL_OUTPUT_DIR = os.path.join(BASE_PROJECT_DIR, "Unified_Batch")
TARGET_SIZE = 640

# --- SPECIAL BATCH CONFIGS ---
SPECIAL_CONFIGS = {
    "BATCH_1": "STREAM",
    "BATCH_2": "STREAM",
    "BATCH_3": "STREAM",
    "BATCH_5": "STREAM",
    "BATCH_6": "PAD_CENTER",
    "BATCH_7": "PAD_CENTER",
    "BATCH_12": "COPY",  # Synthetic 640x640 -> Direct Copy
    "BATCH_13": "COPY",  # Synthetic 640x640 -> Direct Copy
}

# Global output dirs
OUT_IMG_DIR = os.path.join(FINAL_OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(FINAL_OUTPUT_DIR, "masks")


# --- HELPER FUNCTIONS ---
def normalize_pair(img, msk):
    # Ensure correct channels
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

    # Ensure mask is binary
    if len(msk.shape) == 3:
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
    _, msk = cv2.threshold(msk, 127, 255, cv2.THRESH_BINARY)
    return img, msk


def save_pair(img, msk, batch_name, idx):
    img_name = f"{batch_name}_{idx:06d}.jpg"
    msk_name = f"{batch_name}_{idx:06d}.png"
    # Save CLEAN images (High quality) for training-time augmentation
    cv2.imwrite(os.path.join(OUT_IMG_DIR, img_name), img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
    cv2.imwrite(os.path.join(OUT_MSK_DIR, msk_name), msk)


def get_image_mask_pairs(img_dir, msk_dir):
    if not os.path.exists(img_dir) or not os.path.exists(msk_dir):
        return []
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    mask_files = {os.path.splitext(f)[0]: f for f in os.listdir(msk_dir) if f.lower().endswith(valid_exts)}
    pairs = []
    for img_f in os.listdir(img_dir):
        if not img_f.lower().endswith(valid_exts):
            continue
        base_name = os.path.splitext(img_f)[0]
        if base_name in mask_files:
            pairs.append((img_f, mask_files[base_name]))
    return pairs


# --- MODE 1: STREAM STITCHER (GEOMETRY ONLY) ---
class StreamBuffer:
    def __init__(self, batch_name):
        self.name = batch_name
        # Initialize with WHITE background
        self.canvas_img = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
        self.canvas_msk = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255
        self.cx = 0
        self.cy = 0
        self.count = 0

    def save(self):
        # Save raw, no augmentation
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

        self.canvas_img[self.cy : self.cy + copy_h, self.cx : self.cx + copy_w] = img[0:copy_h, 0:copy_w]
        self.canvas_msk[self.cy : self.cy + copy_h, self.cx : self.cx + copy_w] = msk[0:copy_h, 0:copy_w]

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
    if not pairs:
        return

    buffer = StreamBuffer(batch_name)
    # Sort for deterministic processing
    pairs.sort()

    for img_f, msk_f in pairs:
        img = cv2.imread(os.path.join(config["images"], img_f), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(config["masks"], msk_f), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        img, msk = normalize_pair(img, msk)
        buffer.push(img, msk)

    if buffer.cx > 0 or buffer.cy > 0:
        buffer.save()
    print(f"[{batch_name}] Finished.")


# --- MODE 2: PAD CENTER (GEOMETRY ONLY) ---
def process_pad_center_batch(batch_name, config):
    print(f"[{batch_name}] Starting Pad-Center Mode...")
    pairs = get_image_mask_pairs(config["images"], config["masks"])
    if not pairs:
        return

    count = 0
    for img_f, msk_f in pairs:
        img = cv2.imread(os.path.join(config["images"], img_f), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(config["masks"], msk_f), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        img, msk = normalize_pair(img, msk)

        h, w = img.shape[:2]
        # White background
        canvas_img = np.ones((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8) * 255
        canvas_msk = np.ones((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8) * 255

        # Center logic
        src_x, src_y, dst_x, dst_y = 0, 0, 0, 0
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

        canvas_img[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = img[src_y : src_y + copy_h, src_x : src_x + copy_w]
        canvas_msk[dst_y : dst_y + copy_h, dst_x : dst_x + copy_w] = msk[src_y : src_y + copy_h, src_x : src_x + copy_w]

        save_pair(canvas_img, canvas_msk, f"{batch_name}_pad", count)
        count += 1
    print(f"[{batch_name}] Finished.")


# --- MODE 3: COPY (GEOMETRY ONLY) ---
def process_copy_batch(batch_name, config):
    print(f"[{batch_name}] Starting Copy Mode...")
    pairs = get_image_mask_pairs(config["images"], config["masks"])
    if not pairs:
        return

    count = 0
    for img_f, msk_f in pairs:
        img = cv2.imread(os.path.join(config["images"], img_f), cv2.IMREAD_COLOR)
        msk = cv2.imread(os.path.join(config["masks"], msk_f), cv2.IMREAD_GRAYSCALE)
        if img is None or msk is None:
            continue
        img, msk = normalize_pair(img, msk)

        save_pair(img, msk, f"{batch_name}_copy", count)
        count += 1
    print(f"[{batch_name}] Finished.")


# --- PROCESS MANAGER ---
def process_batch_wrapper(batch_item):
    name, cfg = batch_item
    # If key isn't in SPECIAL_CONFIGS, default to COPY is handled in 'Discovery'
    # But here we assume SPECIAL_CONFIGS drives the logic or override
    # We honor the "mode" set during discovery.

    try:
        if cfg["mode"] == "STREAM":
            process_stream_batch(name, cfg)
        elif cfg["mode"] == "PAD_CENTER":
            process_pad_center_batch(name, cfg)
        elif cfg["mode"] == "COPY":
            process_copy_batch(name, cfg)
        else:
            print(f"Unknown mode: {cfg['mode']}")
    except Exception as e:
        print(f"Error processing {name}: {e}")


if __name__ == "__main__":
    # 0. Discovery
    PATHS = {}
    if os.path.exists(BASE_PROJECT_DIR):
        for entry in sorted(os.listdir(BASE_PROJECT_DIR)):
            full_path = os.path.join(BASE_PROJECT_DIR, entry)
            if os.path.isdir(full_path) and entry.lower().startswith("batch_"):
                batch_key = entry.upper()

                # Default mode is COPY unless specified in SPECIAL_CONFIGS
                mode = SPECIAL_CONFIGS.get(batch_key, "COPY")

                # Mark active if it exists
                PATHS[batch_key] = {
                    "mode": mode,
                    "active": True,
                    "images": os.path.join(full_path, "images"),
                    "masks": os.path.join(full_path, "masks"),
                }

    # 1. Setup Output
    if os.path.exists(FINAL_OUTPUT_DIR):
        print("Cleaning previous Unified Output...")
        try:
            shutil.rmtree(FINAL_OUTPUT_DIR)
        except:
            pass

    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    print(f"Starting Clean Geometry Unification...")

    tasks = sorted(list(PATHS.items()), key=lambda x: x[0])
    num_procs = 16

    with multiprocessing.Pool(processes=num_procs) as pool:
        pool.map(process_batch_wrapper, tasks)

    print("\nProcessing Complete.")