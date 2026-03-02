import cv2
import numpy as np
import os
import csv
from glob import glob
from tqdm import tqdm
import random
import re
from collections import defaultdict

# --- CONFIGURATION ---
SMARTDOC_ROOT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\SmartDoc"
OUTPUT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_14"
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(OUTPUT_DIR, "masks")
TARGET_COUNT = 5000


# --- 1. THE HUNTER: Find files wherever they are ---
def find_source_directory(root):
    print("Hunting for 'datasheet' or 'source' directory...")
    # Look for a specific known file like '001.png' or 'letter001.png'
    for root_dir, dirs, files in os.walk(root):
        for f in files:
            if f in ["001.png", "letter001.png", "magazine001.png", "paper001.png"]:
                print(f"   -> Found source images in: {root_dir}")
                return root_dir
    print("   [ERROR] Could not locate source directory! (looked for 001.png)")
    return None


def index_source_files(source_dir):
    source_map = {}
    if not source_dir:
        return {}

    files = glob(os.path.join(source_dir, "*"))
    print(f"Indexing {len(files)} source files...")

    for f_path in files:
        if f_path.lower().endswith((".png", ".jpg", ".jpeg", ".tif")):
            fname = os.path.splitext(os.path.basename(f_path))[0].lower()

            # 1. Map exact name: 'letter001'
            source_map[fname] = f_path

            # 2. Map numeric ID: 'letter001' -> '001' -> '1'
            numeric = re.sub(r"\D", "", fname)
            if numeric:
                source_map[numeric] = f_path
                source_map[str(int(numeric))] = f_path

    return source_map


# --- 2. SETUP (Run Once) ---
REAL_SOURCE_DIR = find_source_directory(SMARTDOC_ROOT)
SOURCE_INDEX = index_source_files(REAL_SOURCE_DIR)

# Locate Images Dir
IMAGES_DIR = os.path.join(SMARTDOC_ROOT, "smart_doc_extracted", "images")
if not os.path.exists(IMAGES_DIR):
    # Try finding it
    for root_dir, dirs, files in os.walk(SMARTDOC_ROOT):
        if "background01-datasheet001-0.png" in files:
            IMAGES_DIR = root_dir
            break
print(f"Images Directory set to: {IMAGES_DIR}")

# Locate CSV
CSV_PATH = os.path.join(SMARTDOC_ROOT, "frame_data.csv")


def get_clean_source_path(filename):
    """'background01-letter003-0.png' -> matches '003' or 'letter003' in index"""
    try:
        parts = filename.split("-")
        if len(parts) < 2:
            return None

        raw_id = parts[1].lower()  # "letter003"
        numeric_id = re.sub(r"\D", "", raw_id)  # "003"

        if raw_id in SOURCE_INDEX:
            return SOURCE_INDEX[raw_id]
        if numeric_id in SOURCE_INDEX:
            return SOURCE_INDEX[numeric_id]
        if numeric_id and str(int(numeric_id)) in SOURCE_INDEX:
            return SOURCE_INDEX[str(int(numeric_id))]
        return None
    except:
        return None


def order_points_clockwise(pts):
    pts = np.array(pts, dtype="float32")
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left_pts = x_sorted[:2, :]
    right_pts = x_sorted[2:, :]
    left_pts = left_pts[np.argsort(left_pts[:, 1]), :]
    (tl, bl) = left_pts
    right_pts = right_pts[np.argsort(right_pts[:, 1]), :]
    (tr, br) = right_pts
    return np.array([tl, tr, br, bl], dtype="float32")


def generate_aligned_pair(captured_path, source_path, points):
    captured_img = cv2.imread(captured_path)
    clean_source = cv2.imread(source_path)

    if captured_img is None or clean_source is None:
        return None, None

    h_cap, w_cap = captured_img.shape[:2]
    h_src, w_src = clean_source.shape[:2]

    dst_pts = order_points_clockwise(points)
    src_pts = np.float32([[0, 0], [w_src, 0], [w_src, h_src], [0, h_src]])

    try:
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped_mask_color = cv2.warpPerspective(clean_source, M, (w_cap, h_cap))
    except:
        return None, None

    warped_mask_gray = cv2.cvtColor(warped_mask_color, cv2.COLOR_BGR2GRAY)
    _, binary_mask = cv2.threshold(warped_mask_gray, 200, 255, cv2.THRESH_BINARY_INV)

    return captured_img, binary_mask


def load_csv_data(csv_path):
    grouped_data = defaultdict(list)
    print(f"Reading CSV: {csv_path}")

    with open(csv_path, "r", encoding="utf-8") as f:
        # Force TAB delimiter based on your sample
        reader = csv.reader(f, delimiter="\t")

        # Skip header logic
        header = next(reader, None)

        for row in reader:
            if not row or len(row) < 5:
                continue

            filename = row[0]
            if filename.startswith("._") or "PaxHeader" in filename:
                continue

            try:
                # Based on your sample: x=col 3, y=col 4
                x = float(row[3])
                y = float(row[4])
                grouped_data[filename].append([x, y])
            except ValueError:
                continue

    valid_data = []
    for fname, points in grouped_data.items():
        if len(points) == 4:
            valid_data.append((fname, points))

    return valid_data


def process_smartdoc():
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        print(f"[ERROR] CSV not found at {CSV_PATH}")
        return

    if not SOURCE_INDEX:
        print("[ERROR] No source files indexed. Cannot proceed.")
        return

    all_samples = load_csv_data(CSV_PATH)
    print(f"Found {len(all_samples)} valid images with 4 points.")

    random.shuffle(all_samples)
    if len(all_samples) > TARGET_COUNT:
        selected_samples = all_samples[:TARGET_COUNT]
    else:
        selected_samples = all_samples

    print(f"Processing {len(selected_samples)} samples...")
    success_count = 0

    for filename, points in tqdm(selected_samples):
        captured_path = os.path.join(IMAGES_DIR, filename)
        source_path = get_clean_source_path(filename)

        if source_path and os.path.exists(captured_path):
            img, mask = generate_aligned_pair(captured_path, source_path, points)

            if img is not None and mask is not None:
                out_name = f"batch14_sd_{filename}"
                msk_name = f"batch14_sd_{os.path.splitext(filename)[0]}.png"
                cv2.imwrite(os.path.join(OUT_IMG_DIR, out_name), img)
                cv2.imwrite(os.path.join(OUT_MSK_DIR, msk_name), mask)
                success_count += 1

    print(f"Successfully generated {success_count} aligned pairs.")

if __name__ == "__main__":
    process_smartdoc()