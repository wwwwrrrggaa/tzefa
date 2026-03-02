import os
import cv2
import numpy as np
import json
from tqdm import tqdm
from datasets import load_dataset
from pathlib import Path
from PIL import Image

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Line_Segmentation")
TARGET_SIZE = 640  # YOLO Standard
USE_DILATION_TRICK = True  # Keep True to merge words into lines


def resize_and_pad(image_pil, target_size=640):
    """
    Resizes image to fit within target_size x target_size while keeping aspect ratio.
    Adds white padding to make it a perfect square.
    """
    w, h = image_pil.size

    # Calculate Scale
    scale = target_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # Resize
    image_pil = image_pil.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # Create new square background (White)
    new_img = Image.new("RGB", (target_size, target_size), (255, 255, 255))

    # Paste resized image in center
    pad_x = (target_size - new_w) // 2
    pad_y = (target_size - new_h) // 2
    new_img.paste(image_pil, (pad_x, pad_y))

    return new_img, scale, pad_x, pad_y


def get_line_bboxes_from_polys(w, h, polys):
    """
    1. Draws polygons on a temp mask.
    2. Dilates to merge words into lines (if configured).
    3. Finds contours of the lines.
    4. Returns bounding boxes [x, y, w, h] of those lines.
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    for poly in polys:
        pts = np.array(poly, np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)

    if USE_DILATION_TRICK:
        # Merge words horizontally
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Find contours of the *merged lines*
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bboxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 10:  # Filter tiny noise
            rect = cv2.boundingRect(cnt)  # x, y, w, h
            bboxes.append(rect)

    return bboxes


def save_sample(batch_folder_name, filename, image_pil, raw_polys):
    """
    1. Resizes Image to 640x640.
    2. Converts raw polygons -> Line BBoxes (using internal mask logic).
    3. Transforms BBoxes to 640x640 coordinates.
    4. Saves Image and YOLO .txt Label.
    """
    batch_dir = BASE_DIR / batch_folder_name
    img_dir = batch_dir / "images"
    lbl_dir = batch_dir / "labels"

    # Create dirs if first run
    if not img_dir.exists():
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)

    # 1. Get Original Dimensions
    w_orig, h_orig = image_pil.size

    # 2. Get Line BBoxes in Original Scale (x, y, w, h)
    orig_bboxes = get_line_bboxes_from_polys(w_orig, h_orig, raw_polys)

    # 3. Resize and Pad Image to 640x640
    new_img, scale, pad_x, pad_y = resize_and_pad(image_pil, TARGET_SIZE)

    # 4. Transform BBoxes to YOLO Format
    yolo_lines = []
    for x, y, w, h in orig_bboxes:
        # Transform from Original -> 640x640 Space
        new_x = (x * scale) + pad_x
        new_y = (y * scale) + pad_y
        new_w = w * scale
        new_h = h * scale

        # Convert to YOLO Normalized Center Format: class x_center y_center w h
        center_x = (new_x + new_w / 2) / TARGET_SIZE
        center_y = (new_y + new_h / 2) / TARGET_SIZE
        norm_w = new_w / TARGET_SIZE
        norm_h = new_h / TARGET_SIZE

        # Clamp values to 0-1
        center_x = min(max(center_x, 0), 1)
        center_y = min(max(center_y, 0), 1)
        norm_w = min(max(norm_w, 0), 1)
        norm_h = min(max(norm_h, 0), 1)

        yolo_lines.append(f"0 {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}")

    # 5. Save Image (Convert RGB -> BGR for OpenCV)
    save_img_np = cv2.cvtColor(np.array(new_img), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_dir / f"{filename}.jpg"), save_img_np)

    # 6. Save Labels
    with open(lbl_dir / f"{filename}.txt", "w") as f:
        f.write("\n".join(yolo_lines))


# --- BATCH 2: CORD (FIXED) ---
def unpack_cord():
    BATCH_NAME = "Batch_2"
    print(f"\n>>> Processing {BATCH_NAME} (CORD)...")
    try:
        ds = load_dataset("naver-clova-ix/cord-v2", split="train")
    except Exception as e:
        print(f"Error loading CORD dataset: {e}")
        return

    for idx, item in enumerate(tqdm(ds)):
        try:
            # Ensure Image is RGB (Handles RGBA/Grayscale issues)
            image = item["image"].convert("RGB")

            # Safe JSON parsing
            ground_truth = item.get("ground_truth", "{}")
            if not ground_truth:
                continue

            gt_data = json.loads(ground_truth)
            polys = []

            # Safe navigation of JSON structure
            valid_lines = gt_data.get("valid_line", [])
            for line in valid_lines:
                words = line.get("words", [])
                for word in words:
                    q = word.get("quad", {})
                    # Ensure coordinates exist
                    if all(k in q for k in ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]):
                        pts = [[q["x1"], q["y1"]], [q["x2"], q["y2"]], [q["x3"], q["y3"]], [q["x4"], q["y4"]]]
                        polys.append(pts)

            if polys:
                save_sample(BATCH_NAME, f"cord_{idx:05d}", image, polys)

        except Exception as e:
            # If one sample fails (e.g., bad JSON), skip it and continue
            continue


# --- BATCH 3: SROIE ---
def unpack_sroie():
    BATCH_NAME = "Batch_3"
    print(f"\n>>> Processing {BATCH_NAME} (SROIE)...")
    ds = load_dataset("darentang/sroie", split="train")

    for idx, item in enumerate(tqdm(ds)):
        polys = []
        try:
            image = item["image"].convert("RGB")
            boxes = item["boxes"]
            for box in boxes:
                x, y, w, h = box
                pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                polys.append(pts)

            if polys:
                save_sample(BATCH_NAME, f"sroie_{idx:05d}", image, polys)
        except:
            continue


# --- BATCH 4: FUNSD ---
def unpack_funsd():
    BATCH_NAME = "Batch_4"
    print(f"\n>>> Processing {BATCH_NAME} (FUNSD)...")
    ds = load_dataset("nielsr/funsd", split="train")

    for idx, item in enumerate(tqdm(ds)):
        try:
            image = item["image"].convert("RGB")
            polys = []
            for box in item["bboxes"]:
                x1, y1, x2, y2 = box
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                polys.append(pts)

            if polys:
                save_sample(BATCH_NAME, f"funsd_{idx:05d}", image, polys)
        except:
            continue


# --- BATCH 5: DOCLAYNET ---
def unpack_doclaynet():
    BATCH_NAME = "Batch_5"
    print(f"\n>>> Processing {BATCH_NAME} (DocLayNet)...")
    ds = load_dataset("pierreguillou/DocLayNet-small", split="train", streaming=True)

    count = 0
    MAX_SAMPLES = 3000

    for item in tqdm(ds):
        if count >= MAX_SAMPLES:
            break

        try:
            image = item["image"].convert("RGB")
            polys = []
            for bbox, cat_id in zip(item["bboxes"], item["category_id"]):
                if cat_id in [1, 2]:
                    x, y, w, h = bbox
                    pts = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    polys.append(pts)

            if polys:
                save_sample(BATCH_NAME, f"doclay_{item['id']}", image, polys)
                count += 1
        except:
            continue


if __name__ == "__main__":
    BASE_DIR.mkdir(parents=True, exist_ok=True)

    unpack_cord()
    unpack_doclaynet()

    print("\nAll datasets unpacked, resized to 640x640, and converted to YOLO labels!")