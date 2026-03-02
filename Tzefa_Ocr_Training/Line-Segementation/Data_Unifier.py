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

# --- CONFIGURATION ---
BASE_DIR = Path(r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Line_Segmentation")
OUTPUT_DIR = BASE_DIR / "Unified_Batch"

# Point to your specific batches
IAM_DIR = BASE_DIR / "Batch_6"
BATCH_1_DIR = BASE_DIR / "Batch_1"

# Binarization Model
BINARIZATION_CKPT = r"E:\Storage\Tzefa_Backups\Tzefa_Models\Binarization\misk\step_810-m3-unet.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 640
PADDING = 50  # Pixels to add above/below the text


# --- BINARIZER (NO ATTENTION) ---
class Binarizer:
    def __init__(self, checkpoint_path):
        print(f"--- Loading Binarizer (mit_b3 / No Attention) ---")
        self.model = smp.Unet(
            encoder_name="mit_b3",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
            state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state)
        except Exception as e:
            print(f"[!] Critical Error loading binarizer: {e}")
            exit(1)

        self.model.to(DEVICE)
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])

    def process(self, img_bgr):
        # Resize to Target Size (640x640)
        img_resized = cv2.resize(img_bgr, (TARGET_SIZE, TARGET_SIZE))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

        img_np = img_rgb.astype(np.float32) / 255.0
        img_np = (img_np - self.mean) / self.std
        img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).unsqueeze(0).float().to(DEVICE)

        with torch.no_grad():
            logits = self.model(img_tensor)
            if logits.shape[-2:] != img_tensor.shape[-2:]:
                logits = torch.nn.functional.interpolate(logits, size=img_tensor.shape[-2:], mode="bilinear")
            mask = (torch.sigmoid(logits) > 0.5).long().squeeze().cpu().numpy()

        bin_img = np.where(mask == 1, 0, 255).astype(np.uint8)

        # QC Check (Skip empty/black images)
        ratio = np.sum(bin_img == 255) / bin_img.size
        if ratio > 0.995 or ratio < 0.005:
            return None

        return bin_img


# --- SETUP ---
def setup_structure():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "labels").mkdir(parents=True, exist_ok=True)

    yaml_content = {"path": str(OUTPUT_DIR.absolute()), "train": "images", "val": "images", "names": {0: "line"}}
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)


# --- PROCESSOR FOR IAM (Smart Label-Based Crop) ---
def process_iam_batch(binarizer):
    print(f"\n>>> Processing IAM (Batch 6) - Calculating Crops from Labels...")
    img_dir = IAM_DIR / "images"
    lbl_dir = IAM_DIR / "labels"

    images = list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))
    if not images:
        print(f"  [!] No images found in {img_dir}")
        return

    count = 0
    for img_path in tqdm(images, desc="IAM"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        orig_h, orig_w = img.shape[:2]

        # Handle "iam_" prefix inconsistency
        lbl_name = img_path.stem
        lbl_path = lbl_dir / f"{lbl_name}.txt"
        if not lbl_path.exists():
            lbl_path = lbl_dir / f"iam_{lbl_name}.txt"

        if not lbl_path.exists():
            continue

        # Read all normalized Y coordinates
        all_ys = []
        lines_data = []

        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                coords = [float(x) for x in parts[1:]]
                lines_data.append(coords)
                for i in range(1, len(coords), 2):
                    all_ys.append(coords[i])

        if not all_ys:
            continue

        # Calculate Dynamic Crop
        min_y_norm = min(all_ys)
        max_y_norm = max(all_ys)
        abs_min_y = int(min_y_norm * orig_h)
        abs_max_y = int(max_y_norm * orig_h)

        crop_y1 = max(0, abs_min_y - PADDING)
        crop_y2 = min(orig_h, abs_max_y + PADDING)

        if (crop_y2 - crop_y1) < 50:
            continue

        # Crop Image
        crop_img = img[crop_y1:crop_y2, :]

        # Binarize
        bin_img = binarizer.process(crop_img)
        if bin_img is None:
            continue

        # Recalculate Labels
        new_lines = []
        for coords in lines_data:
            new_coords = []
            for i in range(0, len(coords), 2):
                abs_x = coords[i] * orig_w
                abs_y = coords[i + 1] * orig_h
                new_y = abs_y - crop_y1  # Shift

                norm_x = max(0, min(1, abs_x / orig_w))
                norm_y = max(0, min(1, new_y / (crop_y2 - crop_y1)))
                new_coords.extend([f"{norm_x:.6f}", f"{norm_y:.6f}"])

            new_lines.append(f"0 {' '.join(new_coords)}\n")

        # Save
        save_name = f"iam_{img_path.stem}"
        cv2.imwrite(str(OUTPUT_DIR / "images" / f"{save_name}.png"), bin_img)
        with open(OUTPUT_DIR / "labels" / f"{save_name}.txt", "w") as f:
            f.writelines(new_lines)
        count += 1

    print(f"  [+] Processed {count} IAM forms.")


# --- PROCESSOR FOR BATCH 1 (Standard Pass-Through) ---
def process_batch_1(binarizer):
    print(f"\n>>> Processing Batch 1...")
    img_dir = BATCH_1_DIR / "images"

    images = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
    for img_path in tqdm(images, desc="Batch 1"):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        bin_img = binarizer.process(img)
        if bin_img is None:
            continue

        lbl_path = Path(str(img_path).replace("images", "labels")).with_suffix(".txt")
        if not lbl_path.exists():
            lbl_path = img_path.with_suffix(".txt")

        if lbl_path.exists():
            save_name = f"b1_{img_path.stem}"
            shutil.copy(lbl_path, OUTPUT_DIR / "labels" / f"{save_name}.txt")
            cv2.imwrite(str(OUTPUT_DIR / "images" / f"{save_name}.png"), bin_img)


# --- VISUALIZATION FUNCTION ---
# --- VISUALIZATION FUNCTION ---
import matplotlib.pyplot as plt  # <--- Add this import at the top

# ... rest of your code ...


# --- VISUALIZATION FUNCTION (Matplotlib Fix) ---
def visualize_random_result():
    print("\n--- Visualizing Random Sample from Unified Batch ---")
    img_dir = OUTPUT_DIR / "images"
    lbl_dir = OUTPUT_DIR / "labels"

    # Get all processed images
    images = list(img_dir.glob("*.png"))
    if not images:
        print("[!] No images found in Unified Batch to visualize.")
        return

    # Pick one random image
    img_path = random.choice(images)
    lbl_path = lbl_dir / f"{img_path.stem}.txt"

    print(f"Displaying: {img_path.name}")

    # Load image (OpenCV loads as BGR)
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]

    # Draw Polygons
    if lbl_path.exists():
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                # YOLO Poly: 0 x1 y1 x2 y2 ...
                coords = [float(x) for x in parts[1:]]

                # Denormalize points
                points = []
                for i in range(0, len(coords), 2):
                    px = int(coords[i] * w)
                    py = int(coords[i + 1] * h)
                    points.append([px, py])

                # Draw Polygon (Red color in BGR is 0,0,255)
                pts = np.array(points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(img, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

    # Convert BGR to RGB for Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Show using Matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(f"Sample: {img_path.name}")
    plt.show()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    setup_structure()

    binarizer = Binarizer(BINARIZATION_CKPT)

    # Process IAM (With Smart Crop)
    process_iam_batch(binarizer)

    # Process Batch 1 (Standard)
    #process_batch_1(binarizer)

    print(f"\n✅ Unification Complete at {OUTPUT_DIR}")

    # Show one result
    visualize_random_result()