"""
Multi-Scale Synthetic Data Generator for Line Segmentation (Batch_7)

Key difference from Data_Generator.py:
  - Each image is generated at a RANDOM canvas size (400–2000 x 300–2500)
  - Then squash-resized to 640x640 (matching inference behavior)
  - Labels are rescaled to the 640x640 coordinate space

This teaches the YOLO-OBB model to handle the exact scale distortion
it will encounter at inference time.
"""

import os
import cv2
import numpy as np
import random
import string
import multiprocessing
import math
import glob
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Line_Segmentation"
BATCH_NAME = "Batch_7"

OUTPUT_DIR = os.path.join(BASE_DIR, BATCH_NAME)
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

NUM_SAMPLES = 8000
TARGET_SIZE = 640  # Final output size (matches inference)

# Random canvas size ranges — simulates real-world image diversity
MIN_W, MAX_W = 400, 2000
MIN_H, MAX_H = 300, 2500


# --- GLOBAL RESOURCES ---
def get_system_fonts():
    fonts = glob.glob(r"C:\Windows\Fonts\*.ttf")
    if not fonts:
        fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    valid_fonts = [f for f in fonts if "dings" not in f.lower() and "symbol" not in f.lower()]
    return valid_fonts if valid_fonts else ["arial.ttf"]


SYSTEM_FONTS = get_system_fonts()


def get_random_word():
    return "".join(random.choices(string.ascii_letters, k=random.randint(3, 10)))


def generate_paper_texture(w, h):
    base_val = random.randint(220, 255)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    tint = (random.randint(-15, 15), random.randint(-15, 15), random.randint(-15, 15))
    img[:] = (
        np.clip(base_val + tint[0], 0, 255),
        np.clip(base_val + tint[1], 0, 255),
        np.clip(base_val + tint[2], 0, 255),
    )
    noise = np.random.normal(0, 8, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# --- GEOMETRY UTILS ---
def rotate_point(x, y, cx, cy, angle_rad):
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    nx = cx + (x - cx) * cos_a - (y - cy) * sin_a
    ny = cy + (x - cx) * sin_a + (y - cy) * cos_a
    return nx, ny


def transform_points(points, ttype="curl", params=None):
    new_points = []
    for x, y in points:
        nx, ny = x, y
        if ttype == "curl":
            amp, freq, phase, axis = params
            if axis == "y":
                ny = y + amp * math.sin(x * freq + phase)
            else:
                nx = x + amp * math.sin(y * freq + phase)
        elif ttype == "perspective":
            M = params
            vec = np.array([x, y, 1.0])
            res = M @ vec
            nx, ny = res[0] / res[2], res[1] / res[2]
        new_points.append([nx, ny])
    return new_points


# --- LAYOUT ENGINE ---
def draw_layout(img_np):
    h, w = img_np.shape[:2]
    base_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)).convert("RGBA")

    font_path = random.choice(SYSTEM_FONTS)
    try:
        # Scale font size relative to canvas — larger canvases get bigger text
        base_font = random.randint(16, 40)
        scale_factor = min(w, h) / 640.0
        font_size = max(12, int(base_font * scale_factor))
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()
        font_size = 14

    margin_top = random.randint(30, max(31, int(h * 0.08)))
    margin_bottom = random.randint(30, max(31, int(h * 0.08)))
    margin_left = random.randint(30, max(31, int(w * 0.08)))
    margin_right = random.randint(30, max(31, int(w * 0.08)))

    cursor_y = margin_top

    try:
        bbox = font.getbbox("Agy")
        line_height = (bbox[3] - bbox[1]) + int(font_size * 0.8)
        ascent = bbox[3] - bbox[1]
    except:
        line_height = font_size + 15
        ascent = font_size

    boxes = []

    while cursor_y < (h - margin_bottom):
        line_text = ""
        dummy = ImageDraw.Draw(base_pil)
        while True:
            word = get_random_word()
            test_line = line_text + word + " "
            txt_bbox = dummy.textbbox((0, 0), test_line, font=font)
            text_w = txt_bbox[2] - txt_bbox[0]
            if margin_left + text_w > (w - margin_right):
                break
            line_text = test_line

        if line_text:
            angle = random.uniform(-5, 5)
            if random.random() > 0.9:
                angle = random.uniform(-15, 15)

            txt_w = int(dummy.textlength(line_text, font=font)) + 10
            txt_h = int(ascent * 1.5) + 10

            txt_layer = Image.new("RGBA", (txt_w, txt_h), (0, 0, 0, 0))
            draw_txt = ImageDraw.Draw(txt_layer)

            ink = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60), 255)
            draw_txt.text((5, 5), line_text, font=font, fill=ink)

            rotated_layer = txt_layer.rotate(angle, expand=True, resample=Image.BICUBIC)

            paste_x = margin_left
            paste_y = int(cursor_y - (rotated_layer.height - txt_h) / 2)

            # Clamp paste position
            paste_x = max(0, min(paste_x, w - rotated_layer.width))
            paste_y = max(0, min(paste_y, h - rotated_layer.height))

            base_pil.alpha_composite(rotated_layer, dest=(paste_x, paste_y))

            cx, cy = txt_w / 2, txt_h / 2
            local_corners = [
                (5, 5),
                (5 + txt_w - 10, 5),
                (5 + txt_w - 10, 5 + ascent),
                (5, 5 + ascent),
            ]

            rad = -math.radians(angle)
            poly = []
            for lx, ly in local_corners:
                rx, ry = rotate_point(lx, ly, cx, cy, rad)
                nw_r, nh_r = rotated_layer.size
                rx += nw_r / 2 - cx
                ry += nh_r / 2 - cy
                gx = rx + paste_x
                gy = ry + paste_y
                poly.append([gx, gy])

            boxes.append(poly)

        cursor_y += line_height
        if random.random() > 0.85:
            cursor_y += line_height

    return cv2.cvtColor(np.array(base_pil.convert("RGB")), cv2.COLOR_RGB2BGR), boxes


# --- DISTORTIONS ---
def apply_distortions(img, boxes):
    h, w = img.shape[:2]

    # Curl
    if random.random() > 0.3:
        amp = random.uniform(5, 30)
        freq = random.uniform(0.005, 0.02)
        phase = random.uniform(0, math.pi * 2)
        axis = "y" if random.random() > 0.5 else "x"

        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        if axis == "y":
            for y_i in range(h):
                map_x[y_i, :] = np.arange(w)
            for x_i in range(w):
                map_y[:, x_i] = np.arange(h) + amp * math.sin(x_i * freq + phase)
        else:
            for x_i in range(w):
                map_y[:, x_i] = np.arange(h)
            for y_i in range(h):
                map_x[y_i, :] = np.arange(w) + amp * math.sin(y_i * freq + phase)

        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        new_boxes = [transform_points(box, "curl", (amp, freq, phase, axis)) for box in boxes]
        boxes = new_boxes

    # Perspective
    if random.random() > 0.3:
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dev = w * 0.1
        dst_pts = np.float32([
            [random.uniform(0, dev), random.uniform(0, dev)],
            [random.uniform(w - dev, w), random.uniform(0, dev)],
            [random.uniform(0, dev), random.uniform(h - dev, h)],
            [random.uniform(w - dev, w), random.uniform(h - dev, h)],
        ])
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)
        new_boxes = [transform_points(box, "perspective", M) for box in boxes]
        boxes = new_boxes

    return img, boxes


def rescale_to_target(img, boxes, orig_w, orig_h):
    """
    Squash-resize image and labels from (orig_w, orig_h) to (TARGET_SIZE, TARGET_SIZE).
    This matches the exact distortion applied at inference time in Line_Segmentation.py.
    """
    resized = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE))

    scale_x = TARGET_SIZE / orig_w
    scale_y = TARGET_SIZE / orig_h

    rescaled_boxes = []
    for poly in boxes:
        new_poly = []
        for x, y in poly:
            new_poly.append([x * scale_x, y * scale_y])
        rescaled_boxes.append(new_poly)

    return resized, rescaled_boxes


def save_label(boxes, path, w, h):
    with open(path, "w") as f:
        for poly in boxes:
            norm_pts = []
            for x, y in poly:
                nx = min(max(0, x), w) / w
                ny = min(max(0, y), h) / h
                norm_pts.append(f"{nx:.6f} {ny:.6f}")
            f.write("0 " + " ".join(norm_pts) + "\n")


# --- WORKER ---
def generate_sample(idx):
    random.seed(os.getpid() + idx)
    np.random.seed(os.getpid() + idx)

    # Random canvas size — the key improvement
    canvas_w = random.randint(MIN_W, MAX_W)
    canvas_h = random.randint(MIN_H, MAX_H)

    img = generate_paper_texture(canvas_w, canvas_h)
    img, boxes = draw_layout(img)
    img, boxes = apply_distortions(img, boxes)

    # Squash to 640x640 (matching inference)
    img, boxes = rescale_to_target(img, boxes, canvas_w, canvas_h)

    fname = f"ms_line_{idx:06d}"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, fname + ".jpg"), img)
    save_label(boxes, os.path.join(OUT_LBL_DIR, fname + ".txt"), TARGET_SIZE, TARGET_SIZE)


if __name__ == "__main__":
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LBL_DIR, exist_ok=True)
    print(f"Generating {NUM_SAMPLES} Multi-Scale Synthetic Samples to {OUTPUT_DIR}...")
    print(f"Canvas range: {MIN_W}-{MAX_W} x {MIN_H}-{MAX_H} -> squashed to {TARGET_SIZE}x{TARGET_SIZE}")

    workers = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(generate_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))

    print(f"\n✅ Done. {NUM_SAMPLES} samples saved to {OUTPUT_DIR}")

