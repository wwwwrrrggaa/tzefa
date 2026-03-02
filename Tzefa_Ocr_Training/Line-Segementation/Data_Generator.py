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
BATCH_NAME = "Batch_1"  # New batch name

OUTPUT_DIR = os.path.join(BASE_DIR, BATCH_NAME)
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_LBL_DIR = os.path.join(OUTPUT_DIR, "labels")

NUM_SAMPLES = 5000
IMG_SIZE = 640


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
    """Rotates a point (x,y) around (cx,cy) by angle_rad."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    nx = cx + (x - cx) * cos_a - (y - cy) * sin_a
    ny = cy + (x - cx) * sin_a + (y - cy) * cos_a
    return nx, ny


def transform_points(points, type="curl", params=None):
    new_points = []
    for x, y in points:
        nx, ny = x, y
        if type == "curl":
            amp, freq, phase, axis = params
            if axis == "y":
                ny = y + amp * math.sin(x * freq + phase)
            else:
                nx = x + amp * math.sin(y * freq + phase)
        elif type == "perspective":
            M = params
            vec = np.array([x, y, 1.0])
            res = M @ vec
            nx, ny = res[0] / res[2], res[1] / res[2]
        new_points.append([nx, ny])
    return new_points


# --- LAYOUT ENGINE ---
def draw_layout(img_np):
    h, w = img_np.shape[:2]
    # We use RGBA for temporary text layers to handle rotation transparency
    base_pil = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)).convert("RGBA")

    font_path = random.choice(SYSTEM_FONTS)
    try:
        font_size = random.randint(20, 45)
        font = ImageFont.truetype(font_path, font_size)
    except:
        font = ImageFont.load_default()

    margin_top = random.randint(50, 100)
    margin_bottom = random.randint(50, 100)
    margin_left = random.randint(50, 90)
    margin_right = random.randint(50, 90)

    cursor_y = margin_top

    try:
        bbox = font.getbbox("Agy")
        line_height = (bbox[3] - bbox[1]) + int(font_size * 0.8)  # More spacing for rotation
        ascent = bbox[3] - bbox[1]
    except:
        line_height = font_size + 15
        ascent = font_size

    boxes = []

    while cursor_y < (h - margin_bottom):
        # 1. Generate Text Content
        line_text = ""
        while True:
            word = get_random_word()
            test_line = line_text + word + " "
            # Dummy draw to measure
            dummy = ImageDraw.Draw(base_pil)
            txt_bbox = dummy.textbbox((0, 0), test_line, font=font)
            text_w = txt_bbox[2] - txt_bbox[0]
            if margin_left + text_w > (w - margin_right):
                break
            line_text = test_line

        if line_text:
            # 2. Determine Per-Line Rotation (-5 to 5 degrees usually, occasionally more)
            angle = random.uniform(-5, 5)
            if random.random() > 0.9:
                angle = random.uniform(-15, 15)  # Crazy line

            # 3. Create a temporary canvas for just this line
            # Size needs to be big enough to hold rotated text
            txt_w = int(dummy.textlength(line_text, font=font)) + 10
            txt_h = int(ascent * 1.5) + 10

            txt_layer = Image.new("RGBA", (txt_w, txt_h), (0, 0, 0, 0))
            draw_txt = ImageDraw.Draw(txt_layer)

            # Ink color
            ink = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60), 255)
            # Draw text horizontally centered in the temp layer
            draw_txt.text((5, 5), line_text, font=font, fill=ink)

            # 4. Rotate the temporary layer
            rotated_layer = txt_layer.rotate(angle, expand=True, resample=Image.BICUBIC)

            # 5. Paste onto main image
            # Calculate position to center it roughly at cursor_y
            # We treat (margin_left, cursor_y) as the anchor for the visual center-left
            paste_x = margin_left
            paste_y = int(cursor_y - (rotated_layer.height - txt_h) / 2)

            base_pil.alpha_composite(rotated_layer, dest=(paste_x, paste_y))

            # 6. Calculate Oriented Bounding Box
            # Original (unrotated) box relative to the temp layer: (5, 5, 5+w, 5+h)
            # Center of rotation was center of txt_layer: (txt_w/2, txt_h/2)
            # Wait, PIL.rotate rotates around center by default.

            cx, cy = txt_w / 2, txt_h / 2

            # Corners of the text in the local temp layer
            local_corners = [
                (5, 5),  # TL
                (5 + txt_w - 10, 5),  # TR
                (5 + txt_w - 10, 5 + ascent),  # BR
                (5, 5 + ascent),  # BL
            ]

            # Transform corners
            rad = -math.radians(angle)  # PIL rotation is counter-clockwise, math is usually CCW too

            poly = []
            for lx, ly in local_corners:
                # 1. Rotate around local center
                rx, ry = rotate_point(lx, ly, cx, cy, rad)

                # 2. Offset by "expand" shift
                # When PIL rotates with expand=True, the center shifts.
                # New width/height
                nw, nh = rotated_layer.size
                # Shift to move (cx, cy) to new center (nw/2, nh/2)
                rx += nw / 2 - cx
                ry += nh / 2 - cy

                # 3. Global offset (paste position)
                gx = rx + paste_x
                gy = ry + paste_y
                poly.append([gx, gy])

            boxes.append(poly)

        cursor_y += line_height
        if random.random() > 0.85:
            cursor_y += line_height

    # Convert back to RGB for OpenCV
    return cv2.cvtColor(np.array(base_pil.convert("RGB")), cv2.COLOR_RGB2BGR), boxes


# --- DISTORTIONS ---
def apply_distortions(img, boxes):
    h, w = img.shape[:2]

    # 1. Curl
    if random.random() > 0.3:
        amp = random.uniform(5, 30)
        freq = random.uniform(0.005, 0.02)
        phase = random.uniform(0, math.pi * 2)
        axis = "y" if random.random() > 0.5 else "x"

        map_x = np.zeros((h, w), np.float32)
        map_y = np.zeros((h, w), np.float32)
        if axis == "y":
            for y in range(h):
                map_x[y, :] = np.arange(w)
            for x in range(w):
                map_y[:, x] = np.arange(h) + amp * math.sin(x * freq + phase)
        else:
            for x in range(w):
                map_y[:, x] = np.arange(h)
            for y in range(h):
                map_x[y, :] = np.arange(w) + amp * math.sin(y * freq + phase)

        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

        new_boxes = []
        for box in boxes:
            new_boxes.append(transform_points(box, "curl", (amp, freq, phase, axis)))
        boxes = new_boxes

    # 2. Perspective
    if random.random() > 0.3:
        src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
        dev = w * 0.1
        dst_pts = np.float32(
            [
                [random.uniform(0, dev), random.uniform(0, dev)],
                [random.uniform(w - dev, w), random.uniform(0, dev)],
                [random.uniform(0, dev), random.uniform(h - dev, h)],
                [random.uniform(w - dev, w), random.uniform(h - dev, h)],
            ]
        )
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        img = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT_101)

        new_boxes = []
        for box in boxes:
            new_boxes.append(transform_points(box, "perspective", M))
        boxes = new_boxes

    return img, boxes


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

    img = generate_paper_texture(IMG_SIZE, IMG_SIZE)
    img, boxes = draw_layout(img)
    img, boxes = apply_distortions(img, boxes)

    fname = f"syn_line_{idx:06d}"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, fname + ".jpg"), img)
    save_label(boxes, os.path.join(OUT_LBL_DIR, fname + ".txt"), IMG_SIZE, IMG_SIZE)


if __name__ == "__main__":
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_LBL_DIR, exist_ok=True)
    print(f"Generating {NUM_SAMPLES} Individual-Line Rotated Samples to {OUTPUT_DIR}...")

    workers = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(workers) as pool:
        list(tqdm(pool.imap_unordered(generate_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))