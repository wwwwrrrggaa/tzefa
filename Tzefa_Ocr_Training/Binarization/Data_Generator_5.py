import os
import cv2
import numpy as np
import random
import string
import multiprocessing
import glob
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# --- CONFIGURATION ---
OUTPUT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_11"
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(OUTPUT_DIR, "masks")
HW_FONT_DIR = r"E:\Storage\handwriting_fonts"

NUM_SAMPLES = 5000
IMG_SIZE = 640


def get_custom_fonts():
    return glob.glob(os.path.join(HW_FONT_DIR, "*.ttf"))


CUSTOM_HW_FONTS = get_custom_fonts()


def get_jittered_color(base_color, variance=15):
    """Applies a random +/- jitter to a base RGB color."""
    r = np.clip(base_color[0] + random.randint(-variance, variance), 0, 255)
    g = np.clip(base_color[1] + random.randint(-variance, variance), 0, 255)
    b = np.clip(base_color[2] + random.randint(-variance, variance), 0, 255)
    return (int(r), int(g), int(b))


def generate_sample(idx):
    random.seed(os.getpid() + idx)

    # 1. Background Initialization
    paper_val = random.randint(240, 255)
    # We build the image as a numpy array for direct pixel manipulation
    img_np = np.full((IMG_SIZE, IMG_SIZE, 3), paper_val, dtype=np.uint8)
    # Mask is pure white (255) background
    mask_np = np.full((IMG_SIZE, IMG_SIZE), 255, dtype=np.uint8)

    # Convert to PIL for font rendering
    mask_pil = Image.fromarray(mask_np)
    m_draw = ImageDraw.Draw(mask_pil)

    # 2. Define Main and Secondary "Ink" for this page
    # Main: Usually Black/Blue. Secondary: Usually Red/Green/Lighter Blue
    main_color = (random.randint(10, 50), random.randint(10, 50), random.randint(30, 80))
    secondary_color = random.choice(
        [
            (150, 20, 20),  # Red
            (20, 120, 20),  # Green
            (20, 60, 180),  # Bright Blue
            (80, 20, 120),  # Purple
        ]
    )

    cursor_y = random.randint(40, 80)
    font_path = random.choice(CUSTOM_HW_FONTS)

    # Create a separate image layer to draw jittered colors
    # We use PIL to draw on a temporary RGB canvas, then blend
    img_pil = Image.fromarray(img_np)
    i_draw = ImageDraw.Draw(img_pil)

    while cursor_y < IMG_SIZE - 60:
        f_size = random.randint(24, 36)
        try:
            font = ImageFont.truetype(font_path, f_size)
        except:
            font = ImageFont.load_default()

        chars = string.ascii_letters + string.digits + " ,.?!-="
        text = "".join(random.choices(chars, k=random.randint(15, 35)))

        # Decide which pen to use for this line
        line_base_color = main_color if random.random() > 0.2 else secondary_color

        x_pos = random.randint(30, 70)

        # Draw character by character for per-char color jitter
        current_x = x_pos
        for char in text:
            char_color = get_jittered_color(line_base_color)
            # Draw on image
            i_draw.text((current_x, cursor_y), char, font=font, fill=char_color)
            # Draw on mask (Fixed 0 for text)
            m_draw.text((current_x, cursor_y), char, font=font, fill=0)

            # Advance X based on character width
            bbox = i_draw.textbbox((current_x, cursor_y), char, font=font)
            current_x += bbox[2] - bbox[0]

        cursor_y += f_size + random.randint(15, 35)

    # 3. Finalize Image and Mask
    img_np = np.array(img_pil)
    mask_final = np.array(mask_pil)

    # 4. Add Notebook Holes (IMAGE ONLY - No Mask corruption)
    if random.random() > 0.4:
        for i in range(5):
            center = (random.randint(15, 25), 80 + i * 120 + random.randint(-10, 10))
            cv2.circle(img_np, center, random.randint(10, 14), (35, 35, 35), -1)

    # 5. Global Shadows
    h, w = img_np.shape[:2]
    shadow = np.ones((h, w), dtype=np.float32)
    ax = np.linspace(random.uniform(0.5, 0.8), 1.0, w)
    if random.random() > 0.5:
        ax = ax[::-1]
    shadow *= np.tile(ax, (h, 1))
    img_np = (img_np.astype(np.float32) * shadow[:, :, np.newaxis]).astype(np.uint8)

    # 6. Save
    filename = f"batch11_{idx:05d}.png"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, filename), cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUT_MSK_DIR, filename), mask_final)


if __name__ == "__main__":
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)
    print("Generating Per-Character Jitter Batch 11...")
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(generate_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))