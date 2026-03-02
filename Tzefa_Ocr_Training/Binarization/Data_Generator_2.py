import os
import cv2
import numpy as np
import random
import string
import multiprocessing
import math

# --- CONFIGURATION ---
OUTPUT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_8"
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(OUTPUT_DIR, "masks")

NUM_SAMPLES = 5000
IMG_SIZE = 640

# --- Helpers ---

def get_random_string(length=None):
    if length is None:
        length = random.randint(3, 10)
    chars = string.ascii_letters + string.digits + string.punctuation + " "
    return ''.join(random.choices(chars, k=length))

def add_noise_grain(img, intensity=0.05):
    """Adds film grain/paper texture noise."""
    noise = np.random.randn(*img.shape) * (255 * intensity)
    img_float = img.astype(np.float32) + noise
    return np.clip(img_float, 0, 255).astype(np.uint8)

def add_coffee_stain(img):
    """Adds ring-like stains simulating coffee cups."""
    if random.random() > 0.4:
        return img

    overlay = img.copy()
    h, w = img.shape[:2]

    # Random location
    center = (random.randint(0, w), random.randint(0, h))
    radius = random.randint(20, 80)
    color = (random.randint(100, 160), random.randint(120, 180), random.randint(140, 200)) # Brownish

    # Draw thick circle
    cv2.circle(overlay, center, radius, color, thickness=random.randint(4, 15))

    # Blur it heavily to make it look like a liquid stain
    overlay = cv2.GaussianBlur(overlay, (41, 41), 0)

    # Blend with multiply effect to darken
    alpha = random.uniform(0.3, 0.6)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def add_paper_lines(img, kind='lined'):
    """Adds notebook lines or graph grids."""
    h, w = img.shape[:2]
    overlay = img.copy()

    # Blueish/Greyish line color typical of notebooks
    line_col = (random.randint(180, 220), random.randint(140, 180), random.randint(140, 180)) # Light Blue/Grey
    margin_col = (100, 100, 200) # Reddish margin

    thickness = 1

    if kind == 'lined':
        step = random.randint(25, 60)
        # Horizontal lines
        for y in range(step, h, step):
            cv2.line(overlay, (0, y), (w, y), line_col, thickness)

        # Vertical Margin
        if random.random() > 0.2:
            margin_x = random.randint(40, 100)
            cv2.line(overlay, (margin_x, 0), (margin_x, h), margin_col, 2)

    elif kind == 'graph':
        step = random.randint(20, 50)
        # Horizontal
        for y in range(0, h, step):
            cv2.line(overlay, (0, y), (w, y), line_col, thickness)
        # Vertical
        for x in range(0, w, step):
            cv2.line(overlay, (x, 0), (x, h), line_col, thickness)

    # Blend lines nicely so they aren't perfect
    alpha = random.uniform(0.6, 0.9)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
    return img

def add_background_watermark(img):
    """Adds faint text that should be IGNORED by the model (not in mask)."""
    if random.random() > 0.5:
        return img

    overlay = img.copy()
    text = random.choice(["DRAFT", "CONFIDENTIAL", "COPY", "SAMPLE", "12345", "URGENT"])
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = random.uniform(2.0, 5.0)
    thickness = random.randint(2, 10)

    (w_txt, h_txt), _ = cv2.getTextSize(text, font, scale, thickness)

    # Random position
    x = random.randint(0, max(1, img.shape[1] - w_txt))
    y = random.randint(h_txt, max(h_txt + 1, img.shape[0]))

    # Grey color
    col = random.randint(180, 220)
    color = (col, col, col)

    # Rotate text slightly
    angle = random.randint(-45, 45)

    # Create a separate canvas for rotation
    txt_mask = np.zeros_like(img)
    # Draw text
    cv2.putText(txt_mask, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    # Rotate
    M = cv2.getRotationMatrix2D((x + w_txt//2, y - h_txt//2), angle, 1)
    txt_mask = cv2.warpAffine(txt_mask, M, (img.shape[1], img.shape[0]))

    # Blend: We want it faint, effectively a stain
    # We turn white pixels in txt_mask to transparent
    non_black = np.any(txt_mask > 0, axis=-1)
    img[non_black] = (img[non_black] * 0.8 + txt_mask[non_black] * 0.2).astype(np.uint8)

    return img

def generate_complex_background(size):
    """Generates a highly varied, hostile document background."""

    # 1. Base Paper Color (Simulate aged paper, recycled paper, bright white)
    # Themes: White, Yellow (Aged), Blue (Legal), Grey
    theme = random.choices(['white', 'aged', 'blue', 'grey'], weights=[0.4, 0.3, 0.1, 0.2])[0]

    base_map = np.zeros((size, size, 3), dtype=np.uint8)

    if theme == 'white':
        base_map[:] = (random.randint(240, 255), random.randint(240, 255), random.randint(240, 255))
    elif theme == 'aged':
        base_map[:] = (random.randint(180, 220), random.randint(210, 240), random.randint(220, 250)) # BGR: Yellowish
    elif theme == 'blue':
        base_map[:] = (random.randint(230, 255), random.randint(210, 240), random.randint(200, 230))
    elif theme == 'grey':
        val = random.randint(200, 230)
        base_map[:] = (val, val, val)

    img = base_map

    # 2. Add Texture/Grain
    img = add_noise_grain(img, intensity=0.03)

    # 3. Add Paper Structure (Lines/Grid)
    paper_type = random.choices(['plain', 'lined', 'graph'], weights=[0.2, 0.5, 0.3])[0]
    if paper_type != 'plain':
        img = add_paper_lines(img, kind=paper_type)

    # 4. Add "Hostile" Background Text (Watermarks)
    # This teaches the model that not all letters are targets
    img = add_background_watermark(img)

    # 5. Add Stains/Damage
    img = add_coffee_stain(img)

    # 6. Large Shadows / Lighting Gradients
    if random.random() > 0.3:
        shadow_mask = np.tile(np.linspace(0.5, 1.0, size), (size, 1))
        if random.choice([True, False]):
            shadow_mask = shadow_mask.T
        # Randomly flip direction
        if random.choice([True, False]):
            shadow_mask = np.flip(shadow_mask)

        for c in range(3):
            img[:, :, c] = (img[:, :, c] * shadow_mask).astype(np.uint8)

    return img

def apply_elastic_transform(img, mask, alpha, sigma):
    """Applies local warping to simulate crumpled paper."""
    rng = np.random.RandomState()
    shape = img.shape[:2]

    dx = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # Image border: White (looks like paper edge)
    distorted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # Mask border: 255 (Background)
    distorted_mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return distorted_img, distorted_mask

def apply_blur_and_compression(img):
    # Blur
    if random.random() > 0.6:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # JPEG Compression Artifacts
    if random.random() > 0.4:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), random.randint(50, 95)]
        result, encimg = cv2.imencode('.jpg', img, encode_param)
        img = cv2.imdecode(encimg, 1)

    return img

fonts = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_ITALIC
]

# --- Worker Function ---
def generate_single_sample(idx):
    # Re-seed random for multiprocessing safety
    random.seed(os.getpid() + idx)
    np.random.seed(os.getpid() + idx)

    # 1. Generate Complex Background
    img = generate_complex_background(IMG_SIZE)

    # Mask: 255=Background
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

    # 2. Fill with Target Text
    num_text_blobs = random.randint(15, 45)

    for _ in range(num_text_blobs):
        text = get_random_string()
        font = random.choice(fonts)
        scale = random.uniform(0.7, 2.3)
        thickness = random.randint(1, 3)
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)

        x = random.randint(-w//4, IMG_SIZE - w//4)
        y = random.randint(h, IMG_SIZE + h//4)

        # Ink Color: Dark but not perfect black (simulates pen variety)
        # Sometimes blue ink, sometimes black ink
        if random.random() > 0.3:
            # Black-ish
            text_color = (random.randint(0, 60), random.randint(0, 60), random.randint(0, 60))
        else:
            # Blue-ish
            text_color = (random.randint(80, 160), random.randint(20, 80), random.randint(0, 50)) # BGR

        cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

        # Mask: Target Text is 0 (Black)
        cv2.putText(mask, text, (x, y), font, scale, 0, thickness, cv2.LINE_AA)

    # 3. Apply Elastic Distortion (Crumpling)
    # This warps BOTH the background lines and the text together
    alpha = random.uniform(150, 450) # Warp intensity
    sigma = random.uniform(15, 30)   # Smoothness of warp
    img, mask = apply_elastic_transform(img, mask, alpha, sigma)

    # 4. Final Quality Degradation
    img = apply_blur_and_compression(img)

    # Ensure mask is binary (0 or 255)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 5. Save
    filename = f"batch8_{idx:05d}.png"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, filename), img)
    cv2.imwrite(os.path.join(OUT_MSK_DIR, filename), mask)

    return idx

# --- Main Execution ---
if __name__ == "__main__":
    # Create directories
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    print(f"Generating {NUM_SAMPLES} Advanced Samples for Batch 8...")
    print(f"Target: {OUTPUT_DIR}")

    # Determine CPUs
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Using {num_workers} worker processes.")

    # Create Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        results = pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES), chunksize=20)

        count = 0
        for _ in results:
            count += 1
            if count % 200 == 0:
                print(f"Generated {count}/{NUM_SAMPLES}")

    print("\nBatch 8 Generation Complete.")