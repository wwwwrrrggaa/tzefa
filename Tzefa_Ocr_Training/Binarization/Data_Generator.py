import os
import cv2
import numpy as np
import random
import string
import multiprocessing
from functools import partial

# --- CONFIGURATION ---
OUTPUT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_4"
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

def generate_background(size):
    """Creates a dirty/noisy background with colored paper themes and dark patches."""
    # 1. Pick a random base color theme (Light/Pastel versions)
    theme = random.choice(['white', 'blue', 'green', 'red'])

    low_res_size = 64
    base_map = np.zeros((low_res_size, low_res_size, 3), dtype=np.uint8)

    # Set base colors based on theme (High values = Light/Pastel)
    if theme == 'white':
        base_map[:,:,0] = np.random.randint(200, 240, (low_res_size, low_res_size)) # B
        base_map[:,:,1] = np.random.randint(210, 250, (low_res_size, low_res_size)) # G
        base_map[:,:,2] = np.random.randint(220, 255, (low_res_size, low_res_size)) # R
    elif theme == 'blue':
        base_map[:,:,0] = np.random.randint(220, 255, (low_res_size, low_res_size))
        base_map[:,:,1] = np.random.randint(190, 230, (low_res_size, low_res_size))
        base_map[:,:,2] = np.random.randint(190, 230, (low_res_size, low_res_size))
    elif theme == 'green':
        base_map[:,:,0] = np.random.randint(190, 230, (low_res_size, low_res_size))
        base_map[:,:,1] = np.random.randint(220, 255, (low_res_size, low_res_size))
        base_map[:,:,2] = np.random.randint(190, 230, (low_res_size, low_res_size))
    elif theme == 'red':
        base_map[:,:,0] = np.random.randint(190, 230, (low_res_size, low_res_size))
        base_map[:,:,1] = np.random.randint(190, 230, (low_res_size, low_res_size))
        base_map[:,:,2] = np.random.randint(220, 255, (low_res_size, low_res_size))

    # Resize to full size
    img = cv2.resize(base_map, (size, size), interpolation=cv2.INTER_CUBIC)

    # 2. Add Darker Patches (Stains/Shadows/Unevenness)
    mask_size = 128
    patch_mask = np.zeros((mask_size, mask_size), dtype=np.uint8)

    for _ in range(random.randint(5, 12)):
        center = (random.randint(0, mask_size), random.randint(0, mask_size))
        axes = (random.randint(10, 40), random.randint(10, 40))
        angle = random.randint(0, 360)
        intensity = random.randint(20, 70)
        cv2.ellipse(patch_mask, center, axes, angle, 0, 360, intensity, -1)

    patch_mask = cv2.GaussianBlur(patch_mask, (41, 41), 0)
    patch_mask = cv2.resize(patch_mask, (size, size), interpolation=cv2.INTER_CUBIC)

    patch_mask_3c = cv2.merge([patch_mask, patch_mask, patch_mask])
    img = cv2.subtract(img, patch_mask_3c)

    # 3. Gradient Overlay
    if random.random() > 0.3:
        gradient_mask = np.tile(np.linspace(0.6, 1.0, size), (size, 1))
        if random.choice([True, False]):
            gradient_mask = gradient_mask.T
        for c in range(3):
            img[:, :, c] = (img[:, :, c] * gradient_mask).astype(np.uint8)

    # 4. Grid Lines
    if random.random() > 0.6:
        step = random.randint(30, 80)
        line_color = (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180))
        thickness = 1
        for y in range(0, size, step):
            cv2.line(img, (0, y), (size, y), line_color, thickness)
        if random.random() > 0.7:
            for x in range(0, size, step):
                cv2.line(img, (x, 0), (x, size), line_color, thickness)

    # 5. Dirt/Speckles
    if random.random() > 0.4:
        noise = np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)
        mask_noise = (noise < 30).astype(np.uint8)
        noise_layer = (mask_noise * 50).astype(np.uint8)
        img = cv2.subtract(img, noise_layer)

    return img

def apply_elastic_transform(img, mask, alpha, sigma):
    """Applies local warping."""
    # Re-seed numpy for this process to ensure uniqueness in multiprocessing
    # (Though standard spawn usually handles this, explicit state is safer)
    rng = np.random.RandomState()
    shape = img.shape[:2]

    dx = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((rng.rand(*shape) * 2 - 1), (0, 0), sigma) * alpha

    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # Border Value: 255 (White) for image
    distorted_img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    # Border Value: 255 (Background) for mask
    distorted_mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)

    return distorted_img, distorted_mask

def apply_blur_and_noise(img):
    if random.random() > 0.5:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    if random.random() > 0.3:
        noise = np.random.normal(0, 10, img.shape).astype(np.float32)
        img_float = img.astype(np.float32) + noise
        img = np.clip(img_float, 0, 255).astype(np.uint8)
    return img

fonts = [
    cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX, cv2.FONT_HERSHEY_COMPLEX_SMALL,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, cv2.FONT_HERSHEY_SCRIPT_COMPLEX, cv2.FONT_ITALIC
]

# --- Worker Function ---
def generate_single_sample(idx):
    # Re-seed random for multiprocessing safety on Windows
    random.seed(os.getpid() + idx)
    np.random.seed(os.getpid() + idx)

    # 1. Setup Canvas (Light/Colored Background)
    img = generate_background(IMG_SIZE)

    # Mask: 255=Background (matches real data standard)
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

    # 2. Fill with Text
    num_text_blobs = random.randint(15, 40)

    for _ in range(num_text_blobs):
        text = get_random_string()
        font = random.choice(fonts)
        scale = random.uniform(0.8, 2.5)
        thickness = random.randint(1, 3)
        (w, h), baseline = cv2.getTextSize(text, font, scale, thickness)

        x = random.randint(-w//4, IMG_SIZE - w//4)
        y = random.randint(h, IMG_SIZE + h//4)

        # Dark Ink (0-80)
        text_color = (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))
        cv2.putText(img, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)

        # Mask: Black Text (0)
        cv2.putText(mask, text, (x, y), font, scale, 0, thickness, cv2.LINE_AA)

    # 3. Apply HEAVY Distortions
    alpha = random.uniform(200, 500)
    sigma = random.uniform(20, 40)
    img, mask = apply_elastic_transform(img, mask, alpha, sigma)

    img = apply_blur_and_noise(img)

    # Threshold Mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # 4. Save
    filename = f"captcha_{idx:05d}.png"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, filename), img)
    cv2.imwrite(os.path.join(OUT_MSK_DIR, filename), mask)

    return idx

# --- Main Execution ---
if __name__ == "__main__":
    # Create directories
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    print(f"Generating {NUM_SAMPLES} Hard Document Samples (Multicore)...")

    # Determine CPUs (leave one free for system)
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f"Using {num_workers} worker processes.")

    # Create Pool
    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for slightly better performance if order doesn't matter
        # Chunksize helps reduce IPC overhead
        results = pool.imap_unordered(generate_single_sample, range(NUM_SAMPLES), chunksize=10)

        # Monitor progress
        count = 0
        for _ in results:
            count += 1
            if count % 100 == 0:
                print(f"Generated {count}/{NUM_SAMPLES}")

    print("\nBatch 4 Generation Complete.")