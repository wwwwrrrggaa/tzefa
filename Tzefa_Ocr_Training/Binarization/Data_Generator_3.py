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
OUTPUT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_9"
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(OUTPUT_DIR, "masks")

NUM_SAMPLES = 5000
IMG_SIZE = 640

# --- GLOBAL RESOURCES ---
def get_system_fonts():
    """Scans Windows font directory for TTF files."""
    # Try standard Windows path
    fonts = glob.glob(r"C:\Windows\Fonts\*.ttf")
    if not fonts:
        # Fallback for Linux/WSL or if dir is empty
        fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)

    # Filter out symbol fonts usually (like webdings) as they don't render text
    valid_fonts = [f for f in fonts if "dings" not in f.lower() and "symbol" not in f.lower()]
    return valid_fonts if valid_fonts else ["arial.ttf"]

SYSTEM_FONTS = get_system_fonts()

# --- TEXT GENERATION ---

def get_random_word(script="latin"):
    """Generates random words resembling different languages."""
    length = random.randint(3, 10)

    if script == "latin":
        chars = string.ascii_letters
    elif script == "latin_extended": # French, Spanish, German style
        chars = string.ascii_letters + "áéíóúñäöüßàèìòùçÁÉÍÓÚÑÄÖÜÀÈÌÒÙÇ"
    elif script == "cyrillic":
        chars = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ"
    elif script == "greek":
        chars = "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
    elif script == "hebrew":
        chars = "אבגדהוזחטיכלמנסעפצקרשת"
    elif script == "vietnamese":
        chars = string.ascii_letters + "ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ"
    else:
        chars = string.ascii_letters

    return ''.join(random.choices(chars, k=length))

# --- TEXTURE & COLOR ---

def apply_strong_tint(img):
    """Applies a global color tint to the image."""
    # 0=None, 1=Sepia/Yellow, 2=Blue/Cool, 3=Red/Pink, 4=Greenish
    mode = random.choices([0, 1, 2, 3, 4], weights=[0.4, 0.2, 0.15, 0.15, 0.1])[0]

    if mode == 0:
        return img, (0,0,0) # No tint shift

    h, w = img.shape[:2]
    overlay = np.zeros_like(img)

    if mode == 1: # Sepia/Old Paper
        overlay[:] = (180, 220, 240) # BGR
    elif mode == 2: # Blue/Cool
        overlay[:] = (240, 200, 180)
    elif mode == 3: # Reddish
        overlay[:] = (180, 180, 240)
    elif mode == 4: # Greenish
        overlay[:] = (180, 240, 180)

    # Blend
    alpha = random.uniform(0.1, 0.4)
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Return the tint color shift for text color calculation
    shift = (
        int(overlay[0,0,0] * alpha),
        int(overlay[0,0,1] * alpha),
        int(overlay[0,0,2] * alpha)
    )
    return img, shift

def generate_paper_texture(w, h):
    base_val = random.randint(230, 255)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = (base_val, base_val, base_val)

    # Add Grain
    noise = np.random.normal(0, 3, (h, w, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img

# --- LAYOUT ---

def draw_document_layout(img_np, mask_np, tint_shift):
    """Uses PIL to draw text with system fonts and variable contrast."""
    h, w = img_np.shape[:2]

    # Convert to PIL
    pil_img = Image.fromarray(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB))
    pil_mask = Image.fromarray(mask_np)

    draw_img = ImageDraw.Draw(pil_img)
    draw_msk = ImageDraw.Draw(pil_mask)

    # Select Font & Language
    font_path = random.choice(SYSTEM_FONTS)
    try:
        font_size = random.randint(18, 42)
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()

    # Pick a dominant script for this document
    script = random.choice(["latin", "latin", "latin_extended", "cyrillic", "greek", "vietnamese"])
    if "times" in font_path.lower() or "arial" in font_path.lower():
        # Standard fonts support more scripts safely
        pass
    else:
        # Fallback to latin for obscure fonts to avoid boxes
        script = "latin"

    # Margins
    margin_top = random.randint(40, 100)
    margin_bottom = random.randint(40, 100)
    margin_left = random.randint(40, 80)
    margin_right = random.randint(40, 80)

    cursor_y = margin_top

    # Estimate line height
    bbox = font.getbbox("Agy")
    line_height = (bbox[3] - bbox[1]) + int(font_size * 0.5)

    # Base paper color (approximate, since texture varies)
    paper_r, paper_g, paper_b = 240, 240, 240

    # Adjust paper color estimate by tint
    paper_b = max(0, min(255, paper_b - tint_shift[0])) # B
    paper_g = max(0, min(255, paper_g - tint_shift[1])) # G
    paper_r = max(0, min(255, paper_r - tint_shift[2])) # R

    while cursor_y < (h - margin_bottom):
        line_text = ""
        while True:
            word = get_random_word(script)
            test_line = line_text + word + " "
            # Measure
            txt_bbox = draw_img.textbbox((0,0), test_line, font=font)
            text_w = txt_bbox[2] - txt_bbox[0]

            if margin_left + text_w > (w - margin_right):
                break
            line_text = test_line

        if line_text:
            # --- COLOR CALCULATION ---
            # To make text "Hard to spot", we want low contrast with background.
            # Normal text: Contrast ~150-200. Hard text: Contrast ~30-60.

            difficulty = random.random()
            if difficulty > 0.7:
                # Hard / Low Contrast / Faded
                contrast = random.randint(40, 80)
            elif difficulty > 0.9:
                # Very Hard
                contrast = random.randint(20, 40)
            else:
                # Normal
                contrast = random.randint(150, 220)

            # Determine Ink Base Color (Black, Blue, Red)
            ink_base = random.choice(['black', 'black', 'black', 'blue', 'red'])

            if ink_base == 'black':
                target_r, target_g, target_b = paper_r - contrast, paper_g - contrast, paper_b - contrast
            elif ink_base == 'blue':
                target_r, target_g, target_b = paper_r - contrast, paper_g - contrast * 0.8, paper_b
            elif ink_base == 'red':
                target_r, target_g, target_b = paper_r, paper_g - contrast, paper_b - contrast

            # Clip values
            final_r = max(0, min(255, int(target_r)))
            final_g = max(0, min(255, int(target_g)))
            final_b = max(0, min(255, int(target_b)))

            # Apply Jitter
            jitter_y = random.randint(-1, 1)

            # Draw
            draw_img.text((margin_left, cursor_y + jitter_y), line_text, font=font, fill=(final_r, final_g, final_b))
            draw_msk.text((margin_left, cursor_y + jitter_y), line_text, font=font, fill=0) # Mask always 0 (black)

        cursor_y += line_height
        if random.random() > 0.85: cursor_y += line_height

    # Convert back to OpenCV
    img_res = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    msk_res = np.array(pil_mask)
    return img_res, msk_res

# --- DISTORTIONS (Same as before) ---

def apply_curl(img, mask):
    h, w = img.shape[:2]
    map_x = np.zeros((h, w), np.float32)
    map_y = np.zeros((h, w), np.float32)

    amp = random.uniform(10, 40)
    freq = random.uniform(0.005, 0.02)
    phase = random.uniform(0, math.pi * 2)

    if random.random() > 0.5:
        for y in range(h): map_x[y, :] = np.arange(w)
        for x in range(w): map_y[:, x] = np.arange(h) + amp * math.sin(x * freq + phase)
    else:
        for x in range(w): map_y[:, x] = np.arange(h)
        for y in range(h): map_x[y, :] = np.arange(w) + amp * math.sin(y * freq + phase)

    img_warped = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(200, 200, 200))
    mask_warped = cv2.remap(mask, map_x, map_y, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return img_warped, mask_warped

def apply_perspective(img, mask):
    h, w = img.shape[:2]
    src_pts = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dev = w * 0.15
    dst_pts = np.float32([
        [random.uniform(0, dev), random.uniform(0, dev)],
        [random.uniform(w - dev, w), random.uniform(0, dev)],
        [random.uniform(0, dev), random.uniform(h - dev, h)],
        [random.uniform(w - dev, w), random.uniform(h - dev, h)]
    ])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_warp = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(220, 220, 220))
    mask_warp = cv2.warpPerspective(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return img_warp, mask_warp

def apply_realistic_lighting(img):
    h, w = img.shape[:2]
    img = img.astype(np.float32)
    gradient = np.tile(np.linspace(0.6, 1.1, w), (h, 1))
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, random.randint(0, 360), 1.0)
    gradient = cv2.warpAffine(gradient, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
    img = img * gradient[:, :, np.newaxis]

    if random.random() > 0.4:
        shadow_mask = np.ones((h, w), dtype=np.float32)
        for _ in range(random.randint(1, 3)):
            cv2.circle(shadow_mask, (random.randint(0, w), random.randint(0, h)), random.randint(100, 300), 0.6, -1)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (151, 151), 0)
        img = img * shadow_mask[:, :, np.newaxis]
    return np.clip(img, 0, 255).astype(np.uint8)

def apply_blur(img):
    r = random.random()
    if r > 0.7:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    elif r > 0.9:
        size = random.randint(3, 7)
        kernel = np.zeros((size, size))
        kernel[int((size-1)/2), :] = np.ones(size)
        kernel /= size
        img = cv2.filter2D(img, -1, kernel)
    return img

# --- WORKER ---

def generate_sample(idx):
    random.seed(os.getpid() + idx)
    np.random.seed(os.getpid() + idx)

    # 1. Base Texture
    img = generate_paper_texture(IMG_SIZE, IMG_SIZE)
    mask = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255

    # 2. Apply Strong Tint
    img, tint_shift = apply_strong_tint(img)

    # 3. Draw Text (PIL) - Uses tint_shift to calculate low-contrast colors
    img, mask = draw_document_layout(img, mask, tint_shift)

    # 4. Distortions
    if random.random() > 0.3: img, mask = apply_curl(img, mask)
    if random.random() > 0.3: img, mask = apply_perspective(img, mask)

    img = apply_realistic_lighting(img)
    img = apply_blur(img)

    if random.random() > 0.5:
        qual = random.randint(70, 95)
        _, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), qual])
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)

    filename = f"batch9_{idx:05d}.png"
    cv2.imwrite(os.path.join(OUT_IMG_DIR, filename), img)
    cv2.imwrite(os.path.join(OUT_MSK_DIR, filename), mask)

if __name__ == "__main__":
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    print(f"Generating {NUM_SAMPLES} Enhanced Photo Documents (Batch 9)...")
    print(f"Fonts found: {len(SYSTEM_FONTS)}")

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(generate_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))

    print("Generation Complete.")