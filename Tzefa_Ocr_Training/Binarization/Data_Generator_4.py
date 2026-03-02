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
OUTPUT_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_10"
OUT_IMG_DIR = os.path.join(OUTPUT_DIR, "images")
OUT_MSK_DIR = os.path.join(OUTPUT_DIR, "masks")

NUM_SAMPLES = 5000
IMG_SIZE = 640


def get_system_fonts():
    fonts = glob.glob(r"C:\Windows\Fonts\*.ttf")
    if not fonts:
        fonts = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True)
    return [f for f in fonts if "dings" not in f.lower() and "symbol" not in f.lower()]


SYSTEM_FONTS = get_system_fonts()


def get_random_word():
    """Returns a word from a randomly selected script."""
    scripts = {
        "latin": string.ascii_letters,
        "latin_ext": string.ascii_letters + "áéíóúñäöüßàèìòùçÁÉÍÓÚÑÄÖÜÀÈÌÒÙÇ",
        "cyrillic": "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ",
        "greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΞΟΠΡΣΤΥΦΧΨΩ",
        "hebrew": "אבגדהוזחטיכלמנסעפצקרשת",
        "vietnamese": string.ascii_letters + "ăâđêôơưàảãáạằẳẵắặầẩẫấậèẻẽéẹềểễếệìỉĩíịòỏõóọồổỗốộờởỡớợùủũúụừửữứựỳỷỹýỵ",
    }
    script = random.choice(list(scripts.values()))
    return "".join(random.choices(script, k=random.randint(3, 10)))


def generate_basic_sample(idx):
    random.seed(os.getpid() + idx)

    # 1. Pure White Background
    img_pil = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    mask_pil = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)  # White = Background

    draw = ImageDraw.Draw(img_pil)
    mask_draw = ImageDraw.Draw(mask_pil)

    cursor_y = random.randint(10, 40)

    while cursor_y < IMG_SIZE - 20:
        # Randomize Font per Line to maximize variety
        try:
            f_path = random.choice(SYSTEM_FONTS)
            # Use smaller sizes for thin writing
            f_size = random.randint(12, 24)
            font = ImageFont.truetype(f_path, f_size)
        except:
            font = ImageFont.load_default()

        # Generate a line with mixed languages
        line_text = " ".join([get_random_word() for _ in range(random.randint(4, 10))])

        # Random Colored Text (Ink)
        # Occasionally use very light colors to challenge the model
        if random.random() > 0.8:
            ink_color = (random.randint(100, 180), random.randint(100, 180), random.randint(100, 180))
        else:
            ink_color = (random.randint(0, 80), random.randint(0, 80), random.randint(0, 80))

        # Center or random indent
        x_pos = random.randint(10, 50)

        # Draw Text
        draw.text((x_pos, cursor_y), line_text, font=font, fill=ink_color)
        # Draw Mask (0 = Text)
        mask_draw.text((x_pos, cursor_y), line_text, font=font, fill=0)

        # Move cursor
        bbox = draw.textbbox((x_pos, cursor_y), line_text, font=font)
        cursor_y += (bbox[3] - bbox[1]) + random.randint(5, 15)

    # Save as PNG to avoid compression artifacts
    filename = f"batch10_{idx:05d}.png"
    img_pil.save(os.path.join(OUT_IMG_DIR, filename))
    mask_pil.save(os.path.join(OUT_MSK_DIR, filename))


if __name__ == "__main__":
    os.makedirs(OUT_IMG_DIR, exist_ok=True)
    os.makedirs(OUT_MSK_DIR, exist_ok=True)

    print(f"Generating {NUM_SAMPLES} Basic 'Babel' Documents (Batch 10)...")

    num_workers = max(1, multiprocessing.cpu_count() - 2)
    with multiprocessing.Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(generate_basic_sample, range(NUM_SAMPLES)), total=NUM_SAMPLES))

    print("Generation Complete.")