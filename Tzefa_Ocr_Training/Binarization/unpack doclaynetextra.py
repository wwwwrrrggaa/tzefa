import os
import json
import shutil
import multiprocessing
import time
from pathlib import Path

# pip install pymupdf opencv-python pillow
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image, ImageDraw, ImageFont

# --- CONFIGURATION ---
FONT_DIR_ROOT = r"E:\Storage\fonts-main\ofl"
DEFAULT_FONT_PATH = r"C:\Windows\Fonts\arial.ttf"
DL_JSON = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\Doclaynet_extra\JSON"
DL_PDF = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\Doclaynet_extra\PDF"
BATCH_OUT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_12"

# Higher DPI = clearer text, but requires font size scaling
RENDER_DPI = 200

# --------------------------
# --- FONT HELPERS ---
# --------------------------


def scan_available_fonts(root_dir):
    """Maps normalized font names to file paths."""
    print(f"Scanning fonts in {root_dir}...")
    font_map = {}
    valid_exts = {".ttf", ".otf"}

    if os.path.exists(DEFAULT_FONT_PATH):
        font_map["default"] = DEFAULT_FONT_PATH
        font_map["arial"] = DEFAULT_FONT_PATH

    if os.path.exists(root_dir):
        for root, _, files in os.walk(root_dir):
            for file in files:
                if Path(file).suffix.lower() in valid_exts:
                    stem = Path(file).stem.lower().split("-")[0]
                    font_map[stem] = os.path.join(root, file)
    return font_map


def get_best_font_match(dirty_name, font_map):
    """Matches JSON font name to local file."""
    if not dirty_name:
        return font_map.get("default", DEFAULT_FONT_PATH)

    clean = dirty_name.strip("/").split("+")[-1].lower()
    clean = clean.replace("mt", "").replace("bold", "").replace("italic", "").replace("ps", "")

    if clean in font_map:
        return font_map[clean]

    for key, path in font_map.items():
        if key in clean or clean in key:
            return path

    return font_map.get("default", DEFAULT_FONT_PATH)


# --------------------------
# --- PROCESSING WORKER ---
# --------------------------


def process_single_pair(args):
    json_path, pdf_root, output_dir, font_map = args

    try:
        # 1. Load JSON
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data.get("metadata", {})
        page_hash = meta.get("page_hash", Path(json_path).stem)

        # Coordinate Space Resolution
        json_w = meta.get("coco_width", meta.get("original_width", 0))
        json_h = meta.get("coco_height", meta.get("original_height", 0))

        if json_w == 0 or json_h == 0:
            return False

        # 2. Render PDF (Input Image)
        pdf_file = Path(pdf_root) / (page_hash + ".pdf")
        if not pdf_file.exists():
            return False

        doc = fitz.open(pdf_file)
        page_idx = meta.get("page_no", 1) - 1
        if page_idx >= len(doc):
            page_idx = 0
        page = doc.load_page(page_idx)

        # Render at high DPI
        zoom = RENDER_DPI / 72.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)

        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        # 3. Create Mask (Target)
        curr_h, curr_w = img_array.shape[:2]
        mask_pil = Image.new("L", (curr_w, curr_h), color=255)
        draw = ImageDraw.Draw(mask_pil)

        # Scale Factor: JSON coords -> Image Pixels
        scale_x = curr_w / json_w
        scale_y = curr_h / json_h

        # 4. Draw Text
        cells = data.get("cells", [])

        # Cache font objects to speed up processing
        font_cache = {}

        for cell in cells:
            bbox = cell.get("bbox", [])
            text_content = cell.get("text", "").strip()

            # Extract Font Info
            font_info = cell.get("font", {})
            json_font_name = font_info.get("name", "")

            # --- KEY FIX: Use JSON Size ---
            # Default to 10pt if missing/zero
            json_font_size = font_info.get("size", 10)
            if json_font_size is None or json_font_size <= 0:
                json_font_size = 10

            # Scale font size from PDF Points (72 DPI) to Render DPI
            # Formula: Size_Points * (Target_DPI / 72)
            render_font_size = int(json_font_size * zoom)

            # Ensure minimum legible size (prevent 0px fonts)
            if render_font_size < 8:
                render_font_size = 8

            if len(bbox) == 4 and text_content:
                # Transform Coords
                jx, jy, jw, jh = bbox
                x = int(jx * scale_x)
                y = int(jy * scale_y)
                w = int(jw * scale_x)
                h = int(jh * scale_y)

                # Get Font Object
                font_key = (json_font_name, render_font_size)
                if font_key not in font_cache:
                    f_path = get_best_font_match(json_font_name, font_map)
                    try:
                        font_cache[font_key] = ImageFont.truetype(f_path, render_font_size)
                    except:
                        font_cache[font_key] = ImageFont.load_default()

                font = font_cache[font_key]

                # Calculate text dimensions to center it
                try:
                    if hasattr(font, "getbbox"):
                        bb = font.getbbox(text_content)
                        txt_w = bb[2] - bb[0]
                        txt_h = bb[3] - bb[1]
                    else:
                        txt_w, txt_h = draw.textsize(text_content, font=font)
                except:
                    txt_w, txt_h = 0, 0

                # Center text in bbox
                draw_x = x + (w - txt_w) // 2
                draw_y = y + (h - txt_h) // 2

                # Draw
                draw.text((draw_x, draw_y), text_content, font=font, fill=0)

        # 5. Save Output
        out_path = Path(output_dir)
        img_out = out_path / "images" / f"{page_hash}.jpg"
        mask_out = out_path / "masks" / f"{page_hash}.png"

        img_out.parent.mkdir(parents=True, exist_ok=True)
        mask_out.parent.mkdir(parents=True, exist_ok=True)

        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(img_out), img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        mask_pil.save(mask_out)

        return True

    except Exception:
        return False


# --------------------------
# --- MAIN ---
# --------------------------

if __name__ == "__main__":
    multiprocessing.freeze_support()

    print("--- Starting Processing (Strict Font Sizing) ---")
    available_fonts = scan_available_fonts(FONT_DIR_ROOT)

    json_path_obj = Path(DL_JSON)
    all_jsons = list(json_path_obj.glob("*.json"))

    tasks = []
    for j_file in all_jsons:
        tasks.append((str(j_file), DL_PDF, BATCH_OUT, available_fonts))

    workers = min(multiprocessing.cpu_count(), 12)
    print(f"Workers: {workers} | Files: {len(tasks)}")

    with multiprocessing.Pool(processes=workers) as pool:
        results = pool.map(process_single_pair, tasks, chunksize=10)

    print(f"Completed: {sum(results)} / {len(tasks)}")