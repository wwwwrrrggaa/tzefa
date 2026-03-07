"""
Tzefa - Complete Pipeline Demo Space
Image → Binarization → Line Segmentation → Word Segmentation → OCR →
Error Correction → Compilation → Execution

Supports:
  - Dialect toggle: 3-word (classic) / 4-word (verbose)
  - Line segmentation toggle: YOLO (trained model) / Surya (general detector)
  - Binarization model toggle: mit_b3 / mit_b5
"""
import os
import gc
import sys
import subprocess
import importlib
import traceback

import cv2
import torch
import numpy as np
from PIL import Image
import gradio as gr
from huggingface_hub import hf_hub_download
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from ultralytics import YOLO

SPACE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SPACE_DIR)

from language.dialects import THREE_WORD, FOUR_WORD, CAPS_ONLY, MIXED_CASE
from language.ErrorCorrection import TzefaParser
from language import topy

# ══════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════
HF_TOKEN = os.environ.get("HF_TOKEN")
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

BIN_B3_REPO     = "WARAJA/Model"
BIN_B3_FILE     = "b3_model.pth"
BIN_B5_REPO     = "WARAJA/b5_model"
BIN_B5_FILE     = "b5_model.pth"
YOLO_REPO       = "WARAJA/Tzefa-Line-Segmentation-YOLO"
YOLO_FILE       = "best.pt"
TROCR_REPO      = "WARAJA/Tzefa-Word-OCR-TrOCR"
TROCR_BASE_PROC = "microsoft/trocr-small-stage1"

TILE_SIZE        = 640
YOLO_IMGSZ       = 640
MAX_DILATE_ITERS = 200

_DIALECT_MAP = {"4-word (verbose)": FOUR_WORD, "3-word (classic)": THREE_WORD}
_CASING_MAP  = {"CAPS only": CAPS_ONLY, "Mixed case": MIXED_CASE}


# ══════════════════════════════════════════════════════════════
# 1. BINARIZATION
# ══════════════════════════════════════════════════════════════
class HighResMAnet(nn.Module):
    def __init__(self, encoder_name="mit_b5", classes=1):
        super().__init__()
        self.base_model = smp.MAnet(
            encoder_name=encoder_name, encoder_weights=None,
            in_channels=3, classes=classes, encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )
        self.high_res_stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
        )
        self.final_fusion = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(16, classes, 1),
        )

    def forward(self, x):
        hr   = self.high_res_stem(x)
        feat = self.base_model.encoder(x)
        dec  = self.base_model.decoder(feat)
        return self.final_fusion(torch.cat([dec, hr], dim=1))


def _load_bin_models():
    models = {}
    b3_path = hf_hub_download(BIN_B3_REPO, BIN_B3_FILE, token=HF_TOKEN, repo_type="space")
    m3 = smp.Unet(encoder_name="mit_b3", encoder_weights=None, in_channels=3, classes=1)
    ckpt3 = torch.load(b3_path, map_location=DEVICE)
    m3.load_state_dict(ckpt3.get("model_state_dict", ckpt3))
    models["mit_b3 (Standard)"] = m3.to(DEVICE).eval()
    b5_path = hf_hub_download(BIN_B5_REPO, BIN_B5_FILE, token=HF_TOKEN, repo_type="model")
    m5 = HighResMAnet(encoder_name="mit_b5")
    ckpt5 = torch.load(b5_path, map_location=DEVICE)
    m5.load_state_dict(ckpt5.get("model_state_dict", ckpt5))
    models["mit_b5 (HighRes)"] = m5.to(DEVICE).eval()
    return models


def _preprocess_tile(pil_img):
    arr  = np.array(pil_img).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return torch.from_numpy(((arr - mean) / std).transpose(2, 0, 1))


def binarize(pil_img, model):
    orig_w, orig_h = pil_img.size
    pad_w  = (TILE_SIZE - orig_w % TILE_SIZE) % TILE_SIZE
    pad_h  = (TILE_SIZE - orig_h % TILE_SIZE) % TILE_SIZE
    padded = Image.new("RGB", (orig_w + pad_w, orig_h + pad_h), (255, 255, 255))
    padded.paste(pil_img, (0, 0))
    nw, nh = padded.size
    canvas = Image.new("L", (nw, nh), 255)
    for y in range(0, nh, TILE_SIZE):
        for x in range(0, nw, TILE_SIZE):
            tile = padded.crop((x, y, x + TILE_SIZE, y + TILE_SIZE))
            t = _preprocess_tile(tile).unsqueeze(0).to(DEVICE).float()
            with torch.no_grad():
                logits = model(t)
                if logits.shape[-2:] != (TILE_SIZE, TILE_SIZE):
                    logits = F.interpolate(logits, (TILE_SIZE, TILE_SIZE), mode="bilinear")
                mask = (torch.sigmoid(logits) > 0.5).float().cpu().numpy()[0, 0]
            canvas.paste(Image.fromarray(((1.0 - mask) * 255).astype(np.uint8)), (x, y))
    return canvas.crop((0, 0, orig_w, orig_h))


# ══════════════════════════════════════════════════════════════
# 2. LINE SEGMENTATION
# ══════════════════════════════════════════════════════════════
def _load_yolo():
    path = hf_hub_download(YOLO_REPO, YOLO_FILE, token=HF_TOKEN, repo_type="model")
    return YOLO(path)


def segment_lines_yolo(bin_arr, yolo_model):
    img_rgb  = cv2.cvtColor(bin_arr, cv2.COLOR_GRAY2RGB) if len(bin_arr.shape) == 2 else bin_arr
    orig_h, orig_w = img_rgb.shape[:2]
    results  = yolo_model.predict(img_rgb, imgsz=YOLO_IMGSZ, conf=0.2, iou=0.2, verbose=False)
    truelines = []
    if len(results) > 0 and results[0].obb is not None:
        obbs = sorted(results[0].obb.xyxyxyxy.cpu().numpy(), key=lambda p: np.min(p[:, 1]))
        for pts in obbs:
            rx0, rx1 = np.min(pts[:, 0]), np.max(pts[:, 0])
            ry0, ry1 = np.min(pts[:, 1]), np.max(pts[:, 1])
            pad = (rx1 - rx0) * 0.12
            x0 = int(np.clip(rx0 - pad, 0, orig_w))
            x1 = int(np.clip(rx1 + pad, 0, orig_w))
            y0 = int(np.clip(ry0, 0, orig_h))
            y1 = int(np.clip(ry1, 0, orig_h))
            if x1 - x0 > 0 and y1 - y0 > 0:
                truelines.append((x0, y0, x1 - x0, y1 - y0))
    return truelines


_surya_predictor = None

def segment_lines_surya(bin_arr):
    global _surya_predictor
    os.environ.setdefault("DETECTOR_TEXT_THRESHOLD", "0.75")
    os.environ.setdefault("DETECTOR_BLANK_THRESHOLD", "0.45")
    try:
        from surya.detection import DetectionPredictor
    except ImportError:
        raise RuntimeError("surya-ocr not installed. Add 'surya-ocr' to requirements.txt.")
    if _surya_predictor is None:
        _surya_predictor = DetectionPredictor()
    img_rgb   = cv2.cvtColor(bin_arr, cv2.COLOR_GRAY2RGB) if len(bin_arr.shape) == 2 else bin_arr
    pil_image = Image.fromarray(img_rgb)
    predictions = _surya_predictor([pil_image])

    CONF_THRESHOLD = 0.6
    raw = []
    if predictions and predictions[0].bboxes:
        for bbox in predictions[0].bboxes:
            conf = getattr(bbox, "confidence", 1.0)
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = bbox.bbox
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                raw.append([float(x1), float(y1), float(x2), float(y2)])

    raw.sort(key=lambda b: (b[1] + b[3]) / 2)

    def overlaps_v(a, b):
        return a[1] < b[3] and b[1] < a[3]

    merged = []
    for box in raw:
        placed = False
        for m in merged:
            if overlaps_v(m, box):
                m[0] = min(m[0], box[0]); m[1] = min(m[1], box[1])
                m[2] = max(m[2], box[2]); m[3] = max(m[3], box[3])
                placed = True; break
        if not placed:
            merged.append(list(box))

    merged.sort(key=lambda b: b[1])
    return [(int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])) for b in merged]


# ══════════════════════════════════════════════════════════════
# 3. WORD SEGMENTATION
# ══════════════════════════════════════════════════════════════
def _get_word_boxes(dilated, min_w, min_h):
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(
        [b for b in [cv2.boundingRect(c) for c in contours] if b[2] >= min_w and b[3] >= min_h],
        key=lambda b: b[0],
    )


def segment_words(bin_arr, lines, target_words):
    words_dict = {}
    for i, (lx, ly, lw, lh) in enumerate(lines):
        ih, iw = bin_arr.shape[:2]
        ly, lx = max(0, ly), max(0, lx)
        lh, lw = min(lh, ih - ly), min(lw, iw - lx)
        if lw <= 0 or lh <= 0:
            continue
        crop   = bin_arr[ly:ly+lh, lx:lx+lw]
        inv    = cv2.bitwise_not(crop)
        min_ww = max(5, int(lw * 0.02))
        min_wh = max(5, int(lh * 0.25))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        dilated, prev, found = inv.copy(), None, False
        for _ in range(MAX_DILATE_ITERS):
            dilated = cv2.dilate(dilated, kernel, iterations=1)
            boxes   = _get_word_boxes(dilated, min_ww, min_wh)
            if len(boxes) == target_words:
                prev = boxes; found = True; break
            elif len(boxes) < target_words:
                break
            else:
                prev = boxes
        if not found and prev and len(prev) > target_words:
            while len(prev) > target_words:
                gaps = [(prev[j+1][0] - (prev[j][0]+prev[j][2]), j) for j in range(len(prev)-1)]
                _, mi = min(gaps)
                b1, b2 = prev[mi], prev[mi+1]
                merged = (
                    min(b1[0],b2[0]), min(b1[1],b2[1]),
                    max(b1[0]+b1[2],b2[0]+b2[2])-min(b1[0],b2[0]),
                    max(b1[1]+b1[3],b2[1]+b2[3])-min(b1[1],b2[1]),
                )
                prev = list(prev); prev[mi] = merged; prev.pop(mi+1)
            found = True
        if not found or not prev or len(prev) != target_words:
            continue
        words_dict[i+1] = {wi+1: (wx, wx+ww) for wi, (wx, wy, ww, wh) in enumerate(prev)}
    return words_dict


# ══════════════════════════════════════════════════════════════
# 4. OCR
# ══════════════════════════════════════════════════════════════
def _load_trocr():
    proc  = TrOCRProcessor.from_pretrained(TROCR_BASE_PROC, use_fast=False)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_REPO, token=HF_TOKEN).to(DEVICE).eval()
    return proc, model


def _pad_aspect(img, max_ratio=4.0):
    w, h = img.size
    if w <= max_ratio * h:
        return img
    th  = int(w / max_ratio)
    pad = th - h
    from PIL import ImageOps
    return ImageOps.expand(img, (0, pad//2, 0, pad - pad//2), fill=(255, 255, 255))


def ocr_word(img_pil, proc, model):
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    img_pil = _pad_aspect(img_pil)
    pv = proc(img_pil, return_tensors="pt").pixel_values.to(DEVICE)
    with torch.no_grad():
        ids = model.generate(pv)
    txt   = proc.batch_decode(ids, skip_special_tokens=True)[0]
    parts = txt.split()
    return max(parts, key=len) if parts else txt


# ══════════════════════════════════════════════════════════════
# 5. VISUALISATION
# ══════════════════════════════════════════════════════════════
def draw_line_bboxes(img_arr, bboxes):
    vis = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB) if len(img_arr.shape) == 2 else img_arr.copy()
    for i, (x, y, w, h) in enumerate(bboxes):
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 50, 50), 2)
        cv2.putText(vis, str(i+1), (x, max(y-5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
    return vis


def draw_word_bboxes(img_arr, word_tuples):
    vis    = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB) if len(img_arr.shape) == 2 else img_arr.copy()
    colors = [(50, 220, 50), (50, 180, 255), (255, 180, 50), (220, 50, 220)]
    for lt in word_tuples:
        for wi, (text, (x1, y1, x2, y2)) in enumerate(lt):
            c = colors[wi % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), c, 2)
            cv2.putText(vis, text, (x1, max(y1-4, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, c, 1)
    return vis


# ══════════════════════════════════════════════════════════════
# 6. UTILITIES
# ══════════════════════════════════════════════════════════════
def clear_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def execute_code(compiled_code):
    try:
        result = subprocess.run(
            [sys.executable, "-c", compiled_code],
            capture_output=True, text=True, timeout=15,
            cwd=SPACE_DIR,
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[Process exited with code {result.returncode}]"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "[Execution timed out after 15 seconds]"
    except Exception as e:
        return f"[Execution error: {e}]"


# ══════════════════════════════════════════════════════════════
# 7. FULL PIPELINE
# ══════════════════════════════════════════════════════════════
def run_full_pipeline(input_image, bin_model_choice, dialect_choice, casing_choice, seg_method):
    if input_image is None:
        return None, None, None, "", "", "", "", "No image provided."

    if isinstance(input_image, np.ndarray):
        pil_img = Image.fromarray(input_image).convert("RGB")
    else:
        pil_img = input_image.convert("RGB")

    dialect = _DIALECT_MAP.get(dialect_choice, FOUR_WORD)
    casing  = _CASING_MAP.get(casing_choice, CAPS_ONLY)
    status  = []

    # Fresh language state for every run
    importlib.reload(topy)
    parser       = TzefaParser(dialect=dialect, casing=casing)
    target_words = parser.expected_words_per_line

    # ── Stage 1: Binarization ──
    try:
        status.append("[1/6] Binarization...")
        bin_models = _load_bin_models()
        bin_pil    = binarize(pil_img, bin_models[bin_model_choice])
        bin_arr    = np.array(bin_pil)
        del bin_models; clear_vram()
        status.append("  OK")
    except Exception as e:
        return None, None, None, "", "", "", "", f"Binarization failed: {e}"

    # ── Stage 2: Line Segmentation ──
    try:
        status.append(f"[2/6] Line Segmentation ({seg_method})...")
        if seg_method == "Surya":
            truelines = segment_lines_surya(bin_arr)
        else:
            yolo_model = _load_yolo()
            truelines  = segment_lines_yolo(bin_arr, yolo_model)
            del yolo_model; clear_vram()
        status.append(f"  OK  {len(truelines)} lines")
        line_vis = draw_line_bboxes(bin_arr, truelines)
    except Exception as e:
        return bin_arr, None, None, "", "", "", "", f"Line Seg failed: {e}\n{traceback.format_exc()}"

    # ── Stage 3: Word Seg + OCR ──
    try:
        status.append("[3/6] Word Segmentation + OCR...")
        words             = segment_words(bin_arr, truelines, target_words)
        proc, trocr_model = _load_trocr()
        all_line_tuples, raw_lines = [], []
        for ln in sorted(words.keys()):
            if ln - 1 >= len(truelines):
                continue
            lx, ly, lw, lh = truelines[ln - 1]
            line_tuples = []
            for wn in sorted(words[ln].keys()):
                wx1, wx2 = words[ln][wn]
                ax1 = max(0, int(lx + wx1))
                ax2 = min(bin_arr.shape[1], int(lx + wx2))
                ay1 = max(0, ly - 20)
                ay2 = min(bin_arr.shape[0], ly + lh + 20)
                text = ocr_word(Image.fromarray(bin_arr[ay1:ay2, ax1:ax2]), proc, trocr_model)
                line_tuples.append((text, (ax1, ay1, ax2, ay2)))
            raw_lines.append(" ".join(t[0] for t in line_tuples))
            all_line_tuples.append(line_tuples)
        del proc, trocr_model; clear_vram()
        word_vis = draw_word_bboxes(bin_arr, all_line_tuples)
        raw_text = "\n".join(raw_lines)
        status.append(f"  OK  {len(raw_lines)} lines recognised")
    except Exception as e:
        return bin_arr, line_vis, None, "", "", "", "", f"OCR failed: {e}\n{traceback.format_exc()}"

    # ── Stage 4: Error Correction ──
    try:
        status.append("[4/6] Error Correction...")
        parser.init_indent_table(len(truelines))
        corrected_lines, bytecode_list = [], []
        for line_entries in all_line_tuples:
            if not line_entries:
                corrected_lines.append("")
                bytecode_list.append(["MAKE", "INTEGER", "TEMPORARY", "0"])
                continue
            raw_tokens = [t[0] for t in line_entries]
            while len(raw_tokens) < target_words:
                raw_tokens.append("")
            raw_tokens  = raw_tokens[:target_words]
            normalised  = parser.normalize_source_line(raw_tokens)
            bytecode    = parser.parse_line(normalised)
            bytecode_list.append(bytecode)
            corrected_lines.append(" ".join(bytecode))   # post-correction output
        corrected_text = "\n".join(corrected_lines)
        status.append("  OK")
    except Exception as e:
        return bin_arr, line_vis, word_vis, raw_text, "", "", "", \
               f"Error Correction failed: {e}\n{traceback.format_exc()}"
