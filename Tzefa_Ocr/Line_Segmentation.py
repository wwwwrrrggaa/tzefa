import json
import os
import subprocess
import sys
import textwrap

import cv2
import numpy as np
from ultralytics import YOLO

# Path to the conda env Python that has surya-ocr + working torch
TZEFA_PYTHON = r"C:\dev\tools\envs\tzefa-env\python.exe"

# Point directly to your newly trained XL model weights
MODEL_PATH = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Line_Segmentation\yolo11x-obb-fresh\weights\best.pt"
INFERENCE_SIZE = 640


def load_model():
    return YOLO(MODEL_PATH)


def segment_lines(img_array, method="yolo"):
    """
    Segment lines using the specified method.

    :param img_array: Input image (RGB or Grayscale)
    :param method: 'yolo' (default) or 'surya'
    """
    if method == "eynollah" or method == "surya":
        return _segment_surya(img_array)
    else:
        return _segment_yolo(img_array)


# ---------------------------------------------------------------------------
# Surya — runs in a subprocess under tzefa-env so torch DLLs load correctly
# ---------------------------------------------------------------------------

_SURYA_SCRIPT = textwrap.dedent("""\
    import sys, json, os
    # Raise detection thresholds before importing surya so Settings picks them up
    os.environ["DETECTOR_TEXT_THRESHOLD"] = "0.75"
    os.environ["DETECTOR_BLANK_THRESHOLD"] = "0.45"
    import cv2
    from PIL import Image
    from surya.detection import DetectionPredictor

    CONF_THRESHOLD = 0.6   # discard boxes below this confidence

    # Read image path from argv
    img_path = sys.argv[1]
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(img_rgb)

    predictor = DetectionPredictor()
    predictions = predictor([pil_image])

    # Collect line boxes, filtering by confidence
    raw = []
    if predictions and predictions[0].bboxes:
        for bbox in predictions[0].bboxes:
            conf = getattr(bbox, 'confidence', 1.0)
            if conf < CONF_THRESHOLD:
                continue
            x1, y1, x2, y2 = bbox.bbox
            if (x2 - x1) > 5 and (y2 - y1) > 5:
                raw.append([float(x1), float(y1), float(x2), float(y2)])

    # Surya is line-level but occasionally splits one line into 2 boxes
    # when there is a gap within the line. Merge boxes whose y-ranges overlap.
    raw.sort(key=lambda b: (b[1] + b[3]) / 2)

    def overlaps_vertically(a, b):
        # True if the two boxes share any vertical extent
        return a[1] < b[3] and b[1] < a[3]

    merged = []
    for box in raw:
        placed = False
        for m in merged:
            if overlaps_vertically(m, box):
                m[0] = min(m[0], box[0])
                m[1] = min(m[1], box[1])
                m[2] = max(m[2], box[2])
                m[3] = max(m[3], box[3])
                placed = True
                break
        if not placed:
            merged.append(list(box))

    # Sort top-to-bottom, output as [x, y, w, h]
    merged.sort(key=lambda b: b[1])
    out = [[int(b[0]), int(b[1]), int(b[2]-b[0]), int(b[3]-b[1])] for b in merged]
    print(json.dumps(out))
""")


def _segment_surya(img_array):
    if not os.path.exists(TZEFA_PYTHON):
        raise RuntimeError(
            f"tzefa-env Python not found at {TZEFA_PYTHON}. "
            "Please verify the conda environment path."
        )

    print("      Line Seg: Using Surya (subprocess via tzefa-env)...")

    import tempfile, traceback as _tb

    # Save image to a temp file so the subprocess can read it
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_img_path = tmp.name

    try:
        if len(img_array.shape) == 2:
            bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        cv2.imwrite(tmp_img_path, bgr)

        result = subprocess.run(
            [TZEFA_PYTHON, "-c", _SURYA_SCRIPT, tmp_img_path],
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Surya subprocess failed (exit {result.returncode}):\n"
                f"STDERR: {result.stderr[-3000:]}\n"
                f"STDOUT: {result.stdout[-500:]}"
            )

        # Parse JSON list of [x, y, w, h] from stdout
        # Filter out non-JSON lines (TF/torch noise), find the last line that starts with '['
        json_line = None
        for line in reversed(result.stdout.strip().splitlines()):
            line = line.strip()
            if line.startswith("["):
                json_line = line
                break

        if not json_line:
            raise RuntimeError(
                f"Surya subprocess produced no JSON output.\n"
                f"STDOUT: {result.stdout[-1000:]}\n"
                f"STDERR: {result.stderr[-1000:]}"
            )

        lines = json.loads(json_line)
        lines = [tuple(b) for b in lines]

        print(f"      Found {len(lines)} lines with Surya")
        return lines

    except Exception as e:
        print(f"      Surya failed: {e}")
        raise e
    finally:
        if os.path.exists(tmp_img_path):
            os.remove(tmp_img_path)


# ---------------------------------------------------------------------------
# YOLO — runs in-process as before
# ---------------------------------------------------------------------------

def _segment_yolo(img_array):
    model = load_model()

    # Convert 1-channel Grayscale to 3-channel RGB for YOLO
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array

    orig_h, orig_w = img_rgb.shape[:2]
    print(f"      Line Seg: input {orig_w}x{orig_h}, letting YOLO letterbox to {INFERENCE_SIZE}")

    results = model.predict(img_rgb, imgsz=INFERENCE_SIZE, conf=0.05, iou=0.05, verbose=False)
    truelines = []
    if len(results) > 0 and results[0].obb is not None:
        obbs = results[0].obb.xyxyxyxy.cpu().numpy()
        obbs = sorted(obbs, key=lambda pts: np.min(pts[:, 1]))

        X_PAD_FRAC = 0.12

        for pts in obbs:
            raw_min_x = np.min(pts[:, 0])
            raw_max_x = np.max(pts[:, 0])
            raw_min_y = np.min(pts[:, 1])
            raw_max_y = np.max(pts[:, 1])

            raw_w = raw_max_x - raw_min_x
            pad = raw_w * X_PAD_FRAC

            min_x = int(np.clip(raw_min_x - pad, 0, orig_w))
            max_x = int(np.clip(raw_max_x + pad, 0, orig_w))
            min_y = int(np.clip(raw_min_y, 0, orig_h))
            max_y = int(np.clip(raw_max_y, 0, orig_h))

            w = max_x - min_x
            h = max_y - min_y
            if w > 0 and h > 0:
                truelines.append((min_x, min_y, w, h))

    print(f"      Found {len(truelines)} lines")
    return truelines

