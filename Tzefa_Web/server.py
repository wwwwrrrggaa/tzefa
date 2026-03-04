"""
Tzefa OCR - Self-hosted Web Interface
Runs the full Tzefa pipeline and displays intermediate results with toggle views.
"""
import sys
import os
import io
import base64
import traceback
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for

# --- Add project paths ---
current_web_dir = Path(__file__).resolve().parent          # Tzefa_Web/
project_root = current_web_dir.parent                       # tzefa/
ocr_dir = project_root / "Tzefa_Ocr"

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
if str(ocr_dir) not in sys.path:
    sys.path.insert(0, str(ocr_dir))

from Tzefa_Ocr.main import image_to_code_pipeline

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = str(current_web_dir / "uploads")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB limit

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def numpy_to_b64_png(img_array: np.ndarray) -> str:
    """Convert a numpy image array to a base64-encoded PNG data URI."""
    if img_array is None:
        return ""
    if len(img_array.shape) == 2:
        pil = Image.fromarray(img_array, mode="L")
    else:
        pil = Image.fromarray(img_array)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def draw_bboxes_on_image(img_array: np.ndarray, bboxes) -> np.ndarray:
    """Draw line bounding boxes on a copy of the image. Returns RGB image."""
    if img_array is None or bboxes is None:
        return img_array

    # Convert grayscale to RGB for colored bboxes
    if len(img_array.shape) == 2:
        vis = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        vis = img_array.copy()

    for i, (x, y, w, h) in enumerate(bboxes):
        color = (255, 50, 50)  # Red
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
        cv2.putText(vis, str(i + 1), (x, max(y - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 255), 2)
    return vis


def draw_word_bboxes_on_image(img_array: np.ndarray, bboxes, word_bboxes) -> np.ndarray:
    """Draw word bboxes (green/blue/orange) with OCR text labels — no line bboxes."""
    if len(img_array.shape) == 2:
        vis = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        vis = img_array.copy()
    if word_bboxes is None:
        return vis

    colors = [(50, 220, 50), (50, 180, 255), (255, 180, 50)]  # green / blue / orange per word slot
    for line_tuples in word_bboxes:
        for w_idx, (text, (x1, y1, x2, y2)) in enumerate(line_tuples):
            color = colors[w_idx % len(colors)]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            # Label: show the OCR'd text above the box
            cv2.putText(vis, text, (x1, max(y1 - 4, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    return vis


@app.route("/", methods=["GET"])
def index():
    return render_template("pipeline.html", result=None)


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return redirect(url_for("index"))

    file = request.files["image"]
    if file.filename == "":
        return redirect(url_for("index"))

    try:
        # Read image directly from upload into numpy
        file_bytes = file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img_bgr is None:
            return render_template("pipeline.html", result={"error": "Could not decode image."})
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Run pipeline
        pipeline = image_to_code_pipeline(img_rgb)

        # Build display data
        display = {
            "stage": pipeline["stage"],
            "error": pipeline["error"],
            "original_img": numpy_to_b64_png(img_rgb),
            "binarized_img": "",
            "binarized_bbox_img": "",
            "word_bbox_img": "",
            "raw_ocr": "",
            "corrected": "",
            "compiled_code": "",
            "execution_output": "",
        }

        if pipeline["binarized_img"] is not None:
            display["binarized_img"] = numpy_to_b64_png(pipeline["binarized_img"])

        if pipeline["binarized_img"] is not None and pipeline["truelines"] is not None:
            bbox_vis = draw_bboxes_on_image(pipeline["binarized_img"], pipeline["truelines"])
            display["binarized_bbox_img"] = numpy_to_b64_png(bbox_vis)

        if pipeline["binarized_img"] is not None and pipeline["truelines"] is not None and pipeline["word_bboxes"] is not None:
            word_vis = draw_word_bboxes_on_image(pipeline["binarized_img"], pipeline["truelines"], pipeline["word_bboxes"])
            display["word_bbox_img"] = numpy_to_b64_png(word_vis)

        if pipeline["raw_ocr_lines"] is not None:
            display["raw_ocr"] = "\n".join(pipeline["raw_ocr_lines"])

        if pipeline["corrected_lines"] is not None:
            display["corrected"] = "\n".join(pipeline["corrected_lines"])

        if pipeline["compiled_code"] is not None:
            display["compiled_code"] = pipeline["compiled_code"]

        if pipeline["execution_output"] is not None:
            display["execution_output"] = pipeline["execution_output"]

        return render_template("pipeline.html", result=display)

    except Exception as e:
        tb = traceback.format_exc()
        return render_template("pipeline.html", result={
            "stage": "crash",
            "error": f"Server error: {e}\n{tb}",
            "original_img": "",
            "binarized_img": "",
            "binarized_bbox_img": "",
            "word_bbox_img": "",
            "raw_ocr": "",
            "corrected": "",
            "compiled_code": "",
            "execution_output": "",
        })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

