import cv2
import numpy as np
from ultralytics import YOLO

# Point directly to your newly trained XL model weights
MODEL_PATH = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Line_Segmentation\run_fresh_XL_speed\weights\best.pt"
INFERENCE_SIZE = 640

def load_model():
    return YOLO(MODEL_PATH)

def segment_lines(img_array):
    model = load_model()

    # Convert 1-channel Grayscale to 3-channel RGB for YOLO
    if len(img_array.shape) == 2:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif len(img_array.shape) == 3 and img_array.shape[2] == 1:
        img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = img_array

    orig_h, orig_w = img_rgb.shape[:2]

    # Force resize to 640x640 for inference.
    # This gives accurate Y-axis detection (height and vertical position).
    resized = cv2.resize(img_rgb, (INFERENCE_SIZE, INFERENCE_SIZE))
    scale_x = orig_w / INFERENCE_SIZE
    scale_y = orig_h / INFERENCE_SIZE

    print(f"      Line Seg: {orig_w}x{orig_h} -> {INFERENCE_SIZE}x{INFERENCE_SIZE}, scale_x={scale_x:.3f} scale_y={scale_y:.3f}")

    results = model.predict(resized, imgsz=INFERENCE_SIZE, conf=0.25, verbose=False)

    truelines = []
    if len(results) > 0 and results[0].obb is not None:
        obbs = results[0].obb.xyxyxyxy.cpu().numpy()

        # Sort lines from top to bottom
        obbs = sorted(obbs, key=lambda pts: np.min(pts[:, 1]))

        for pts in obbs:
            # Scale both axes from 640 space back to original resolution
            min_x = int(np.min(pts[:, 0]) * scale_x)
            max_x = int(np.max(pts[:, 0]) * scale_x)
            min_y = int(np.min(pts[:, 1]) * scale_y)
            max_y = int(np.max(pts[:, 1]) * scale_y)

            # The image was squashed to 640x640, so YOLO's X detection is
            # tighter than the actual line width. Pad X by 50% on each side.
            w = max_x - min_x
            pad_x = int(w * 0.5)
            min_x = max(0, min_x - pad_x)
            max_x = min(orig_w, max_x + pad_x)

            min_y = max(0, min_y)
            max_y = min(orig_h, max_y)

            w = max_x - min_x
            h = max_y - min_y
            if w > 0 and h > 0:
                truelines.append((min_x, min_y, w, h))

    print(f"      Found {len(truelines)} lines")
    return truelines