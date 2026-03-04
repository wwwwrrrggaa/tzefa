import cv2
import numpy as np
from ultralytics import YOLO

# Point directly to your newly trained XL model weights
MODEL_PATH = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Line_Segmentation\yolo11x-obb-fresh\weights\best.pt"
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
    print(f"      Line Seg: input {orig_w}x{orig_h}, letting YOLO letterbox to {INFERENCE_SIZE}")

    # Pass the original image directly — YOLO letterboxes it internally while
    # preserving aspect ratio, and returns xyxyxyxy already scaled back to
    # original pixel coordinates. No manual rescaling needed.
    # Bug fix: previously the image was manually squashed to 640x640 (distorting
    # aspect ratio), then coords were scaled back with wrong scale_x, which is
    # why a 50% X-pad hack was needed. Removed both.
    results = model.predict(img_rgb, imgsz=INFERENCE_SIZE, conf=0.2, iou=0.45, verbose=False)
    truelines = []
    if len(results) > 0 and results[0].obb is not None:
        # xyxyxyxy coords are already in original image space when the original
        # image (not a pre-resized copy) is passed to predict().
        obbs = results[0].obb.xyxyxyxy.cpu().numpy()

        # Sort lines from top to bottom
        obbs = sorted(obbs, key=lambda pts: np.min(pts[:, 1]))

        for pts in obbs:
            min_x = int(np.clip(np.min(pts[:, 0]), 0, orig_w))
            max_x = int(np.clip(np.max(pts[:, 0]), 0, orig_w))
            min_y = int(np.clip(np.min(pts[:, 1]), 0, orig_h))
            max_y = int(np.clip(np.max(pts[:, 1]), 0, orig_h))

            w = max_x - min_x
            h = max_y - min_y
            if w > 0 and h > 0:
                truelines.append((min_x, min_y, w, h))

    print(f"      Found {len(truelines)} lines")
    return truelines