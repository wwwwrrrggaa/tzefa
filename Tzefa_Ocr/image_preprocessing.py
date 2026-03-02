import os
import cv2


def UV_unwrap(image_path):
    """Loads image from path as RGB numpy array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} does not exist")
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


TARGET_WORDS = 3
MAX_ITERATIONS = 200


def _get_word_boxes(dilated, min_word_w, min_word_h):
    """Find filtered, sorted bounding boxes from a dilated image."""
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours]
    boxes = [b for b in boxes if b[2] >= min_word_w and b[3] >= min_word_h]
    boxes.sort(key=lambda b: b[0])
    return boxes


def linestowords(binarized_img, lines_bboxes):
    """
    Small fixed kernel, repeatedly applied to the SAME image until
    exactly 3 connected components remain.
    A Tzefa instruction is ALWAYS 3 tokens (COMMAND ARG1 ARG2).
    """
    print("      Segmenting lines into words (Repeated Dilation)...")
    words_dict = {}

    for i, line_box in enumerate(lines_bboxes):
        lx, ly, lw, lh = line_box

        # Safety check for image bounds
        img_h, img_w = binarized_img.shape[:2]
        ly = max(0, ly)
        lx = max(0, lx)
        lh = min(lh, img_h - ly)
        lw = min(lw, img_w - lx)

        if lw <= 0 or lh <= 0:
            continue

        # Crop from full resolution binary
        line_crop = binarized_img[ly : ly + lh, lx : lx + lw]

        # Invert (Text=White) for contour detection
        inverted = cv2.bitwise_not(line_crop)

        # Noise filter thresholds
        min_word_w = max(3, int(lw * 0.01))
        min_word_h = max(3, int(lh * 0.1))

        # Small fixed kernel — just keep applying it to the same image
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))

        dilated = inverted.copy()
        prev_boxes = None
        found = False

        for iteration in range(MAX_ITERATIONS):
            dilated = cv2.dilate(dilated, kernel, iterations=1)
            boxes = _get_word_boxes(dilated, min_word_w, min_word_h)

            if len(boxes) == TARGET_WORDS:
                prev_boxes = boxes
                found = True
                break
            elif len(boxes) < TARGET_WORDS:
                # Overshot — use prev_boxes (last time we had >3)
                break
            else:
                prev_boxes = boxes

        if not found and prev_boxes is not None and len(prev_boxes) > TARGET_WORDS:
            # Merge closest pairs down to 3
            while len(prev_boxes) > TARGET_WORDS:
                min_gap = float('inf')
                merge_idx = 0
                for j in range(len(prev_boxes) - 1):
                    gap = prev_boxes[j + 1][0] - (prev_boxes[j][0] + prev_boxes[j][2])
                    if gap < min_gap:
                        min_gap = gap
                        merge_idx = j
                b1, b2 = prev_boxes[merge_idx], prev_boxes[merge_idx + 1]
                merged = (
                    min(b1[0], b2[0]),
                    min(b1[1], b2[1]),
                    max(b1[0] + b1[2], b2[0] + b2[2]) - min(b1[0], b2[0]),
                    max(b1[1] + b1[3], b2[1] + b2[3]) - min(b1[1], b2[1]),
                )
                prev_boxes = list(prev_boxes)
                prev_boxes[merge_idx] = merged
                prev_boxes.pop(merge_idx + 1)
            found = True

        if not found or prev_boxes is None or len(prev_boxes) != TARGET_WORDS:
            print(f"      Line {i+1}: WARNING — got {len(prev_boxes) if prev_boxes else 0} words (expected {TARGET_WORDS}), skipping")
            continue

        print(f"      Line {i+1}: OK — {len(prev_boxes)} words")

        line_words = {}
        for w_idx, (wx, wy, ww, wh) in enumerate(prev_boxes):
            line_words[w_idx + 1] = (wx, wx + ww)

        words_dict[i + 1] = line_words

    return words_dict
