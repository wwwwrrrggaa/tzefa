from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import torch

# --- CONFIG ---
WORD_MODEL_PATH = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Word_Model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- LAZY-LOADED GLOBALS ---
_processor = None
_model_words = None


def _load_word_model():
    """Load the fine-tuned word OCR model on first use (not on import)."""
    global _processor, _model_words
    if _model_words is not None:
        return

    print(f"Loading Word OCR Model from {WORD_MODEL_PATH} on {DEVICE}...")
    # use_fast=False is CRITICAL — prevents tokenizer conversion crash
    _processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1", use_fast=False)
    _model_words = VisionEncoderDecoderModel.from_pretrained(WORD_MODEL_PATH).to(DEVICE)
    _model_words.eval()
    print("Word OCR Model loaded.")


def _pad_to_aspect_ratio(image, max_ratio=4.0):
    """
    If the image is wider than max_ratio * height, pad top and bottom with
    white so the aspect ratio doesn't exceed max_ratio.  This prevents TrOCR
    from receiving an extreme landscape crop (e.g. 1195×94) that collapses to
    an unreadable strip when rescaled to the model's 384×384 input.
    """
    w, h = image.size
    if w <= max_ratio * h:
        return image  # already fine

    target_h = int(w / max_ratio)
    pad_total = target_h - h
    pad_top = pad_total // 2
    pad_bottom = pad_total - pad_top

    from PIL import ImageOps
    return ImageOps.expand(image, border=(0, pad_top, 0, pad_bottom), fill=(255, 255, 255))


def ocr_word(image):
    """Run OCR on a single word crop. Returns recognized text string."""
    _load_word_model()

    if image.mode != "RGB":
        image = image.convert("RGB")

    # Guard against extreme aspect ratios that confuse TrOCR's 384×384 resize
    image = _pad_to_aspect_ratio(image, max_ratio=4.0)

    pixel_values = _processor(image, return_tensors="pt").pixel_values.to(DEVICE)

    with torch.no_grad():
        generated_ids = _model_words.generate(pixel_values)

    generated_text = _processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


if __name__ == "__main__":
    import os
    test_img_path = r"C:\Users\yonat\Downloads\image.png"
    if os.path.exists(test_img_path):
        print(f"Testing OCR on: {test_img_path}")
        print(f"Result: {ocr_word(Image.open(test_img_path))}")
    else:
        print(f"Test image not found at {test_img_path}")