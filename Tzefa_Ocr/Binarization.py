import os
from pathlib import Path
import torch
import numpy as np
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Constants
DEFAULT_CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "Tzefa_Models" / "Binarization" / "run_2"
# Fallback to the specific checkpoint we know works
DEFAULT_CHECKPOINT_PATH = DEFAULT_CHECKPOINT_DIR / "checkpoint-5050"

# Model Configuration
TILE_SIZE = 640
BATCH_SIZE = 16

class BinarizationModel:
    def __init__(self, checkpoint_path=None):
        """
        Initializes the SegFormer Binarization model.
        checkpoint_path: Path to the model checkpoint. If None, tries to find the best one automatically.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = checkpoint_path

        if self.checkpoint is None:
            self.checkpoint = self._find_best_checkpoint()

        print(f"Loading Binarization Model from: {self.checkpoint} on {self.device}")

        try:
            self.model = SegformerForSemanticSegmentation.from_pretrained(self.checkpoint)
            # Load processor from base model config to ensure consistency
            self.processor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512")
            self.processor.do_resize = False
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"CRITICAL ERROR loading Binarization Model: {e}")
            raise

    def _find_best_checkpoint(self):
        """Helper to find the latest checkpoint if path not provided."""
        if os.path.exists(DEFAULT_CHECKPOINT_PATH):
            return str(DEFAULT_CHECKPOINT_PATH)

        # Fallback search logic
        if not os.path.exists(DEFAULT_CHECKPOINT_DIR):
            raise FileNotFoundError(f"Model directory not found: {DEFAULT_CHECKPOINT_DIR}")

        checkpoints = list(DEFAULT_CHECKPOINT_DIR.glob("checkpoint-*"))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found in {DEFAULT_CHECKPOINT_DIR}")

        # Sort by number and take the last one
        checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
        return str(checkpoints[-1])

    def _infer_batch(self, batch_tiles):
        """Internal method to run GPU inference on a batch."""
        if not batch_tiles:
            return []

        inputs = self.processor(images=batch_tiles, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            # Upsample
            target_size = batch_tiles[0].size[::-1]
            logits = torch.nn.functional.interpolate(
                logits,
                size=target_size,
                mode="bilinear",
                align_corners=False,
            )

            pred_segs = logits.argmax(dim=1)

        return pred_segs

    def clean(self, image_input) -> np.ndarray:
        """
        Main Inference Function.
        Args:
            image_input: str (filepath) or PIL.Image or numpy array (OpenCV).
        Returns:
            numpy.ndarray: The binarized image (Grayscale, 0=Ink, 255=Paper).
        """
        # 1. Standardize Input to PIL
        if isinstance(image_input, str) or isinstance(image_input, Path):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            image = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image format. Use path, PIL Image, or Numpy Array.")

        orig_w, orig_h = image.size

        # 2. Pad image
        pad_w = (TILE_SIZE - (orig_w % TILE_SIZE)) % TILE_SIZE
        pad_h = (TILE_SIZE - (orig_h % TILE_SIZE)) % TILE_SIZE
        new_w, new_h = orig_w + pad_w, orig_h + pad_h

        padded_image = Image.new("RGB", (new_w, new_h), (255, 255, 255))
        padded_image.paste(image, (0, 0))

        # 3. Process Tiles
        result_canvas = Image.new("L", (new_w, new_h), 255)
        batch_tiles = []
        batch_coords = []

        for y in range(0, new_h, TILE_SIZE):
            for x in range(0, new_w, TILE_SIZE):
                box = (x, y, x + TILE_SIZE, y + TILE_SIZE)
                tile = padded_image.crop(box)
                batch_tiles.append(tile)
                batch_coords.append((x, y))

                if len(batch_tiles) >= BATCH_SIZE:
                    pred_segs = self._infer_batch(batch_tiles)
                    self._paste_results(result_canvas, pred_segs, batch_coords)
                    batch_tiles, batch_coords = [], []

        # Process remaining
        if batch_tiles:
            pred_segs = self._infer_batch(batch_tiles)
            self._paste_results(result_canvas, pred_segs, batch_coords)

        # 4. Crop & Return as Numpy (for OpenCV compatibility)
        final_result = result_canvas.crop((0, 0, orig_w, orig_h))
        return np.array(final_result)

    def _paste_results(self, canvas, pred_segs, coords):
        """Internal helper to paste GPU results onto canvas."""
        for i, pred_seg in enumerate(pred_segs):
            mask_np = pred_seg.cpu().numpy().astype(np.uint8)
            # Logic: Class 1 = Background (White 255), Class 0 = Ink (Black 0)
            binary_tile_np = np.where(mask_np == 1, 255, 0).astype(np.uint8)
            tile_img = Image.fromarray(binary_tile_np)
            canvas.paste(tile_img, coords[i])

# For quick testing
if __name__ == "__main__":
    model = BinarizationModel()
    print("Model initialized successfully.")