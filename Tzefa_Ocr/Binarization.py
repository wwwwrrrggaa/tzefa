import os
from pathlib import Path

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# Constants
DEFAULT_CHECKPOINT_PATH = Path(r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization\run_mitb5_highres_manet\step_3200.pth")

TILE_SIZE = 640
BATCH_SIZE = 16

# --- 1. MODEL ARCHITECTURE (Strictly mit_b5 MAnet) ---
class HighResMAnet(nn.Module):
    def __init__(self, encoder_name="mit_b5", classes=1):
        super().__init__()
        self.base_model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=classes,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )

        # Stage 0 High-Res Stem
        self.high_res_stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_fusion = nn.Sequential(
            nn.Conv2d(16 + 32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, classes, kernel_size=1),
        )

    def forward(self, x):
        high_res_features = self.high_res_stem(x)
        features = self.base_model.encoder(x)
        decoder_output = self.base_model.decoder(features)
        combined = torch.cat([decoder_output, high_res_features], dim=1)
        logits = self.final_fusion(combined)
        return logits


# --- 2. INFERENCE CLASS ---
class BinarizationModel:
    def __init__(self, checkpoint_path=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.checkpoint = checkpoint_path if checkpoint_path else str(DEFAULT_CHECKPOINT_PATH)

        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"Model checkpoint not found at: {self.checkpoint}")

        print(f"Loading HighResMAnet (mit_b5) Binarization Model from: {self.checkpoint} on {self.device}")

        try:
            # Init model specifically as mit_b5
            self.model = HighResMAnet(encoder_name="mit_b5", classes=1)

            # Load weights
            checkpoint_data = torch.load(self.checkpoint, map_location=self.device)
            if "model_state_dict" in checkpoint_data:
                self.model.load_state_dict(checkpoint_data["model_state_dict"])
            else:
                self.model.load_state_dict(checkpoint_data)

            self.model.to(self.device)
            self.model.eval()

            # Optional: Channels last optimization for inference speed
            self.model = self.model.to(memory_format=torch.channels_last)

            # Preprocessing transform (ImageNet normalization expected by mit_b5)
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

        except Exception as e:
            print(f"CRITICAL ERROR loading Binarization Model: {e}")
            raise

    def _infer_batch(self, batch_tiles_np):
        if not batch_tiles_np:
            return []

        tensor_list = [self.transform(image=tile)["image"] for tile in batch_tiles_np]
        batch_tensor = torch.stack(tensor_list).to(self.device, memory_format=torch.channels_last)

        with torch.no_grad():
            with torch.amp.autocast(self.device, enabled=(self.device == "cuda")):
                logits = self.model(batch_tensor)
                probs = torch.sigmoid(logits)
                probs_np = probs.squeeze(1).cpu().numpy()

        return probs_np

    def clean(self, image_input) -> np.ndarray:
        if isinstance(image_input, (str, Path)):
            image = Image.open(image_input).convert("RGB")
            img_np = np.array(image)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img_np = image_input
            else:
                img_np = np.array(Image.fromarray(image_input).convert("RGB"))
        elif isinstance(image_input, Image.Image):
            img_np = np.array(image_input.convert("RGB"))
        else:
            raise ValueError("Unsupported image format. Use path, PIL Image, or Numpy Array.")

        orig_h, orig_w = img_np.shape[:2]

        pad_w = (TILE_SIZE - (orig_w % TILE_SIZE)) % TILE_SIZE
        pad_h = (TILE_SIZE - (orig_h % TILE_SIZE)) % TILE_SIZE

        if pad_w > 0 or pad_h > 0:
            img_np = cv2.copyMakeBorder(img_np, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=[255, 255, 255])

        new_h, new_w = img_np.shape[:2]

        result_canvas = np.full((new_h, new_w), 255, dtype=np.uint8)
        batch_tiles = []
        batch_coords = []

        for y in range(0, new_h, TILE_SIZE):
            for x in range(0, new_w, TILE_SIZE):
                tile = img_np[y:y+TILE_SIZE, x:x+TILE_SIZE]
                batch_tiles.append(tile)
                batch_coords.append((x, y))

                if len(batch_tiles) >= BATCH_SIZE:
                    pred_probs = self._infer_batch(batch_tiles)
                    self._paste_results(result_canvas, pred_probs, batch_coords)
                    batch_tiles, batch_coords = [], []

        if batch_tiles:
            pred_probs = self._infer_batch(batch_tiles)
            self._paste_results(result_canvas, pred_probs, batch_coords)

        final_result = result_canvas[0:orig_h, 0:orig_w]
        return final_result

    def _paste_results(self, canvas, pred_probs, coords):
        for i, prob in enumerate(pred_probs):
            x, y = coords[i]
            # Thresholding: prob > 0.5 means it's Ink. 0 for Ink, 255 for Paper.
            binary_tile_np = np.where(prob > 0.5, 0, 255).astype(np.uint8)
            canvas[y:y+TILE_SIZE, x:x+TILE_SIZE] = binary_tile_np

if __name__ == "__main__":
    model = BinarizationModel()
    print("HighResMAnet Model initialized successfully.")