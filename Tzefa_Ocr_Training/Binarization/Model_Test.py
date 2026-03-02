import os
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn.functional as F

# --- CONFIGURATION ---
TILE_SIZE = 640
# Updated Encoder and Architecture
ENCODER_NAME = "mit_b3"
MODEL_ARCH = "Unet"

# Updated Default Paths
DEFAULT_CHECKPOINT = (
    r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization\run_mitb3_focal_tversky_clean\step_690.pth")
DEFAULT_IMAGE = r"E:\Storage\tests\test97.jpg"


def load_model(checkpoint_path):
    print(f"--- Loading {ENCODER_NAME}-{MODEL_ARCH} from {checkpoint_path} ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize SMP Architecture (UnetPlusPlus)
    # Note: We use UnetPlusPlus here as requested
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights=None,  # Loading our own weights
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    )

    # 2. Smart Weight Loading
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Check if it's "Full State" (dict with keys) or "Raw Weights"
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        print(" -> Detected full training state. Extracting weights...")
        state_dict = checkpoint["model_state_dict"]
    else:
        print(" -> Detected raw weights dictionary.")
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Apply channels_last for potential speed boost
    model = model.to(memory_format=torch.channels_last)

    return model, device


def preprocess(pil_img):
    """Manually apply the same normalization used in training."""
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    # ImageNet normalization (Standard for EfficientNet as well)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    # HWC -> CHW
    img_np = img_np.transpose(2, 0, 1)
    return torch.from_numpy(img_np)


def infer_batch(model, device, batch_tiles):
    if not batch_tiles:
        return []

    # Prepare batch tensor
    tensors = [preprocess(tile) for tile in batch_tiles]
    input_tensor = torch.stack(tensors).to(device, memory_format=torch.channels_last).float()

    with torch.no_grad():
        # AMP for speed/memory efficiency
        with torch.amp.autocast("cuda"):
            logits = model(input_tensor)

            # Ensure output matches TILE_SIZE (Unet++ sometimes has slight padding diffs)
            if logits.shape[-2:] != (TILE_SIZE, TILE_SIZE):
                logits = F.interpolate(logits, size=(TILE_SIZE, TILE_SIZE), mode="bilinear", align_corners=False)

            probs = torch.sigmoid(logits)
            # Binary threshold: > 0.5 is Text
            pred_masks = (probs > 0.5).float().squeeze(1)  # [Batch, H, W]

    return pred_masks.cpu().numpy()


def binarize_image_tiled(model, device, image_path, batch_size=4):
    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return None, None

    orig_w, orig_h = original_image.size

    # Pad image to be divisible by TILE_SIZE
    pad_w = (TILE_SIZE - (orig_w % TILE_SIZE)) % TILE_SIZE
    pad_h = (TILE_SIZE - (orig_h % TILE_SIZE)) % TILE_SIZE
    new_w = orig_w + pad_w
    new_h = orig_h + pad_h

    padded_image = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded_image.paste(original_image, (0, 0))

    result_canvas = Image.new("L", (new_w, new_h), 255)
    print(f"Processing: {Path(image_path).name}")

    tiles = []
    coords = []

    # Extract tiles
    for y in range(0, new_h, TILE_SIZE):
        for x in range(0, new_w, TILE_SIZE):
            box = (x, y, x + TILE_SIZE, y + TILE_SIZE)
            tiles.append(padded_image.crop(box))
            coords.append((x, y))

    # Batch processing
    for i in range(0, len(tiles), batch_size):
        batch_tiles = tiles[i : i + batch_size]
        batch_coords = coords[i : i + batch_size]

        preds = infer_batch(model, device, batch_tiles)

        for j, pred_mask in enumerate(preds):
            # 1 = Text (Black/0), 0 = Background (White/255)
            # Inverting mask so text is black
            binary_tile = ((1.0 - pred_mask) * 255).astype(np.uint8)
            tile_img = Image.fromarray(binary_tile)
            result_canvas.paste(tile_img, batch_coords[j])

    final_result = result_canvas.crop((0, 0, orig_w, orig_h))
    return final_result, original_image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    model, device = load_model(args.checkpoint)

    input_path = Path(args.image)
    files = [input_path] if not input_path.is_dir() else list(input_path.glob("*.[jp][pn]g"))

    for img_file in files:
        res, orig = binarize_image_tiled(model, device, str(img_file), batch_size=args.batch_size)

        if res:
            out_path = img_file.parent / (img_file.stem + "_binarized_effb5.png")
            res.save(out_path)
            print(f"--- Saved to: {out_path.name} ---")

            if len(files) == 1:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(orig)
                plt.title("Input")
                plt.axis("off")
                plt.subplot(1, 2, 2)
                plt.imshow(res, cmap="gray")
                plt.title(f"Binarized ({ENCODER_NAME})")
                plt.axis("off")
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    main()