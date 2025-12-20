import os
import argparse
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

# Fix for potential memory fragmentation on inference
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

# Defaults
DEFAULT_CHECKPOINT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization\run_1\segformer_step_4300.pth"
DEFAULT_IMAGE = r"E:\Storage\tests\test4.jpg"
BASE_MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"

def load_model(checkpoint_path):
    print(f"Loading model from {checkpoint_path}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Initialize Processor (Standard for all)
    processor = SegformerImageProcessor.from_pretrained(BASE_MODEL_NAME)
    processor.do_resize = False
    processor.do_rescale = True
    processor.do_normalize = True

    # 2. Load Model based on Format
    if os.path.isdir(checkpoint_path):
        # Case A: Hugging Face Directory (Old format)
        print(" -> Detected HuggingFace checkpoint folder.")
        try:
            model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_path)
        except Exception:
            # Fallback if config is missing but bin exists
            print(" -> Warning: Standard load failed, trying to force architecture.")
            model = SegformerForSemanticSegmentation.from_pretrained(
                BASE_MODEL_NAME,
                num_labels=1,
                ignore_mismatched_sizes=True,
                reshape_last_stage=True
            )
            # Try loading weights manually if bin exists? No, mostly from_pretrained handles it.
            pass

    elif os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pth"):
        # Case B: PyTorch .pth File (New format)
        print(" -> Detected PyTorch .pth checkpoint file.")

        # Initialize Architecture
        model = SegformerForSemanticSegmentation.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=1,
            ignore_mismatched_sizes=True,
            reshape_last_stage=True
        )

        # Load State Dict
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Handle "Full State" dict (model + optimizer) vs "Weights Only"
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            print(" -> Loading weights from full training state...")
            state_dict = checkpoint["model_state_dict"]
        else:
            print(" -> Loading raw weights dictionary...")
            state_dict = checkpoint

        # Load into model
        msg = model.load_state_dict(state_dict, strict=False)
        print(f" -> Weights Loaded. Missing keys (expected for head replacement): {len(msg.missing_keys)}")

    else:
        raise ValueError(f"Checkpoint path not recognized: {checkpoint_path}")

    model.to(device)
    model.eval()
    return model, processor, device

def infer_batch(model, processor, device, batch_tiles):
    """Runs inference on a batch of tiles."""
    if not batch_tiles:
        return []

    # Process batch
    inputs = processor(images=batch_tiles, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

        # Upsample to 640x640 (or whatever tile size was)
        # We assume all tiles in batch are same size
        target_size = batch_tiles[0].size[::-1]

        logits = torch.nn.functional.interpolate(
            logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )

        # Sigmoid for binary classification (Since we trained with BCE)
        # Output is [Batch, 1, H, W]
        # We want masks where > 0.5 is Text (1), < 0.5 is Background (0)
        # Training logic: Mask=1 (Text).
        # So we want prob > 0.5
        probs = torch.sigmoid(logits)
        pred_masks = (probs > 0.5).long().squeeze(1) # [Batch, H, W]

    return pred_masks

def binarize_image_tiled(model, processor, device, image_path, tile_size=640, batch_size=4):
    try:
        original_image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Could not open image {image_path}: {e}")
        return None, None

    orig_w, orig_h = original_image.size

    # Pad to tile size
    pad_w = (tile_size - (orig_w % tile_size)) % tile_size
    pad_h = (tile_size - (orig_h % tile_size)) % tile_size
    new_w = orig_w + pad_w
    new_h = orig_h + pad_h

    padded_image = Image.new("RGB", (new_w, new_h), (255, 255, 255))
    padded_image.paste(original_image, (0, 0))

    # Canvas for result (White background)
    result_canvas = Image.new("L", (new_w, new_h), 255)

    print(f"Processing {Path(image_path).name} ({orig_w}x{orig_h})...")

    batch_tiles = []
    batch_coords = []

    for y in range(0, new_h, tile_size):
        for x in range(0, new_w, tile_size):
            box = (x, y, x + tile_size, y + tile_size)
            tile = padded_image.crop(box)
            batch_tiles.append(tile)
            batch_coords.append((x, y))

            if len(batch_tiles) >= batch_size:
                preds = infer_batch(model, processor, device, batch_tiles)

                for i, pred_mask in enumerate(preds):
                    mask_np = pred_mask.cpu().numpy().astype(np.uint8)
                    # Training: 1=Text, 0=Bg
                    # Output Image: 0=Text(Black), 255=Bg(White)
                    # So if mask==1 (Text), we want 0. Else 255.
                    binary_tile = np.where(mask_np == 1, 0, 255).astype(np.uint8)

                    tile_img = Image.fromarray(binary_tile)
                    result_canvas.paste(tile_img, batch_coords[i])

                batch_tiles = []
                batch_coords = []

    # Final batch
    if batch_tiles:
        preds = infer_batch(model, processor, device, batch_tiles)
        for i, pred_mask in enumerate(preds):
            mask_np = pred_mask.cpu().numpy().astype(np.uint8)
            binary_tile = np.where(mask_np == 1, 0, 255).astype(np.uint8)
            tile_img = Image.fromarray(binary_tile)
            result_canvas.paste(tile_img, batch_coords[i])

    # Crop back
    final_result = result_canvas.crop((0, 0, orig_w, orig_h))
    return final_result, original_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--image", type=str, default=DEFAULT_IMAGE)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return

    model, processor, device = load_model(args.checkpoint)

    # Handle Directory or File
    input_path = Path(args.image)
    if input_path.is_dir():
        files = [f for f in input_path.iterdir() if f.suffix.lower() in ['.jpg', '.png', '.jpeg']]
    else:
        files = [input_path]

    for img_file in files:
        res, orig = binarize_image_tiled(model, processor, device, str(img_file), batch_size=args.batch_size)

        if res:
            out_path = img_file.parent / (img_file.stem + "_clean_v4.png")
            res.save(out_path)
            print(f"Saved: {out_path}")

            # Display if single file
            if len(files) == 1:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(orig)
                plt.title("Original")
                plt.axis("off")

                plt.subplot(1, 2, 2)
                plt.imshow(res, cmap='gray')
                plt.title("Cleaned (SegFormer B2)")
                plt.axis("off")
                plt.tight_layout()
                plt.show()

if __name__ == "__main__":
    main()