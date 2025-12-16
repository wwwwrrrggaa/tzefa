import os
# --- Configuration & Tuning ---
# MUST be set before importing torch to take effect
# "expandable_segments" is NVIDIA-only. For AMD/HIP, we use max_split_size_mb to prevent fragmentation.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import argparse
from pathlib import Path
import numpy as np
import torch
import random
import glob
import re
from PIL import Image, ImageFilter, ImageOps, ImageEnhance
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from transformers import (
    SegformerForSemanticSegmentation,
    SegformerImageProcessor,
    TrainingArguments,
    Trainer
)

TARGET_SIZE = 640  # Increased to 640 for maximum detail (B2 fits this easily)

class AugmentationUtils:
    """Helper class for synchronizing image/mask transformations."""

    @staticmethod
    def add_noise(image, intensity=0.05):
        """Adds random Gaussian noise to the image."""
        if random.random() > 0.5:
            return image

        img_arr = np.array(image)
        noise = np.random.normal(0, intensity * 255, img_arr.shape).astype(np.int16)
        img_arr = img_arr.astype(np.int16) + noise
        img_arr = np.clip(img_arr, 0, 255).astype(np.uint8)
        return Image.fromarray(img_arr)

    @staticmethod
    def add_blur(image):
        """Adds blur to simulate out-of-focus scans."""
        if random.random() > 0.7:
            # Randomly choose between BoxBlur and GaussianBlur
            if random.random() > 0.5:
                return image.filter(ImageFilter.BoxBlur(random.uniform(0.5, 1.5)))
            else:
                return image.filter(ImageFilter.GaussianBlur(random.uniform(0.5, 1.5)))
        return image

    @staticmethod
    def adjust_color(image):
        """Randomly changes brightness and contrast."""
        if random.random() > 0.5:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
        if random.random() > 0.5:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(random.uniform(0.7, 1.3))
        return image

    @staticmethod
    def letterbox_pad(image, mask, target_size):
        """
        Resizes image to fit within target_size while maintaining aspect ratio,
        then pads the rest with black (0). used for Batch_1.
        """
        w, h = image.size
        scale = min(target_size / w, target_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        image = image.resize((new_w, new_h), Image.BILINEAR)
        mask = mask.resize((new_w, new_h), Image.NEAREST)

        # Create new canvas
        new_img = Image.new("RGB", (target_size, target_size), (0, 0, 0))
        new_mask = Image.new("L", (target_size, target_size), 0)

        # Paste in center
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2

        new_img.paste(image, (paste_x, paste_y))
        new_mask.paste(mask, (paste_x, paste_y))

        return new_img, new_mask

    @staticmethod
    def random_crop(image, mask, target_size):
        """
        Randomly crops a target_size x target_size patch.
        If image is smaller than target, it pads it first.
        Used for Batch_2+.
        """
        w, h = image.size

        # If image is smaller than target, pad it first
        if w < target_size or h < target_size:
            return AugmentationUtils.letterbox_pad(image, mask, target_size)

        # Random crop coordinates
        x = random.randint(0, w - target_size)
        y = random.randint(0, h - target_size)

        crop_box = (x, y, x + target_size, y + target_size)
        image = image.crop(crop_box)
        mask = mask.crop(crop_box)

        return image, mask

class BinarizationDataset(Dataset):
    """
    Advanced Dataset capable of handling different processing logic
    based on the 'batch_name' of the data.
    """
    def __init__(self, data_entries, processor, img_size=640, is_training=True):
        self.data_entries = data_entries # List of dicts: {path, mask_path, batch_name}
        self.processor = processor
        self.img_size = img_size
        self.is_training = is_training

    def __len__(self):
        return len(self.data_entries)

    def __getitem__(self, idx):
        entry = self.data_entries[idx]

        # 1. Load Images
        try:
            image = Image.open(entry['path']).convert("RGB")
            mask = Image.open(entry['mask_path']).convert("L")
        except Exception as e:
            # Fallback for corrupted images to avoid crashing training
            print(f"Error loading {entry['path']}: {e}")
            # return a dummy black image
            image = Image.new("RGB", (self.img_size, self.img_size))
            mask = Image.new("L", (self.img_size, self.img_size))

        batch_name = entry['batch_name']

        # 2. Batch-Specific Logic
        # "Batch_1" is old, small data -> Pad it.
        # "Batch_2" (and future) is new, large data -> Random Crop it.
        if "Batch_1" in batch_name:
            image, mask = AugmentationUtils.letterbox_pad(image, mask, self.img_size)
        else:
            # Logic for Batch_2 and future batches
            if self.is_training:
                image, mask = AugmentationUtils.random_crop(image, mask, self.img_size)
            else:
                # Validation logic for large images: Center Crop or Resize
                # We'll use resize for validation to see full context
                image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
                mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        # 3. General Augmentations (Only during training)
        if self.is_training:
            image = AugmentationUtils.add_blur(image)
            image = AugmentationUtils.add_noise(image)
            image = AugmentationUtils.adjust_color(image)

        # 4. Binarize Mask (0 or 1)
        mask_arr = np.array(mask)
        mask_arr = (mask_arr > 127).astype(np.int64)

        # 5. Processor handles normalization
        encoded = self.processor(
            images=image,
            segmentation_maps=mask_arr,
            return_tensors="pt",
            do_resize=False # We handled resizing manually above
        )

        return {
            "pixel_values": encoded.pixel_values.squeeze(),
            "labels": encoded.labels.squeeze()
        }

class BalancedTrainer(Trainer):
    """
    Custom Trainer to enforce 50/50 (or 1/n) sampling between batches
    regardless of dataset size differences.
    """
    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset

        # Calculate weights for 1/n sampling strategy
        # 1. Count items per batch
        batch_counts = {}
        batch_indices = {}

        for idx, entry in enumerate(train_dataset.data_entries):
            b_name = entry['batch_name']
            batch_counts[b_name] = batch_counts.get(b_name, 0) + 1
            if b_name not in batch_indices: batch_indices[b_name] = []
            batch_indices[b_name].append(idx)

        n_batches = len(batch_counts)
        total_samples = len(train_dataset)

        print(f"\n--- Data Distribution: {batch_counts} ---")

        # 2. Assign weight to each sample
        # Weight = 1 / (Number of Batches * Number of samples in this batch)
        # This ensures the sum of weights for Batch A equals sum of weights for Batch B
        weights = np.zeros(total_samples)

        for b_name, count in batch_counts.items():
            weight_per_sample = 1.0 / (count * n_batches)
            for idx in batch_indices[b_name]:
                weights[idx] = weight_per_sample

        weights = torch.DoubleTensor(weights)

        # 3. Create WeightedRandomSampler
        sampler = WeightedRandomSampler(weights, len(weights))

        return DataLoader(
            train_dataset,
            batch_size=self.args.train_batch_size,
            sampler=sampler,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            persistent_workers=True if self.args.dataloader_num_workers > 0 else False,
        )

def scan_dataset(root_dir):
    """
    Recursively finds 'images' and 'masks' folders inside Batch folders.
    Returns a list of dicts.
    """
    root = Path(root_dir)
    data_entries = []

    # Find all batch directories (Batch_1, Batch_2, etc.)
    batch_dirs = [d for d in root.iterdir() if d.is_dir() and "Batch" in d.name]

    if not batch_dirs:
        # Fallback if no batch folders, treat root as one batch
        batch_dirs = [root]

    for batch_dir in batch_dirs:
        img_dir = batch_dir / "images"
        mask_dir = batch_dir / "masks"

        if not img_dir.exists() or not mask_dir.exists():
            print(f"Skipping {batch_dir.name}: missing images or masks folder")
            continue

        print(f"Scanning {batch_dir.name}...")
        images = list(img_dir.glob("*"))

        for img_path in images:
            if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png', '.bmp']:
                continue

            # Try to find corresponding mask
            mask_path = mask_dir / f"{img_path.stem}.png"
            if not mask_path.exists():
                mask_path = mask_dir / f"{img_path.stem}.jpg"

            if mask_path.exists():
                data_entries.append({
                    'path': str(img_path),
                    'mask_path': str(mask_path),
                    'batch_name': batch_dir.name
                })

    if not data_entries:
        raise ValueError(f"No valid pairs found in {root_dir}")

    return data_entries

def get_next_run_dir(base_dir):
    """Auto-increments run directory."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        run_dir = base_path / f"run_{i}"
        if not run_dir.exists():
            print(f"--- Starting New Training Session: {run_dir.name} ---")
            return str(run_dir)
        i += 1

def find_latest_run_dir(base_dir):
    """Finds the run directory with the highest index (e.g. run_5)."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return None

    runs = []
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                num = int(d.name.split("_")[1])
                runs.append((num, d.name))
            except ValueError:
                continue

    if not runs:
        return None

    # Sort by number descending
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0][1] # Return folder name e.g., "run_5"

def find_last_checkpoint(run_dir):
    """Find the most recent checkpoint in a run directory"""
    run_path = Path(run_dir)
    if not run_path.exists():
        return None

    checkpoints = [d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if not checkpoints:
        return None

    # Sort by number in checkpoint-XXXX
    checkpoints.sort(key=lambda x: int(x.name.split("-")[1]))
    return str(checkpoints[-1])

def main():
    parser = argparse.ArgumentParser()
    # Updated default paths - Changed hyphen to underscore to likely match user's structure
    parser.add_argument("--dataset_dir", default=r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization")
    parser.add_argument("--output_base", default=r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization")
    parser.add_argument("--base_model", default="nvidia/segformer-b2-finetuned-ade-512-512") # CHANGED TO B2 for better stability
    parser.add_argument("--resume_from_run", type=str, default=None, help="Name of run folder to resume weights from. Defaults to latest run if None.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8) # Bumping to 8 for speed, kept below 128 to avoid OOM
    parser.add_argument("--lr", type=float, default=6e-5)
    args = parser.parse_args()

    # Determine model to load (Base vs Previous Run)
    model_load_path = args.base_model
    resume_run_name = args.resume_from_run

    # RESTORED AUTO-DETECT: Since old incompatible runs are gone, this is safe again.
    if resume_run_name is None:
        latest_run = find_latest_run_dir(args.output_base)
        if latest_run:
            print(f"--- Auto-detected latest run for resume: {latest_run} ---")
            resume_run_name = latest_run
        else:
            print("--- No previous runs found. Starting fresh from base model. ---")

    if resume_run_name:
        previous_run_dir = Path(args.output_base) / resume_run_name
        last_checkpoint = find_last_checkpoint(previous_run_dir)
        if last_checkpoint:
            print(f"--- Found checkpoint in {resume_run_name}: {last_checkpoint} ---")
            print("--- Loading weights from this checkpoint for the new run ---")
            model_load_path = last_checkpoint
        else:
            print(f"Warning: No checkpoint found in {previous_run_dir}. Starting from base model.")

    # 1. Setup Data
    print(f"Scanning dataset at {args.dataset_dir}...")

    # Smart correction for Dataset path (Underscore vs Hyphen)
    dataset_path = Path(args.dataset_dir)
    if not dataset_path.exists():
        # Try swapping underscore/hyphen
        if "Tzefa_Datasets" in args.dataset_dir:
            alt_path = args.dataset_dir.replace("Tzefa_Datasets", "Tzefa-Datasets")
            if Path(alt_path).exists():
                print(f"--- Correcting path to: {alt_path} ---")
                args.dataset_dir = alt_path
        elif "Tzefa-Datasets" in args.dataset_dir:
            alt_path = args.dataset_dir.replace("Tzefa-Datasets", "Tzefa_Datasets")
            if Path(alt_path).exists():
                print(f"--- Correcting path to: {alt_path} ---")
                args.dataset_dir = alt_path

    # Pre-check to avoid ugly traceback if path is STILL wrong
    if not Path(args.dataset_dir).exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}\nPlease check if the folder exists or update --dataset_dir argument.")

    all_data = scan_dataset(args.dataset_dir)
    print(f"Found {len(all_data)} total image/mask pairs.")

    # Split (stratified by batch name to ensure val set has both types)
    batch_labels = [x['batch_name'] for x in all_data]
    train_data, val_data = train_test_split(
        all_data, test_size=0.1, random_state=42, stratify=batch_labels
    )

    # Load processor from the BASE model (config doesn't change during fine-tuning)
    # attempting to load from checkpoint often fails if preprocessor_config.json wasn't saved there
    processor = SegformerImageProcessor.from_pretrained(args.base_model)
    processor.do_resize = False # We handle resizing in Dataset

    train_ds = BinarizationDataset(train_data, processor, img_size=TARGET_SIZE, is_training=True)
    val_ds = BinarizationDataset(val_data, processor, img_size=TARGET_SIZE, is_training=False)

    # 2. Setup Output Directory
    run_output_dir = get_next_run_dir(args.output_base)

    # 3. Setup Model
    print(f"Loading model from: {model_load_path}")
    model = SegformerForSemanticSegmentation.from_pretrained(
        model_load_path,
        num_labels=2,
        id2label={0: "background", 1: "ink"},
        label2id={"background": 0, "ink": 1},
        ignore_mismatched_sizes=True
    )

    # 4. Setup Trainer
    training_args = TrainingArguments(
        output_dir=run_output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=1, # Standard training (B2 fits easily)
        gradient_checkpointing=False,
        save_strategy="epoch",
        eval_strategy="epoch",
        save_total_limit=2, # Keep last 2 checkpoints
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4, # Lowered to 4 to reduce system strain
        logging_steps=10,
    )

    # Use custom BalancedTrainer
    trainer = BalancedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting Training with Multi-Batch Balancing...")
    trainer.train()

    # Final Save
    model.save_pretrained(run_output_dir)
    processor.save_pretrained(run_output_dir)
    print(f"Model saved to {run_output_dir}")

if __name__ == "__main__":
    main()