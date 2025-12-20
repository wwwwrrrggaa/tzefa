import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from transformers import SegformerForSemanticSegmentation
from pathlib import Path
import gc

# --- CONFIGURATION ---
# Memory Optimization for ROCm/CUDA to prevent fragmentation
os.environ['PYTORCH_ALLOC_CONF'] = 'max_split_size_mb:128'

# Fixed Path: Added 'Binarization' subfolder to match the Unifier's output
DATASET_ROOT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Unified_Batch"
OUTPUT_BASE_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization"

# Model Config
BASE_MODEL = "nvidia/segformer-b2-finetuned-ade-512-512"

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
LEARNING_RATE = 6e-5
EPOCHS = 30
NUM_WORKERS = 4       # Reverted to 4 as requested previously (assuming stability)

# Save Strategy
SAVE_INTERVAL_STEPS = 100
MAX_CHECKPOINTS = 10

# --- Helper Functions for Run Management ---
def find_latest_run_dir(base_dir):
    """Finds the run directory with the highest index (e.g. run_5)."""
    base_path = Path(base_dir)
    if not base_path.exists(): return None
    runs = []
    for d in base_path.iterdir():
        if d.is_dir() and d.name.startswith("run_"):
            try:
                num = int(d.name.split("_")[1])
                runs.append((num, d.name))
            except ValueError: continue
    if not runs: return None
    runs.sort(key=lambda x: x[0], reverse=True)
    return runs[0][1]

def get_next_run_dir(base_dir):
    """Auto-increments run directory."""
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    i = 1
    while True:
        run_dir = base_path / f"run_{i}"
        if not run_dir.exists():
            print(f"--- Starting New Training Session: {run_dir.name} ---")
            run_dir.mkdir(parents=True, exist_ok=True)
            return str(run_dir)
        i += 1

def find_last_checkpoint(run_dir_name, base_dir):
    """
    Finds the most recent checkpoint in a specific run directory.
    Supports both custom .pth files AND HuggingFace checkpoint-X folders.
    """
    run_path = Path(base_dir) / run_dir_name
    if not run_path.exists(): return None
    candidates = []

    # 1. Custom Loop .pth files (segformer_step_100.pth)
    checkpoints_pth = [f for f in run_path.iterdir() if f.suffix == ".pth" and "step_" in f.name]
    if checkpoints_pth:
        # Sort by step number
        checkpoints_pth.sort(key=lambda x: int(x.stem.split("_")[-1]))
        candidates.append((int(checkpoints_pth[-1].stem.split("_")[-1]), str(checkpoints_pth[-1])))

    # 2. HuggingFace folders (checkpoint-5050)
    checkpoints_hf = [d for d in run_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")]
    if checkpoints_hf:
        # Sort by number in folder name (e.g. checkpoint-5050 -> 5050)
        checkpoints_hf.sort(key=lambda x: int(x.name.split("-")[1]))
        candidates.append((int(checkpoints_hf[-1].name.split("-")[1]), str(checkpoints_hf[-1])))

    if not candidates:
        # Check for final if no steps
        final_ckpt = run_path / "segformer_final.pth"
        if final_ckpt.exists(): return str(final_ckpt)
        return None

    # Return the one with the highest step count
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]

# --- 1. Dataset Class (Color Support) ---
class UnifiedBinarizationDataset(Dataset):
    def __init__(self, root_dir, augment=True):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, "images")
        self.msk_dir = os.path.join(root_dir, "masks")

        # Debug print to verify path visibility
        if not os.path.exists(self.img_dir):
            print(f"DEBUG: Looking for images at: {os.path.abspath(self.img_dir)}")
            raise FileNotFoundError(f"Dataset images not found at: {self.img_dir}\nDid you run Unified_Data_Processor.py?")

        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])

        if augment:
            self.transform = A.Compose([
                A.Rotate(limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.HorizontalFlip(p=0.2),
                A.RGBShift(p=0.3),
                A.RandomBrightnessContrast(p=0.4),
                # Heavy geometric transforms
                A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.1),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.1, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        fname = self.images[idx]
        img_path = os.path.join(self.img_dir, fname)

        # FIX: Replace image extension with .png for mask path
        # Unified_Data_Processor saves masks as .png regardless of input
        mask_name = os.path.splitext(fname)[0] + ".png"
        msk_path = os.path.join(self.msk_dir, mask_name)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(msk_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            # Fallback debug print if needed
            # print(f"Warning: Failed to load {fname}")
            return torch.zeros((3, 640, 640)), torch.zeros((640, 640))

        try:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        except Exception as e:
            print(f"Augmentation error: {e}")
            return torch.zeros((3, 640, 640)), torch.zeros((640, 640))

        mask = (mask > 127).long()
        mask = 1 - mask

        return image, mask

# --- 2. Training Loop ---
def train_model():
    print(f"--- Starting SegFormer (Color) Training on {DEVICE} ---")

    latest_run = find_latest_run_dir(OUTPUT_BASE_DIR)
    checkpoint_path = None

    if latest_run:
        checkpoint_path = find_last_checkpoint(latest_run, OUTPUT_BASE_DIR)
        if checkpoint_path:
            print(f"--- Found previous run: {latest_run} ---")
            print(f"--- Checkpoint detected: {checkpoint_path} ---")
        else:
            print(f"--- Found {latest_run} but no valid checkpoints. Starting fresh. ---")
    else:
        print("--- No previous runs found. Starting fresh. ---")

    current_run_dir = get_next_run_dir(OUTPUT_BASE_DIR)

    # AUGMENTATION DISABLED (As requested)
    dataset = UnifiedBinarizationDataset(DATASET_ROOT, augment=False)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # Init Base Model
    model = SegformerForSemanticSegmentation.from_pretrained(
        BASE_MODEL,
        num_labels=1,
        ignore_mismatched_sizes=True,
        reshape_last_stage=True
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=EPOCHS * len(loader), power=1.0)
    scaler = torch.amp.GradScaler('cuda')

    global_step = 0
    start_epoch = 0
    saved_checkpoints = []

    # --- Load Weights & State if Checkpoint Exists ---
    if checkpoint_path:
        print(f"Loading checkpoint data from: {checkpoint_path}")

        # Case A: Hugging Face Directory (Folder)
        if os.path.isdir(checkpoint_path):
            # THIS REPLACES THE MODEL OBJECT
            model = SegformerForSemanticSegmentation.from_pretrained(
                checkpoint_path,
                num_labels=1,
                ignore_mismatched_sizes=True,
                reshape_last_stage=True
            ).to(DEVICE)
            print("Loaded HF model weights. Optimizer reset (incompatible format).")

        # Case B: PyTorch File (.pth)
        elif os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)

            # Check if it's a full checkpoint dict or just weights
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])

                try:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    if "scaler_state_dict" in checkpoint:
                        scaler.load_state_dict(checkpoint["scaler_state_dict"])

                    start_epoch = checkpoint.get("epoch", 0)
                    global_step = checkpoint.get("global_step", 0)
                    print(f"Resumed full training state from Epoch {start_epoch}, Step {global_step}")
                except Exception as e:
                    print(f"Warning: Could not load optimizer state ({e}). Resuming with fresh optimizer.")
            else:
                # Old style: just weights
                model.load_state_dict(checkpoint)
                print("Loaded model weights only (no optimizer state).")

    # --- Epoch Loop ---
    optimizer.zero_grad(set_to_none=True) # Init gradients

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for i, (images, masks) in enumerate(loop):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE).float()

            with torch.amp.autocast('cuda'):
                outputs = model(pixel_values=images).logits
                outputs = nn.functional.interpolate(outputs, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss = nn.functional.binary_cross_entropy_with_logits(outputs.squeeze(1), masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

            global_step += 1

            if global_step % SAVE_INTERVAL_STEPS == 0:
                save_name = f"segformer_step_{global_step}.pth"
                save_path = os.path.join(current_run_dir, save_name)

                # FULL STATE SAVING
                checkpoint_dict = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "epoch": epoch,
                    "global_step": global_step
                }

                torch.save(checkpoint_dict, save_path)
                saved_checkpoints.append(save_path)

                if len(saved_checkpoints) > MAX_CHECKPOINTS:
                    oldest_ckpt = saved_checkpoints.pop(0)
                    if os.path.exists(oldest_ckpt):
                        try:
                            os.remove(oldest_ckpt)
                        except OSError as e:
                            print(f"Error removing old checkpoint: {e}")

            # Removed in-loop garbage collection to prevent stutter

        print(f"Epoch {epoch+1} Avg Loss: {epoch_loss/len(loader):.4f}")

        # Cleanup only at END of epoch
        gc.collect()
        torch.cuda.empty_cache()

    final_dict = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "epoch": EPOCHS,
        "global_step": global_step
    }
    torch.save(final_dict, os.path.join(current_run_dir, "segformer_final.pth"))

if __name__ == "__main__":
    train_model()