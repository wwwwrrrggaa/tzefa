import gc
import glob
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import lmdb
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.optim as optim
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- CONFIGURATION ---
DATASET_ROOT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Unified_LMDB"
CHECKPOINT_DIR = Path(r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization\run_mitb5_highres_manet")
PRETRAINED_CHECKPOINT = ""

MAP_SIZE = 10 * 1024 * 1024 * 1024

MODEL_ARCH = "MAnet"
ENCODER_NAME = "mit_b5"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training Config
BATCH_SIZE = 2
TARGET_BATCH_SIZE = 64
ACCUMULATION_STEPS = max(1, TARGET_BATCH_SIZE // BATCH_SIZE)

LEARNING_RATE = 1e-5
EPOCHS = 50
NUM_WORKERS = 4

# --- INPUT SIZE ---
PATCH_SIZE = 640

# Saving
SAVE_INTERVAL_STEPS = 10
MAX_CHECKPOINTS = 10


# --- 1. AUGMENTATION PIPELINE (Corrected & Enhanced) ---
import albumentations as A


import cv2
import numpy as np
import random
from albumentations.pytorch import ToTensorV2


# --- Custom Augmentation Class ---
class AddGridLines(A.ImageOnlyTransform):
    def __init__(self, p=0.5):
        super().__init__(p=p)

    def apply(self, img, **params):
        # Retrieve the mask to ensure we draw lines BEHIND the text
        mask = params.get("mask")
        if mask is None:
            return img

        img = img.copy()
        h, w, _ = img.shape

        # Randomize grid parameters for realism
        grid_size = random.randint(30, 60)
        thickness = random.randint(1, 2)
        # Colors: Faded blue, gray, or dull red/green
        color = [random.randint(170, 210), random.randint(170, 210), random.randint(190, 240)]

        grid_layer = np.zeros_like(img)

        # Randomize direction: 0=Rows, 1=Columns, 2=Grid (Both)
        mode = random.randint(0, 2)

        # Draw Horizontal Lines (Rows)
        if mode == 0 or mode == 2:
            for y in range(0, h, grid_size):
                cv2.line(grid_layer, (0, y), (w, y), color, thickness)

        # Draw Vertical Lines (Columns)
        if mode == 1 or mode == 2:
            for x in range(0, w, grid_size):
                cv2.line(grid_layer, (x, 0), (x, h), color, thickness)

        # Apply logic: Draw lines only where grid_layer has pixels AND mask is 0 (Background)
        # Ensure mask is squeezed to match dimensions (H, W) for boolean indexing
        mask_sq = mask.squeeze() if len(mask.shape) > 2 else mask

        # Create boolean condition
        # (grid_layer > 0).any(axis=-1) checks if a pixel has color in any channel
        condition = (grid_layer > 0).any(axis=-1) & (mask_sq == 0)

        # Apply the grid lines to the image
        img[condition] = grid_layer[condition]

        return img


# --- Main Pipeline ---
def get_train_transforms():
    """
    Structure:
    1. Geometric Distortions (Shape/Position) - Applied First
    2. Synthetic Elements (Grid Lines) - Applied Middle
    3. Pixel-Level Distortions (Color/Noise/Invert) - Applied Last
    4. Normalization - Applied Final
    """
    return A.Compose(
        [
            # === BLOCK A: OPTIONAL AUGMENTATIONS (50% Chance) ===
            A.Compose(
                [
                    # --- 1. Geometric (Slicing/Warping/Resizing) ---
                    A.OneOf(
                        [
                            # "Floating Patch": Randomly shrink then pad back
                            A.Compose(
                                [
                                    A.RandomCrop(
                                        height=random.randint(300, 550), width=random.randint(300, 550), p=1.0
                                    ),
                                    A.PadIfNeeded(
                                        min_height=PATCH_SIZE,
                                        min_width=PATCH_SIZE,
                                        border_mode=cv2.BORDER_CONSTANT,
                                        p=1.0,
                                    ),
                                ]
                            ),
                            # Standard affine/warp
                            A.Affine(scale=(0.9, 1.1), rotate=(-10, 10), translate_percent=(-0.1, 0.1), p=1.0),
                            A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                            A.ElasticTransform(alpha=1, sigma=50, p=1.0),
                        ],
                        p=0.3,
                    ),
                    # --- 2. Synthetic Grid (INSERTED HERE) ---
                    # Adds lines/grids behind text before we add noise/blur
                    AddGridLines(p=0.4),
                    # --- 3. "Destructive" Augs (Holes/Shadows) ---
                    A.CoarseDropout(
                        num_holes_range=(1, 3), hole_height_range=(20, 50), hole_width_range=(20, 50), p=0.1
                    ),
                    A.RandomShadow(num_shadows_limit=(1, 2), shadow_dimension=5, p=0.1),
                    # --- 4. Pixel / Color Logic ---
                    # INVERT: Solves Dark Mode.
                    # Applied AFTER grid so we get inverted grids (white lines on black) too.
                    A.InvertImg(p=0.2),
                    # Color/Noise
                    A.OneOf(
                        [
                            A.ISONoise(p=1.0),
                            A.GaussNoise(p=1.0),
                            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                            A.MotionBlur(blur_limit=5, p=1.0),
                            A.ImageCompression(quality_range=(20, 50), p=1.0),
                        ],
                        p=0.2,
                    ),
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                            A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                            A.CLAHE(clip_limit=2.0, p=1.0),
                        ],
                        p=0.2,
                    ),
                ],
                p=0.5,
            ),
            # === BLOCK B: MANDATORY ===
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )
# --- 2. MODEL ARCHITECTURE ---
class HighResMAnet(nn.Module):
    def __init__(self, encoder_name, encoder_weights, classes=1):
        super().__init__()
        self.base_model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=classes,
            encoder_depth=5,
            decoder_channels=(256, 128, 64, 32, 16),
        )

        # Stage 0 High-Res Stem (Stride 1) - Preserves fine details
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


# --- 3. LOSS FUNCTION (Recall-Heavy) ---
def compute_shift_invariant_loss(pred, target, criterion, max_shift=4, step=2):
    """
    Computes loss robust to slight misalignments (1-4 pixels) between ground truth and prediction.
    Useful because DocLayNet/Synthetic data masks aren't always pixel-perfect.
    """
    min_loss = criterion(pred, target)
    # Reduced max_shift to 4 to speed up training slightly
    shifts = range(-max_shift, max_shift + 1, step)
    for dx in shifts:
        for dy in shifts:
            if dx == 0 and dy == 0:
                continue
            # Shift target to match prediction
            shifted_target = torch.roll(target, shifts=(dy, dx), dims=(2, 3))
            loss = criterion(pred, shifted_target)
            if loss < min_loss:
                min_loss = loss
    return min_loss


# --- 4. DATASET CLASS ---
class UnifiedBinarizationDataset(Dataset):
    def __init__(self, root_dir):
        self.lmdb_path = root_dir
        self.env = None
        if not os.path.exists(self.lmdb_path):
            raise FileNotFoundError(f"LMDB not found at {self.lmdb_path}")

        temp_env = lmdb.open(self.lmdb_path, readonly=True, lock=False, map_size=MAP_SIZE)
        with temp_env.begin(write=False) as txn:
            self.length = int(txn.get("length".encode("ascii")).decode("ascii"))
        temp_env.close()

        self.transform = get_train_transforms()

    def _init_db(self):
        self.env = lmdb.open(
            self.lmdb_path, readonly=True, lock=False, readahead=False, meminit=False, map_size=MAP_SIZE
        )

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.env is None:
            self._init_db()

        with self.env.begin(write=False) as txn:
            img_bytes = txn.get(f"image_{idx}".encode("ascii"))
            msk_bytes = txn.get(f"mask_{idx}".encode("ascii"))

        image = cv2.cvtColor(
            cv2.imdecode(np.frombuffer(img_bytes, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )
        mask = cv2.imdecode(np.frombuffer(msk_bytes, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)

        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented["image"]
        # Convert mask: 0=Background, 1=Text (Target is usually white text on black background for loss)
        mask_tensor = 1.0 - (augmented["mask"].float() / 255.0)

        return img_tensor, mask_tensor


# --- 5. UTILS ---
def cleanup_old_checkpoints():
    files = glob.glob(str(CHECKPOINT_DIR / "step_*.pth"))
    if len(files) <= MAX_CHECKPOINTS:
        return
    try:
        files.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        for f in files[:-MAX_CHECKPOINTS]:
            os.remove(f)
    except Exception as e:
        print(f"Error cleaning checkpoints: {e}")


# --- 6. TRAINING LOOP ---
def train_model():
    print(f"--- INITIALIZING HighRes {ENCODER_NAME}-{MODEL_ARCH} (Resuming from {CHECKPOINT_DIR}) ---")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = HighResMAnet(encoder_name=ENCODER_NAME, encoder_weights="imagenet", classes=1).to(DEVICE)
    model = model.to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-7)
    scaler = torch.amp.GradScaler("cuda")

    start_epoch = 0
    global_step = 0

    # --- RESUME LOGIC ---
    local_checkpoints = glob.glob(str(CHECKPOINT_DIR / "step_*.pth"))

    if local_checkpoints:
        local_checkpoints.sort(key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]))
        load_path = local_checkpoints[-1]
        print(f"--- RESUMING LOCAL RUN: {load_path} ---")

        checkpoint = torch.load(load_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

        start_epoch = checkpoint["epoch"]
        global_step = checkpoint["global_step"]
        print(f"--- Resumed successfully at Global Step: {global_step} (Epoch {start_epoch}) ---")

    elif os.path.exists(PRETRAINED_CHECKPOINT):
        print(f"--- LOADING PRETRAINED: {PRETRAINED_CHECKPOINT} ---")
        model.load_state_dict(torch.load(PRETRAINED_CHECKPOINT, map_location=DEVICE), strict=False)

    # Dataset
    dataset = UnifiedBinarizationDataset(DATASET_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # --- LOSS CONFIGURATION (STRONGER RECALL) ---
    loss_focal = smp.losses.FocalLoss(mode="binary", gamma=2.0)

    # Tversky Loss: alpha=FP penalty, beta=FN penalty.
    # beta=0.7 means we punish False Negatives (erased text) heavily.
    loss_tversky = smp.losses.TverskyLoss(mode="binary", alpha=0.3, beta=0.7)

    def base_criterion(pred, target):
        return loss_focal(pred, target) + loss_tversky(pred, target)

    print("--- STARTING TRAINING ---")
    for epoch in range(start_epoch, EPOCHS):
        model.train()
        loop = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for i, (images, masks) in enumerate(loop):
            images = images.to(DEVICE, non_blocking=True, memory_format=torch.channels_last)
            masks = masks.to(DEVICE, non_blocking=True).float().unsqueeze(1)

            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(images)
                loss = compute_shift_invariant_loss(outputs, masks, base_criterion)
                loss = loss / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % SAVE_INTERVAL_STEPS == 0:
                    save_path = CHECKPOINT_DIR / f"step_{global_step}.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "scheduler_state_dict": scheduler.state_dict(),
                            "scaler_state_dict": scaler.state_dict(),
                        },
                        save_path,
                    )
                    cleanup_old_checkpoints()

            loop.set_postfix(loss=loss.item(), step=global_step)

        scheduler.step()
        gc.collect()

    torch.save(model.state_dict(), CHECKPOINT_DIR / "final_model.pth")
    print("--- DONE ---")

if __name__ == "__main__":
    train_model()