from pathlib import Path
import shutil
from PIL import Image
import os
import random

def add_pair_to_dataset(image_path: str,
                        mask_path: str,
                        out_dir: str,
                        pad: int = 8,
                        resize_to: tuple = None) -> tuple:
    """
    Add one image + binarized mask pair to a growing dataset.

    - resize_to: If None, keeps original size (CRITICAL for Batch 2).
                 If set to (256, 256), performs resize (for Batch 1).
    """
    out = Path(out_dir)
    imgs_dir = out / "images"
    masks_dir = out / "masks"
    index_file = out / "next_index.txt"

    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    img_src = Path(image_path)
    mask_src = Path(mask_path)

    if not img_src.exists():
        print(f"Skipping: Image not found: {img_src}")
        return None, None
    if not mask_src.exists():
        print(f"Skipping: Mask not found: {mask_src}")
        return None, None

    # Read or create next index
    if index_file.exists():
        try:
            idx = int(index_file.read_text().strip())
        except Exception:
            idx = 0
    else:
        idx = 0

    # --- PROCESS IMAGE ---
    try:
        with Image.open(img_src) as im:
            im = im.convert("RGB") # Force 3 channels

            # ONLY resize if specifically requested (e.g. for Batch 1)
            # For Batch 2, we skip this to keep full resolution.
            if resize_to is not None:
                if im.size != resize_to:
                    im = im.resize(resize_to, Image.BILINEAR)

            # --- PROCESS MASK ---
            with Image.open(mask_src) as m:
                m = m.convert("L") # Force grayscale

                if resize_to is not None:
                    if m.size != resize_to:
                        m = m.resize(resize_to, Image.NEAREST)
                else:
                    # If we didn't resize, ensure mask matches image size exactly
                    # (Sometimes datasets have 1px off errors)
                    if m.size != im.size:
                        m = m.resize(im.size, Image.NEAREST)

                # Binarize mask -> 0 or 255 (Clean up gray artifacts)
                m = m.point(lambda p: 255 if p > 127 else 0).convert("L")

            # --- SAVE ---
            # Rename strategy: 00000001.jpg / 00000001.png
            base_name = f"{idx:0{pad}d}"

            # Force Images to JPG (Efficiency)
            new_img = imgs_dir / (base_name + ".jpg")
            # Force Masks to PNG (Lossless - Mandatory for segmentation)
            new_mask = masks_dir / (base_name + ".png")

            # Handle index collision if file exists manually
            while new_img.exists() or new_mask.exists():
                idx += 1
                base_name = f"{idx:0{pad}d}"
                new_img = imgs_dir / (base_name + ".jpg")
                new_mask = masks_dir / (base_name + ".png")

            im.save(new_img, format="JPEG", quality=95)
            m.save(new_mask, format="PNG")

            # Update index
            next_idx = idx + 1
            index_file.write_text(str(next_idx))

            return new_img, new_mask

    except Exception as e:
        print(f"Error processing {img_src.name}: {e}")
        return None, None


def find_mask_for_image(img_path, gt_folder_path):
    """
    Tries to find the matching mask in the GT folder.
    Handles variations like: img.jpg -> img.png, img_GT.bmp, img_gt.tiff
    """
    img_stem = img_path.stem # 'image_01'

    # List of possible mask filenames to look for
    candidates = [
        img_stem,                 # match exact name
        img_stem + "_gt",         # common suffix
        img_stem + "_GT",         # common suffix
        img_stem.replace("image", "gt"), # rare variation
    ]

    # Extensions to check
    extensions = ['.png', '.bmp', '.tiff', '.tif', '.jpg', '.jpeg']

    for cand in candidates:
        for ext in extensions:
            potential_path = gt_folder_path / (cand + ext)
            if potential_path.exists():
                return potential_path

    return None

def unpack_Restomer(dir_path, write_path):
    """
    Unpacks Restomer dataset structure: train/train_gt, test/test_gt, etc.
    Forces resize to 256x256 as per original logic for patches.
    """
    dir_names = ["train", "test", "valid"]
    for dir_name in dir_names:
        data_path = os.path.join(dir_path, dir_name)
        gt_path = data_path + "_gt"

        if not os.path.exists(data_path):
            print(f"Skipping {data_path}, not found.")
            continue

        print(f"Processing Restomer folder: {dir_name}")
        for file_name in os.listdir(data_path):
            data_file_name = os.path.join(data_path, file_name)
            gt_file_name = os.path.join(gt_path, file_name)

            # Using resize_to=(256, 256) to match your original implicit behavior
            add_pair_to_dataset(data_file_name, gt_file_name, write_path, pad=8, resize_to=(256, 256))


def unpack_Kaggle_Batch2(source_root_dir, output_dir):
    """
    Iterates through dataset folders in source_root_dir, finds IMG/GT subfolders,
    and unpacks them. Keeps original resolution (resize_to=None).
    """
    source_path = Path(source_root_dir)
    print(f"Scanning {source_path} for datasets...")

    count = 0

    # Iterate over immediate subdirectories (The separate datasets: DIBCO 2009, etc.)
    for dataset_dir in source_path.iterdir():
        if not dataset_dir.is_dir():
            continue

        # Check subdirectories of this dataset folder for IMG and GT
        # Map uppercase name to actual name to handle case sensitivity
        subdirs = {d.name.upper(): d.name for d in dataset_dir.iterdir() if d.is_dir()}

        if "IMG" in subdirs and "GT" in subdirs:
            print(f"--> Found dataset in: {dataset_dir.name}")

            img_folder = dataset_dir / subdirs["IMG"]
            gt_folder = dataset_dir / subdirs["GT"]

            # Iterate over all images in IMG folder
            valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            images = [f for f in img_folder.iterdir() if f.suffix.lower() in valid_extensions]

            for img_file in images:
                # Find matching mask
                mask_file = find_mask_for_image(img_file, gt_folder)

                if mask_file:
                    add_pair_to_dataset(
                        image_path=str(img_file),
                        mask_path=str(mask_file),
                        out_dir=output_dir,
                        resize_to=None  # <--- IMPORTANT: KEEP ORIGINAL SIZE
                    )
                    count += 1
                else:
                    print(f"    Warning: No mask found for {img_file.name}")
        else:
            print(f"Skipping {dataset_dir.name}: Could not find both IMG and GT folders.")

    print(f"Done. Processed {count} full-page images into {output_dir}")

def unpack_Challenge1(source_dir, output_dir):
    """
    Unpacks Challenge-1 structure where images map to multiple GT files (GT1, GT2).
    Alternates between GT1 and GT2 for consecutive images to balance data.
    """
    source = Path(source_dir)
    if not source.exists():
        print(f"Source directory not found: {source}")
        return

    print(f"Processing Challenge-1 Dataset from: {source}")

    # Get all jpg images
    images = sorted([f for f in source.glob("*.jpg")])
    count = 0

    for i, img_path in enumerate(images):
        stem = img_path.stem

        # Alternating logic: Even index -> try GT1, Odd index -> try GT2
        if i % 2 == 0:
            primary_suffix = "_GT1.bmp"
            secondary_suffix = "_GT2.bmp"
        else:
            primary_suffix = "_GT2.bmp"
            secondary_suffix = "_GT1.bmp"

        # Try finding primary first, then fallback to secondary
        mask_path = source / (stem + primary_suffix)

        if not mask_path.exists():
            # Fallback
            mask_path = source / (stem + secondary_suffix)

        if mask_path.exists():
            add_pair_to_dataset(
                image_path=str(img_path),
                mask_path=str(mask_path),
                out_dir=output_dir,
                resize_to=None # Keep Original Size
            )
            count += 1
        else:
            print(f"Warning: No GT (1 or 2) found for {stem}")

    print(f"Done. Processed {count} images into Batch 5.")

def unpack_NoisyOffice(image_source_dir, mask_source_dir, output_dir):
    """
    Unpacks NoisyOffice dataset.
    Matches 'Font..._Clean_...' masks to corresponding 'Font..._Noise..._...' images.
    Finds ALL available noise variants for a specific mask and randomly selects ONE.
    """
    img_src = Path(image_source_dir)
    msk_src = Path(mask_source_dir)

    if not img_src.exists() or not msk_src.exists():
        print("Source directories for NoisyOffice not found.")
        return

    print(f"Processing NoisyOffice from: {img_src}")

    # Noise variants to check
    noise_types = ['Noisef', 'Noisec', 'Noisep', 'Noisew']

    # Get all Clean masks
    masks = sorted([f for f in msk_src.glob("*_Clean_*.png")])
    count = 0

    for mask_path in masks:
        # Example: Fontfre_Clean_TE.png -> parts: ['Fontfre', 'Clean', 'TE']
        parts = mask_path.stem.split('_')

        if 'Clean' not in parts:
            continue

        clean_idx = parts.index('Clean')
        candidates = []

        # Find all valid noise images for this mask
        for noise_type in noise_types:
            img_parts = parts.copy()
            img_parts[clean_idx] = noise_type
            img_name = "_".join(img_parts) + mask_path.suffix

            img_candidate = img_src / img_name
            if img_candidate.exists():
                candidates.append(img_candidate)

        if candidates:
            # Pick ONE random variant to ensure diversity (prevents bias toward specific noise type)
            selected_img = random.choice(candidates)

            add_pair_to_dataset(
                image_path=str(selected_img),
                mask_path=str(mask_path),
                out_dir=output_dir,
                resize_to=None # Keep Original Size
            )
            count += 1
        else:
            print(f"Warning: No matching noise image found for {mask_path.name}")

    print(f"Done. Processed {count} NoisyOffice pairs into Batch 7.")

# --- EXECUTION ---
if __name__ == '__main__':
    # 3. Unpack NoisyOffice (Batch 7)
    NOISY_IMG_SOURCE = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\images"
    NOISY_MSK_SOURCE = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\masks"
    BATCH_7_OUTPUT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_7"

    unpack_NoisyOffice(NOISY_IMG_SOURCE, NOISY_MSK_SOURCE, BATCH_7_OUTPUT)

    # --- Previous Batches (Commented out) ---
    # CHALLENGE_SOURCE = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\Challenge-1-ForTrain\train-50"
    # BATCH_5_OUTPUT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_5"
    # unpack_Challenge1(CHALLENGE_SOURCE, BATCH_5_OUTPUT)