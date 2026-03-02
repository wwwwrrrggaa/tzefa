from pathlib import Path
import shutil

import cv2
import numpy as np
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
                resize_to=None,  # Keep Original Size
            )
            count += 1
        else:
            print(f"Warning: No matching noise image found for {mask_path.name}")

    print(f"Done. Processed {count} NoisyOffice pairs into Batch 7.")


def unpack_DIVA_Batch12(temp_root_dir, output_dir):
    """
    Unpacks DIVA-HisDB using Otsu's Binarization gated by the Layout GT.
    """
    temp_root = Path(temp_root_dir)
    out_path = Path(output_dir)

    manuscripts = ["CB55", "CS18", "CS863"]
    subsets = ["training", "validation", "public-test"]

    print("--- Starting Batch 12: DIVA-HisDB (Otsu-Gated Ink Extraction) ---")

    count = 0
    for manuscript in manuscripts:
        for subset in subsets:
            data_folder = temp_root / f"img-{manuscript}" / "img" / subset
            gt_folder = temp_root / f"pixel-level-gt-{manuscript}" / "pixel-level-gt" / subset

            if not data_folder.exists():
                continue

            for img_path in data_folder.glob("*.jpg"):
                gt_path = gt_folder / img_path.name.replace(".jpg", ".png")

                if gt_path.exists():
                    try:
                        # 1. Load Original Image (Grayscale)
                        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)

                        # 2. Load GT Layout Mask
                        gt_img = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED)

                        # Handle potential 3-channel or 1-channel GT
                        if len(gt_img.shape) == 3:
                            gt_indices = gt_img[:, :, 0]  # Blue channel often contains index
                        else:
                            gt_indices = gt_img

                        # Create the Region Mask (Select Main Text Body zones)
                        # Value 1 and 8 are standard for 'Main Text' in HisDB
                        region_mask = np.zeros_like(img_gray)
                        region_mask[(gt_indices == 1) | (gt_indices == 8)] = 255

                        # 3. Apply Otsu's Binarization
                        # THRESH_BINARY results in 0 for Ink and 255 for Paper
                        _, ink_mask = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                        # 4. Gating: Keep Ink only inside the Region Mask
                        # Start with a pure white background (255)
                        final_mask = np.full(img_gray.shape, 255, dtype=np.uint8)

                        # Only where region_mask is active, use the Otsu result
                        final_mask[region_mask == 255] = ink_mask[region_mask == 255]

                        # 5. Save and Cleanup
                        temp_mask_name = f"temp_otsu_{manuscript}_{img_path.stem}.png"
                        cv2.imwrite(temp_mask_name, final_mask)

                        add_pair_to_dataset(
                            image_path=str(img_path),
                            mask_path=temp_mask_name,
                            out_dir=str(out_path),
                            resize_to=None,
                        )
                        count += 1

                        if os.path.exists(temp_mask_name):
                            os.remove(temp_mask_name)

                    except Exception as e:
                        print(f"Error on {img_path.name}: {e}")

    print(f"Done. Processed {count} DIVA images using Region-Gated Otsu.")


def unpack_Obs_Dataset(input_dir, gt_dir, output_dir):
    """
    Unpacks the 'Obs' dataset and inverts the GT masks.
    Logic: .scanned.png (Input) -> .clean.png (GT)
    Mask Inversion: Black (0) becomes White (255), White (255) becomes Black (0).
    """
    input_path = Path(input_dir)
    gt_path = Path(gt_dir)

    print(f"--- Processing Batch 14: Obs Dataset (with Mask Inversion) ---")

    # Match pairs based on the filename stem before '.scanned' or '.clean'
    # Example: 3_nouvel-obs...scanned.png
    images = list(input_path.glob("*.scanned.png"))
    count = 0

    for img_file in images:
        # Construct the matching GT filename
        # replace '.scanned.png' with '.clean.png'
        gt_file_name = img_file.name.replace(".scanned.png", ".clean.png")
        full_gt_path = gt_path / gt_file_name

        if full_gt_path.exists():
            try:
                # Load the GT Mask
                mask = cv2.imread(str(full_gt_path), cv2.IMREAD_GRAYSCALE)

                # REVERSE PIXELS: (black to white and white to black)
                # This ensures Ink is 0 and Background is 255 (standard for binarization)
                # or vice-versa depending on your training needs.
                inverted_mask = cv2.bitwise_not(mask)

                # Save to a temporary file for add_pair_to_dataset to pick up
                temp_mask_name = f"temp_inverted_{img_file.stem}.png"
                cv2.imwrite(temp_mask_name, inverted_mask)

                # Add to dataset
                add_pair_to_dataset(
                    image_path=str(img_file),
                    mask_path=temp_mask_name,
                    out_dir=output_dir,
                    resize_to=None,  # Keep original resolution
                )

                # Cleanup temp file
                if os.path.exists(temp_mask_name):
                    os.remove(temp_mask_name)

                count += 1
            except Exception as e:
                print(f"Error processing {img_file.name}: {e}")
        else:
            print(f"Warning: No GT found for {img_file.name}")

    print(f"Done. Processed {count} inverted pairs into Batch 14.")

# --- EXECUTION ---
if __name__ == "__main__":
    OBS_INPUT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\obs\input"
    OBS_GT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\obs\gt\bin"
    BATCH_14_OUT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Batch_14"

    unpack_Obs_Dataset(OBS_INPUT, OBS_GT, BATCH_14_OUT)