import os
import lmdb
import cv2
import pickle
import numpy as np
from tqdm import tqdm

# --- CONFIGURATION ---
SOURCE_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Unified_Batch"
OUTPUT_LMDB = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Binarization\Unified_LMDB"
MAP_SIZE = 10 * 1024 * 1024 * 1024  # 5GB

def create_lmdb():
    img_dir = os.path.join(SOURCE_DIR, "images")
    msk_dir = os.path.join(SOURCE_DIR, "masks")

    if not os.path.exists(img_dir):
        print(f"Error: Source not found at {img_dir}")
        return

    # Get file pairs
    images = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
    print(f"Found {len(images)} images. Packing into LMDB...")

    # Open LMDB environment
    # map_size must be larger than expected DB size. Windows requires this to be explicit.
    env = lmdb.open(OUTPUT_LMDB, map_size=MAP_SIZE)

    # Write to DB
    with env.begin(write=True) as txn:
        # Save dataset length
        txn.put("length".encode("ascii"), str(len(images)).encode("ascii"))

        for idx, img_name in enumerate(tqdm(images)):
            # Determine mask name (Unified Processor uses same stem + .png)
            base_name = os.path.splitext(img_name)[0]
            mask_name = base_name + ".png"

            img_path = os.path.join(img_dir, img_name)
            msk_path = os.path.join(msk_dir, mask_name)

            # Read Raw Bytes (We don't decode here, we store compressed bytes to save space)
            with open(img_path, 'rb') as f:
                img_bytes = f.read()

            with open(msk_path, 'rb') as f:
                msk_bytes = f.read()

            # Keys: image_0, mask_0, image_1, mask_1...
            txn.put(f"image_{idx}".encode("ascii"), img_bytes)
            txn.put(f"mask_{idx}".encode("ascii"), msk_bytes)

            # Commit every 1000 images to ensure safety
            if idx % 1000 == 0:
                pass # txn autocommits at end of context, but for huge sets manual commit helps RAM

    env.close()
    print(f"\nSuccess! LMDB created at: {OUTPUT_LMDB}")

if __name__ == "__main__":
    create_lmdb()