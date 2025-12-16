# python
from pathlib import Path
import shutil
import os

def add_pair_to_dataset(image_path: str, mask_path: str, out_dir: str, pad: int = 8) -> tuple:
    """
    Add one image+mask pair to a growing dataset.

    - image_path, mask_path: source file paths (strings or Path-like)
    - out_dir: dataset root directory (will contain `images/`, `masks/`, and `next_index.txt`)
    - pad: zero padding width for filenames (default 8 -> 00000001)

    Returns: (new_image_path, new_mask_path) as Path objects.
    """
    out = Path(out_dir)
    imgs_dir = out / "images"
    masks_dir = out / "masks"
    index_file = out / "next_index.txt"

    imgs_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # read or create next index
    if index_file.exists():
        try:
            idx = int(index_file.read_text().strip())
        except Exception:
            idx = 0
    else:
        idx = 0

    # compute target names
    img_src = Path(image_path)
    mask_src = Path(mask_path)
    if not img_src.exists():
        raise FileNotFoundError(f"Image not found: {img_src}")
    if not mask_src.exists():
        raise FileNotFoundError(f"Mask not found: {mask_src}")

    img_ext = img_src.suffix or ".jpg"
    mask_ext = mask_src.suffix or ".png"

    base_name = f"{idx:0{pad}d}"
    new_img = imgs_dir / (base_name + img_ext)
    new_mask = masks_dir / (base_name + mask_ext)

    # avoid overwriting accidentally: if file exists, increment until free
    while new_img.exists() or new_mask.exists():
        idx += 1
        base_name = f"{idx:0{pad}d}"
        new_img = imgs_dir / (base_name + img_ext)
        new_mask = masks_dir / (base_name + mask_ext)

    # copy files
    shutil.copy2(img_src, new_img)
    shutil.copy2(mask_src, new_mask)

    # write next index (next free)
    next_idx = idx + 1
    index_file.write_text(str(next_idx))

    return new_img, new_mask

def unpack_Restomer(dir_path,write_path):
    dir_names=["train","test","valid"]
    for dir_name in dir_names:
        path=os.join(dir_path,dir_name)

if __name__ == '__main__':
    unpack_Restomer(r"/Tzefa_Datasets\temp\Restomer_data", r"/Tzefa_Datasets\Binarization")