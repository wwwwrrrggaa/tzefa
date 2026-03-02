import os
import shutil
import zipfile
from pathlib import Path

import pandas as pd
import requests
from huggingface_hub import snapshot_download
from tqdm import tqdm

# Ensure pandas and pyarrow are installed for reading Parquet files
# pip install pandas pyarrow


def download_file(url, dest_path):
    """Helper to download a file with a progress bar."""
    if dest_path.exists():
        print(f"File {dest_path.name} already exists. Skipping download.")
        return

    print(f"Downloading {url}...")
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with (
            open(dest_path, "wb") as f,
            tqdm(
                desc=dest_path.name,
                total=total_size,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            ) as bar,
        ):
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if dest_path.exists():
            dest_path.unlink()  # cleanup partial


# --- 1. SROIE (Receipts) ---
def unpack_sroie(target_dir="Batch_3"):
    print(f"\n>>> Processing {target_dir} (SROIE)...")
    output_path = Path(target_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # We use the GitHub archive of the dataset to avoid HF script issues
    url = "https://github.com/zzzDavid/ICDAR-2019-SROIE/archive/refs/heads/master.zip"
    zip_path = Path("sroie_master.zip")

    download_file(url, zip_path)

    print("Extracting SROIE...")
    with zipfile.ZipFile(zip_path, "r") as z:
        # The repo has data/img and data/box
        # We want to match images with their text
        files = z.namelist()
        img_files = [f for f in files if f.endswith(".jpg") and "data/img" in f]

        for img_file in tqdm(img_files):
            # Find corresponding txt file (in data/box or data/key usually)
            # This repo structure: data/img/000.jpg -> data/box/000.txt (coords) or data/key/000.txt (text)
            # We usually want the text. Let's grab the 'key' (ground truth text).

            # Construct the matching text path in the zip
            # Note: The zip creates a root folder "ICDAR-2019-SROIE-master"
            txt_file = img_file.replace("data/img", "data/key").replace(".jpg", ".txt")  # trying key first

            if txt_file not in files:
                # Fallback to box if key doesn't exist (though SROIE usually needs key for OCR text)
                txt_file = img_file.replace("data/img", "data/box").replace(".jpg", ".txt")

            if txt_file in files:
                # Extract image
                img_name = Path(img_file).name
                dest_img = output_path / f"sroie_{img_name}"
                with open(dest_img, "wb") as f:
                    f.write(z.read(img_file))

                # Extract text
                # We need to read the content and save it
                dest_txt = output_path / f"sroie_{img_name.replace('.jpg', '.txt')}"
                content = z.read(txt_file).decode("utf-8", errors="ignore")

                # SROIE specific cleaning (sometimes it's JSON-like or CSV)
                # For simplicity, we just dump the raw content.
                # Ideally, you'd parse the JSON if it's the 'key' folder.
                with open(dest_txt, "w", encoding="utf-8") as f:
                    f.write(content)

    print("SROIE Complete.")


# --- 2. FUNSD (Forms) ---
def unpack_funsd(target_dir="Batch_4"):
    print(f"\n>>> Processing {target_dir} (FUNSD)...")
    output_path = Path(target_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    url = "https://guillaumejaume.github.io/FUNSD/dataset.zip"
    zip_path = Path("funsd.zip")

    download_file(url, zip_path)

    print("Extracting FUNSD...")
    with zipfile.ZipFile(zip_path, "r") as z:
        files = z.namelist()
        # Filter for training images
        img_files = [f for f in files if "training_data/images/" in f and f.endswith(".png")]

        for img_file in tqdm(img_files):
            filename = Path(img_file).name

            # Annotation is in 'annotations' folder with .json extension
            ann_file = img_file.replace("images/", "annotations/").replace(".png", ".json")

            if ann_file in files:
                dest_img = output_path / f"funsd_{filename}"
                dest_ann = output_path / f"funsd_{filename.replace('.png', '.json')}"

                with open(dest_img, "wb") as f:
                    f.write(z.read(img_file))
                with open(dest_ann, "wb") as f:
                    f.write(z.read(ann_file))

    print("FUNSD Complete.")


# --- 3. DocLayNet (Printed Docs) ---
def unpack_doclaynet(target_dir="Batch_5"):
    print(f"\n>>> Processing {target_dir} (DocLayNet)...")
    output_path = Path(target_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Use a direct HF download of the zip if available, or parse parquet
    # Parsing parquet is more reliable for newer HF datasets
    print("Downloading/Loading DocLayNet (Small) via Parquet...")

    try:
        # We download the parquet files directly using snapshot_download
        repo_path = snapshot_download(
            repo_id="pierreguillou/DocLayNet-small", repo_type="dataset", allow_patterns="*.parquet"
        )

        # Find parquet files
        parquet_files = list(Path(repo_path).rglob("*.parquet"))

        for p_file in parquet_files:
            if "train" in p_file.name:  # Only process training split
                df = pd.read_parquet(p_file)

                print(f"Extracting {len(df)} images from {p_file.name}...")
                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    # 'image' column usually contains bytes or dictionary
                    image_data = row["image"]

                    # Extract bytes
                    if isinstance(image_data, dict) and "bytes" in image_data:
                        img_bytes = image_data["bytes"]
                    elif isinstance(image_data, bytes):
                        img_bytes = image_data
                    else:
                        continue  # Skip if format is unexpected

                    file_name = f"doclaynet_{idx}.png"

                    # Save Image
                    with open(output_path / file_name, "wb") as f:
                        f.write(img_bytes)

                    # Save Text/Layout info
                    # DocLayNet has 'texts' and 'bboxes' columns
                    txt_name = file_name.replace(".png", ".txt")
                    with open(output_path / txt_name, "w", encoding="utf-8") as f:
                        # Simple format: just dumping the text lines
                        # You can enhance this to save bboxes if needed
                        if "texts" in row and row["texts"] is not None:
                            for line in row["texts"]:
                                f.write(str(line) + "\n")

    except Exception as e:
        print(f"DocLayNet extraction failed: {e}")

    print("DocLayNet Complete.")


# --- 4. IAM Handwriting (FKI) ---
def unpack_iam(target_dir="Batch_6"):
    print(f"\n>>> Processing {target_dir} (IAM Handwriting)...")
    output_path = Path(target_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # IAM official is gated. We use the 'Teklia/IAM-line' mirror on HF which is open.
    print("Downloading IAM via Teklia mirror...")
    try:
        repo_path = snapshot_download(repo_id="Teklia/IAM-line", repo_type="dataset", allow_patterns="*.parquet")
        parquet_files = list(Path(repo_path).rglob("*.parquet"))

        for p_file in parquet_files:
            if "train" in p_file.name:
                df = pd.read_parquet(p_file)
                print(f"Extracting {len(df)} lines...")

                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    # Columns usually: 'image', 'text'
                    if "image" not in row or "text" not in row:
                        continue

                    image_data = row["image"]
                    text_data = row["text"]

                    if isinstance(image_data, dict) and "bytes" in image_data:
                        img_bytes = image_data["bytes"]
                    else:
                        continue

                    fname = f"iam_{idx}.png"

                    with open(output_path / fname, "wb") as f:
                        f.write(img_bytes)

                    with open(output_path / fname.replace(".png", ".txt"), "w", encoding="utf-8") as f:
                        f.write(str(text_data))

    except Exception as e:
        print(f"IAM extraction failed: {e}")

    print("IAM Complete.")


# --- 5. CTW1500 (Curved Lines) ---
def unpack_ctw1500(target_dir="Batch_7"):
    print(f"\n>>> Processing {target_dir} (CTW1500)...")
    output_path = Path(target_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Using MiXaiLL76/CTW1500_OCR mirror
    print("Downloading CTW1500 via HF Mirror...")
    try:
        repo_path = snapshot_download(repo_id="MiXaiLL76/CTW1500_OCR", repo_type="dataset", allow_patterns="*.parquet")
        parquet_files = list(Path(repo_path).rglob("*.parquet"))

        for p_file in parquet_files:
            if "train" in p_file.name:
                df = pd.read_parquet(p_file)
                print(f"Extracting {len(df)} images...")

                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    if "image" not in row:
                        continue

                    image_data = row["image"]
                    # CTW might have 'file_name' in row
                    orig_name = row.get("file_name", f"{idx}.jpg")

                    if isinstance(image_data, dict) and "bytes" in image_data:
                        img_bytes = image_data["bytes"]
                    else:
                        continue

                    fname = f"ctw_{Path(orig_name).stem}.jpg"

                    with open(output_path / fname, "wb") as f:
                        f.write(img_bytes)

                    # Annotations might be in a column or separate.
                    # This mirror might be Image-only or Image-Text.
                    # We save what text we can find (if any column exists)
                    # Often these OCR datasets have 'ground_truth' or 'text'

    except Exception as e:
        print(f"CTW1500 extraction failed: {e}")

    print("CTW1500 Complete.")


if __name__ == "__main__":
    # Remove old dirs if you want a clean slate
    # shutil.rmtree("Batch_3", ignore_errors=True)
    unpack_iam()  # Batch 5
