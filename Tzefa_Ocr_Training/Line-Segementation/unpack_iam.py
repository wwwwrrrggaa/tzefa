import os
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from tqdm import tqdm
import cv2
import shutil

# --- CONFIGURATION ---
# 1. Path to where you extracted the Kaggle "forms" (images)
IMAGES_SOURCE_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\IAM_Forms\data"

# 2. Path to where you extracted the Kaggle "xml" files
XML_SOURCE_DIR = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\temp\xml_fle"

# 3. Output Directory
TARGET_DIR = os.path.join(r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Datasets\Line_Segmentation","Batch_6")


def get_line_bbox_from_xml(xml_path):
    """
    Parses IAM XML to find line coordinates.
    Merges all words inside a line to get the full line bounding box.
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        line_bboxes = []

        # Iterate over every <line> tag
        for line in root.findall(".//line"):
            x_coords = []
            y_coords = []

            # Words are the reliable source of coords in IAM XML
            for word in line.findall("word"):
                cmp = word.find("cmp")  # "cmp" = component
                if cmp is not None:
                    try:
                        x = int(cmp.get("x"))
                        y = int(cmp.get("y"))
                        w = int(cmp.get("width"))
                        h = int(cmp.get("height"))

                        # Add corners of this word
                        x_coords.append(x)
                        x_coords.append(x + w)
                        y_coords.append(y)
                        y_coords.append(y + h)
                    except (ValueError, TypeError):
                        continue

            # If we found words, create a bounding box for the whole line
            if x_coords and y_coords:
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                line_bboxes.append((min_x, min_y, max_x, max_y))

        return line_bboxes
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return []


def normalize_yolo_obb(x1, y1, x2, y2, img_w, img_h):
    """
    Converts standard Box (x1,y1, x2,y2) -> YOLO OBB Polygon (Normalized 0-1)
    Format: x1 y1 x2 y1 x2 y2 x1 y2 (Top-Left -> Top-Right -> Bottom-Right -> Bottom-Left)
    """
    poly = [
        x1,
        y1,  # Top Left
        x2,
        y1,  # Top Right
        x2,
        y2,  # Bottom Right
        x1,
        y2,  # Bottom Left
    ]
    # Normalize
    return [p / img_w if i % 2 == 0 else p / img_h for i, p in enumerate(poly)]


def process_iam():
    print(f"Matching images from: {IMAGES_SOURCE_DIR}")
    print(f"Matching XMLs from:   {XML_SOURCE_DIR}")

    output_path = Path(TARGET_DIR)
    images_out = output_path / "images"
    labels_out = output_path / "labels"

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Find all images (png, jpg)
    image_files = list(Path(IMAGES_SOURCE_DIR).rglob("*.png")) + list(Path(IMAGES_SOURCE_DIR).rglob("*.jpg"))

    print(f"Found {len(image_files)} images. Starting processing...")

    success_count = 0

    for img_path in tqdm(image_files):
        file_id = img_path.stem  # e.g. "a01-000u"

        # Try to find the matching XML
        # Kaggle XML folders sometimes have nested structure (xml/a01/a01-000u.xml)
        # We use rglob to find it anywhere inside XML_SOURCE_DIR
        matched_xmls = list(Path(XML_SOURCE_DIR).rglob(f"{file_id}.xml"))

        if not matched_xmls:
            # Skip if no label found
            continue

        xml_file = matched_xmls[0]

        # 1. Read Image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        # 2. Parse XML
        line_boxes = get_line_bbox_from_xml(xml_file)
        if not line_boxes:
            continue

        # 3. Create Label Content
        label_lines = []
        for box in line_boxes:
            # Convert to OBB Polygon
            poly = normalize_yolo_obb(*box, w, h)
            # Format: class_id x1 y1 x2 y2 ...
            poly_str = " ".join([f"{x:.6f}" for x in poly])
            label_lines.append(f"0 {poly_str}")

        # 4. Save Image & Label to Batch_6
        shutil.copy(img_path, images_out / f"iam_{file_id}.png")

        with open(labels_out / f"iam_{file_id}.txt", "w") as f:
            f.write("\n".join(label_lines))

        success_count += 1

    print(f"\n✅ Processing Complete.")
    print(f"Successfully paired {success_count} forms.")
    print(f"Saved to: {os.path.abspath(TARGET_DIR)}")

if __name__ == "__main__":
    process_iam()