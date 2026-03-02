import os
import cv2
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import segmentation_models_pytorch as smp
from ultralytics import YOLO
import matplotlib.pyplot as plt

# --- CONFIG ---
# Ensure this path points to your actual .pth file for Segformer
BINARIZATION_CKPT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Binarization\run_mitb3_unet_640\step_1290.pth"

# Pointing to your latest YOLO run
YOLO_CKPT = r"C:\dev\projects\PycharmProjects\tzefa\Tzefa_Models\Line_Segmentation\run_3\weights\best.pt"

TEST_IMAGE = r"E:\Storage\tests\test4.jpg"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Binarizer:
    def __init__(self, checkpoint_path):
        print("Loading Binarizer (Segformer / mit_b3)...")

        # CHANGED: Switched to Unet with mit_b3 encoder
        self.model = smp.Unet(
            encoder_name="mit_b3",
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
        )

        try:
            ckpt = torch.load(checkpoint_path, map_location=DEVICE)
            # Handle cases where checkpoint is a dict (state_dict) or the model itself
            state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            self.model.load_state_dict(state)
        except Exception as e:
            print(f"❌ Error loading Binarizer checkpoint: {e}")
            print("Check if the checkpoint path matches the model architecture (mit_b3).")
            exit(1)

        self.model.to(DEVICE)
        self.model.eval()

        # Standard ImageNet normalization (required for mit_b3)
        self.tf = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

    def clean(self, img_rgb):
        pil_img = Image.fromarray(img_rgb)

        # Resize to closest 32 multiple (Transformer encoders strictly require this)
        w, h = pil_img.size
        nw = (w // 32) * 32
        nh = (h // 32) * 32
        if (nw, nh) != (w, h):
            pil_img = pil_img.resize((nw, nh))

        inp = self.tf(pil_img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            out = self.model(inp)
            # Resize mask back to original size if needed (though resize above handles aspect ratio well)
            mask = (torch.sigmoid(out) > 0.5).long().squeeze().cpu().numpy()

        # 0=Text, 255=Bg (Inverting logic: mask 1 is text -> 0 black)
        return np.where(mask == 1, 0, 255).astype(np.uint8)


def run_pipeline():
    # 1. Load Image
    if not os.path.exists(TEST_IMAGE):
        print(f"Image not found: {TEST_IMAGE}")
        return

    raw_bgr = cv2.imread(TEST_IMAGE)
    raw_rgb = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)

    # 2. Binarize
    binarizer = Binarizer(BINARIZATION_CKPT)
    clean_img = binarizer.clean(raw_rgb)

    # Convert clean image to RGB for YOLO (YOLO expects 3 channels)
    clean_rgb = cv2.cvtColor(clean_img, cv2.COLOR_GRAY2RGB)

    # 3. Detect Lines
    print(f"Loading YOLO from {YOLO_CKPT}...")
    if not os.path.exists(YOLO_CKPT):
        print("⚠️ YOLO checkpoint not found! Falling back to yolov8n-obb.pt for testing.")
        model_path = "yolov8n-obb.pt"
    else:
        model_path = YOLO_CKPT

    yolo = YOLO(model_path)

    # Run Inference
    # Note: increased imgsz to 1024 for better small text detection
    print("Running Inference...")
    results = yolo.predict(clean_rgb, imgsz=1024, conf=0.25, verbose=False)

    # 4. Visualize
    res_plotted = results[0].plot()  # YOLO's plotting

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(raw_rgb)
    plt.title("1. Raw Input")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(clean_img, cmap="gray")
    plt.title("2. Binarized (Segformer mit_b3)")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(res_plotted)
    plt.title("3. YOLO OBB Result")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_pipeline()