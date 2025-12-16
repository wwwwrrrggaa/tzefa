from flask import Flask, render_template, request, redirect, url_for
import sys
import os
from PIL import Image
import cv2
import numpy as np
from werkzeug.utils import secure_filename

# --- Add Project Root to sys.path ---
current_web_dir = os.path.dirname(os.path.abspath(__file__))  # .../Tzefa_Web
project_root = os.path.dirname(current_web_dir)  # .../tzefa (parent of Tzefa_Web)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import from TzefaOcr package, now that project_root is in sys.path
    from TzefaOcr.image_preprocessing import preprocess_image
except ImportError as e:
    print(f"Error importing preprocessing module from {project_root}: {e}")

    # Define a dummy function if import fails
    def preprocess_image(image_path, output_image_path=None):
        print(
            f"CRITICAL ERROR: image_preprocessing module not found or import failed: {e}"
        )
        return None, []


app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(
    current_web_dir, "uploads"
)  # Ensure UPLOAD_FOLDER is absolute to Tzefa_Web
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file and file.filename:  # Ensure filename is not None
            # Secure the filename
            s_filename = secure_filename(file.filename)
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], s_filename)
            file.save(filepath)

            preprocess_result = process_uploaded_image(
                filepath
            )  # Changed to use the saved filepath

            return render_template(
                "index.html", filename=s_filename, preprocess_result=preprocess_result
            )
    return render_template("index.html", filename=None, preprocess_result=None)


def process_uploaded_image(image_filepath):
    """
    Processes an uploaded image file using preprocess_image.
    """
    print(f"Starting preprocessing for {image_filepath}...")

    # Define a path for the binarized image if you want to save/debug it
    base_filename = os.path.basename(image_filepath)
    binarized_image_filename = "binarized_" + base_filename
    binarized_image_save_path = os.path.join(
        app.config["UPLOAD_FOLDER"], binarized_image_filename
    )

    # Step 1: Preprocess the image
    binary_image_np, word_bboxes = preprocess_image(
        image_filepath, output_image_path=binarized_image_save_path
    )

    if binary_image_np is None:
        print("Error: Preprocessing failed, binary_image_np is None.")
        return "Image preprocessing failed."

    print(
        f"Preprocessing complete. Binarized image potentially saved to {binarized_image_save_path}."
    )
    return "Image preprocessing completed successfully."


if __name__ == "__main__":
    app.run(debug=True)
