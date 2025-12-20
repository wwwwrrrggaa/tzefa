# Tzefa Project

Tzefa is a multi-component project aimed at creating an integrated system for extracting code from images using Optical Character Recognition (OCR), interpreting this code via a custom programming language ("Tzefa"), and providing tools for image processing and model training.

## Core Components

The project is organized into the following main modules:

*   **`Tzefa_Ocr/`**:
    *   Handles the OCR inference pipeline.
    *   **Current State**: Uses `PaddleOCR` for unwarping and `Sauvola` binarization (`image_preprocessing.py`).
    *   **Future State**: Transitioning to custom Deep Learning models for:
        *   **Binarization**: Cleaning and preparing document images (replacing `sbb_binarize`).
        *   **Line Segmentation**: Detecting and isolating lines of code.
        *   **Text Recognition**: Transformer-based models (TrOCR) for recognizing text (`OCR.py`).

*   **`Tzefa_Ocr_Training/`**:
    *   Contains training scripts and dataset generators for the OCR pipeline.
    *   **Binarization**: `Binarization/` contains `Model.py`, `Training.py`, and `Data.py` for the new custom cleaning model.
    *   **Line-Segementation**: Placeholder for future segmentation model development.
    *   **Word/Number Models**: Tools for generating synthetic data and training recognition models (`Words-Dataset-Generator.py`, `Numbers-Dataset-Generator.py`).

*   **`Tzefa_Language/`**:
    *   Implements the "Tzefa" custom programming language.
    *   **Compiler**: `topy.py` translates Tzefa instructions (e.g., `MAKEINTEGER`, `NEWLIST`) into executable Python code.
    *   **Error Correction**: `ErrorCorrection.py` uses `fast_edit_distance` to correct OCR errors based on defined language grammar (`listfunctions`).
    *   **Planned Features**:
        *   **Immediate Types**: shifting from digits to words (e.g., "SEVENTEEN") to maximize error correction coverage.
        *   **Custom Levenshtein**: Weighted distance metrics to handle specific OCR confusions (e.g., 'l' vs 'I').
        *   **Usability**: Support for lowercase and simplified function names.
    *   Key files: `main.py` (entry point), `topy.py`, `ErrorCorrection.py`.

*   **`Tzefa_Web/`**:
    *   A Flask-based web application (`app.py`) that provides a user interface for:
        *   Uploading images.
        *   Performing OCR (currently using `Tzefa_Ocr` pipeline).
        *   Displaying results.
    *   **Server Interface**: Plans to expose a dedicated API/Server interface.

*   **`Tzefa_Datasets/`** & **`Tzefa_Models/`**:
    *   Directories for managing datasets (images, masks) and storing trained model checkpoints.

## Overall Goal

The broader vision for Tzefa includes:
*   A complete end-to-end system for executing handwritten code from images.
*   An OCR system specifically optimized for Tzefa code syntax.
*   A runtime environment for the Tzefa language.
*   Solutions aimed at educational tools for learning programming through handwritten code(Automatic test checking,Executing code from white board at class,Data visualization tools,Llm debugger,Ipad Ide,and more) .

## Getting Started

Each component directory may contain its own `README.md` and `requirements.txt` for specific setup and usage instructions. The `Tzefa-Web` application can serve as an initial entry point for testing the image-to-code pipeline.