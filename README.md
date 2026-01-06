# Tzefa: Programming Language & OCR System
I didnt write the readme its just a filler for now 

Tzefa is an innovative project that combines a custom-designed programming language with an Optical Character Recognition (OCR) system. This allows users to write Tzefa code, capture it as an image, and then have the system recognize, interpret, and execute the code. The project also includes tools for training the OCR models and a graphical user interface (GUI) for a more interactive experience.

## Table of Contents

-   [Overview](#overview)
-   [Project Structure](#project-structure)
-   [Core Components](#core-components)
    -   [Tzefa Language (`Tzefa-Language/`)](#tzefa-language-tzefa-language)
    -   [Tzefa OCR (`Tzefa-Ocr/`)](#tzefa-ocr-tzefa-ocr)
    -   [OCR Model Training (`Tzefa-Ocr-Training/`)](#ocr-model-training-tzefa-ocr-training)
    -   [Tzefa GUI Application (`Tzefa-App/`)](#tzefa-gui-application-tzefa-app)
-   [Workflow](#workflow)
-   [Setup](#setup)
-   [Usage](#usage)
-   [Key Technologies](#key-technologies)
-   [Future Development / TODO](#future-development--todo)
-   [Contributing](#contributing)

## Overview

The Tzefa project aims to provide a unique coding experience where physical or digital images of code can be brought to life. It consists of several interconnected modules:

1.  **The Tzefa Programming Language**: A custom language with its own syntax and interpreter/compiler.
2.  **The Tzefa OCR System**: A pipeline to process images of Tzefa code, segment them, and recognize the characters.
3.  **OCR Model Training**: Scripts and tools to train machine learning models for accurate text recognition of Tzefa code.
4.  **Tzefa GUI Application**: A user-friendly interface to interact with the OCR and language execution functionalities.

## Project Structure

```
tzefa/
├── README.md                 # This main README file
├── TODO.md                   # Project-wide to-do list
│
├── Tzefa-Language/           # Tzefa programming language interpreter/compiler
│   ├── main.py               # Entry point for language execution
│   ├── topy.py               # Tzefa to Python compiler/translator
│   ├── ErrorCorrection.py    # Language-specific error correction
│   └── README.md             # (Recommended) Detailed language documentation
│
├── Tzefa-Ocr/                # OCR processing module
│   ├── main.py               # Entry point for OCR processing
│   ├── image_preprocessing.py # Image binarization, line/word segmentation
│   ├── ocr.py                # Text recognition from word images
│   └── requirements.txt      # Python dependencies for OCR
│
├── Tzefa-Ocr-Training/       # Scripts and tools for training OCR models
│   ├── Words-Model-Training.py # Script for training word recognition models
│   ├── Numbers-Model-Training.py # Script for training number recognition models
│   └── Captcha-Generator.py  # Tool for generating synthetic training data
│
└── Tzefa-App/                # GUI Application
    ├── Tzefa-Gui.py          # Main GUI application file
    ├── Languageinterface.py  # Interface for language module
    └── Ocrinterface.py       # Interface for OCR module
```

## Core Components

### Tzefa Language (`Tzefa-Language/`)

This module is responsible for understanding and executing Tzefa code.
-   **Interpreter/Compiler (`topy.py`, `main.py`)**: Parses and runs Tzefa code.
-   **Error Correction (`ErrorCorrection.py`)**: Provides language-specific error handling and suggestions.
-   *(Refer to `Tzefa-Language/README.md` for detailed language specifications if available).*

### Tzefa OCR (`Tzefa-Ocr/`)

This module extracts Tzefa code from images.
-   **Image Preprocessing (`image_preprocessing.py`)**:
    -   Binarization: Converts images to black and white. (Currently uses placeholder models, needs robust deep learning integration).
    -   Line Segmentation: Identifies lines of text. (Currently uses placeholder models, needs robust deep learning integration).
    -   Word Segmentation: Uses a dilation-based method to segment lines into three words (a Tzefa language characteristic).
-   **Text Recognition (`ocr.py`)**: Performs OCR on segmented word images. (Currently a placeholder, needs OCR model integration like TrOCR).

### OCR Model Training (`Tzefa-Ocr-Training/`)

This module contains the necessary tools to create datasets and train custom OCR models tailored for the Tzefa language's specific characters or fonts.
-   **Dataset Generation (`Captcha-Generator.py`, `Words-Dataset-Generator.py`, `Numbers-Dataset-Generator.py`)**: Tools to create synthetic images of Tzefa words and numbers.
-   **Model Training Scripts (`Words-Model-Training.py`, `Numbers-Model-Training.py`)**: Scripts to train models (e.g., TrOCR) on the generated datasets.

### Tzefa GUI Application (`Tzefa-App/`)

Provides a graphical interface for users to:
-   Load images of Tzefa code.
-   Visualize OCR results.
-   Edit recognized code.
-   Execute Tzefa code and see the output.

## Workflow

1.  **Write Tzefa Code**: User writes code in the Tzefa language.
2.  **Capture Image**: The code is captured as an image (e.g., a photo or screenshot).
3.  **OCR Processing (`Tzefa-Ocr/`)**:
    -   The image is loaded into the system.
    -   It undergoes preprocessing (binarization, segmentation).
    -   The OCR engine recognizes characters in the segmented words.
4.  **Code Interpretation/Execution (`Tzefa-Language/`)**:
    -   The recognized text (Tzefa code) is passed to the language module.
    -   The code is parsed, potentially corrected, and executed.
5.  **Output**: The results of the Tzefa code execution are displayed.
    *(The GUI application in `Tzefa-App/` can facilitate this entire workflow.)*

## Setup

Each module might have its own specific setup instructions and dependencies.

1.  **General (Root Directory)**:
    ```bash
    # Clone the repository (if you haven't already)
    # git clone <repository-url>
    # cd tzefa
    ```

2.  **Tzefa OCR (`Tzefa-Ocr/`)**:
    ```bash
    cd Tzefa-Ocr
    pip install -r requirements.txt
    cd ..
    ```

3.  **Tzefa Language, OCR Training, App**:
    -   These modules may have their own `requirements.txt` or specific setup steps. Please refer to any README files within those directories or ensure necessary dependencies (like PyTorch for training, PyQt/Tkinter for GUI) are installed in your environment.

## Usage

-   **Running the OCR**:
    ```bash
    python Tzefa-Ocr/main.py path/to/your/tzefa_image.png
    ```
-   **Running the Tzefa Language Interpreter** (Example, actual command might vary):
    ```bash
    python Tzefa-Language/main.py path/to/your/tzefa_code_file.tzefa
    ```
-   **Running the GUI Application**:
    ```bash
    python Tzefa-App/Tzefa-Gui.py
    ```
-   **Training OCR Models**:
    -   Follow instructions or run scripts within the `Tzefa-Ocr-Training/` directory.

## Key Technologies

-   **Python**: Core programming language for all modules.
-   **OpenCV**: For image processing tasks in the OCR module.
-   **PyTorch & Transformers (Hugging Face)**: For deep learning model implementation and training (intended for OCR).
-   **(GUI Framework)**: Such as PyQt, Tkinter, or Kivy for `Tzefa-App/`.
-   **(Version Control)**: Git.

## Future Development / TODO

(Refer to `TODO.md` for a detailed list. High-level goals include:)

-   **Language Module**:
    -   Expand language features and standard library.
    -   Improve compiler/interpreter performance and error reporting.
-   **OCR Module**:
    -   Integrate robust deep learning models for binarization (e.g., U-Net, T2T-BinFormer).
    -   Integrate robust deep learning models for line segmentation.
    -   Fully implement and integrate a trained TrOCR (or similar) model in `ocr.py`.
    -   Develop a comprehensive testing and evaluation suite for OCR accuracy.
-   **OCR Training Module**:
    -   Enhance dataset generation capabilities.
    -   Streamline the model training and fine-tuning process.
-   **GUI Application**:
    -   Improve user experience and add more features (e.g., real-time OCR, debugging tools).
-   **Overall**:
    -   Improve integration between modules.
    -   Create comprehensive documentation for each module.

## Contributing

Contributions to the Tzefa project are welcome! Please follow these steps:
1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes.
4.  Commit your changes and push them to your fork.
5.  Submit a pull request.

Please ensure your code adheres to any existing coding standards and include tests for new features.
