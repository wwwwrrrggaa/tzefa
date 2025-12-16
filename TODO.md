## Project design:

- Progress needs to be made in all 3 branches together (ocr, language, app).
- OCR and language need to remain independent for debugging.
- **Current Major Shift**: Moving away from boilerplate image processing methods (Sauvola, PaddleOCR unwarping) to custom Deep Learning models for Binarization and Line Segmentation.

## To do current

### OCR Pipeline (Deep Learning Focus)
- [ ] **Binarization**:
    - [x] Create synthetic data for document binarization.
    - [x] Implement training loop (`Tzefa_Ocr_Training/Binarization/Training.py`).
    - [ ] Train DL model for document binarization (In Progress).
    - [ ] Evaluate and tune binarization model.
    - [ ] Replace `sbb_binarize` in `image_preprocessing.py` with new model inference.
- [ ] **Line Segmentation**:
    - [ ] Research and select architecture for line segmentation (currently empty folder).
    - [ ] Create/Synthesize dataset for line segmentation.
    - [ ] Train DL model for line segmentation.
- [ ] **Text Recognition**:
    - [ ] Continue improving TrOCR based models for words and numbers.

### Integration & App
- [ ] Integrate new DL Binarization model into `Tzefa_Ocr`.
- [ ] Integrate new Line Segmentation model into `Tzefa_Ocr`.
- [ ] Connect updated OCR pipeline with `Tzefa_Web` GUI.
- [ ] Implement multiprocessing for better UX.
- [ ] **Server Interface**: Develop a robust server interface/API for the backend.

### Language & Error Correction
- [ ] **Immediate Types**: Replace digit recognition with number words (e.g., "SEVENTEEN") to enable full error correction.
- [ ] **Lowercase Support**: Update compiler and OCR to support lowercase characters.
- [ ] **Syntax Improvements**: Simplify function names and make coding easier/more intuitive.
- [ ] **Custom Levenshtein**: Implement weighted Levenshtein distance for OCR error correction (e.g., low cost for 'l' vs 'I', high for 'X' vs 'O').
- [ ] Improve compiler (`topy.py`) and error-correction (`ErrorCorrection.py`).
- [ ] Expand `listfunctions` to support more language features.

## OCR Strategy:

    1. **Preprocessing (Binarization)**: Train a custom DL model to handle various lighting/noise conditions, replacing static algorithms like Sauvola.
    2. **Detection (Line Segmentation)**: Train a custom DL model to accurately segment lines of code, replacing heuristic methods.
    3. **Recognition**: Use TrOCR (already working well), expand dataset.

## GUI Strategy:

    1. Photo editing with adjustable filters.
    2. Visualization of detection output (lines).
    3. Error correction interface for recognized text.
    4. Code execution and editing.
