---
title: Tzefa
emoji: "\U0001F40D"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "5.8.0"
app_file: app.py
pinned: true
license: cc-by-nc-3.0
pipeline_tag: image-to-text
tags:
- vision
- ocr
- document-processing
- binarization
- yolo
- trocr
- handwriting-recognition
- programming-language
- compiler
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/6645e2ce9c6ed6c615e56bf0/eTdBX9sR1-qzBuEPVWwZA.jpeg
---

# Tzefa - Handwritten Code to Execution

**Tzefa** is an end-to-end system that photographs handwritten code on a whiteboard,
recognizes it via a custom OCR pipeline, compiles it through a custom programming language,
and executes it -- all from a single image upload.

## Pipeline
```
Image Upload
  |-- Stage 1: Binarization        -- HighResMAnet (mit_b5), tiled 640x640
  |-- Stage 2: Line Segmentation   -- YOLO11x-OBB, oriented bounding boxes
  |-- Stage 3: Word Segmentation   -- Morphological dilation (exactly 3 words/line)
  |-- Stage 4: Word OCR            -- Fine-tuned TrOCR
  |-- Stage 5: Error Correction    -- Edit-distance matching against Tzefa vocabulary
  |-- Stage 6: Compilation         -- Tzefa instructions -> Python code
  '-- Stage 7: Execution           -- Subprocess with 15s timeout
```

## Modular Design
All models load from their own HuggingFace repos. Push new weights to any repo
and this Space picks them up on next run.

| Component | Model Repo |
|-----------|-----------|
| Binarization (b3) | [WARAJA/Model](https://huggingface.co/spaces/WARAJA/Model) |
| Binarization (b5) | [WARAJA/b5_model](https://huggingface.co/WARAJA/b5_model) |
| Line Segmentation | [WARAJA/Tzefa-Line-Segmentation-YOLO](https://huggingface.co/WARAJA/Tzefa-Line-Segmentation-YOLO) |
| Word OCR | [WARAJA/Tzefa-Word-OCR-TrOCR](https://huggingface.co/WARAJA/Tzefa-Word-OCR-TrOCR) |

## Tzefa Language
Every instruction is exactly 3 tokens: `COMMAND ARG1 ARG2`

```
MAKEINTEGER NUMY FIVE       -- create integer NUMY = 5
MULTIPLY RESULT BIGLY       -- RESULT = RESULT * BIGLY
PRINTINTEGER TEMPORARY BREAK -- print TEMPORARY with newline
```

Numbers are written as words (ZERO through ONEHUNDRED) to maximize OCR error correction.

## Related
- [Binarization Demo](https://huggingface.co/spaces/WARAJA/Tzefa-Binarization)
- [Binarization Dataset](https://huggingface.co/datasets/WARAJA/Tzefa-Binarization-Dataset)
- [Line Segmentation Dataset](https://huggingface.co/datasets/WARAJA/Tzefa-Line-Segmentation-Dataset)
- [Word OCR Dataset](https://huggingface.co/datasets/WARAJA/Tzefa-Word-OCR-Dataset)

