# Tzefa
<img width="2712" height="1207" alt="image" src="https://github.com/user-attachments/assets/fbb52b28-6673-421a-bf7b-604048e0b078" />


**Tzefa** is an end-to-end system that photographs handwritten code on a whiteboard, recognizes it via a custom OCR pipeline, compiles it through a custom programming language, and executes it — all from a single image upload in a self-hosted web UI.

---

## Pipeline

```
Image Upload (Flask web UI)
    │
    ├─ Stage 1: Binarization ──────── HighResMAnet (mit_b5), tiled 640×640, custom trained
    ├─ Stage 2: Line Segmentation ─── YOLO-OBB XL, resize to 640×640, custom trained
    ├─ Stage 3: Word Segmentation ─── Morphological dilation (repeated small kernel → 3 components)
    ├─ Stage 4: Word OCR ──────────── Fine-tuned TrOCR (custom trained word model)
    ├─ Stage 5: Error Correction ──── Edit distance matching against Tzefa vocabulary
    ├─ Stage 6: Compilation ───────── Tzefa instructions → Python code generation
    └─ Stage 7: Execution ─────────── Subprocess with 15s timeout, captures stdout
```

All three DL models (Binarization, Line Segmentation, Word OCR) are custom trained. No stock/pretrained models in the inference path.

---

## Project Structure

| Directory | Purpose |
|-----------|---------|
| `Tzefa_Ocr/` | Inference pipeline: binarization, line/word segmentation, OCR, pipeline orchestration |
| `Tzefa_Language/` | Tzefa programming language: compiler (`topy.py`), error correction, VM runtime (`createdpython.py`) |
| `Tzefa_Web/` | Flask web server + HTML template with toggle views for every pipeline stage |
| `Tzefa_Ocr_Training/` | Training scripts and dataset generators for all models |
| `Tzefa_Models/` | Trained model checkpoints (Binarization, Line Segmentation, Word Model) |
| `Tzefa_Datasets/` | Training datasets (Binarization, Line Segmentation) |

### Key Files

| File | Role |
|------|------|
| `Tzefa_Ocr/main.py` | Pipeline orchestrator — 7 stages with crash isolation |
| `Tzefa_Ocr/Binarization.py` | HighResMAnet (mit_b5) with high-res stem, tiled inference |
| `Tzefa_Ocr/Line_Segmentation.py` | YOLO-OBB, resize to 640×640, Y-axis detection + X scaling |
| `Tzefa_Ocr/image_preprocessing.py` | Word segmentation via repeated dilation to exactly 3 components |
| `Tzefa_Ocr/OCR.py` | Lazy-loaded fine-tuned TrOCR word model |
| `Tzefa_Language/ErrorCorrection.py` | Edit distance correction against Tzefa vocabulary per argument type |
| `Tzefa_Language/topy.py` | Compiler: Tzefa instructions → Python code strings |
| `Tzefa_Language/createdpython.py` | VM runtime: variables, lists, conditions, functions, I/O |
| `Tzefa_Web/server.py` | Flask server with base64 image rendering and toggle UI |

---

## Web UI

The self-hosted web page at `localhost:5000` provides:

**Left panel (Image):**
- Binarized image
- Binarized + line bounding boxes overlay
- Original uploaded image

**Right panel (Text):**
- Raw OCR output (what TrOCR read)
- Error-corrected output (matched to Tzefa vocabulary)
- Compiled Python code
- Execution output (stdout from running the generated program)

Each view is toggled via buttons. If the pipeline crashes at any stage, all prior results remain visible.

---

## Current Status (March 2026)

### Working
- ✅ Full 7-stage pipeline runs end-to-end from image to execution
- ✅ Binarization model performs well (HighResMAnet mit_b5)
- ✅ Word OCR model performs well (fine-tuned TrOCR)
- ✅ Error correction corrects all 3 tokens per line (command + arg1 + arg2)
- ✅ Compilation and execution with subprocess isolation
- ✅ Web UI with full toggle views for debugging every stage
- ✅ Word segmentation via repeated dilation enforces exactly 3 words per line

### Needs Work
- ⚠️ **Line Segmentation model** is the weakest link — detection accuracy requires X-axis padding compensation due to 640×640 squash. Needs more training data and/or architectural improvements.
- ⚠️ **ErrorCorrection + topy global state** requires `importlib.reload()` between runs
- ⚠️ **createdpython.py** has 3 error-handler call-site bugs (wrong arg counts)

---

## Getting Started

```bash
# 1. Install dependencies
pip install -r Tzefa_Ocr/requirements.txt

# 2. Ensure model checkpoints exist at paths in:
#    Tzefa_Ocr/Binarization.py (DEFAULT_CHECKPOINT_PATH)
#    Tzefa_Ocr/Line_Segmentation.py (MODEL_PATH)
#    Tzefa_Ocr/OCR.py (WORD_MODEL_PATH)

# 3. Run the web server
python Tzefa_Web/server.py

# 4. Open http://localhost:5000, upload a handwritten Tzefa program image
```

---

## Tzefa Language

Every instruction is exactly 3 tokens: `COMMAND ARG1 ARG2`

Examples:
```
MAKEINTEGER NUMY FIVE       -- create integer NUMY with value 5
MULTIPLY RESULT BIGLY       -- RESULT = RESULT * BIGLY
SUBTRACT NUMY ONE           -- NUMY = NUMY - ONE
WHILETRUE JUSTBIGGER FOURTEEN  -- while JUSTBIGGER is true, loop until line 14
PRINTINTEGER TEMPORARY BREAK   -- print TEMPORARY with newline
```

Numbers are written as words (ZERO through ONEHUNDRED) to maximize OCR error correction coverage. The compiler (`topy.py`) generates Python code that runs on the Tzefa VM (`createdpython.py`).
