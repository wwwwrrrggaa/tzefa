# Tzefa
<img width="2712" height="1207" alt="image" src="https://github.com/user-attachments/assets/fbb52b28-6673-421a-bf7b-604048e0b078" />


**Tzefa** is an end-to-end system that photographs handwritten code on a whiteboard, recognizes it via a custom OCR pipeline, compiles it through a custom programming language, and executes it — all from a single image upload in a self-hosted web UI.
Its goal is to be a prototype for a joint dsl/ocr system that complement eachother
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

All three DL models (Binarization, Line Segmentation, Word OCR) are custom trained.

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
- ✅ Error correction with edit-distance matching against full vocabulary
- ✅ Dual-dialect support: **3-word** (classic) and **4-word** (verbose) compile to same bytecode
- ✅ Casing modes: **CAPS_ONLY** and **MIXED_CASE** (title-cased commands, lowercase vars)
- ✅ Compilation and execution with subprocess isolation
- ✅ Web UI with full toggle views for debugging every stage + dialect/casing toggles
- ✅ Type system with functions, loops, conditions, lists, arithmetic
- ✅ PEP 8 compliance, type hints throughout, clean function/variable names
- ✅ Comprehensive documentation: `SYNTAX_GUIDE.md`, `ARCHITECTURE.md`, `COMPARISON.md`

### Needs Work
- ⚠️ **Line Segmentation model** is the weakest link — detection accuracy requires X-axis padding compensation due to 640×640 squash. Needs more training data and/or architectural improvements.
- ⚠️ **Dialect/casing toggle** in web UI not yet wired to backend (`server.py` needs POST parameter handling)
- ⚠️ **Word segmentation** currently enforces 3 words per line; should adapt to dialect choice

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

Tzefa is a **typed, stack-based language** designed for handwritten OCR recognition. Source code supports **two dialects** that compile to the same **4-word bytecode**:

### **3-Word Dialect (Classic)** – `OPCODE ARG1 ARG2`
```
MAKEINTEGER COUNTER FIVE       -- create integer COUNTER with value 5
ADDVALUES COUNTER ONE          -- TEMPORARY = COUNTER + 1
ASSSIGNINT COUNTER TEMPORARY   -- COUNTER = TEMPORARY
PRINTINTEGER COUNTER BREAK     -- print COUNTER with newline
```

### **4-Word Dialect (Verbose)** – `VERB TYPE ARG1 ARG2`
```
MAKE INTEGER COUNTER FIVE      -- create integer COUNTER with value 5
ADD RESULT COUNTER ONE         -- RESULT = COUNTER + 1
SET INTEGER COUNTER RESULT     -- COUNTER = RESULT
PRINT INTEGER COUNTER BREAK    -- print COUNTER with newline
```

**Key features:**
- Numbers written as English words (`ZERO` through `ONEHUNDRED`) for OCR reliability
- Full type system: `INTEGER`, `STRING`, `BOOLEAN`, `LIST`
- Functions with single input/output via `LOCALINT`, `LOCALSTR`, `LOCALLIST`
- Loop constructs: `WHILE`, `ITERATE` with end-line markers
- Conditions: `NEW CONDITION` with `EQUALS`, `BIGEQUALS`, `BIGGER` operators
- List operations: fixed-size typed containers with index-based access
- Two casing modes: `CAPS_ONLY` (all uppercase) and `MIXED_CASE` (commands title-cased, vars lowercase)

The compiler (`topy.py`) transpiles to Python code executed by the Tzefa VM (`createdpython.py`). See `SYNTAX_GUIDE.md` and `ARCHITECTURE.md` for complete documentation.
