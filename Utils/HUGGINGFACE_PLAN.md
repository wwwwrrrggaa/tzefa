# Hugging Face Publishing — Tzefa

> Current state of all Tzefa HuggingFace repos and how to maintain them.

---

## HF Repo Inventory

| HF Repo | Type | Status |
|---------|------|--------|
| `WARAJA/b5_model` | Model (Binarization mit_b5 HighResMAnet) | Uploaded |
| `WARAJA/Model` | Space (holds b3_model.pth) | Pre-existing |
| `WARAJA/Tzefa-Line-Segmentation-YOLO` | Model (YOLO11x-OBB) | Uploaded |
| `WARAJA/Tzefa-Word-OCR-TrOCR` | Model (fine-tuned TrOCR) | Uploaded |
| `WARAJA/Tzefa-Binarization-Dataset` | Dataset (40k image+mask pairs) | Card uploaded, **bulk data pending** |
| `WARAJA/Tzefa-Line-Segmentation-Dataset` | Dataset (YOLO OBB format) | Card uploaded, **bulk data pending** |
| `WARAJA/Tzefa-Word-OCR-Dataset` | Dataset (links to existing) | Card uploaded |
| `WARAJA/Tzefa-Binarization` | Space (Binarization-only demo) | Pre-existing |
| `WARAJA/Tzefa` | Space (Full pipeline demo) | Uploaded |

All repos are **private** with **cc-by-nc-3.0** license.

---

## Project Structure

```
Tzefa_HF/
  Space/              -- Complete Gradio Space (uploaded to WARAJA/Tzefa)
    app.py            -- Full pipeline: binarize -> line-seg -> word-seg -> OCR -> correction -> compile -> execute
    requirements.txt
    README.md         -- HF Space card (YAML frontmatter + description)
    demo.png          -- Example image for gr.Examples
    language/         -- Patched copies of Tzefa_Language/ (imports adjusted)
      __init__.py
      Number2Name.py
      ErrorCorrection.py
      topy.py
      createdpython.py
  scripts/
    upload_space.py   -- Patches language modules + uploads Space to HF
    upload_models.py  -- Uploads model weights to their HF repos
    upload_datasets.py -- Uploads bulk dataset files (40k+ images, long-running)
  logs/               -- Upload logs
```

---

## Modular Design

All models are loaded from their own HF repos via `hf_hub_download()` / `from_pretrained()`.
To update any model, just push new weights to the corresponding repo — the Space picks them
up automatically on next run.

| Pipeline Stage | Model Source |
|---------------|-------------|
| Binarization (b3) | `hf_hub_download("WARAJA/Model", "b3_model.pth", repo_type="space")` |
| Binarization (b5) | `hf_hub_download("WARAJA/b5_model", "b5_model.pth")` |
| Line Segmentation | `hf_hub_download("WARAJA/Tzefa-Line-Segmentation-YOLO", "best.pt")` |
| Word OCR | `VisionEncoderDecoderModel.from_pretrained("WARAJA/Tzefa-Word-OCR-TrOCR")` |

---

## How to Update

### Update a model
1. Train new weights locally
2. Run: `python Tzefa_HF/scripts/upload_models.py`
3. The Space will use the new weights on next request

### Update the Space code
1. Edit files in `Tzefa_HF/Space/`
2. Run: `python Tzefa_HF/scripts/upload_space.py`
   - This automatically patches `Tzefa_Language/` -> `language/` imports
   - Copies latest language files before upload

### Upload dataset files
1. Run: `python Tzefa_HF/scripts/upload_datasets.py`
   - Binarization: uploads `Tzefa_Datasets/Binarization/Unified_Batch/` (images/ + masks/)
   - Line Seg: uploads `Tzefa_Datasets/Line_Segmentation/Unified_Batch/` (train/ + val/)
   - **This is long-running** (~40k binarization + ~25k line-seg files)

---

## Manual Steps Still Needed

- [ ] Set GPU hardware on `WARAJA/Tzefa` Space (Settings > Hardware > T4 Small or ZeroGPU)
- [ ] Add `HF_TOKEN` secret to the Space (Settings > Secrets > add your HF token as `HF_TOKEN`)
- [ ] Run `python Tzefa_HF/scripts/upload_datasets.py` to upload bulk dataset files
- [ ] Test the Space end-to-end
- [ ] Make repos public when ready (Settings > change visibility)

