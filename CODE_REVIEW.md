# Tzefa — Code Review

*March 2, 2026 — Post-session review*

---

## Summary

Tzefa is a 7-stage pipeline: image → binarization → line segmentation → word segmentation → OCR → error correction → compilation → execution. The pipeline runs end-to-end and produces correct output. The primary bottleneck is the line segmentation model quality, which required several inference-time workarounds (640×640 squash, X-axis padding, repeated dilation to force 3 words).

---

## File-by-File Review

### `Tzefa_Ocr/main.py` (287 lines) — ✅ Good
Pipeline orchestrator with 7 stages, each wrapped in try/except so that if a later stage crashes, earlier results (e.g. binarized image) are still visible in the UI. `importlib.reload()` resets ErrorCorrection/topy global state between runs.

**Notes:**
- Stage 4 uppercases all OCR tokens and corrects all 3 tokens (command, arg1, arg2) against the right vocabulary list — this was a bug fix this session
- Stage 7 runs compiled code in a subprocess with 15s timeout — clean isolation
- `_execute_compiled_code` uses `subprocess.run` with `cwd=project_root` so imports resolve correctly

### `Tzefa_Ocr/Binarization.py` (165 lines) — ✅ Solid
Custom `HighResMAnet` architecture (mit_b5 encoder + high-res stem + fusion head). Tiled inference at 640×640 with batching. Clean memory lifecycle (model loaded in constructor, VRAM freed by caller).

**Notes:**
- `torch.amp.autocast` guarded for CPU with `enabled=(self.device == "cuda")`
- `channels_last` memory format for inference speed — good practice

### `Tzefa_Ocr/Line_Segmentation.py` (62 lines) — ⚠️ Works but Fragile
Force resizes to 640×640 (squashing aspect ratio), runs YOLO-OBB inference, scales Y coordinates back accurately, applies 50% X-axis padding to compensate for squash-induced narrow detections.

**Issues:**
- YOLO model reloaded from disk every call — should cache
- 50% X padding is a tuned constant that may break on different image aspect ratios
- The fundamental problem is the model was trained on letterboxed images but runs on squashed input. Training on squashed images (or letterboxing at inference time with correct coordinate unmapping) would eliminate the X padding hack.

### `Tzefa_Ocr/image_preprocessing.py` (116 lines) — ✅ Good
`linestowords`: Small fixed kernel `(5,3)`, repeatedly dilated on the same image until exactly 3 connected components remain. If it overshoots (goes from >3 to <3), takes the last >3 result and merges closest pairs down to 3. Up to 200 iterations.

**Notes:**
- Noise filtering by minimum word width/height
- This approach is resolution-independent — the fixed kernel accumulates naturally regardless of image size
- Only skips a line if the very first iteration already produces <3 components (barely any content)

### `Tzefa_Ocr/OCR.py` (51 lines) — ✅ Clean
Lazy-loaded TrOCR word model. Only loads on first `ocr_word()` call, not on import. Uses `use_fast=False` for the processor (prevents tokenizer conversion crash). Single model (word only — number model not yet trained).

### `Tzefa_Language/ErrorCorrection.py` (362 lines) — ⚠️ Works but Needs Refactor
Vocabulary-based error correction using `fast_edit_distance`. Each token is corrected against its type-specific vocabulary list (integers, strings, lists, conditions, number names, etc.). The `toline()` function parses corrected lines and produces 3-element instruction lists.

**Bugs fixed this session:**
- `updatesizelistofindnets` missing `global` keyword — was creating a local variable
- `findword` shadowed `min` builtin, had unsafe 3-element default return
- `sendlines` undersized `listofindents` array — caused index out of range
- Control flow indent check used `"COMPARE" in name` which caught `CHANGECOMPARE` — now uses explicit set

**Remaining issues:**
- 15+ module-level globals mutated during compilation (`counter`, `thetype`, `insidefunction`, `listfunctions`, etc.)
- `importlib.reload()` is the only way to reset state between runs
- `tosimple()` uses a mixed-type list (initialized as strings, then overwritten with ints) — confuses static analysis

### `Tzefa_Language/topy.py` (456 lines) — ⚠️ Works, Some Fixed Bugs
Code generator: maps Tzefa instructions to Python code strings. Each instruction has a dedicated function (MAKEINTEGER, WHILE, COMPARE, etc.) that returns a string of Python code.

**Bugs fixed this session:**
- `SIMPLEDIVIDE` was dispatched to `COPYLIST` function — programs using simple division silently did list copy
- `PRINTSTRING` used bare `name` instead of `tostri(name)` — generated invalid Python code
- `PRINTSTRING` had inverted BREAK/STAY logic vs `PRINTINTEGER`
- Duplicate dispatch entries (ADDSIZE, COPYLIST registered twice)

**Remaining issues:**
- Parameter names shadow builtins: `type`, `bool` used in multiple functions
- `listofindentchanges` is module-level mutable state (same reload problem as ErrorCorrection)
- `dictofinstructions` bootstrapped with `{i: "thetext" ...}` then overwritten with functions — confuses type analysis

### `Tzefa_Language/createdpython.py` (763 lines) — ⚠️ Has Known Bugs
The Tzefa VM runtime. Implements variables (VALUE), lists (LIST), conditions (COND), functions, a call stack, and I/O. Generated Python code calls into this module's functions.

**Known bugs (not yet fixed):**
- `errore.makeeindexrror(newindex, self.size, self.name)` — expects 4 args, called with 3 (lines ~499, ~507)
- `errore.readerror()` — expects 1 arg, called with 0 (line ~528)
- `COND.givetype()` references `self.type` which is never assigned in `__init__`

**Other issues:**
- `LIST.__init__` had `self.currentindex` where all methods used `self.index` — fixed this session
- `pow()` function shadows `builtins.pow`
- All VM state is module-level — subprocess execution isolates this but in-process re-runs get corrupt state

### `Tzefa_Language/Number2Name.py` — ✅ Clean, Limited
Maps integers 0–100 to uppercase word names. Hardcoded limit — programs needing values >100 can't express them.

### `Tzefa_Web/server.py` (147 lines) — ✅ Clean
Flask server. Reads uploaded image, calls `image_to_code_pipeline()`, converts numpy arrays to base64 PNGs, draws bbox overlays, passes everything to the Jinja template. Error fallback dict includes all keys so template never crashes on missing data.

### `Tzefa_Web/templates/pipeline.html` (432 lines) — ✅ Clean
Toggle buttons for image views (binarized, bboxes, original) and text views (raw OCR, corrected, compiled, execution output). Spinner on submit. File name display. Clean CSS.

### `Tzefa_Web/app.py` — ❌ Delete
Imports from `TzefaOcr.image_preprocessing` (wrong package name). Always crashes on import.

### `Tzefa_Web/Tzefa_website.py` — ❌ Delete
Calls `"\n".join(result)` where `result` is a dict. Always crashes at runtime.

### `Tzefa_Ocr/requirements.txt` — ✅ Cleaned
Was 50+ lines including unused packages (paddleocr, easyocr, layoutlm, diffusers, vit-pytorch, etc.). Now 12 actual dependencies.

### `pyproject.toml` — ✅ Fixed
`requires-python` was `>=3.14` (doesn't exist). Now `>=3.11`.

---

## Architecture Observations

### What's Good
1. **Crash isolation** — each stage is independently try/excepted, so partial results always show up
2. **VRAM lifecycle** — sequential load/infer/flush, no model competition for GPU memory
3. **Subprocess execution** — naturally isolates createdpython.py global state problem
4. **Lazy loading** — OCR model only loads when first word needs recognition, not on import
5. **Repeated dilation** — resolution-independent word segmentation that always hits exactly 3 components
6. **3-token invariant** — enforced at word segmentation, trusted by all downstream stages. No fallback gymnastics.

### What's Fragile
1. **Line Segmentation X-axis** — 50% padding hack compensates for aspect ratio squash. Works for current test images but may break on different aspect ratios.
2. **Global state reset** — `importlib.reload()` is a hack. If any module holds a stale reference, second run produces garbage.
3. **Error handler call sites** — 3 call-site bugs in createdpython.py will crash when specific error conditions are triggered at runtime.

### Recommendation
The single highest-impact improvement would be **retraining the line segmentation model on binarized images at the correct aspect ratio** (letterboxed, not squashed). This would eliminate the X-axis padding hack, make the word segmentation's job easier, and improve the entire pipeline's end-to-end accuracy. Everything downstream of line segmentation is solid.

