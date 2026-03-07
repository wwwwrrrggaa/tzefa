# Tzefa — TODO

*Updated: March 2, 2026*

---

## P0 — Fix Now

### VM Runtime Bugs (`createdpython.py`)
- [x] `makeeindexrror` called with 3 args, expects 4 (lines ~499, ~507) — crashes on list index-out-of-bounds
- [x] `readerror()` called with no args, expects 1 (line ~528) — crashes on unreadable variable access
- [x] `COND.givetype()` references `self.type` which is never set in `__init__` — AttributeError

### Files to Delete
- [ ] `Tzefa_Web/app.py` — broken, imports wrong package name, fully superseded by `server.py`
- [ ] `Tzefa_Web/Tzefa_website.py` — broken, calls `.join()` on a dict, fully superseded by `server.py`

---

## P1 — Do Soon

### Line Segmentation (Weakest Model)
The YOLO-OBB line segmentation model is the primary bottleneck. The current workarounds:
- Force resize to 640×640 (squashes aspect ratio) — gives good Y-axis detection
- X-axis needs 50% padding compensation because squash makes detections too narrow
- Word segmentation uses repeated dilation to enforce exactly 3 components per line

**Recommended improvements:**
- [ ] **Train on more data** — more handwritten Tzefa images with line-level annotations
- [ ] **Train on binarized images** — the model sees binarized input at inference time, train on the same distribution
- [ ] **Evaluate letterboxing** — train with aspect-ratio-preserving letterbox instead of squash, so detection widths are accurate without padding hacks
- [ ] **Cache the YOLO model** — currently reloaded from disk for every image (`load_model()` called in `segment_lines()`)


### Add `.gitignore`
- [ ] Exclude: `__pycache__/`, `uploads/`, `*.pyc`, `Tzefa_Models/` (multi-GB weights), `Tzefa_Datasets/`, `*.prof`

---

## P2 — Architecture Improvements

### Eliminate Global State
- [x] **Refactor `ErrorCorrection.py` into a class** — 15+ module-level globals (`counter`, `thetype`, `insidefunction`, `listfunctions`, `listezfunc`, `listall`, etc.) mutated during compilation. Currently requires `importlib.reload()` between web requests. One pipeline run = one class instance = clean state.
- [ ] **Refactor `topy.py` into a class** — same pattern. `listofindentchanges`, `infunction`, `dictofinstructions` become instance state.
- [ ] **Add reset to `createdpython.py`** — VM state (`allthevars`, `alltheconds`, stacks) is module-level. Subprocess execution isolates this for now, but in-process re-runs get corrupt state.

### Rename Builtin Shadows
- [ ] `type` → `var_type` (4+ functions in `topy.py`)
- [ ] `bool` → `bool_name` (3 functions in `topy.py`)
- [ ] `pow()` → `mathpow_impl()` in `createdpython.py` (shadows `builtins.pow`)

---

## P3 — Nice to Have

### Language
- [ ] Extend `Number2Name.py` beyond 100 (currently hardcoded 0–100)
- [ ] Rename typos: `ASSSIGNINT`→`ASSIGNINT`, `handelfirstword`→`handlefirstword`, `makeparenthasis`→`makeparenthesis`, `errore`→`error_handler`, `makeeindexrror`→`make_index_error`
- [ ] Clean up `Tzefa_Language/main.py` — uses `import test` for side-effect execution (module cache prevents re-runs)

### Web UI
- [ ] Add word-level bbox overlay (third image toggle) — shows where each word was cropped, helps debug OCR issues
- [ ] Add confidence scores per corrected line — show edit distance from raw OCR to corrected vocabulary
- [ ] Manual correction interface — let user fix OCR mistakes before compilation

### OCR Pipeline
- [ ] Batch word OCR inference (currently one word at a time)
- [ ] Add custom Levenshtein with weighted OCR confusion pairs (l/I, O/0, S/5, etc.) — `ocr_edit_distance` exists in ErrorCorrection.py but isn't used

---

## Future Vision

### Language Evolution
- [ ] **4-word syntax** — expand instruction format from `COMMAND ARG1 ARG2` to `COMMAND ARG1 ARG2 ARG3` for richer expressions
- [ ] **Lowercase support** — allow lowercase characters in the language and update compiler, OCR, and error correction accordingly

### Automated Testing / Education Platform
- [ ] **Teacher API** — Python API where a teacher defines a problem: predefined input variables the student receives, expected output, and test cases
- [ ] **Test runner** — automatically runs student's handwritten code against the test suite and reports pass/fail per test case
- [ ] **Classroom integration** — batch processing of student submissions with grading output

### iPad / Smart Pen IDE
- [ ] **Real-time recognition** — stream handwriting from iPad or smart pen directly into the OCR pipeline
- [ ] **Live feedback** — show error correction suggestions as the student writes
- [ ] **Hybrid IDE** — combine handwriting input with a code editor for corrections

### Data Visualization & LLM Debugging
- [ ] **Data visualization libraries** — add plotting/charting support to the Tzefa language (e.g. bar charts, line graphs for teaching data concepts)
- [ ] **LLM-powered debugging** — give an LLM context from inside the VM (variable state, execution trace, error messages) to provide natural language debugging hints to students

---

## Done (This Session — March 2, 2026)

- [x] Built self-hosted web UI with toggle views for every pipeline stage
- [x] Added execution output (Stage 7) — subprocess with 15s timeout
- [x] Fixed SIMPLEDIVIDE dispatch bug (was mapped to COPYLIST)
- [x] Fixed PRINTSTRING (bare `name` instead of `tostri(name)`, inverted BREAK logic)
- [x] Fixed LIST `self.currentindex` → `self.index` mismatch
- [x] Fixed `updatesizelistofindnets` missing `global` keyword
- [x] Fixed `findword` shadowing `min` builtin + unsafe 3-element default return
- [x] Fixed `torch.amp.autocast` crash on CPU
- [x] Fixed `listofindents` index out of range (undersized array)
- [x] Fixed `CHANGECOMPARE` matching control flow indent check (`"COMPARE" in name` was too broad)
- [x] Fixed lowercase in corrected output (all tokens now uppercased, all 3 corrected against vocabulary)
- [x] Rewrote word segmentation — repeated small kernel dilation until exactly 3 components
- [x] Rewrote Line Segmentation — force 640×640, scale Y back, scale X with padding
- [x] Deleted 200 lines of dead stock-model code from `image_preprocessing.py`
- [x] Removed stock number model + stale paths from `OCR.py`, made it lazy-load
- [x] Gutted `requirements.txt` from ~30 packages to actual dependencies
- [x] Fixed `pyproject.toml` requires-python from >=3.14 to >=3.11
- [x] Removed duplicate dispatch entries + unused variables across multiple files
