# Tzefa — TODO

*Updated: March 2, 2026*

---

## P0 — Fix Now

### VM Runtime Bugs (`createdpython.py`)
- [x] `makeeindexrror` called with 3 args, expects 4 (lines ~499, ~507) — crashes on list index-out-of-bounds
- [x] `readerror()` called with no args, expects 1 (line ~528) — crashes on unreadable variable access
- [x] `COND.givetype()` references `self.type` which is never set in `__init__` — AttributeError


---

## P1 — Do Soon

### Line Segmentation (Weakest Model)
The YOLO-OBB line segmentation model is the primary bottleneck. The current workarounds:
- Force resize to 640×640 (squashes aspect ratio) — gives good Y-axis detection
- X-axis needs 50% padding compensation because squash makes detections too narrow
- Word segmentation uses repeated dilation to enforce exactly 3 components per line

**Recommended improvements:**
- [ ] **Train on more data** — more handwritten Tzefa images with line-level annotations
- [ ] **Evaluate letterboxing** — train with aspect-ratio-preserving letterbox instead of squash, so detection widths are accurate without padding hacks



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


### Language Evolution
- [ ] **4-word syntax** — expand instruction format from `COMMAND ARG1 ARG2` to `COMMAND ARG1 ARG2 ARG3` for richer expressions
- [ ] **Lowercase support** — allow lowercase characters in the language and update compiler, OCR, and error correction accordingly
- [ ] Rename typos: `ASSSIGNINT`→`ASSIGNINT`, `handelfirstword`→`handlefirstword`, `makeparenthasis`→`makeparenthesis`, `errore`→`error_handler`, `makeeindexrror`→`make_index_error`
- [ ] Clean up `Tzefa_Language/main.py` — uses `import test` for side-effect execution (module cache prevents re-runs)


---

## P3 — Nice to Have

### Web UI
- [ ] Manual correction interface — let user fix OCR mistakes before compilation

### OCR Pipeline
- [ ] Batch word OCR inference (currently one word at a time)
- [ ] Add custom Levenshtein with weighted OCR confusion pairs (l/I, O/0, S/5, etc.) — `ocr_edit_distance` exists in ErrorCorrection.py but isn't used

---

## Future Vision


### Automated Testing / Education Platform
- [ ] **Teacher API** — Python API where a teacher defines a problem: predefined input variables the student receives, expected output, and test cases
- [ ] **Test runner** — automatically runs student's handwritten code against the test suite and reports pass/fail per test case
- [ ] **Classroom integration** — batch processing of student submissions with grading output

### iPad / Smart Pen IDE
- [ ] **Real-time recognition** — stream handwriting from iPad or smart pen directly into the OCR pipeline
- [ ] **Live feedback** — show error correction suggestions as the student writes
- [ ] **Hybrid IDE** — combine handwriting input with a code editor for corrections

### Data Visualization & LLM Debugging
- [ ] **Data visualization libraries** — add plotting/charting support to the Tzefa language for teaching data concepts
- [ ] **LLM-powered debugging** — give an LLM context from inside the VM (variable state, execution trace, error messages) to provide natural language debugging hints to students

---

