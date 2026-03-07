# Tzefa Language – Internal Architecture

This document describes how Tzefa source text is compiled and executed.

---

## Overview

```
Handwritten paper
      │
      ▼
  OCR (TrOCR)              Tzefa_Ocr
      │  raw tokens (3 or 4 per line)
      ▼
  Dialect Normalisation     dialects.normalize_line()
      │  canonical 4-word CAPS tuple
      ▼
  Error Correction          ErrorCorrection.TzefaParser.parse_line()
      │  validated 4-word bytecode
      ▼
  Code Generation           topy.make_instruction()
      │  Python source string
      ▼
  Subprocess exec           _execute_compiled_code()
      │
      ▼
  VM (createdpython)        Value / VmList / Condition + helpers
```

---

## Bytecode Format

The internal representation is a **4-word tuple**:

```
[VERB, TYPE, ARG1, ARG2]
```

- **VERB** — the action (`MAKE`, `SET`, `GET`, `WRITE`, `ADD`, `WHILE`, `IF`, …)
- **TYPE** — what it acts on (`INTEGER`, `STRING`, `LIST`, `CONDITION`, `VALUES`, …)
- **ARG1, ARG2** — operands (variable names or resolved numeric literals)

This is the canonical form that `topy.py` dispatches on. Both the 3-word and 4-word source dialects normalise into this form.

---

## Dialect & Casing Layer (`dialects.py`)

| Dialect | Tokens/line | Example |
|---|---|---|
| `THREE_WORD` (default) | 3 | `MAKEINTEGER COUNTER FIVE` → `["MAKE","INTEGER","COUNTER","5"]` |
| `FOUR_WORD` | 4 | `Make Integer counter Five` → `["MAKE","INTEGER","COUNTER","5"]` |

For 3-word input, `THREE_TO_FOUR` maps each monolithic opcode to a `(VERB, TYPE)` pair. Unknown opcodes are treated as function calls: `FUNCNAME A B` → `["CALL","FUNCNAME","A","B"]`.

| Casing | Rule |
|---|---|
| `CAPS_ONLY` (default) | All tokens uppercased |
| `MIXED_CASE` | Commands Titlecase, user vars lowercase; all uppercased before processing |

---

## Error Correction (`ErrorCorrection.py`)

`TzefaParser` processes 4-word tuples one at a time via `parse_line()`.

### Instruction table

`_BUILTIN_INSTRUCTIONS` defines every valid `[VERB, TYPE, ARG1_KIND, ARG2_KIND]` combination. The `ARG_KIND` values determine how each argument is validated:

| Kind | Meaning | Bucket |
|---|---|---|
| `NEWINT` / `NEWSTR` / `NEWLIST` / `NEWBOOL` / `NEWCOND` / `NEWFUNC` | Declares a new name (registered, not corrected) | 0–4, 7 |
| `INT` / `STR` / `LIST` / `BOOL` / `COND` | Existing variable (fuzzy-matched) | 0–4 |
| `STATE` | `STAY` or `BREAK` | 5 |
| `TYPE` | `INTEGER`, `STRING`, `LIST`, `BOOLEAN` | 6 |
| `TRUTH` | `TRUE`, `FALSE` | 8 |
| `COMPARE` | `EQUALS`, `BIGEQUALS`, `BIGGER` | 9 |
| `NUMNAME` | `ZERO` … `ONEHUNDRED` (resolved to integer) | 10 |
| `TEXT` | Free text (no correction) | 11 |
| `VALUE` | Context-dependent (resolved at parse time) | — |

### `match_opcode(verb, type_word)`

Exact-matches `(verb, type_word)` against the instruction table. On miss, fuzzy-matches the combined `VERB_TYPE` string against bucket 7.

### `parse_line(quad)`

1. Identifies the instruction via `match_opcode`.
2. For `FUNCTION` definitions: registers the function name in `topy`, adds a `CALL` opcode.
3. For `RETURN`: resolves the return variable against the function's type, tracks indent.
4. For `CALL`: passes through (args are input/output variable names).
5. For everything else: resolves each argument via `_resolve_arg()`.
6. Tracks control-flow indentation.

### `_resolve_arg(kind, raw)`

- **NEW kinds**: appends the raw name to the bucket (no correction).
- **Existing kinds**: fuzzy-matches against the bucket.
- **NUMNAME**: replaces the matched word with its integer value.

### `ocr_edit_distance(word1, word2)`

Custom Levenshtein with cost `0.5` for common OCR shape confusions (`O↔0`, `I↔1`, `I↔L`, etc.) and cost `2.0` for all other substitutions.

### `find_word(name_list, word, use_ocr_weights)`

Returns the closest match by edit distance. Ties broken by closest length.

---

## Code Generation (`topy.py`)

`make_instruction(quad, line_num)` dispatches on `quad[0]` (the verb) to a handler function that receives `(type_word, arg1, arg2, line_num)` and returns a Python source line.

### Dispatch table

```python
_DISPATCH = {
    "MAKE": _make,  "NEW": _new,  "SET": _set,  "CHANGE": _change,
    "WHILE": _while,  "IF": _if,  "ELIF": _elif,  "ITERATE": _iterate,
    "PRINT": _print,  "GET": _get,  "WRITE": _write,  "ADD": _add,
    "SUBTRACT": _subtract,  "MULTIPLY": _multiply,  "DIVIDE": _divide,
    "SIMPLEDIVIDE": _simpledivide,  "MODULO": _modulo,  "POWER": _power,
    "COMBINE": _combine,  "PAD": _pad,  "TYPE": _type,
    "FUNCTION": _function,  "RETURN": _return,  "CALL": _call,
}
```

Each handler uses the `type_word` to distinguish sub-operations (e.g. `_set("INTEGER", ...)` → `vm_assign_int`, `_set("INDEX", ...)` → `change_index`).

### Indent tracking

Control-flow handlers write `+1` / `-1` into `_indent_changes[]`. `make_py_file()` accumulates these to produce correct Python indentation.

### User-defined functions

`register_user_function(name, input_type, output_type)` is called by the parser when a `FUNCTION` instruction is encountered. The `_call` handler emits `enter_function_call(...)`.

---

## VM (`createdpython.py`)

The VM is a set of Python module-level globals and helper functions.

### Core types

- **`Value`** — named, typed variable with `readable`/`writable` flags.
- **`VmList`** — fixed-size list of `Value` objects with an active index.
- **`Condition`** — compares two `Value` objects with `EQUALS`/`BIGEQUALS`/`BIGGER`.

### Registries

```python
var_registry  : Dict[str, Dict[str, Value | VmList]]
cond_registry : Dict[str, Condition]
```

### Function call protocol

1. `enter_function_call` pushes locals/conditions onto stacks, resets `LOCAL*` slots, copies input, calls the function, copies the return value.
2. `exit_function_call` pops and restores everything.

### Safety limits

| Global | Default | Effect |
|---|---|---|
| `line_limit` | 1000 | `sys.exit` |
| `function_limit` | 25 | `sys.exit` |

---

## File Map

| File | Role |
|---|---|
| `dialects.py` | Dialect/casing normalisation (3-word → 4-word, mixed → CAPS) |
| `ErrorCorrection.py` | Parser: OCR error correction → validated 4-word bytecode |
| `topy.py` | Bytecode → Python code generator |
| `createdpython.py` | VM runtime |
| `main.py` | Compiler entry point and profiler |
| `Number2Name.py` | Integer → English name lookup (0–100) |
| `test.py` | Generated output (do not edit by hand) |

---

## Execution Flow (web server)

```
Tzefa_Web/server.py  (reads dialect + casing from form toggles)
  └─ Tzefa_Ocr/main.image_to_code_pipeline(img, dialect, casing)
       ├─ Binarization.BinarizationModel.clean(img)
       ├─ Line_Segmentation.segment_lines(img)
       ├─ image_preprocessing.linestowords(img, lines, target_words)
       ├─ OCR.run_ocr(word_crops)           → raw tokens
       ├─ dialects.normalize_line()          → 4-word CAPS tuples
       ├─ TzefaParser.parse_line()           → validated bytecode
       ├─ topy.make_instruction()            → Python source lines
       └─ subprocess(sys.executable, "-c", compiled_code)
            └─ createdpython VM             → stdout captured
```

The subprocess runs with a 15-second timeout.  
`topy` module state is reset between requests via `importlib.reload()`.  
`TzefaParser` is instantiated fresh for each request.

