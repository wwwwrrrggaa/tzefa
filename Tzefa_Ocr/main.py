import sys
import gc
import traceback
import subprocess
import torch
from pathlib import Path
from PIL import Image
import importlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import image_preprocessing
import Line_Segmentation
import Binarization
import OCR  # Moved to top level
from Tzefa_Language import ErrorCorrection, topy

def clear_vram(model_ref=None):
    """Aggressively clears VRAM to prevent OOM errors."""
    if model_ref is not None:
        del model_ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_error_correction():
    """Reload ErrorCorrection to reset all global state between runs."""
    importlib.reload(ErrorCorrection)
    importlib.reload(topy)


def _execute_compiled_code(compiled_code: str) -> str:
    """
    Execute compiled Tzefa code in a subprocess and capture output.
    Returns the stdout/stderr as a string. Timeout after 15 seconds
    to prevent infinite loops from crashing the server.
    """
    project_root = str(Path(__file__).resolve().parents[1])

    try:
        result = subprocess.run(
            [sys.executable, "-c", compiled_code],
            capture_output=True,
            text=True,
            timeout=15,
            cwd=project_root,
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- STDERR ---\n" + result.stderr
        if result.returncode != 0:
            output += f"\n[Process exited with code {result.returncode}]"
        return output.strip() if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "[Execution timed out after 15 seconds — possible infinite loop]"
    except Exception as e:
        return f"[Execution error: {e}]"


def image_to_code_pipeline(img_array):
    """
    Runs the full Tzefa OCR pipeline and returns a dict of intermediate results.
    Each stage is wrapped so that if it crashes, all prior results are still available.

    Returns dict with keys:
        binarized_img    - np.ndarray (grayscale) or None
        truelines        - list of (x,y,w,h) bboxes or None
        raw_ocr_lines    - list of str (raw OCR per line) or None
        corrected_lines  - list of str (error-corrected) or None
        compiled_code    - str (final Python code) or None
        execution_output - str (stdout from running the code) or None
        error            - str description of where it crashed, or None
        stage            - str name of last completed stage
    """
    result = {
        "binarized_img": None,
        "truelines": None,
        "word_bboxes": None,   # list of lists: [[( text, (x1,y1,x2,y2) ), ...], ...]
        "raw_ocr_lines": None,
        "corrected_lines": None,
        "compiled_code": None,
        "execution_output": None,
        "error": None,
        "stage": "init",
    }

    # Reset global state in ErrorCorrection / topy between runs
    _reset_error_correction()

    # --- STAGE 1: DL Binarization ---
    try:
        print("Loading Binarization Model to GPU...")
        bin_model = Binarization.BinarizationModel()
        binarified_img_array = bin_model.clean(img_array)
        print("Flushing Binarization Model from VRAM...")
        clear_vram(bin_model)
        result["binarized_img"] = binarified_img_array
        result["stage"] = "binarization"
    except Exception as e:
        clear_vram()
        result["error"] = f"Binarization failed: {e}\n{traceback.format_exc()}"
        return result

    # --- STAGE 2: DL Line Segmentation ---
    try:
        print("Loading Line Segmentation Model to GPU...")
        truelines = Line_Segmentation.segment_lines(binarified_img_array)
        print("Flushing Line Segmentation Model from VRAM...")
        clear_vram()
        result["truelines"] = truelines
        result["stage"] = "line_segmentation"
    except Exception as e:
        clear_vram()
        result["error"] = f"Line Segmentation failed: {e}\n{traceback.format_exc()}"
        return result

    # --- STAGE 3: Word Segmentation + OCR ---
    try:
        words = image_preprocessing.linestowords(binarified_img_array, truelines)

        all_lines_tuples = []
        raw_ocr_lines = []

        print("Starting OCR Inference...")
        for line_number in sorted(words.keys()):
            line_tuples = []

            if line_number - 1 < len(truelines):
                line_bbox = truelines[line_number - 1]
                line_x, line_y, line_w, line_h = line_bbox
            else:
                continue

            sorted_word_keys = sorted(words[line_number].keys())

            for word_number in sorted_word_keys:
                word_x1, word_x2 = words[line_number][word_number]

                abs_x1 = line_x + word_x1
                abs_x2 = line_x + word_x2
                abs_y1 = line_y - 20
                abs_y2 = line_y + line_h + 20

                img_h, img_w = img_array.shape[:2]
                final_x1 = max(0, int(abs_x1))
                final_y1 = max(0, int(abs_y1))
                final_x2 = min(img_w, int(abs_x2))
                final_y2 = min(img_h, int(abs_y2))

                crop_array = img_array[final_y1:final_y2, final_x1:final_x2]
                cropped_pil = Image.fromarray(crop_array)

                recognized_text = OCR.ocr_word(cropped_pil)
                # TrOCR can predict spaces inside a single-word crop (e.g. "BIGLY Y"
                # instead of "BIGLY"), which inflates the token count and breaks
                # downstream correction. Each word crop must be exactly one token.
                # Keep the longest part — it's almost always the real word.
                parts = recognized_text.split()
                recognized_text = max(parts, key=len) if parts else recognized_text
                line_tuples.append((recognized_text, (final_x1, final_y1, final_x2, final_y2)))

            raw_line_text = " ".join([t[0] for t in line_tuples])
            raw_ocr_lines.append(raw_line_text)
            all_lines_tuples.append(line_tuples)
            print(f"      Line {line_number}: {len(line_tuples)} tokens -> {raw_line_text}")

        print("Flushing OCR Model from VRAM...")
        clear_vram()

        result["raw_ocr_lines"] = raw_ocr_lines
        result["word_bboxes"] = all_lines_tuples
        result["stage"] = "ocr"
    except Exception as e:
        clear_vram()
        result["error"] = f"OCR failed: {e}\n{traceback.format_exc()}"
        return result

    # --- STAGE 4: Error Correction ---
    try:
        ErrorCorrection.sendlines(len(truelines))
        index_list = []
        corrected_lines = []

        for line_entries in all_lines_tuples:
            if not line_entries:
                corrected_lines.append("")
                index_list.append(0)
                continue

            # Uppercase all tokens — TrOCR outputs lowercase,
            # but the entire Tzefa vocabulary is uppercase.
            # Pad/trim to exactly 3 tokens (command, arg1, arg2).
            raw_tokens = [t[0].upper() for t in line_entries]
            while len(raw_tokens) < 3:
                raw_tokens.append("")
            raw_tokens = raw_tokens[:3]

            # Correct the command word against the function list to get its index.
            # All argument correction is intentionally left to toline() in Stage 5,
            # which processes lines in order and registers new variable names into
            # listall as it goes — so later lines can fuzzy-match those names.
            # Doing arg correction here (before var names are registered) caused
            # wrong fuzzy matches for user-defined variable names.
            cleaned_first, index, _ = ErrorCorrection.handelfirstword(raw_tokens[0])
            index_list.append(index)

            # For "new variable" declarations (simpler[1] == 0), register the raw
            # arg1 token into the correct listall bucket right now, so that toline()
            # in Stage 5 can match it when it encounters later lines that reference it.
            simpler = ErrorCorrection.listsimplefunc[index]
            if simpler[1] == 0:
                bucket_idx = simpler[2]
                if isinstance(bucket_idx, int) and bucket_idx < len(ErrorCorrection.listall):
                    bucket = ErrorCorrection.listall[bucket_idx]
                    if raw_tokens[1] and raw_tokens[1] not in bucket:
                        bucket.append(raw_tokens[1])

            # Rebuild as a clean 3-word string — toline() will do all vocab matching.
            full_line_text = cleaned_first + " " + raw_tokens[1] + " " + raw_tokens[2]
            corrected_lines.append(full_line_text)

        result["corrected_lines"] = corrected_lines
        result["stage"] = "error_correction"
    except Exception as e:
        result["error"] = f"Error Correction failed: {e}\n{traceback.format_exc()}"
        return result

    # --- STAGE 5: Compilation to Python ---
    try:
        linelist = []
        for i in range(len(corrected_lines)):
            current_indent = index_list[i] if i < len(index_list) else 0
            line_obj = ErrorCorrection.toline(corrected_lines[i], current_indent, ErrorCorrection.giveindents())
            linelist.append(line_obj)

        listfunctions_out, listezfunctions_out = ErrorCorrection.giveinstructions()
        topy.getinstructions(listfunctions_out, listezfunctions_out)

        # Instead of writing to file, capture the code as a string
        compiled_lines = []
        compiled_lines.append("from Tzefa_Language.createdpython import *")
        counterindent = 0
        indent = "    "
        for i in range(1, len(linelist) + 1):
            counterindent += topy.listofindentchanges[i]
            compiled_lines.append(indent * counterindent + topy.makepredict(linelist[i - 1], i))
        compiled_lines.append("printvars()")

        result["compiled_code"] = "\n".join(compiled_lines)
        result["stage"] = "compilation"
        print("Compilation Complete!")
    except Exception as e:
        result["error"] = f"Compilation failed: {e}\n{traceback.format_exc()}"
        return result

    # --- STAGE 6: Execute the compiled code ---
    try:
        print("Executing compiled Tzefa program...")
        execution_output = _execute_compiled_code(result["compiled_code"])
        result["execution_output"] = execution_output
        result["stage"] = "execution"
        print("Execution Complete!")
    except Exception as e:
        result["error"] = f"Execution failed: {e}\n{traceback.format_exc()}"
        return result

    return result


def image_to_code(img_array, debug_mode=False):
    """Legacy wrapper - runs the full pipeline."""
    pipeline_result = image_to_code_pipeline(img_array)

    if debug_mode and pipeline_result["binarized_img"] is not None:
        Image.fromarray(pipeline_result["binarized_img"]).show()

    if pipeline_result["error"]:
        print(f"Pipeline error: {pipeline_result['error']}")

    if pipeline_result["compiled_code"]:
        # Also write to file for backwards compat
        outfile = Path(__file__).parent.parent / "Tzefa_Language" / "test.py"
        with outfile.open("w+", encoding="utf-8") as f:
            f.write(pipeline_result["compiled_code"])

    return pipeline_result


def main():
    image_path = r"E:\Storage\tests\test2.png"
    if not Path(image_path).exists():
        print(f"Path not found: {image_path}")
        return

    img = image_preprocessing.UV_unwrap(image_path)
    image_to_code(img, debug_mode=True)

if __name__ == '__main__':
    main()