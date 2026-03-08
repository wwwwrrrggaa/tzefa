import gc
import importlib
import subprocess
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import Binarization
import image_preprocessing
import Line_Segmentation
import OCR

from Tzefa_Language import topy
from Tzefa_Language.dialects import CAPS_ONLY, THREE_WORD
from Tzefa_Language.ErrorCorrection import TzefaParser


def clear_vram(model_ref=None):
    """Aggressively clears VRAM to prevent OOM errors."""
    if model_ref is not None:
        del model_ref
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reset_topy():
    """Reload topy to reset its module-level globals between runs."""
    importlib.reload(topy)


def _execute_compiled_code(compiled_code: str) -> str:
    """
    Execute compiled Tzefa code in a subprocess and capture output.
    Timeout after 15 seconds to prevent infinite loops.
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


def image_to_code_pipeline(
    img_array,
    dialect: str = THREE_WORD,
    casing: str = CAPS_ONLY,
    segmentation_method: str = "yolo",
):
    """
    Run the full Tzefa OCR pipeline and return a dict of intermediate results.

    Parameters
    ----------
    img_array : np.ndarray
        Input image (RGB).
    dialect : str
        ``THREE_WORD`` or ``FOUR_WORD``.
    casing : str
        ``CAPS_ONLY`` or ``MIXED_CASE``.
    segmentation_method : str
        ``yolo`` or ``eynollah``.
    """
    result = {
        "binarized_img": None,
        "truelines": None,
        "word_bboxes": None,
        "raw_ocr_lines": None,
        "corrected_lines": None,
        "compiled_code": None,
        "execution_output": None,
        "error": None,
        "stage": "init",
    }

    # Fresh parser and topy state for every run
    _reset_topy()
    parser = TzefaParser(dialect=dialect, casing=casing)
    target_words = parser.expected_words_per_line

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
        print(f"Loading Line Segmentation Model ({segmentation_method}) to GPU...")
        truelines = Line_Segmentation.segment_lines(binarified_img_array, method=segmentation_method)
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
        words = image_preprocessing.linestowords(
            binarified_img_array, truelines, target_words=target_words,
        )

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
                ### bug if out of bounds and also i dont get it
                abs_x1 = line_x + word_x1
                abs_x2 = line_x + word_x2
                abs_y1 = line_y - 20
                abs_y2 = line_y + line_h + 20

                img_h, img_w = binarified_img_array.shape[:2]
                final_x1 = max(0, int(abs_x1))
                final_y1 = max(0, int(abs_y1))
                final_x2 = min(img_w, int(abs_x2))
                final_y2 = min(img_h, int(abs_y2))

                crop_array = binarified_img_array[final_y1:final_y2, final_x1:final_x2]
                cropped_pil = Image.fromarray(crop_array)

                recognized_text = OCR.ocr_word(cropped_pil)
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

    # --- STAGE 4: Error Correction + Compilation ---
    try:
        #if its a class go all the way,give the text list and return the corrected code,why do we need to init the indent table and go line by line
        parser.init_indent_table(len(truelines))
        corrected_lines = []
        bytecode_list = []

        for line_entries in all_lines_tuples:
            if not line_entries:
                corrected_lines.append("")
                bytecode_list.append(["MAKE", "INTEGER", "TEMPORARY", "0"])  # no-op placeholder
                continue

            # Extract raw OCR tokens and pad/trim to expected word count
            # I also dont get this
            raw_tokens = [t[0] for t in line_entries]
            while len(raw_tokens) < target_words:
                raw_tokens.append("")
            raw_tokens = raw_tokens[:target_words]

            # Normalise to canonical 4-word CAPS tuple
            normalised = parser.normalize_source_line(raw_tokens)

            # Parse and error-correct into validated 4-word bytecode
            bytecode = parser.parse_line(normalised)
            bytecode_list.append(bytecode)

            # Show the corrected (post-error-correction) form
            corrected_lines.append(" ".join(bytecode))

        result["corrected_lines"] = corrected_lines
        result["stage"] = "error_correction"
    except Exception as e:
        result["error"] = f"Error Correction failed: {e}\n{traceback.format_exc()}"
        return result

    # --- STAGE 5: Compilation to Python ---
    # again, why do we need to go line by line here, why cant we just compile the whole thing at once, also why do we need to init the indent table in the parser
    # make it also a class that gets input and gives output[cant believe im saying this when i always preferred functional programming but this code is just a mess ngl]
    try:
        compiled_lines = ["from Tzefa_Language.createdpython import *"]
        indent_level = 0
        indent_unit = "    "
        for i in range(1, len(bytecode_list) + 1):
            indent_level += topy._indent_changes[i]
            compiled_lines.append(
                indent_unit * max(0, indent_level) + topy.make_instruction(bytecode_list[i - 1], i)
            )
        compiled_lines.append("print_vars()")

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


def image_to_code(img_array, debug_mode=False, dialect=THREE_WORD, casing=CAPS_ONLY):
    """Legacy wrapper – runs the full pipeline."""
    pipeline_result = image_to_code_pipeline(img_array, dialect=dialect, casing=casing)

    if debug_mode and pipeline_result["binarized_img"] is not None:
        Image.fromarray(pipeline_result["binarized_img"]).show()

    if pipeline_result["error"]:
        print(f"Pipeline error: {pipeline_result['error']}")

    if pipeline_result["compiled_code"]:
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