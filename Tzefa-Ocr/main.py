import cv2
from PIL import Image, ImageFilter, ImageOps
import image_preprocessing
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Tzefa_Language import ErrorCorrection as ErrorCorrection


def image_to_code(img):
    global binarified_img, Truelines, saveimg, listoftruth
    binarified_img = image_preprocessing.binarize_with_colors(img,image_preprocessing.find_colors(img),'whatever')
    Truelines = image_preprocessing.segment_lines(binarified_img)
    print(Truelines)
    binarified_img = Image.fromarray(binarified_img)  # Convert NumPy array to PIL Image
    words = image_preprocessing.linestowords(binarified_img, Truelines)
    print(words)
    # Translate word bounding boxes into correct readings using error correction
    corrected_lines = []
    for line_num, word_dict in words.items():
        line_bbox = Truelines[line_num - 1]  # Line numbers start at 1, list at 0
        x, y, w, h = line_bbox
        line_text = []
        for word_num, word_pos in word_dict.items():
            word_x1, word_x2 = word_pos
            word_y1, word_y2 = y, y + h
            word_bbox = (x + word_x1, word_y1, word_x2 - word_x1, word_y2 - word_y1)
            # First, OCR as word to get initial text
            ocr_text = image_preprocessing.ocr_word(word_bbox, binarified_img)
            # Classify based on error correction to decide if it's a word or number
            type_ = ErrorCorrection.classify_word(ocr_text)
            if type_ == 'number':
                # Re-OCR as number if classified as such
                ocr_text = image_preprocessing.ocr_number(word_bbox, binarified_img)
            # Apply error correction to the appropriate list
            if type_ == 'word':
                corrected_text = ErrorCorrection.findword(ErrorCorrection.listezfunc, ocr_text)[0]
            else:
                corrected_text = ErrorCorrection.findword(ErrorCorrection.listintegers, ocr_text)[0]
            line_text.append(corrected_text)
        corrected_lines.append(" ".join(line_text))
    return corrected_lines


def main():
    listoftruth = []
    img = cv2.imread(r"D:\downloads\Test2.jpg")
    stre = ""
    result = image_to_code(img)
    print("Corrected lines:", result)

if __name__ == '__main__':
    main()