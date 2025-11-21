import PIL
import cv2
from PIL import Image, ImageFilter, ImageOps
import image_preprocessing
import sys
from pathlib import Path
import numpy as np
import OCR

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Tzefa_Language import ErrorCorrection as ErrorCorrection
from Tzefa_Language import topy as topy



def image_to_code(img):
    global binarified_img, Truelines, saveimg, listoftruth
    from PIL import ImageDraw

    binarified_img = image_preprocessing.binarize_with_colors(img, image_preprocessing.find_colors(img), 'whatever')
    img = PIL.Image.fromarray(img)
    Truelines = image_preprocessing.segment_lines_paddle(binarified_img)
    binarified_img = Image.fromarray(binarified_img)
    binarified_img.show()

    # create a copy of the original PIL image to draw bounding boxes on
    boxed_img = img.copy()
    draw = ImageDraw.Draw(boxed_img)

    words = image_preprocessing.linestowords(binarified_img, Truelines)


    # Translate word bounding boxes into correct readings using error correction
    corrected_lines = []
    ErrorCorrection.sendlines(len(Truelines))
    index_list = []
    for line_number in words:

        for word_number in words[line_number]:
            line_bbox = Truelines[line_number - 1]  # Line numbers start at 1, list at 0
            x, y, w, h = line_bbox
            word_x1, word_x2 = words[line_number][word_number]
            word_y1, word_y2 = y, y + h

            # compute crop bbox (keep original behavior but fix absolute x2)
            word_bbox = (x + word_x1, word_y1 - 20, x + word_x2, word_y2 + 20)
            # clamp coordinates to image bounds
            img_w, img_h = boxed_img.size
            x1 = max(0, int(word_bbox[0]))
            y1 = max(0, int(word_bbox[1]))
            x2 = min(img_w, int(word_bbox[2]))
            y2 = min(img_h, int(word_bbox[3]))

            # draw rectangle for this word
            draw.rectangle([x1, y1, x2, y2], outline='red', width=2)

            cropped_img = img.crop((x1, y1, x2, y2))

            if (word_number == 1):
                recognized_text = OCR.ocr_word(cropped_img)
                firstword, index, numflag = ErrorCorrection.handelfirstword(recognized_text)
                index_list.append(index)
                text_line = firstword
            else:
                if numflag == 1 and word_number == 3:
                    recognized_text = OCR.ocr_number(cropped_img)
                    text_line += " " + recognized_text
                else:
                    recognized_text = OCR.ocr_word(cropped_img)
                    text_line += " " + recognized_text
        corrected_lines.append(text_line)

    # show the image with bounding boxes
    boxed_img.show()

    print(corrected_lines)
    linelist = [ErrorCorrection.toline(corrected_lines[i], index_list[i], ErrorCorrection.giveindents()) for i in range(len(corrected_lines))]
    print(linelist)
    listfunctions, listezfunctions = ErrorCorrection.giveinstructions()
    topy.getinstructions(listfunctions, listezfunctions)
    topy.makepyfile(linelist)
    from Tzefa_Language import test





def main():
    img=image_preprocessing.UV_unwrap(r"D:\downloads\Test2.jpg")
    result = image_to_code(img)

if __name__ == '__main__':
    main()