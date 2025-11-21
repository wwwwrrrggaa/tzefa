from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

word_model=r"C:\Users\yonat\PycharmProjects\tzefa\word_model"
number_model=r"C:\Users\yonat\PycharmProjects\tzefa\number_model"

processor_words = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1", use_fast=False)
model_words = VisionEncoderDecoderModel.from_pretrained(word_model)

processor_numbers = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1", use_fast=False)
model_numbers = VisionEncoderDecoderModel.from_pretrained(number_model)


def ocr_word(image):
    pixel_values = processor_words(image, return_tensors="pt").pixel_values
    generated_ids = model_words.generate(pixel_values)

    generated_text = processor_words.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text
def ocr_number(image):
    pixel_values = processor_numbers(image, return_tensors="pt").pixel_values
    generated_ids = model_numbers.generate(pixel_values)

    generated_text = processor_numbers.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if __name__ == '__main__':
    print(ocr_word(Image.open(r"D:\downloads\Word_test.jpg"))) # Replace 'image.jpg' with your image file pathz