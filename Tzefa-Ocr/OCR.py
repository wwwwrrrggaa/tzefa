from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import requests
from PIL import Image

processor = TrOCRProcessor.from_pretrained("microsoft/trocr-small-stage1")
model = VisionEncoderDecoderModel.from_pretrained("C:\Storage\Models\checkpoint-1052300")
def ocr(image):
    pixel_values = processor(image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)

    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

if __name__ == '__main__':
    print(ocr(Image.open(r"C:\DataSets\Words\kYXosdQEIbxOznnvHHjV\genbatch2-number1.jpg"))) # Replace 'image.jpg' with your image file path