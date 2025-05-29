import sys
from image_preprocessing import preprocess_image
#from ocr import extract_tzefa_code
import cv2

def main():
    if len(sys.argv) != 2 and False:
        print("Usage: python main.py <image_file>")
        return
    
    #image_file = sys.argv[1]
    image_file =r"E:\Storage\tests\longtime.png"
    
    # Preprocess image
    preprocessed_image, word_bboxes = preprocess_image(image_file)
    cv2.imshow("Preprocessed Image", preprocessed_image)
    cv2.waitKey(0)
    # Extract Tzefa code
    #tzefa_code = extract_tzefa_code(preprocessed_image, word_bboxes)
    
    #print(tzefa_code)

if __name__ == "__main__":
    main()
