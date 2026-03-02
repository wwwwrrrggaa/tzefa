import numpy as np
from PIL import Image, ImageDraw, ImageFont
from main import image_to_code

def create_and_run_test():
    test_code = """MAKEINTEGER NUMY FIVE
MAKEINTEGER BIGLY SIX
MAKEINTEGER RESULT ONE
BASICCONDITION JUSTBIGGER
MAKEINTEGER ZERO ZERO
MAKEINTEGER ONE ONE
SUBTRACT ONE ZERO
LEFTSIDE JUSTBIGGER NUMY
RIGHTSIDE JUSTBIGGER ZERO
WHILE JUSTBIGGER FOURTEEN
MULTIPLY RESULT BIGLY
ASSSIGNINT RESULT TEMPORARY
SUBTRACT NUMY ONE
ASSSIGNINT NUMY TEMPORARY"""

    print("Generatiing virtual whiteboard image...")

    # Create a blank white image (1000x800)
    img = Image.new('RGB', (1000, 800), color='white')
    draw = ImageDraw.Draw(img)

    # Try to load a standard font, fallback to default if not found
    try:
        # You might need to adjust the font name based on your OS (e.g., 'arial.ttf' for Windows)
        font = ImageFont.truetype("arial.ttf", 36)
    except IOError:
        print("Standard font not found, falling back to default.")
        font = ImageFont.load_default()

    # Draw text onto the image
    y_text = 40
    for line in test_code.split('\n'):
        draw.text((40, y_text), line.strip(), font=font, fill='black')
        y_text += 45  # Line spacing

    # Show the generated image temporarily
    img.show(title="Generated Tzefa Code")

    # Convert PIL Image to cv2/NumPy array (RGB)
    img_array = np.array(img)

    print("Feeding image array into main Tzefa pipeline...")
    # Run through the pipeline
    image_to_code(img_array, debug_mode=True)

if __name__ == "__main__":
    create_and_run_test()