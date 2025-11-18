import cv2
import numpy as np
from PIL import Image


def find_colors(image):
    """Detect dominant text color and estimate background color."""
    if image is None:
        return None, None

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges (adjust thresholds as needed)
    color_ranges = {
        'blue': (np.array([90, 50, 50]), np.array([130, 255, 255])),
        'red': (np.array([0, 50, 50]), np.array([10, 255, 255])),
        'green': (np.array([40, 50, 50]), np.array([80, 255, 255]))
    }

    # Count pixels for each color
    color_counts = {'black': 0, 'blue': 0, 'red': 0, 'green': 0}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create text mask
    text_mask = np.zeros_like(gray)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            cv2.drawContours(text_mask, [contour], -1, 255, -1)

    # Sample text pixels for color detection
    text_pixels = hsv[text_mask > 0]
    if len(text_pixels) > 0:
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv, lower, upper)
            masked_text = cv2.bitwise_and(mask, text_mask)
            color_counts[color] = cv2.countNonZero(masked_text)
        # Check for black (low saturation and value)
        low_sat_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 30, 100]))
        masked_black = cv2.bitwise_and(low_sat_mask, text_mask)
        color_counts['black'] = cv2.countNonZero(masked_black)

    # Determine dominant text color
    dominant_color = max(color_counts, key=color_counts.get)
    if color_counts[dominant_color] < 1000:  # Threshold for noise
        dominant_color = 'black'  # Default

    # Estimate background color from non-text areas
    bg_mask = cv2.bitwise_not(text_mask)
    bg_pixels = image[bg_mask > 0]
    if len(bg_pixels) > 0:
        avg_bg = np.mean(bg_pixels, axis=0).astype(int)  # BGR average
    else:
        avg_bg = [255, 255, 255]  # Default white

    return dominant_color, tuple(avg_bg)


def binarize_with_colors(image, text_color, bg_color):
    """Binarize image based on detected text and background colors."""
    if image is None or text_color is None:
        return None

    if text_color[0] == 'black':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        if text_color[0] == 'blue':
            lower, upper = np.array([90, 50, 50]), np.array([130, 255, 255])
        elif text_color[0] == 'red':
            lower1, upper1 = np.array([0, 50, 50]), np.array([10, 255, 255])
            lower2, upper2 = np.array([170, 50, 50]), np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        elif text_color[0] == 'green':
            lower, upper = np.array([40, 50, 50]), np.array([80, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
        binary = cv2.bitwise_not(mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return binary
def segment_lines(binary_image):
    """Segment lines using statistical thresholding on horizontal projection."""
    if binary_image is None:
        return []

    height, width = binary_image.shape
    horizontal_projection = np.sum(binary_image == 0, axis=1)

    # Compute statistical thresholds
    mean_proj = np.mean(horizontal_projection)
    std_proj = np.std(horizontal_projection)
    threshold = mean_proj + 1 * std_proj  # Adjust multiplier as needed

    line_bboxes = []
    in_line = False
    start_y = 0
    min_line_height = 10

    for y in range(height):
        if horizontal_projection[y] > threshold and not in_line:
            start_y = y
            in_line = True
        elif horizontal_projection[y] <= threshold and in_line:
            end_y = y
            if end_y - start_y > min_line_height:
                line_bboxes.append((0, start_y, width, end_y - start_y))
            in_line = False

    if in_line and height - start_y > min_line_height:
        line_bboxes.append((0, start_y, width, height - start_y))

    return line_bboxes



def segment_words(binary_image, line_bboxes):
    """Use repeated dilation until exactly 3 connected components per line"""
    word_bboxes = []
    
    # Process each line
    for line_bbox in line_bboxes:
        x, y, w, h = line_bbox
        line_roi = binary_image[y:y+h, x:x+w]
        
        # Use repeated dilation to get exactly 3 connected components
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        current_image = line_roi.copy()
        iterations = 0
        max_iterations = 20  # Safety limit
        
        while iterations < max_iterations:
            # Apply dilation
            current_image = cv2.dilate(current_image, kernel, iterations=1)
            
            # Count connected components
            contours, _ = cv2.findContours(current_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter out very small components
            valid_contours = []
            for contour in contours:
                _, _, w_c, h_c = cv2.boundingRect(contour)
                if w_c > 5 and h_c > 5:  # Minimum size threshold
                    valid_contours.append(contour)
            
            # If we have exactly 3 components, we're done
            if len(valid_contours) == 3:
                break
            # If we have fewer than 3, we might need more dilation (but be careful)
            elif len(valid_contours) < 3 and iterations > 10:
                break  # Stop if we're not making progress
            
            iterations += 1
        
        # Extract word bounding boxes from final components
        line_words = []
        for contour in valid_contours:
            word_x, word_y, word_w, word_h = cv2.boundingRect(contour)
            
            # Convert back to global coordinates
            global_x = x + word_x
            global_y = y + word_y
            line_words.append((global_x, global_y, word_w, word_h))
        
        # Sort words by x-coordinate (left to right)
        line_words.sort(key=lambda w: w[0])
        
        word_bboxes.extend(line_words)
    
    return word_bboxes

def splitline2(img):
    checked = False

    width, height = img.size
    linempty = True
    inproc = False
    lines = []
    rows = []
    counter = 1
    newlines = {}
    pixels = img.load()
    for i in range(width - 1):
        for j in range(height - 1):
            if (0 == pixels[i, j]):
                linempty = False
                if (inproc == False):
                    inproc = True
                    checked = True
                    lines.append(j)
                    newlines[counter] = [i, "n/a"]
        if (linempty == True):
            if (inproc == True):
                lines.append(j)
                checked = False
                newlines[counter][1] = i
                counter = counter + 1
                # pixels[b, i] = 0
                inproc = False
        linempty = True
    rowempty = True
    inproc = False
    rows = {}
    if (checked):
        if (newlines[counter][1] == "n/a"):
            newlines[counter][1] = width
    return newlines


def linestowords(binarified_img, line_bboxes):
    words = {}
    line_num = 1
    img_width, img_height = binarified_img.size
    cv2_img = np.asarray(binarified_img, dtype=np.uint8)
    kernel = np.ones((5, 5), np.uint8)

    for bbox in line_bboxes:
        x, y, w, h = bbox
        line_roi = cv2_img[y:y + h, 0:img_width]
        eroded_img = cv2.erode(line_roi, kernel, iterations=0)
        pil_img = Image.fromarray(eroded_img)
        words[line_num] = splitline2(pil_img)

        while len(words[line_num]) > 3:
            eroded_img = cv2.erode(eroded_img, kernel, iterations=1)
            pil_img = Image.fromarray(eroded_img)
            words[line_num] = splitline2(pil_img)

        line_num += 1

    return words

def ocr_word(bbox, binarified_img):
    """Filler function for OCR of words. Crops the word from binarified_img using bbox and returns placeholder text."""
    x, y, w, h = bbox
    word_img = binarified_img.crop((x, y, x + w, y + h))
    # Placeholder: return dummy text. Replace with actual OCR model later.
    return "WORD"


def ocr_number(bbox, binarified_img):
    """Filler function for OCR of numbers. Crops the number from binarified_img using bbox and returns placeholder text."""
    x, y, w, h = bbox
    number_img = binarified_img.crop((x, y, x + w, y + h))
    # Placeholder: return dummy text. Replace with actual OCR model later.
    return "NUMBER"
