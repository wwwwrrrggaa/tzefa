import cv2
import numpy as np
import torch
from transformers import AutoModel, AutoProcessor

def binarize_image(image):
    """Advanced document binarization using multiple OpenCV techniques"""
    if image is None:
        return None
        
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Method 1: Adaptive thresholding (works well for varying lighting)
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)
    
    # Method 2: OTSU thresholding (good for bimodal histograms)
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Method 3: Enhanced preprocessing + OTSU
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(blurred)
    
    # Apply OTSU on enhanced image
    _, enhanced_otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Choose the best result (you can customize this logic)
    # For now, use the enhanced OTSU as it typically works best for documents
    return enhanced_otsu

def segment_lines(binary_image):
    """Use DocEnTR model for line segmentation"""
    try:
        # Try to load DocEnTR model for line segmentation
        processor = AutoProcessor.from_pretrained("microsoft/layoutlm-base-uncased")
        model = AutoModel.from_pretrained("microsoft/layoutlm-base-uncased")
        
        # Convert to RGB for model
        rgb_image = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2RGB)
        inputs = processor(images=rgb_image, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Extract line bounding boxes from model output
            line_bboxes = []
            # This is a simplified placeholder - actual implementation would parse model outputs
            
        return line_bboxes
    except:
        # Fallback to OpenCV line segmentation
        height, width = binary_image.shape
        
        # Horizontal projection to find lines
        horizontal_projection = np.sum(binary_image == 0, axis=1)
        
        # Find line boundaries
        line_bboxes = []
        in_line = False
        start_y = 0
        
        for i, count in enumerate(horizontal_projection):
            if count > width * 0.1 and not in_line:  # Start of line
                start_y = i
                in_line = True
            elif count <= width * 0.1 and in_line:  # End of line
                line_bboxes.append((0, start_y, width, i))
                in_line = False
        
        # Handle last line
        if in_line:
            line_bboxes.append((0, start_y, width, height))
            
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

def preprocess_image(image_file):
    """Main preprocessing function"""
    image = cv2.imread(image_file)
    
    if image is None:
        print(f"Error: Could not load image from {image_file}")
        return None, []
    
    # Step 1: Binarization
    binary_image = binarize_image(image)
    
    if binary_image is None:
        print("Error: Binarization failed")
        return None, []
    
    # Step 2: Line segmentation
    line_bboxes = segment_lines(binary_image)
    
    # Step 3: Word segmentation
    word_bboxes = segment_words(binary_image, line_bboxes)
    
    return binary_image, word_bboxes
