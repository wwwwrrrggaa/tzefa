from flask import Flask, request, render_template_string
import cv2
import numpy as np
import sys
import os
from pathlib import Path

# Add paths to import Tzefa-OCR modules
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'Tzefa-Ocr'))

from main import image_to_code

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Tzefa OCR</title>
</head>
<body>
    <h1>Upload an Image for OCR</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept="image/*" required>
        <input type="submit" value="Upload and Process">
    </form>
    {% if result %}
    <h2>OCR Result:</h2>
    <pre>{{ result }}</pre>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Read image from uploaded file
            img_array = np.frombuffer(file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            # Process with Tzefa-OCR
            result = image_to_code(img)

            # Format result as string
            result_str = "\n".join(result)

            return render_template_string(HTML_TEMPLATE, result=result_str)
    return render_template_string(HTML_TEMPLATE, result=None)

if __name__ == '__main__':
    app.run(debug=True)
