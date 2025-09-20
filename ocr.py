import io
import os
from google.cloud import vision

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'keytoken.json'

def detect_text_from_file(image_file):
    """Detects text from a local image file and returns the OCR result."""
    client = vision.ImageAnnotatorClient()

    with io.open(image_file, 'rb') as file:
        content = file.read()

    image = vision.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations

    if texts:
        full_text = texts[0].description
        return full_text
    else:
        return "No text detected."

# Path to your image file
image_path = 'test.jpeg'

# Get the OCR text
try:
    ocr_result = detect_text_from_file(image_path)
    print("--- OCR Result ---")
    print(ocr_result)
    print("------------------")
except Exception as e:
    print(f"An error occurred: {e}")