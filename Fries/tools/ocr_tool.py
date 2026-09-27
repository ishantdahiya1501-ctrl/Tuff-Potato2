import os
from PIL import Image
import pytesseract

def extract_text_from_image(file_path):
    if not file_path:
        return "Error: No image path provided."
    if not os.path.exists(file_path):
        return f"Error: File not found: {file_path}"
    supported_extensions = {
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".bmp",
        ".tiff",
        ".tif"
    }
    extension = os.path.splitext(file_path)[1].lower()
    if extension not in supported_extensions:
        return (
            f"Error: Unsupported image format '{extension}'. "
            f"Supported formats: PNG, JPG, JPEG, WEBP, BMP, TIFF."
        )
    try:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        text = text.strip()
        if not text:
            return "No text could be detected in the image."
        return text
    except Exception as e:
        return f"OCR Error: {str(e)}"