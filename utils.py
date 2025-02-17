import io  # io fo
import os
import docx

import fitz  # PyMuPDF for handling PDF files
import pytesseract  # pytesseract for OCR
from PIL import Image  # Pillow for image processing


os.environ["TESSDATA_PREFIX"] = "./tessdata"


def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file, including OCR for images."""
    full_ocr_result = ""

    # Open the PDF
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")

    # Extract images and perform OCR
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")

        image = Image.open(io.BytesIO(img_bytes))
        page_text = pytesseract.image_to_string(image, lang="pol")
        full_ocr_result += page_text + "\n"

        # Check for embedded images
        for img in page.get_images(full=True):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            try:
                image = Image.open(io.BytesIO(image_bytes))
                image_text = pytesseract.image_to_string(image, lang="pol")
                full_ocr_result += image_text + "\n"
            except Exception as e:
                print(f"Error processing image on page {page_num}: {e}")

    doc.close()
    return full_ocr_result


def extract_text_from_docx(docx_file):
    """Extract text from a DOCX file, including OCR for images."""
    full_ocr_result = ""

    # Open DOCX and extract text
    doc = docx.Document(docx_file)
    for para in doc.paragraphs:
        full_ocr_result += para.text + "\n"

    # Extract images and perform OCR
    for rel in doc.part.rels:
        if "image" in doc.part.rels[rel].target_ref:
            image_data = doc.part.rels[rel].target_part.blob
            try:
                image = Image.open(io.BytesIO(image_data))
                image_text = pytesseract.image_to_string(image, lang="pol")
                full_ocr_result += image_text + "\n"
            except Exception as e:
                print(f"Error processing image in DOCX: {e}")

    return full_ocr_result


def convert_document_to_text(file, file_type):

    if file_type == "pdf":
        return extract_text_from_pdf(file)
    elif file_type == "docx":
        return extract_text_from_docx(file)
    else:
        raise ValueError("Unsupported file type. Use 'pdf' or 'docx'.")
