import io  # io fo
import os

import fitz  # PyMuPDF for handling PDF files
import pytesseract  # pytesseract for OCR
from PIL import Image  # Pillow for image processing

os.environ["TESSDATA_PREFIX"] = "./tessdata"


def convert_pdf_to_images(pdf):
    # Reopen the PDF file
    doc = fitz.open(stream=pdf.read(), filetype="pdf")

    switch_to_ocr_full_page = False
    # Reinitialize an empty string to store OCR extracted text
    ocr_text = ""
    better_ocr_result = ""
    # Extract images from each page and perform OCR
    for page_num in range(len(doc)):
        # Get the page
        page = doc.load_page(page_num)

        pix = page.get_pixmap()
        img_bytes = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_bytes))
        ocr_result = pytesseract.image_to_string(image, lang="pol")

        ocr_text += ocr_result

        # Extract images from the page
        # # Process each image
        for image_index, img in enumerate(page.get_images(full=True)):
            # Get the image from the XREF table
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]

            # Convert bytes to PIL Image
            try:
                image = Image.open(io.BytesIO(image_bytes))
            except Exception as e:
                switch_to_ocr_full_page = True
                print(f"Error: {e}")
                continue

            # image_filename = f"image_{page_num}_{image_index}.png"
            # with open(image_filename, "wb") as image_file:
            #     image_file.write(image_bytes)
            # Perform OCR using pytesseract
            ocr_result = pytesseract.image_to_string(image, lang="pol")

            better_ocr_result += ocr_result
    doc.close()

    return ocr_text if switch_to_ocr_full_page else better_ocr_result


# text = convert_pdf_to_images("example.pdf")
