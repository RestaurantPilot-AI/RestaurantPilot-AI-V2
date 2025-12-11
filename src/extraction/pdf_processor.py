import os
from datetime import datetime
from pypdf import PdfReader

def extract_text_from_pdf(file_path):
    """
    Extracts text and metadata from a PDF file.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        tuple: (extracted_text, filename, text_length, page_count, extraction_timestamp)
    """
    # 1. Get the filename from the path
    filename = os.path.basename(file_path)

    # 2. Generate the timestamp
    extraction_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    extracted_text = ""
    page_count = 0

    try:
        # Open the PDF file
        reader = PdfReader(file_path)
        page_count = len(reader.pages)

        # Iterate over each page and extract text
        for page in reader.pages:
            text = page.extract_text()
            if text:
                extracted_text += text + "\n" # Add a newline between pages

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        extracted_text = ""
    except Exception as e:
        print(f"An error occurred while reading the PDF: {e}")
        extracted_text = ""

    # 3. Calculate text length
    text_length = len(extracted_text)

    # Return values in the exact order requested
    return extracted_text, filename, text_length, page_count, extraction_timestamp