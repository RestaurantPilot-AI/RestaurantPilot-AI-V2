# Invoice Automation: Extraction Module

This module is responsible for the intelligent extraction of text and data from various document types, primarily PDFs and images, within the Invoice Automation system. It employs a robust pipeline that combines native PDF parsing with an adaptive Optical Character Recognition (OCR) gateway to ensure accurate and efficient data acquisition.

## Module Components

The `src/extraction` directory contains the following key components:

*   **[`file_processor.py`](invoice-automation/src/extraction/file_processor.py)**: The central orchestrator for handling different input file types. It directs incoming documents to the appropriate processing sub-modules based on their format.
*   **[`pdf_processor.py`](invoice-automation/src/extraction/pdf_processor.py)**: Specializes in extracting text directly from PDF documents. It utilizes native PDF parsing capabilities and can also convert PDF pages into image formats for subsequent OCR processing when direct text extraction is insufficient or unavailable.
*   **[`ocr_processor.py`](invoice-automation/src/extraction/ocr_processor.py)**: The core of the OCR functionality. This module includes:
    *   **Image Preprocessing**: Routines to enhance image quality for optimal OCR performance.
    *   **OCR Gateway (`OCRRouter` class)**: An intelligent routing mechanism that evaluates image quality metrics (e.g., sharpness, contrast, brightness) to dynamically select the most suitable OCR engine.
        *   **Tesseract OCR**: Utilized for its speed, typically chosen for high-quality images (the "fast path").
        *   **EasyOCR**: Employed for its higher accuracy, especially on lower-quality or more complex images (the "accurate path").
    *   **Tesseract Integration**: Interfaces with the Tesseract OCR engine via `pytesseract`.
    *   **EasyOCR Integration**: Integrates the `easyocr` library for robust and versatile OCR capabilities.
    *   **Dependency Management**: Manages the paths for the Tesseract executable and EasyOCR models, and includes dynamic adjustments to the system `PATH` for seamless standalone execution and debugging on Windows systems.
*   **[`invoice_extractor.py`](invoice-automation/src/extraction/invoice_extractor.py)**: Orchestrates the overall extraction workflow, coordinating between the file processing, PDF parsing, and OCR components to acquire raw text from invoices. It may also include initial text structuring or normalization routines.
*   **[`config.py`](invoice-automation/src/extraction/config.py)**: A centralized configuration file for the entire extraction process, defining critical parameters:
    *   **`THRESHOLDS`**: Image quality criteria (e.g., `sharpness`, `contrast`, `brightness_min`, `brightness_max`) that guide the OCR Gateway's routing decisions.
    *   **`CONFIG`**: General operational settings such as Tesseract's OEM/PSM modes, EasyOCR's GPU usage, supported languages, maximum image size for memory efficiency, logging preferences, and a `debug_metrics` flag for enabling standalone testing of `ocr_processor.py`.
    *   **`PATHS`**: Defines essential file system locations, including directories for test images, output results, log files, and the critical `tesseract_cmd` executable path (relative to the project root).
*   **[`regularize_file.py`](invoice-automation/src/extraction/regularize_file.py)**: Responsible for standardizing and cleaning the extracted raw text. This may involve applying regex-based corrections, normalizing various data formats, and performing initial field extractions.
