# ============================================================================
# Gateway Thresholds (CORRECT VALUES FROM OCR ENGINE PDF)
# ============================================================================
THRESHOLDS = {
    'sharpness': 100.0,      # Min variance of Laplacian (blur detection)
    'contrast': 40.0,        # Min standard deviation (fade detection)
    'brightness_min': 40,    # Min pixel mean (too dark check)
    'brightness_max': 230    # Max pixel mean (over-exposed check)
}

# Processing Config
CONFIG = {
    'tesseract_oem': '1',           # OEM 1 = LSTM Engine
    'tesseract_psm': '4',           # PSM 4 = Variable column text (invoices)
    'easyocr_detail': 0,            # 0 = text only, 1 = with confidence
    'easyocr_gpu': False,           # Force CPU (8GB RAM constraint)
    'easyocr_languages': ['en'],    # Language model
    'max_image_size': 1920,         # Max width for memory efficiency
    'enable_logging': True,         # Console output
    'debug_metrics': True          # Show quality scores
}

# Paths
PATHS = {
    'test_images': './test_images', # These paths might need adjustment if used
    'output_results': './results',  # from within invoice-automation
    'log_file': './ocr_router.log',
    'tesseract_cmd': 'src/extraction/tesseract_models/tesseract.exe' # Path relative to the invoice-automation directory
}