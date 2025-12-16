# ============================================================================
# Gateway Thresholds (CORRECT VALUES FROM OCR ENGINE PDF)
# ============================================================================
# Processing Config
CONFIG = {
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
}