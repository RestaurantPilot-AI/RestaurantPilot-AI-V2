import pytesseract
import easyocr
import os
import sys
from typing import Tuple, Optional
import cv2
from datetime import datetime

# Import local config
if __name__ == "__main__":
    # If run directly, add the parent directory to sys.path to resolve local imports
    # This allows `config.py` to be imported directly.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Add Tesseract dependencies to PATH for standalone execution
    dependencies_path = os.path.join(current_dir, 'dependencies')
    tesseract_exec_path = os.path.join(current_dir, 'tesseract_models') # Directory containing tesseract.exe

    # Ensure both paths are in the PATH environment variable
    current_os_path = os.environ.get('PATH', '')
    paths_to_add = []
    if os.path.exists(dependencies_path) and dependencies_path not in current_os_path:
        paths_to_add.append(dependencies_path)
    if os.path.exists(tesseract_exec_path) and tesseract_exec_path not in current_os_path:
        paths_to_add.append(tesseract_exec_path)
    
    if paths_to_add:
        os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + current_os_path
    
    import config
    CONFIG = config.CONFIG
    THRESHOLDS = config.THRESHOLDS
    PATHS = config.PATHS
    # Conditionally set pytesseract.tesseract_cmd for standalone debugging
    if CONFIG['debug_metrics']:
        project_root = os.path.dirname(os.path.dirname(current_dir))
        pytesseract.tesseract_cmd = os.path.join(project_root, PATHS['tesseract_cmd'])
else:
    # When imported as part of a package, use relative import
    from .config import CONFIG, THRESHOLDS, PATHS


class ImageProcessor:
    """
    Processes images and extracts quality metrics for OCR routing.
    Optimized for 8GB RAM systems.
    """
    
    @staticmethod
    def load_image(image_path):
        """
        Load image in grayscale format.
        Returns: numpy array or None if loading fails
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not decode image: {image_path}")
        
        return img
    
    @staticmethod
    def resize_for_memory(img):
        """
        Resize image if it exceeds max size (memory optimization).
        Keeps aspect ratio.
        """
        import cv2
        height, width = img.shape
        max_size = CONFIG['max_image_size']
        
        if width > max_size:
            scale = max_size / width
            new_height = int(height * scale)
            img = cv2.resize(img, (max_size, new_height), interpolation=cv2.INTER_AREA)
        
        return img
    
    @staticmethod
    def get_sharpness(img):
        """
        Calculate sharpness using Laplacian variance.
        Higher = sharper image.
        
        Metric A: Sharpness (Variance of Laplacian)
        - Detects blur in images
        - Tesseract struggles with blurry images
        """
        import cv2
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()
        return round(sharpness, 2)
    
    @staticmethod
    def get_contrast(img):
        """
        Calculate RMS contrast using standard deviation.
        Higher = better text-background separation.
        
        Metric B: RMS Contrast
        - Detects faded ink or washed-out scans
        - Low contrast confuses OCR engines
        """
        contrast = img.std()
        return round(contrast, 2)
    
    @staticmethod
    def get_brightness(img):
        """
        Calculate brightness using pixel mean.
        Should be between 40 (dark) and 230 (bright).
        
        Metric C: Brightness Check
        - Rejects over-exposed (all white) images
        - Rejects under-exposed (all black) images
        """
        brightness = img.mean()
        return round(brightness, 2)
    
    @staticmethod
    def get_image_metrics(image_path):
        """
        Extract all 3 quality metrics.
        Returns: dict with sharpness, contrast, brightness
        """
        try:
            img = ImageProcessor.load_image(image_path)
            img = ImageProcessor.resize_for_memory(img)
            
            metrics = {
                'sharpness': ImageProcessor.get_sharpness(img),
                'contrast': ImageProcessor.get_contrast(img),
                'brightness': ImageProcessor.get_brightness(img),
                'image_shape': img.shape,
                'image_size_mb': img.nbytes / (1024 * 1024)
            }
            
            return metrics
        
        except Exception as e:
            raise Exception(f"Error extracting metrics from {image_path}: {str(e)}")
    
    @staticmethod
    def log_metrics(metrics, image_path):
        """Pretty-print metrics for debugging."""
        if CONFIG['debug_metrics']:
            print("\n" + "="*60)
            print(f"📊 QUALITY METRICS for: {os.path.basename(image_path)}")
            print("="*60)
            print(f"  Sharpness:    {metrics['sharpness']:>8.2f}  (threshold: {THRESHOLDS['sharpness']})")
            print(f"  Contrast:     {metrics['contrast']:>8.2f}  (threshold: {THRESHOLDS['contrast']})")
            print(f"  Brightness:   {metrics['brightness']:>8.2f}  (range: {THRESHOLDS['brightness_min']}-{THRESHOLDS['brightness_max']})")
            print(f"  Image Size:   {metrics['image_size_mb']:.2f} MB")
            print("="*60 + "\n")


class OCRRouter:
    """
    Intelligent OCR gateway that routes images based on quality.
    Optimized for 8GB RAM, CPU-only execution.
    """
    
    def __init__(self):
        """Initialize OCR engines (EasyOCR loads on startup)."""
        self.tesseract_available = self._check_tesseract()
        self.easyocr_reader = self._initialize_easyocr()
        self.routing_stats = {'tesseract': 0, 'easyocr': 0}
    
    def _check_tesseract(self):
        """Verify Tesseract installation."""
        try:
            # Set Tesseract command path explicitly if specified in config
            if 'tesseract_cmd' in PATHS and os.path.exists(PATHS['tesseract_cmd']):
                pytesseract.pytesseract.tesseract_cmd = PATHS['tesseract_cmd']
            
            pytesseract.get_tesseract_version()
            if CONFIG['enable_logging']:
                print("✓ Tesseract OCR detected")
            return True
        except Exception as e:
            print(f"✗ Tesseract not found: {e}")
            print("  Please ensure Tesseract is installed and its path is correctly configured.")
            print("  See README file for more information, or install: https://github.com/UB-Mannheim/tesseract/wiki")
            return False
    
    def _initialize_easyocr(self):
        """
        Initialize EasyOCR reader (CPU-only for 8GB RAM).
        First call loads model (~100MB) - this happens once at startup.
        """
        try:
            if CONFIG['enable_logging']:
                print("Loading EasyOCR model...")
            
            reader = easyocr.Reader(
                CONFIG['easyocr_languages'],
                gpu=CONFIG['easyocr_gpu'],  # False for 8GB RAM
                model_storage_directory=os.path.join(os.path.dirname(__file__), 'easyocr_models')
            )
            
            if CONFIG['enable_logging']:
                print("✓ EasyOCR initialized (CPU mode)")
            
            return reader
        
        except Exception as e:
            print(f"✗ EasyOCR initialization failed: {e}")
            sys.exit(1)
    
    def evaluate_image_quality(self, metrics):
        """
        Decision Logic: Evaluate if image is "Easy" (Tesseract-ready).
        
        An image is considered HIGH QUALITY if:
        - Sharpness > threshold (not too blurry)
        - Contrast > threshold (text visible)
        - Brightness within range (not over/under exposed)
        """
        is_sharp = metrics['sharpness'] > THRESHOLDS['sharpness']
        is_contrasted = metrics['contrast'] > THRESHOLDS['contrast']
        is_exposed = (THRESHOLDS['brightness_min'] < metrics['brightness'] <
                     THRESHOLDS['brightness_max'])
        
        return {
            'is_sharp': is_sharp,
            'is_contrasted': is_contrasted,
            'is_exposed': is_exposed,
            'is_high_quality': is_sharp and is_contrasted and is_exposed
        }
    
    def route_image(self, image_path):
        """
        Main routing logic: Gateway Check → Decision → Route to OCR engine.
        
        Returns: (extracted_text, route_taken, metrics, evaluation)
        """
        try:
            # STEP 1: Gateway Check - Extract metrics
            if CONFIG['enable_logging']:
                print(f"\n🔍 Gateway Check: {os.path.basename(image_path)}")
            
            metrics = ImageProcessor.get_image_metrics(image_path)
            ImageProcessor.log_metrics(metrics, image_path)
            
            # STEP 2: Decision Logic - Evaluate quality
            evaluation = self.evaluate_image_quality(metrics)
            
            # STEP 3: Route to OCR engine
            if evaluation['is_high_quality']:
                # PATH A: High quality → Fast Tesseract
                if self.tesseract_available:
                    if CONFIG['enable_logging']:
                        print("✓ Route: TESSERACT (Fast Path)")
                    text = self._run_tesseract(image_path)
                    self.routing_stats['tesseract'] += 1
                    route = 'tesseract'
                else:
                    print("⚠ Tesseract unavailable, falling back to EasyOCR")
                    text = self._run_easyocr(image_path)
                    self.routing_stats['easyocr'] += 1
                    route = 'easyocr'
            else:
                # PATH B: Low quality → Accurate EasyOCR
                if CONFIG['enable_logging']:
                    print("✓ Route: EASYOCR (Accuracy Path)")
                text = self._run_easyocr(image_path)
                self.routing_stats['easyocr'] += 1
                route = 'easyocr'
            
            return text, route, metrics, evaluation
        
        except Exception as e:
            return f"ERROR: {str(e)}", "error", {}, {}
    
    def _run_tesseract(self, image_path):
        """
        Route A: Tesseract OCR (Fast, CPU-efficient)
        
        Config:
        - OEM 1 = LSTM Engine (good accuracy/speed balance)
        - PSM 4 = Assume variable column text (invoices)
        """
        try:
            config = f'--oem {CONFIG["tesseract_oem"]} --psm {CONFIG["tesseract_psm"]}'
            text = pytesseract.image_to_string(image_path, config=config)
            return text.strip()
        
        except Exception as e:
            return f"Tesseract Error: {str(e)}"
    
    def _run_easyocr(self, image_path):
        """
        Route B: EasyOCR (Accurate, deep learning-based)
        
        More robust to:
        - Blurry images
        - Poor contrast
        - Complex layouts
        - Handwriting (multilingual support)
        """
        try:
            results = self.easyocr_reader.readtext(
                image_path,
                detail=CONFIG['easyocr_detail']  # 0 = text only
            )
            # EasyOCR results are a list of (bbox, text, confidence) if detail=1, or just text if detail=0
            # If detail=0, results is already a list of strings
            if CONFIG['easyocr_detail'] == 0:
                text = " ".join(results)
            else:
                text = " ".join([res[1] for res in results]) # Extract text from detailed results
            return text.strip()
        
        except Exception as e:
            return f"EasyOCR Error: {str(e)}"
    
    def get_stats(self):
        """Return routing statistics."""
        return self.routing_stats
    
    def print_stats(self):
        """Print routing statistics."""
        total = sum(self.routing_stats.values())
        if total > 0:
            print("\n" + "="*60)
            print("📊 ROUTING STATISTICS")
            print("="*60)
            print(f"  Tesseract (Fast):  {self.routing_stats['tesseract']} images")
            print(f"  EasyOCR (Accurate): {self.routing_stats['easyocr']} images")
            print(f"  Total:             {total} images")
            print("="*60 + "\n")


ocr_router_instance = OCRRouter()

def extract_text_from_ocr(image_path: str) -> Optional[Tuple[str, str, int, int, str]]:
    """
    Extracts text from an image using the intelligent OCR router.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
        Optional[Tuple[str, str, int, int, str]]: A tuple containing:
            (raw_data, filename, text_length, page_count, extraction_timestamp).
            Returns None if an error occurs.
    """
    try:
        # Pre-calculate metadata
        filename = os.path.basename(image_path)
        extraction_timestamp = datetime.now().isoformat()
        
        extracted_text, route, metrics, evaluation = ocr_router_instance.route_image(image_path)
        
        if route == "error":
            print(f"ERROR: OCR processing failed for {image_path}. Details: {extracted_text}")
            return None
        
        print(f"DEBUG: OCR Processor called for {image_path}. Route: {route}")
        
        if CONFIG['enable_logging']:
            ocr_router_instance.print_stats()
            ImageProcessor.log_metrics(metrics, image_path)
        
        # specific requirement: page_count is fixed at 1
        page_count = 1
        text_length = len(extracted_text)
        
        
        return (extracted_text, filename, text_length, page_count, extraction_timestamp)

    except Exception as e:
        print(f"ERROR: Unhandled exception in extract_text_from_ocr for {image_path}: {str(e)}")
        return None


if __name__ == "__main__":
    # For manual testing, ensure a test image is available.
    # This path assumes 'test_images' directory is at the root of the workspace.
    # Adjust this path if your test image is located elsewhere.
    test_image_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'test_images', 'high_quality.jpg')
    
    # Add the parent directory to sys.path for direct script execution to resolve relative imports
    # This allows `from .config import ...` to work when running `ocr_processor.py` directly.
    
    # IMPORTANT NOTE ON CONFIGURATION AND EXECUTION:
    # 1. 'from .config import CONFIG, THRESHOLDS, PATHS': This is a relative import.
    #    The sys.path modification above handles this for direct execution.
    #    Ensure 'invoice-automation/src/extraction/config.py' exists.
    # 2. Tesseract Path: The Tesseract executable path is configured in 'config.py' (PATHS['tesseract_cmd']).
    #    For direct execution, ensure this path is either absolute or correctly relative to where 'ocr_processor.py'
    #    is being executed from, or is configured to be found in your system's PATH.
    # 3. EasyOCR Models: The model storage directory is set relative to 'ocr_processor.py':
    #    'os.path.join(os.path.dirname(__file__), 'easyocr_models')'.
    #    Ensure the 'easyocr_models' folder (containing 'craft_mlt_25k.pth' and 'english_g2.pth')
    #    is located in the same directory as 'ocr_processor.py' or adjusted accordingly for testing.

    print(f"--- Starting OCR Processor Standalone Test ---")
    print(f"Attempting OCR on: {os.path.abspath(test_image_path)}")
    
    try:
        # Re-initialize ocr_router_instance after sys.path modification
        ocr_router_instance = OCRRouter()
        
        extracted_text_list = extract_text_from_ocr(test_image_path)
        
        if extracted_text_list:
            print("\n--- OCR Result ---")
            print("Extracted Text:")
            for text in extracted_text_list:
                print(text)
            print("------------------")
        else:
            print("\n--- OCR Result ---")
            print("No text extracted or an error occurred.")
            print("------------------")
            
        ocr_router_instance.print_stats()

    except FileNotFoundError as e:
        print(f"Error during manual test: {e}")
        print("Please ensure the test image path is correct and the image file exists.")
    except Exception as e:
        print(f"An unexpected error occurred during manual test: {e}")
