# src/extraction/ocr_processor.py

import pytesseract
import easyocr
import os
import sys
import shutil  # <--- Added for Linux detection
from typing import Tuple, Optional
import cv2
from datetime import datetime

# Import local config
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    import config
    CONFIG = config.CONFIG
    THRESHOLDS = config.THRESHOLDS
    PATHS = config.PATHS
else:
    from .config import CONFIG, THRESHOLDS, PATHS


class ImageProcessor:
    """
    Processes images and extracts quality metrics for OCR routing.
    """
    
    @staticmethod
    def load_image(image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            raise ValueError(f"Could not decode image: {image_path}")
        
        return img
    
    @staticmethod
    def resize_for_memory(img):
        height, width = img.shape
        max_size = CONFIG['max_image_size']
        
        if width > max_size:
            scale = max_size / width
            new_height = int(height * scale)
            img = cv2.resize(img, (max_size, new_height), interpolation=cv2.INTER_AREA)
        
        return img
    
    @staticmethod
    def get_sharpness(img):
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        sharpness = laplacian.var()
        return round(sharpness, 2)
    
    @staticmethod
    def get_contrast(img):
        contrast = img.std()
        return round(contrast, 2)
    
    @staticmethod
    def get_brightness(img):
        brightness = img.mean()
        return round(brightness, 2)
    
    @staticmethod
    def get_image_metrics(image_path):
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
    """
    
    def __init__(self):
        self.tesseract_available = self._check_tesseract()
        self.easyocr_reader = self._initialize_easyocr()
        self.routing_stats = {'tesseract': 0, 'easyocr': 0}
    
    def _check_tesseract(self):
        """Verify Tesseract installation (Linux & Windows compatible)."""
        try:
            # 1. Check if a specific path is provided in config (Local Windows dev)
            if PATHS.get('tesseract_cmd') and os.path.exists(PATHS['tesseract_cmd']):
                pytesseract.pytesseract.tesseract_cmd = PATHS['tesseract_cmd']
            
            # 2. Check system PATH (Linux/Streamlit Cloud)
            elif shutil.which("tesseract"):
                pytesseract.pytesseract.tesseract_cmd = shutil.which("tesseract")
            
            # 3. Fallback: Common Windows installation path
            elif os.path.exists(r"C:\Program Files\Tesseract-OCR\tesseract.exe"):
                 pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

            # Verify version
            version = pytesseract.get_tesseract_version()
            if CONFIG['enable_logging']:
                print(f"✓ Tesseract OCR detected (Version: {version})")
            return True
            
        except Exception as e:
            print(f"✗ Tesseract not found: {e}")
            print("  Falling back to EasyOCR only.")
            return False
    
    def _initialize_easyocr(self):
        """
        Initialize EasyOCR reader using local .pth models if available.
        """
        try:
            if CONFIG['enable_logging']:
                print("Loading EasyOCR model...")
            
            # Construct path to models folder relative to this script
            model_dir = os.path.join(os.path.dirname(__file__), 'easyocr_models')
            
            reader = easyocr.Reader(
                CONFIG['easyocr_languages'],
                gpu=CONFIG['easyocr_gpu'],
                model_storage_directory=model_dir,
                download_enabled=False # Don't download if models are present
            )
            
            if CONFIG['enable_logging']:
                print("✓ EasyOCR initialized")
            
            return reader
        
        except Exception as e:
            print(f"✗ EasyOCR initialization failed: {e}")
            sys.exit(1)
    
    def evaluate_image_quality(self, metrics):
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
        try:
            if CONFIG['enable_logging']:
                print(f"\n🔍 Gateway Check: {os.path.basename(image_path)}")
            
            metrics = ImageProcessor.get_image_metrics(image_path)
            ImageProcessor.log_metrics(metrics, image_path)
            
            evaluation = self.evaluate_image_quality(metrics)
            
            # PATH A: High quality → Fast Tesseract
            if evaluation['is_high_quality'] and self.tesseract_available:
                if CONFIG['enable_logging']:
                    print("✓ Route: TESSERACT (Fast Path)")
                text = self._run_tesseract(image_path)
                self.routing_stats['tesseract'] += 1
                route = 'tesseract'
            # PATH B: Low quality or Tesseract missing → Accurate EasyOCR
            else:
                if CONFIG['enable_logging']:
                    if not evaluation['is_high_quality']:
                        print("✓ Route: EASYOCR (Quality Issues Detected)")
                    else:
                        print("✓ Route: EASYOCR (Tesseract Unavailable)")
                
                text = self._run_easyocr(image_path)
                self.routing_stats['easyocr'] += 1
                route = 'easyocr'
            
            return text, route, metrics, evaluation
        
        except Exception as e:
            return f"ERROR: {str(e)}", "error", {}, {}
    
    def _run_tesseract(self, image_path):
        try:
            config = f'--oem {CONFIG["tesseract_oem"]} --psm {CONFIG["tesseract_psm"]}'
            text = pytesseract.image_to_string(image_path, config=config)
            return text.strip()
        except Exception as e:
            return f"Tesseract Error: {str(e)}"
    
    def _run_easyocr(self, image_path):
        try:
            results = self.easyocr_reader.readtext(
                image_path,
                detail=CONFIG['easyocr_detail']
            )
            if CONFIG['easyocr_detail'] == 0:
                text = " ".join(results)
            else:
                text = " ".join([res[1] for res in results])
            return text.strip()
        except Exception as e:
            return f"EasyOCR Error: {str(e)}"
    
    def print_stats(self):
        total = sum(self.routing_stats.values())
        if total > 0:
            print("\n" + "="*60)
            print("📊 ROUTING STATISTICS")
            print("="*60)
            print(f"  Tesseract: {self.routing_stats['tesseract']} images")
            print(f"  EasyOCR:   {self.routing_stats['easyocr']} images")
            print(f"  Total:     {total} images")
            print("="*60 + "\n")


ocr_router_instance = OCRRouter()

def extract_text_from_ocr(image_path: str) -> Optional[Tuple[str, str, int, int, str]]:
    try:
        filename = os.path.basename(image_path)
        extraction_timestamp = datetime.now().isoformat()
        
        extracted_text, route, metrics, evaluation = ocr_router_instance.route_image(image_path)
        
        if route == "error":
            print(f"ERROR: OCR processing failed for {image_path}. Details: {extracted_text}")
            return None
        
        if CONFIG['enable_logging']:
            ocr_router_instance.print_stats()
        
        page_count = 1
        text_length = len(extracted_text)
        
        return (extracted_text, filename, text_length, page_count, extraction_timestamp)

    except Exception as e:
        print(f"ERROR: Unhandled exception for {image_path}: {str(e)}")
        return None

if __name__ == "__main__":
    print("--- OCR Processor Standalone Mode ---")
    # Add your testing logic here