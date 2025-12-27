import easyocr
import os
import sys
from pathlib import Path
from typing import Tuple, Optional
import cv2
from datetime import datetime
from .config import CONFIG

# Import local config
if __name__ == "__main__":
    # If run directly, add the parent directory to sys.path to resolve local imports
    # This allows `config.py` to be imported directly.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    dependencies_path = os.path.join(current_dir, 'dependencies')
    # Ensure both paths are in the PATH environment variable
    current_os_path = os.environ.get('PATH', '')
    paths_to_add = []
    if os.path.exists(dependencies_path) and dependencies_path not in current_os_path:
        paths_to_add.append(dependencies_path)
    
    if paths_to_add:
        os.environ['PATH'] = os.pathsep.join(paths_to_add) + os.pathsep + current_os_path


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

class OCRRouter:
    """
    Intelligent OCR gateway that routes images based on quality.
    Optimized for 8GB RAM, CPU-only execution.
    """
    
    def __init__(self):
        """Initialize OCR engines lazily (EasyOCR loads on first use)."""
        self._easyocr_reader = None  # Lazy initialization
        self.routing_stats = {'easyocr': 0}
    
    @property
    def easyocr_reader(self):
        """Lazy-load EasyOCR reader only when needed."""
        if self._easyocr_reader is None:
            self._easyocr_reader = self._initialize_easyocr()
        return self._easyocr_reader
    
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
                model_storage_directory=str(Path(__file__).parent / 'easyocr_models')
            )
            
            if CONFIG['enable_logging']:
                print("✓ EasyOCR initialized (CPU mode)")
            
            return reader
        
        except Exception as e:
            print(f"✗ EasyOCR initialization failed: {e}")
            sys.exit(1)
    
    def route_image(self, image_path):
        """
        Main routing logic: Gateway Check → Decision → Route to OCR engine.
        
        Returns: (extracted_text, route_taken, metrics, evaluation)
        """
        try:
            if CONFIG['enable_logging']:
                print("✓ Route: EASYOCR (Accuracy Path)")
                text = self._run_easyocr(image_path)
                self.routing_stats['easyocr'] += 1
                route = 'easyocr'
            return text, route
        
        except Exception as e:
            return f"ERROR: {str(e)}", "error", {}, {}
    
    def _run_easyocr(self, image_path):
        """
        EasyOCR (Accurate, deep learning-based)
        
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
        filename = Path(image_path).name
        extraction_timestamp = datetime.now().isoformat()
        
        extracted_text, route = ocr_router_instance.route_image(image_path)
        
        if route == "error":
            print(f"ERROR: OCR processing failed for {image_path}. Details: {extracted_text}")
            return None
        
        print(f"DEBUG: OCR Processor called for {image_path}. Route: {route}")
        
        page_count = 1
        text_length = len(extracted_text)
        
        return (extracted_text, filename, text_length, page_count, extraction_timestamp)

    except Exception as e:
        print(f"ERROR: Unhandled exception in extract_text_from_ocr for {image_path}: {str(e)}")
        return None

def extract_only_text_from_ocr(image_path: str) -> str:
    """
    Extracts text from an image using the intelligent OCR router.
    
    Args:
        image_path (str): The path to the image file.
        
    Returns:
            Extracted text as a single string.
            Returns an empty string if extraction fails..
    """
    try:

        
        extracted_text, route = ocr_router_instance.route_image(image_path)
        
        if route == "error":
            print(f"ERROR: OCR processing failed for {image_path}. Details: {extracted_text}")
            return None
        
        print(f"DEBUG: OCR Processor called for {image_path}. Route: {route}")
        
        return extracted_text

    except Exception as e:
        print(f"ERROR: Unhandled exception in extract_text_from_ocr for {image_path}: {str(e)}")
        return None
