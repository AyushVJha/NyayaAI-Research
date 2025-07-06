"""
OCR Pipeline for Indian Legal Documents
Handles multilingual OCR processing with advanced preprocessing and error handling
"""

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
import base64
from typing import Dict, List, Optional, Tuple, Union
import os
from pathlib import Path

from config.settings import OCR_CONFIG
from utils.logger import logger
from utils.error_handler import handle_exceptions, OCRError, ValidationError

class IndianLegalOCR:
    """
    Advanced OCR processor specifically designed for Indian legal documents
    Supports multiple Indian languages and handles poor quality scans
    """
    
    def __init__(self):
        self.languages = OCR_CONFIG["languages"]
        self.confidence_threshold = OCR_CONFIG["confidence_threshold"]
        self.tesseract_cmd = OCR_CONFIG["tesseract_cmd"]
        
        # Set Tesseract command path
        if os.path.exists(self.tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        
        # Verify Tesseract installation
        self._verify_tesseract()
        
    def _verify_tesseract(self):
        """Verify Tesseract installation and language support"""
        try:
            # Test basic Tesseract functionality
            test_image = Image.new('RGB', (100, 50), color='white')
            pytesseract.image_to_string(test_image)
            
            # Check available languages
            available_langs = pytesseract.get_languages()
            missing_langs = [lang for lang in self.languages if lang not in available_langs]
            
            if missing_langs:
                logger.warning(f"Missing Tesseract language packs: {missing_langs}")
                # Filter out missing languages
                self.languages = [lang for lang in self.languages if lang in available_langs]
            
            logger.info(f"OCR initialized with languages: {self.languages}")
            
        except Exception as e:
            raise OCRError(
                "Tesseract OCR not properly installed or configured",
                "TESSERACT_NOT_FOUND",
                {"error": str(e), "tesseract_cmd": self.tesseract_cmd}
            )
    
    @handle_exceptions(OCRError, "OCR processing failed")
    def process_image(self, image_input: Union[str, bytes, Image.Image], 
                     language: Optional[str] = None,
                     preprocessing: bool = True) -> Dict[str, any]:
        """
        Process an image and extract text using OCR
        
        Args:
            image_input: Image file path, bytes, or PIL Image
            language: Specific language code (if None, uses auto-detection)
            preprocessing: Whether to apply image preprocessing
            
        Returns:
            Dictionary containing extracted text, confidence scores, and metadata
        """
        logger.info("Starting OCR processing")
        
        # Load and validate image
        image = self._load_image(image_input)
        original_size = image.size
        
        # Preprocess image if enabled
        if preprocessing:
            image = self._preprocess_image(image)
        
        # Determine language(s) to use
        ocr_languages = self._determine_languages(image, language)
        
        # Extract text with confidence scores
        results = self._extract_text_with_confidence(image, ocr_languages)
        
        # Post-process results
        processed_results = self._post_process_results(results, original_size)
        
        logger.info(f"OCR completed. Extracted {len(processed_results['text'])} characters with {processed_results['avg_confidence']:.1f}% confidence")
        
        return processed_results
    
    def _load_image(self, image_input: Union[str, bytes, Image.Image]) -> Image.Image:
        """Load image from various input formats"""
        try:
            if isinstance(image_input, str):
                # File path
                if not os.path.exists(image_input):
                    raise ValidationError(f"Image file not found: {image_input}", "FILE_NOT_FOUND")
                image = Image.open(image_input)
                
            elif isinstance(image_input, bytes):
                # Bytes data
                image = Image.open(io.BytesIO(image_input))
                
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = image_input
                
            else:
                raise ValidationError("Unsupported image input type", "INVALID_IMAGE_TYPE")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Validate image size
            if image.size[0] < 50 or image.size[1] < 50:
                raise ValidationError("Image too small for OCR processing", "IMAGE_TOO_SMALL")
            
            return image
            
        except Exception as e:
            if isinstance(e, (ValidationError, OCRError)):
                raise e
            else:
                raise OCRError(f"Failed to load image: {str(e)}", "IMAGE_LOAD_FAILED")
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy"""
        logger.debug("Applying image preprocessing")
        
        try:
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Apply preprocessing steps
            if OCR_CONFIG["preprocessing"]["denoise"]:
                cv_image = self._denoise_image(cv_image)
            
            if OCR_CONFIG["preprocessing"]["contrast_enhancement"]:
                cv_image = self._enhance_contrast(cv_image)
            
            if OCR_CONFIG["preprocessing"]["deskew"]:
                cv_image = self._deskew_image(cv_image)
            
            # Additional preprocessing
            cv_image = self._sharpen_image(cv_image)
            cv_image = self._binarize_image(cv_image)
            
            # Convert back to PIL
            processed_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            
            logger.debug("Image preprocessing completed")
            return processed_image
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {str(e)}. Using original image.")
            return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Remove noise from image"""
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """Enhance image contrast"""
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back
        enhanced = cv2.merge([l, a, b])
        return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """Correct image skew"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using Hough transform
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        
        if lines is not None:
            # Calculate average angle
            angles = []
            for rho, theta in lines[:10]:  # Use first 10 lines
                angle = theta * 180 / np.pi
                if angle < 45:
                    angles.append(angle)
                elif angle > 135:
                    angles.append(angle - 180)
            
            if angles:
                avg_angle = np.mean(angles)
                
                # Rotate image to correct skew
                if abs(avg_angle) > 0.5:  # Only rotate if significant skew
                    height, width = image.shape[:2]
                    center = (width // 2, height // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
                    image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                         flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return image
    
    def _sharpen_image(self, image: np.ndarray) -> np.ndarray:
        """Sharpen image for better text recognition"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def _binarize_image(self, image: np.ndarray) -> np.ndarray:
        """Convert image to binary (black and white)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use adaptive thresholding for better results with varying lighting
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Convert back to 3-channel for consistency
        return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    
    def _determine_languages(self, image: Image.Image, specified_language: Optional[str]) -> str:
        """Determine which languages to use for OCR"""
        if specified_language and specified_language in self.languages:
            return specified_language
        
        # For now, use all available languages
        # In production, could implement language detection
        return '+'.join(self.languages)
    
    def _extract_text_with_confidence(self, image: Image.Image, languages: str) -> Dict[str, any]:
        """Extract text with confidence scores and word-level details"""
        try:
            # Configure Tesseract
            custom_config = f'--oem 3 --psm 6 -l {languages}'
            
            # Extract text
            text = pytesseract.image_to_string(image, config=custom_config)
            
            # Get detailed data with confidence scores
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            # Extract word-level information
            words_info = []
            for i in range(len(data['text'])):
                if int(data['conf'][i]) > 0:  # Filter out empty detections
                    word_info = {
                        'text': data['text'][i],
                        'confidence': int(data['conf'][i]),
                        'bbox': {
                            'x': data['left'][i],
                            'y': data['top'][i],
                            'width': data['width'][i],
                            'height': data['height'][i]
                        },
                        'block_num': data['block_num'][i],
                        'par_num': data['par_num'][i],
                        'line_num': data['line_num'][i],
                        'word_num': data['word_num'][i]
                    }
                    words_info.append(word_info)
            
            return {
                'text': text,
                'words': words_info,
                'languages_used': languages
            }
            
        except Exception as e:
            raise OCRError(f"Text extraction failed: {str(e)}", "EXTRACTION_FAILED")
    
    def _post_process_results(self, results: Dict[str, any], original_size: Tuple[int, int]) -> Dict[str, any]:
        """Post-process OCR results"""
        text = results['text']
        words = results['words']
        
        # Calculate statistics
        total_words = len(words)
        confident_words = [w for w in words if w['confidence'] >= self.confidence_threshold]
        avg_confidence = np.mean([w['confidence'] for w in words]) if words else 0
        
        # Filter low-confidence words if needed
        filtered_text = ' '.join([w['text'] for w in confident_words])
        
        # Detect potential issues
        issues = []
        if avg_confidence < self.confidence_threshold:
            issues.append("Low overall confidence")
        if len(confident_words) / max(total_words, 1) < 0.7:
            issues.append("Many low-confidence words")
        if len(text.strip()) < 50:
            issues.append("Very short text extracted")
        
        return {
            'text': text,
            'filtered_text': filtered_text,
            'words': words,
            'confident_words': confident_words,
            'statistics': {
                'total_words': total_words,
                'confident_words': len(confident_words),
                'avg_confidence': avg_confidence,
                'confidence_ratio': len(confident_words) / max(total_words, 1),
                'character_count': len(text),
                'original_image_size': original_size
            },
            'languages_used': results['languages_used'],
            'quality_issues': issues,
            'processing_successful': len(issues) == 0 and avg_confidence >= self.confidence_threshold
        }
    
    def process_multi_page_document(self, image_paths: List[str]) -> Dict[str, any]:
        """Process multiple pages of a document"""
        logger.info(f"Processing multi-page document with {len(image_paths)} pages")
        
        all_results = []
        combined_text = []
        total_confidence = 0
        total_words = 0
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing page {i+1}/{len(image_paths)}")
                page_result = self.process_image(image_path)
                
                page_result['page_number'] = i + 1
                all_results.append(page_result)
                
                combined_text.append(f"--- Page {i+1} ---\n{page_result['text']}")
                total_confidence += page_result['statistics']['avg_confidence'] * page_result['statistics']['total_words']
                total_words += page_result['statistics']['total_words']
                
            except Exception as e:
                logger.error(f"Failed to process page {i+1}: {str(e)}")
                all_results.append({
                    'page_number': i + 1,
                    'error': str(e),
                    'processing_successful': False
                })
        
        # Calculate overall statistics
        successful_pages = [r for r in all_results if r.get('processing_successful', False)]
        avg_confidence = total_confidence / max(total_words, 1)
        
        return {
            'combined_text': '\n\n'.join(combined_text),
            'page_results': all_results,
            'document_statistics': {
                'total_pages': len(image_paths),
                'successful_pages': len(successful_pages),
                'overall_confidence': avg_confidence,
                'total_words': total_words,
                'total_characters': len('\n\n'.join(combined_text))
            },
            'processing_successful': len(successful_pages) > 0
        }

# Convenience functions
def extract_text_from_image(image_input: Union[str, bytes, Image.Image], 
                           language: Optional[str] = None) -> str:
    """Simple function to extract text from an image"""
    ocr = IndianLegalOCR()
    result = ocr.process_image(image_input, language)
    return result['text']

def process_legal_document_images(image_paths: List[str]) -> Dict[str, any]:
    """Process multiple images as a single legal document"""
    ocr = IndianLegalOCR()
    return ocr.process_multi_page_document(image_paths)

def get_ocr_confidence(image_input: Union[str, bytes, Image.Image]) -> float:
    """Get OCR confidence score for an image"""
    ocr = IndianLegalOCR()
    result = ocr.process_image(image_input)
    return result['statistics']['avg_confidence']
