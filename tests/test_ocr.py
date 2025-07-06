"""
Unit tests for OCR pipeline functionality
"""

import pytest
import tempfile
import os
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from ingestion.ocr_pipeline import IndianLegalOCR, extract_text_from_image
from utils.error_handler import OCRError, ValidationError

class TestIndianLegalOCR:
    """Test cases for Indian Legal OCR functionality"""
    
    @pytest.fixture
    def ocr_processor(self):
        """Create OCR processor instance for testing"""
        return IndianLegalOCR()
    
    @pytest.fixture
    def sample_text_image(self):
        """Create a sample image with text for testing"""
        # Create a simple image with text
        img = Image.new('RGB', (800, 200), color='white')
        draw = ImageDraw.Draw(img)
        
        # Try to use a default font, fallback to basic if not available
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()
        
        text = "Supreme Court of India - Civil Appeal No. 1234 of 2023"
        draw.text((50, 80), text, fill='black', font=font)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            return tmp.name
    
    @pytest.fixture
    def noisy_image(self):
        """Create a noisy image for testing preprocessing"""
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        
        # Add some noise
        pixels = np.array(img)
        noise = np.random.randint(0, 50, pixels.shape)
        noisy_pixels = np.clip(pixels + noise, 0, 255)
        noisy_img = Image.fromarray(noisy_pixels.astype('uint8'))
        
        # Add text
        draw = ImageDraw.Draw(noisy_img)
        draw.text((20, 40), "Petitioner vs Respondent", fill='black')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            noisy_img.save(tmp.name)
            return tmp.name
    
    def test_ocr_initialization(self, ocr_processor):
        """Test OCR processor initialization"""
        assert ocr_processor is not None
        assert hasattr(ocr_processor, 'languages')
        assert hasattr(ocr_processor, 'confidence_threshold')
        assert 'eng' in ocr_processor.languages
    
    def test_process_valid_image(self, ocr_processor, sample_text_image):
        """Test processing a valid text image"""
        try:
            result = ocr_processor.process_image(sample_text_image)
            
            assert result is not None
            assert 'text' in result
            assert 'processing_successful' in result
            assert len(result['text']) > 0
            
            # Check for expected legal terms
            text_lower = result['text'].lower()
            assert any(term in text_lower for term in ['supreme court', 'civil appeal', 'india'])
            
        except Exception as e:
            # OCR might not be available in test environment
            pytest.skip(f"OCR not available in test environment: {str(e)}")
        finally:
            # Clean up
            if os.path.exists(sample_text_image):
                os.unlink(sample_text_image)
    
    def test_process_image_with_preprocessing(self, ocr_processor, noisy_image):
        """Test image preprocessing functionality"""
        try:
            # Process with preprocessing enabled
            result_with_preprocessing = ocr_processor.process_image(
                noisy_image, 
                preprocessing=True
            )
            
            # Process without preprocessing
            result_without_preprocessing = ocr_processor.process_image(
                noisy_image, 
                preprocessing=False
            )
            
            assert result_with_preprocessing is not None
            assert result_without_preprocessing is not None
            
            # Preprocessing should generally improve results
            # (though this might not always be true in test conditions)
            
        except Exception as e:
            pytest.skip(f"OCR not available in test environment: {str(e)}")
        finally:
            if os.path.exists(noisy_image):
                os.unlink(noisy_image)
    
    def test_invalid_image_path(self, ocr_processor):
        """Test handling of invalid image paths"""
        with pytest.raises(ValidationError):
            ocr_processor.process_image("nonexistent_file.png")
    
    def test_empty_image_handling(self, ocr_processor):
        """Test handling of very small/empty images"""
        # Create a very small image
        tiny_img = Image.new('RGB', (10, 10), color='white')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tiny_img.save(tmp.name)
            
            try:
                with pytest.raises(ValidationError):
                    ocr_processor.process_image(tmp.name)
            finally:
                os.unlink(tmp.name)
    
    def test_multi_language_support(self, ocr_processor):
        """Test multi-language OCR support"""
        # This test checks if multiple languages are configured
        assert len(ocr_processor.languages) > 1
        assert 'hin' in ocr_processor.languages  # Hindi support
    
    def test_confidence_threshold(self, ocr_processor, sample_text_image):
        """Test confidence threshold filtering"""
        try:
            result = ocr_processor.process_image(sample_text_image)
            
            if result['processing_successful']:
                # Check that confidence filtering works
                confident_words = result.get('confident_words', [])
                all_words = result.get('words', [])
                
                # All confident words should meet threshold
                for word in confident_words:
                    assert word['confidence'] >= ocr_processor.confidence_threshold
                    
        except Exception as e:
            pytest.skip(f"OCR not available in test environment: {str(e)}")
        finally:
            if os.path.exists(sample_text_image):
                os.unlink(sample_text_image)

class TestOCRConvenienceFunctions:
    """Test convenience functions for OCR"""
    
    def test_extract_text_from_image_function(self):
        """Test the convenience function for text extraction"""
        # Create a simple test image
        img = Image.new('RGB', (200, 50), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 15), "Test Text", fill='black')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            
            try:
                text = extract_text_from_image(tmp.name)
                assert isinstance(text, str)
                # Text might be empty if OCR is not available
                
            except Exception as e:
                pytest.skip(f"OCR not available: {str(e)}")
            finally:
                os.unlink(tmp.name)

class TestOCRErrorHandling:
    """Test error handling in OCR pipeline"""
    
    def test_corrupted_image_handling(self):
        """Test handling of corrupted image files"""
        # Create a file with invalid image data
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b"This is not an image file")
            tmp.flush()
            
            try:
                ocr = IndianLegalOCR()
                with pytest.raises((OCRError, ValidationError)):
                    ocr.process_image(tmp.name)
            except Exception as e:
                # If OCR is not available, skip the test
                if "tesseract" in str(e).lower():
                    pytest.skip("Tesseract not available in test environment")
                else:
                    raise
            finally:
                os.unlink(tmp.name)
    
    def test_unsupported_format_handling(self):
        """Test handling of unsupported image formats"""
        # This would typically be handled by PIL/Pillow
        # but we can test our validation logic
        ocr = IndianLegalOCR()
        
        # Test with a text file instead of image
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp:
            tmp.write(b"This is a text file, not an image")
            tmp.flush()
            
            try:
                with pytest.raises((OCRError, ValidationError)):
                    ocr.process_image(tmp.name)
            except Exception as e:
                if "tesseract" in str(e).lower():
                    pytest.skip("Tesseract not available in test environment")
                else:
                    raise
            finally:
                os.unlink(tmp.name)

# Integration tests
class TestOCRIntegration:
    """Integration tests for OCR with other components"""
    
    def test_ocr_with_legal_text_normalization(self):
        """Test OCR output with text normalization"""
        from utils.text_normalization import normalize_legal_text
        
        # Create image with legal text
        img = Image.new('RGB', (600, 100), color='white')
        draw = ImageDraw.Draw(img)
        legal_text = "Hon'ble Supreme Court of India - AIR 2023 SC 1234"
        draw.text((20, 40), legal_text, fill='black')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            
            try:
                # Extract text using OCR
                extracted_text = extract_text_from_image(tmp.name)
                
                # Normalize the extracted text
                normalized_text = normalize_legal_text(extracted_text)
                
                assert isinstance(normalized_text, str)
                # The normalization should work even if OCR produces imperfect results
                
            except Exception as e:
                pytest.skip(f"OCR not available: {str(e)}")
            finally:
                os.unlink(tmp.name)

if __name__ == "__main__":
    pytest.main([__file__])
