"""
Unit tests for text normalization functionality
"""

import pytest
from utils.text_normalization import (
    IndianLegalTextNormalizer,
    normalize_legal_text,
    extract_document_sections,
    augment_legal_text
)
from utils.error_handler import DocumentProcessingError

class TestIndianLegalTextNormalizer:
    """Test cases for Indian Legal Text Normalizer"""
    
    @pytest.fixture
    def normalizer(self):
        """Create normalizer instance for testing"""
        return IndianLegalTextNormalizer()
    
    @pytest.fixture
    def sample_legal_text(self):
        """Sample legal document text for testing"""
        return """
        IN THE SUPREME COURT OF INDIA
        
        CIVIL APPEAL NO. 1234 OF 2023
        
        BETWEEN:
        
        ABC Company Ltd.                    ... Petitioner
        
        VERSUS
        
        XYZ Corporation                     ... Respondent
        
        CORAM: HON'BLE MR. JUSTICE JOHN DOE
        
        JUDGMENT
        
        This appeal arises out of the judgement dated 15.03.2023 passed by the High Court of Delhi in Writ Petition No. 567/2022. The petitioner has challenged the order passed by the respondent under Section 147 of the Income Tax Act, 1961.
        
        The facts of the case are as follows:
        1. The petitioner is a company incorporated under the Companies Act, 2013.
        2. The respondent issued a notice under Section 148 of the IT Act.
        3. The petitioner filed a writ petition challenging the said notice.
        
        HELD:
        
        After hearing the learned counsel for both parties, this Court is of the view that the impugned order is liable to be set aside. The appeal is allowed.
        """
    
    @pytest.fixture
    def messy_text(self):
        """Text with formatting issues for testing cleanup"""
        return """
        This    is   a    text   with     irregular    spacing.
        
        
        
        Multiple   line   breaks   and   inconsistent   formatting.
        
        Some unicode characters: "smart quotes" and – em dashes.
        
        OCR errors like: judgrnent, petiti0ner, resp0ndent, c0urt.
        """
    
    def test_normalizer_initialization(self, normalizer):
        """Test normalizer initialization"""
        assert normalizer is not None
        assert hasattr(normalizer, 'legal_abbreviations')
        assert hasattr(normalizer, 'court_names')
        assert hasattr(normalizer, 'legal_terms')
        assert len(normalizer.legal_abbreviations) > 0
        assert len(normalizer.court_names) > 0
    
    def test_basic_text_normalization(self, normalizer, sample_legal_text):
        """Test basic text normalization functionality"""
        normalized = normalizer.normalize_text(sample_legal_text)
        
        assert isinstance(normalized, str)
        assert len(normalized) > 0
        assert len(normalized) <= len(sample_legal_text)  # Should not expand significantly
        
        # Check that basic cleanup happened
        assert "   " not in normalized  # Multiple spaces should be cleaned
        
    def test_unicode_normalization(self, normalizer):
        """Test Unicode character normalization"""
        unicode_text = "This has "smart quotes" and – em dashes and … ellipsis"
        normalized = normalizer._normalize_unicode(unicode_text)
        
        assert '"' in normalized  # Smart quotes converted
        assert '--' in normalized or '-' in normalized  # Em dash converted
        assert '...' in normalized  # Ellipsis converted
    
    def test_whitespace_cleaning(self, normalizer, messy_text):
        """Test whitespace cleaning functionality"""
        # Test with structure preservation
        cleaned_with_structure = normalizer._clean_whitespace(messy_text, preserve_structure=True)
        assert '\n\n' in cleaned_with_structure  # Paragraph breaks preserved
        
        # Test without structure preservation
        cleaned_without_structure = normalizer._clean_whitespace(messy_text, preserve_structure=False)
        assert '\n\n' not in cleaned_without_structure  # All whitespace normalized
    
    def test_abbreviation_expansion(self, normalizer):
        """Test legal abbreviation expansion"""
        text_with_abbrevs = "The SC held that the HC was wrong. The CrPC and CPC apply."
        normalized = normalizer._normalize_abbreviations(text_with_abbrevs)
        
        assert "Supreme Court" in normalized
        assert "High Court" in normalized
        assert "Criminal Procedure Code" in normalized
        assert "Civil Procedure Code" in normalized
    
    def test_citation_standardization(self, normalizer):
        """Test legal citation standardization"""
        text_with_citations = "See AIR 1950 SC 124 and (2020) 1 SCC 456"
        normalized = normalizer._standardize_citations(text_with_citations)
        
        # Should contain standardized citation format
        assert "AIR" in normalized
        assert "SCC" in normalized
        assert "1950" in normalized
        assert "2020" in normalized
    
    def test_ocr_error_correction(self, normalizer):
        """Test OCR error correction"""
        text_with_errors = "The judgrnent was passed by the c0urt. The petiti0ner and resp0ndent were present."
        corrected = normalizer._fix_ocr_errors(text_with_errors)
        
        assert "judgment" in corrected
        assert "court" in corrected
        assert "petitioner" in corrected
        assert "respondent" in corrected
    
    def test_legal_term_normalization(self, normalizer):
        """Test legal terminology normalization"""
        text_with_terms = "The PETITIONER filed a WRIT PETITION against the RESPONDENT."
        normalized = normalizer._normalize_legal_terms(text_with_terms)
        
        # Should convert to lowercase for consistency
        assert "petitioner" in normalized
        assert "respondent" in normalized
    
    def test_section_extraction(self, normalizer, sample_legal_text):
        """Test extraction of document sections"""
        sections = normalizer.extract_key_sections(sample_legal_text)
        
        assert isinstance(sections, dict)
        # Should extract some sections from the sample text
        if sections:
            assert any(key in sections for key in ['case_title', 'court_name', 'judges'])
    
    def test_text_augmentation(self, normalizer):
        """Test text data augmentation"""
        simple_text = "The petitioner filed a case against the respondent in the court."
        augmented = normalizer.augment_text_data(simple_text, augmentation_factor=3)
        
        assert isinstance(augmented, list)
        assert len(augmented) == 3
        assert augmented[0] == simple_text  # First should be original
        
        # Other versions should be different
        for i in range(1, len(augmented)):
            assert augmented[i] != simple_text
    
    def test_empty_text_handling(self, normalizer):
        """Test handling of empty or invalid text"""
        with pytest.raises(DocumentProcessingError):
            normalizer.normalize_text("")
        
        with pytest.raises(DocumentProcessingError):
            normalizer.normalize_text(None)
    
    def test_very_long_text_handling(self, normalizer):
        """Test handling of very long texts"""
        long_text = "This is a test sentence. " * 1000  # 1000 repetitions
        normalized = normalizer.normalize_text(long_text)
        
        assert isinstance(normalized, str)
        assert len(normalized) > 0

class TestConvenienceFunctions:
    """Test convenience functions for text normalization"""
    
    def test_normalize_legal_text_function(self):
        """Test the convenience function for text normalization"""
        test_text = "The SC held that the judgrnent was correct."
        normalized = normalize_legal_text(test_text)
        
        assert isinstance(normalized, str)
        assert "Supreme Court" in normalized
        assert "judgment" in normalized
    
    def test_extract_document_sections_function(self):
        """Test the convenience function for section extraction"""
        test_text = """
        IN THE SUPREME COURT OF INDIA
        CORAM: HON'BLE MR. JUSTICE JOHN DOE
        DATED: 15th March, 2023
        """
        sections = extract_document_sections(test_text)
        
        assert isinstance(sections, dict)
        # May or may not extract sections depending on text format
    
    def test_augment_legal_text_function(self):
        """Test the convenience function for text augmentation"""
        test_text = "The court delivered the judgment."
        augmented = augment_legal_text(test_text, factor=2)
        
        assert isinstance(augmented, list)
        assert len(augmented) == 2
        assert augmented[0] == test_text

class TestSpecialCases:
    """Test special cases and edge conditions"""
    
    def test_multilingual_content(self):
        """Test handling of multilingual content"""
        # Text with Hindi/Devanagari content
        multilingual_text = "The petitioner राम कुमार filed a case against respondent श्याम लाल."
        normalized = normalize_legal_text(multilingual_text)
        
        assert isinstance(normalized, str)
        # Should preserve non-English content
        assert "राम कुमार" in normalized
        assert "श्याम लाल" in normalized
    
    def test_complex_citations(self):
        """Test handling of complex citation formats"""
        citation_text = """
        Refer to AIR 1950 SC 124, (2020) 1 SCC 456, 2019 (2) Bom LR 789,
        and Madan Mohan vs State of UP, AIR 1954 All 445.
        """
        normalized = normalize_legal_text(citation_text)
        
        assert isinstance(normalized, str)
        # Should handle various citation formats
        assert "AIR" in normalized
        assert "SCC" in normalized
    
    def test_special_characters_and_symbols(self):
        """Test handling of special legal characters and symbols"""
        special_text = "Section 147(a)(i) & 148(b)(ii) of the I.T. Act, 1961 @ 15% p.a."
        normalized = normalize_legal_text(special_text)
        
        assert isinstance(normalized, str)
        assert "Section" in normalized
        assert "Act" in normalized
    
    def test_date_formats(self):
        """Test handling of various date formats"""
        date_text = "Dated 15.03.2023, 15th March 2023, March 15, 2023"
        normalized = normalize_legal_text(date_text)
        
        assert isinstance(normalized, str)
        assert "2023" in normalized
    
    def test_case_sensitivity(self):
        """Test case sensitivity handling"""
        mixed_case_text = "THE SUPREME COURT held that the petitioner was RIGHT."
        normalized = normalize_legal_text(mixed_case_text)
        
        assert isinstance(normalized, str)
        # Should maintain proper capitalization for legal terms
        assert "Supreme Court" in normalized or "supreme court" in normalized

class TestPerformance:
    """Test performance aspects of text normalization"""
    
    def test_large_document_processing(self):
        """Test processing of large documents"""
        # Create a large document
        large_text = """
        This is a sample legal document with multiple paragraphs.
        It contains various legal terms, citations, and formatting.
        The document discusses important legal principles and precedents.
        """ * 100  # Repeat 100 times
        
        import time
        start_time = time.time()
        normalized = normalize_legal_text(large_text)
        end_time = time.time()
        
        assert isinstance(normalized, str)
        assert len(normalized) > 0
        
        # Should complete within reasonable time (adjust threshold as needed)
        processing_time = end_time - start_time
        assert processing_time < 10.0  # Should complete within 10 seconds
    
    def test_memory_efficiency(self):
        """Test memory efficiency with repeated processing"""
        test_text = "The Supreme Court held that the judgment was correct."
        
        # Process the same text multiple times
        for _ in range(100):
            normalized = normalize_legal_text(test_text)
            assert isinstance(normalized, str)
        
        # If we reach here without memory errors, the test passes

if __name__ == "__main__":
    pytest.main([__file__])
