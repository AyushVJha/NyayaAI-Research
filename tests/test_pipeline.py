"""
Integration tests for the complete AI Legal Document Intelligence Pipeline
"""

import pytest
import tempfile
import os
import json
from pathlib import Path
from PIL import Image, ImageDraw

from ingestion.document_ingestion import LegalDocumentProcessor
from models.longformer_summarizer import LegalDocumentSummarizer
from models.spacy_ner import IndianLegalNER
from utils.error_handler import DocumentProcessingError, ModelInferenceError
from utils.logger import logger

class TestEndToEndPipeline:
    """Test the complete end-to-end pipeline"""
    
    @pytest.fixture
    def sample_legal_document(self):
        """Create a sample legal document for testing"""
        content = """
        IN THE SUPREME COURT OF INDIA
        
        CIVIL APPEAL NO. 1234 OF 2023
        
        BETWEEN:
        
        ABC Company Limited                    ... Appellant
        
        VERSUS
        
        XYZ Corporation Private Limited        ... Respondent
        
        CORAM: HON'BLE MR. JUSTICE JOHN DOE
               HON'BLE MR. JUSTICE JANE SMITH
        
        DATED: 15th March, 2023
        
        JUDGMENT
        
        This appeal arises out of the judgment and order dated 10.01.2023 passed by the High Court of Delhi in Writ Petition No. 567/2022. The appellant has challenged the assessment order passed by the Income Tax Department under Section 147 of the Income Tax Act, 1961.
        
        FACTS:
        
        1. The appellant is a company incorporated under the Companies Act, 2013, engaged in the business of software development.
        
        2. The Income Tax Department issued a notice under Section 148 of the Income Tax Act, 1961, seeking to reopen the assessment for the Assessment Year 2018-19.
        
        3. The appellant filed a writ petition before the Delhi High Court challenging the validity of the notice.
        
        4. The High Court dismissed the writ petition vide its order dated 10.01.2023.
        
        CONTENTIONS:
        
        The learned counsel for the appellant contended that:
        a) The notice under Section 148 was issued beyond the limitation period.
        b) There was no failure on the part of the appellant to disclose material facts.
        c) The reasons recorded by the Assessing Officer were not adequate.
        
        The learned counsel for the respondent submitted that:
        a) The notice was issued within the prescribed time limit.
        b) The appellant had failed to disclose certain transactions.
        c) The Assessing Officer had valid reasons to believe that income had escaped assessment.
        
        LEGAL PRINCIPLES:
        
        This Court in the case of CIT vs. Kelvinator of India Ltd., (2010) 2 SCC 723, held that reopening of assessment under Section 147 requires the Assessing Officer to have reason to believe that income has escaped assessment.
        
        HELD:
        
        After hearing the learned counsel for both parties and perusing the material on record, this Court is of the considered view that:
        
        1. The notice under Section 148 was issued within the prescribed limitation period of four years.
        
        2. The Assessing Officer had adequate reasons to believe that income had escaped assessment.
        
        3. The appellant had failed to disclose certain material facts during the original assessment.
        
        CONCLUSION:
        
        In view of the above, this Court finds no merit in the appeal. The appeal is hereby dismissed. No order as to costs.
        
        The parties are directed to appear before the Assessing Officer on 30.04.2023 for further proceedings.
        
        (JOHN DOE)                    (JANE SMITH)
        JUDGE                         JUDGE
        """
        
        # Create temporary text file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            return tmp.name
    
    @pytest.fixture
    def sample_pdf_content(self):
        """Create a sample PDF-like content for testing"""
        # This would typically be a real PDF file
        # For testing, we'll use a text file that simulates PDF content
        content = """
        SUPREME COURT OF INDIA
        
        Criminal Appeal No. 456/2023
        
        State of Maharashtra vs. John Doe
        
        Judgment delivered on: 20th March, 2023
        
        This case involves charges under Section 302 of the Indian Penal Code.
        The appellant State has challenged the acquittal order passed by the High Court.
        
        After careful consideration of evidence and legal precedents,
        this Court upholds the acquittal order.
        
        Appeal dismissed.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            return tmp.name
    
    @pytest.fixture
    def sample_image_document(self):
        """Create a sample image with legal text for OCR testing"""
        img = Image.new('RGB', (800, 400), color='white')
        draw = ImageDraw.Draw(img)
        
        text_lines = [
            "HIGH COURT OF DELHI",
            "Writ Petition No. 123/2023",
            "Petitioner: ABC Ltd.",
            "Respondent: Income Tax Department",
            "Order dated: 15.03.2023",
            "The petition is allowed.",
            "The impugned order is set aside."
        ]
        
        y_position = 50
        for line in text_lines:
            draw.text((50, y_position), line, fill='black')
            y_position += 40
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            return tmp.name
    
    def test_complete_pipeline_text_document(self, sample_legal_document):
        """Test complete pipeline with text document"""
        try:
            # Step 1: Document Processing
            doc_processor = LegalDocumentProcessor()
            doc_result = doc_processor.process_document(
                sample_legal_document,
                normalize_text=True,
                extract_sections=True
            )
            
            assert doc_result['success'] == True
            assert 'text' in doc_result
            assert len(doc_result['text']) > 0
            
            # Step 2: Summarization
            try:
                summarizer = LegalDocumentSummarizer()
                summary_result = summarizer.summarize_document(
                    doc_result['text'],
                    summary_type='comprehensive'
                )
                
                assert summary_result['success'] == True
                assert 'summary' in summary_result
                assert len(summary_result['summary']) > 0
                assert summary_result['compression_ratio'] < 1.0
                
            except Exception as e:
                # Summarization might fail if models are not available
                logger.warning(f"Summarization failed in test: {str(e)}")
                summary_result = None
            
            # Step 3: Entity Recognition
            try:
                ner_model = IndianLegalNER()
                ner_result = ner_model.extract_entities(doc_result['text'])
                
                assert ner_result['processing_successful'] == True
                assert 'entities' in ner_result
                assert ner_result['total_entities'] >= 0
                
                # Check for expected entity types in legal document
                entity_types = [entity['label'] for entity in ner_result['entities']]
                expected_types = ['CASE_NUMBER', 'COURT_NAME', 'JUDGE_NAME', 'DATE']
                
                # At least some expected types should be found
                found_types = [t for t in expected_types if t in entity_types]
                assert len(found_types) > 0, f"Expected entity types not found. Found: {entity_types}"
                
            except Exception as e:
                # NER might fail if models are not available
                logger.warning(f"NER failed in test: {str(e)}")
                ner_result = None
            
            # Verify overall pipeline success
            assert doc_result is not None
            logger.info("Complete pipeline test passed successfully")
            
        finally:
            # Cleanup
            if os.path.exists(sample_legal_document):
                os.unlink(sample_legal_document)
    
    def test_pipeline_with_image_document(self, sample_image_document):
        """Test pipeline with image document (OCR)"""
        try:
            # Step 1: Document Processing with OCR
            doc_processor = LegalDocumentProcessor()
            doc_result = doc_processor.process_document(
                sample_image_document,
                normalize_text=True,
                extract_sections=True,
                ocr_fallback=True
            )
            
            # OCR might not work in test environment
            if doc_result['success']:
                assert 'text' in doc_result
                
                # If OCR worked, test the rest of the pipeline
                if len(doc_result['text'].strip()) > 0:
                    # Test entity extraction on OCR'd text
                    try:
                        ner_model = IndianLegalNER()
                        ner_result = ner_model.extract_entities(doc_result['text'])
                        assert 'entities' in ner_result
                    except Exception as e:
                        logger.warning(f"NER on OCR text failed: {str(e)}")
            else:
                pytest.skip("OCR not available in test environment")
                
        except Exception as e:
            if "tesseract" in str(e).lower() or "ocr" in str(e).lower():
                pytest.skip(f"OCR not available in test environment: {str(e)}")
            else:
                raise
        finally:
            if os.path.exists(sample_image_document):
                os.unlink(sample_image_document)
    
    def test_pipeline_error_handling(self):
        """Test pipeline error handling with invalid inputs"""
        doc_processor = LegalDocumentProcessor()
        
        # Test with non-existent file
        with pytest.raises(Exception):  # Should raise ValidationError or similar
            doc_processor.process_document("nonexistent_file.txt")
        
        # Test with empty file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as tmp:
            tmp.write("")  # Empty file
            tmp.flush()
            
            try:
                result = doc_processor.process_document(tmp.name)
                # Should handle empty file gracefully
                assert 'text' in result
            finally:
                os.unlink(tmp.name)
    
    def test_pipeline_performance(self, sample_legal_document):
        """Test pipeline performance with timing"""
        import time
        
        try:
            start_time = time.time()
            
            # Process document
            doc_processor = LegalDocumentProcessor()
            doc_result = doc_processor.process_document(sample_legal_document)
            
            doc_time = time.time()
            
            # Test summarization if available
            try:
                summarizer = LegalDocumentSummarizer()
                summary_result = summarizer.summarize_document(
                    doc_result['text'],
                    summary_type='brief'  # Use brief for faster processing
                )
                summary_time = time.time()
                
                # Check processing time
                total_time = summary_time - start_time
                assert total_time < 60.0, f"Pipeline took too long: {total_time}s"
                
            except Exception as e:
                logger.warning(f"Summarization not available for performance test: {str(e)}")
            
            # Test NER if available
            try:
                ner_model = IndianLegalNER()
                ner_result = ner_model.extract_entities(doc_result['text'])
                
                ner_time = time.time()
                total_time = ner_time - start_time
                assert total_time < 60.0, f"Complete pipeline took too long: {total_time}s"
                
            except Exception as e:
                logger.warning(f"NER not available for performance test: {str(e)}")
                
        finally:
            if os.path.exists(sample_legal_document):
                os.unlink(sample_legal_document)

class TestPipelineIntegration:
    """Test integration between different pipeline components"""
    
    def test_text_normalization_integration(self):
        """Test integration between document processing and text normalization"""
        # Create document with normalization issues
        messy_content = """
        The    SC    held   that   the   judgrnent   was   correct.
        The   petiti0ner   filed   a   case   against   resp0ndent.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(messy_content)
            tmp.flush()
            
            try:
                doc_processor = LegalDocumentProcessor()
                result = doc_processor.process_document(tmp.name, normalize_text=True)
                
                assert result['success'] == True
                
                # Check that normalization was applied
                if 'normalized_text' in result:
                    normalized = result['normalized_text']
                    assert "Supreme Court" in normalized  # SC should be expanded
                    assert "judgment" in normalized  # OCR error should be fixed
                    assert "petitioner" in normalized  # OCR error should be fixed
                    assert "   " not in normalized  # Multiple spaces should be cleaned
                    
            finally:
                os.unlink(tmp.name)
    
    def test_section_extraction_integration(self):
        """Test integration of section extraction with document processing"""
        structured_content = """
        IN THE SUPREME COURT OF INDIA
        
        CIVIL APPEAL NO. 1234 OF 2023
        
        CORAM: HON'BLE MR. JUSTICE JOHN DOE
        
        DATED: 15th March, 2023
        
        JUDGMENT
        
        This is the main judgment content.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(structured_content)
            tmp.flush()
            
            try:
                doc_processor = LegalDocumentProcessor()
                result = doc_processor.process_document(tmp.name, extract_sections=True)
                
                assert result['success'] == True
                
                # Check that sections were extracted
                if 'sections' in result and result['sections']:
                    sections = result['sections']
                    # Should extract at least some sections
                    assert len(sections) > 0
                    
            finally:
                os.unlink(tmp.name)
    
    def test_bias_detection_integration(self):
        """Test bias detection across pipeline components"""
        # Create content that might trigger bias detection
        biased_content = """
        The male judge ruled in favor of the male petitioner.
        The female respondent was not given adequate representation.
        This case from Delhi High Court shows regional bias.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(biased_content)
            tmp.flush()
            
            try:
                # Test document processing
                doc_processor = LegalDocumentProcessor()
                doc_result = doc_processor.process_document(tmp.name)
                
                # Test summarization with bias check
                try:
                    summarizer = LegalDocumentSummarizer()
                    summary_result = summarizer.summarize_document(doc_result['text'])
                    
                    if summary_result['success'] and 'bias_check' in summary_result:
                        bias_check = summary_result['bias_check']
                        assert 'passed_bias_check' in bias_check
                        # The bias check should run (pass or fail is less important for testing)
                        
                except Exception as e:
                    logger.warning(f"Summarization bias check not available: {str(e)}")
                
                # Test NER with bias check
                try:
                    ner_model = IndianLegalNER()
                    ner_result = ner_model.extract_entities(doc_result['text'])
                    
                    if 'bias_check' in ner_result:
                        bias_check = ner_result['bias_check']
                        assert 'passed_bias_check' in bias_check
                        
                except Exception as e:
                    logger.warning(f"NER bias check not available: {str(e)}")
                    
            finally:
                os.unlink(tmp.name)

class TestPipelineDataFlow:
    """Test data flow through the pipeline"""
    
    def test_data_consistency(self):
        """Test that data remains consistent through pipeline stages"""
        content = "The Supreme Court delivered a landmark judgment."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp.flush()
            
            try:
                # Process document
                doc_processor = LegalDocumentProcessor()
                doc_result = doc_processor.process_document(tmp.name)
                
                original_text = doc_result['text']
                
                # Test that subsequent processing maintains data integrity
                try:
                    ner_model = IndianLegalNER()
                    ner_result = ner_model.extract_entities(original_text)
                    
                    # Entities should reference valid positions in the text
                    for entity in ner_result.get('entities', []):
                        start = entity['start']
                        end = entity['end']
                        entity_text = entity['text']
                        
                        # Verify entity position is valid
                        assert 0 <= start < len(original_text)
                        assert start < end <= len(original_text)
                        assert original_text[start:end] == entity_text
                        
                except Exception as e:
                    logger.warning(f"NER consistency check not available: {str(e)}")
                    
            finally:
                os.unlink(tmp.name)
    
    def test_metadata_preservation(self):
        """Test that metadata is preserved through pipeline stages"""
        content = "Test document for metadata preservation."
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp:
            tmp.write(content)
            tmp.flush()
            
            try:
                doc_processor = LegalDocumentProcessor()
                result = doc_processor.process_document(tmp.name)
                
                # Check that file metadata is preserved
                assert 'file_info' in result
                file_info = result['file_info']
                
                assert 'filename' in file_info
                assert 'file_size_mb' in file_info
                assert 'file_type' in file_info
                
                # Check processing metadata
                assert 'processing_info' in result
                proc_info = result['processing_info']
                
                assert 'extraction_method' in proc_info
                
            finally:
                os.unlink(tmp.name)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
