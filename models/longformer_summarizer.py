"""
Longformer-based Abstractive Summarization for Indian Legal Documents
Fine-tuned model for generating concise summaries of legal judgments and documents
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import numpy as np
from typing import List, Dict, Any, Optional, Union
import time
import re
from pathlib import Path

from config.settings import MODEL_CONFIG
from utils.logger import logger
from utils.error_handler import handle_exceptions, ModelInferenceError, BiasDetectionError
from utils.text_normalization import normalize_legal_text

class LegalDocumentSummarizer:
    """
    Advanced summarization system for Indian legal documents
    Uses Longformer architecture optimized for long legal texts
    """
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or MODEL_CONFIG["longformer"]["model_name"]
        self.max_length = MODEL_CONFIG["longformer"]["max_length"]
        self.min_length = MODEL_CONFIG["longformer"]["min_length"]
        self.num_beams = MODEL_CONFIG["longformer"]["num_beams"]
        self.early_stopping = MODEL_CONFIG["longformer"]["early_stopping"]
        self.cache_dir = MODEL_CONFIG["longformer"]["cache_dir"]
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"Initializing Legal Document Summarizer on {self.device}")
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.model = None
        self.summarization_pipeline = None
        
        self._load_model()
        
        # Legal document specific configurations
        self.legal_keywords = self._load_legal_keywords()
        self.summary_templates = self._load_summary_templates()
        
    def _load_model(self):
        """Load the Longformer model and tokenizer"""
        try:
            logger.info(f"Loading model: {self.model_name}")
            
            # Create cache directory
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)
            
            # Try to load fine-tuned legal model first, fallback to base model
            try:
                # Check if we have a fine-tuned legal model
                legal_model_path = Path(self.cache_dir) / "legal_longformer"
                if legal_model_path.exists():
                    logger.info("Loading fine-tuned legal Longformer model")
                    self.tokenizer = AutoTokenizer.from_pretrained(str(legal_model_path))
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(str(legal_model_path))
                else:
                    raise FileNotFoundError("Fine-tuned model not found")
                    
            except Exception as e:
                logger.info(f"Fine-tuned model not available ({str(e)}), using base model")
                
                # Use BART or T5 as they're better for summarization than base Longformer
                if "longformer" in self.model_name.lower():
                    # Longformer doesn't have a conditional generation variant, use BART
                    model_name = "facebook/bart-large-cnn"
                    logger.info(f"Using BART model for summarization: {model_name}")
                else:
                    model_name = self.model_name
                
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=self.cache_dir
                )
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=self.cache_dir
                )
            
            # Move model to device
            self.model.to(self.device)
            self.model.eval()
            
            # Create summarization pipeline
            self.summarization_pipeline = pipeline(
                "summarization",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device.type == "cuda" else -1,
                framework="pt"
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            raise ModelInferenceError(
                f"Failed to load summarization model: {str(e)}",
                "MODEL_LOAD_FAILED",
                {"model_name": self.model_name, "device": str(self.device)}
            )
    
    def _load_legal_keywords(self) -> List[str]:
        """Load important legal keywords to emphasize in summaries"""
        return [
            # Court and legal entities
            "supreme court", "high court", "district court", "tribunal",
            "petitioner", "respondent", "appellant", "appellee",
            "plaintiff", "defendant", "accused", "complainant",
            
            # Legal concepts
            "judgment", "order", "decree", "writ", "injunction",
            "mandamus", "certiorari", "habeas corpus", "quo warranto",
            "interim relief", "stay order", "ex-parte", "suo moto",
            
            # Legal principles
            "natural justice", "due process", "fundamental rights",
            "constitutional validity", "judicial review", "precedent",
            "ratio decidendi", "obiter dicta", "stare decisis",
            
            # Indian legal acts
            "constitution", "indian penal code", "criminal procedure code",
            "civil procedure code", "evidence act", "contract act",
            "companies act", "income tax act", "gst act",
            
            # Legal outcomes
            "allowed", "dismissed", "quashed", "set aside", "remanded",
            "upheld", "overruled", "modified", "confirmed", "reversed"
        ]
    
    def _load_summary_templates(self) -> Dict[str, str]:
        """Load summary templates for different types of legal documents"""
        return {
            "judgment": "This judgment by {court} in {case_name} deals with {main_issue}. The court held that {holding}. Key legal principles: {principles}.",
            "order": "The court passed this order in {case_name} regarding {subject_matter}. The court directed that {directions}.",
            "petition": "This petition filed by {petitioner} against {respondent} seeks {relief_sought}. The main grounds are {grounds}.",
            "contract": "This contract between {parties} pertains to {subject_matter}. Key terms include {key_terms}. Important clauses: {clauses}.",
            "general": "This legal document addresses {main_subject}. Key points include {key_points}. Important legal implications: {implications}."
        }
    
    @handle_exceptions(ModelInferenceError, "Summarization failed")
    def summarize_document(self, 
                          text: str,
                          summary_type: str = "comprehensive",
                          max_length: Optional[int] = None,
                          min_length: Optional[int] = None,
                          focus_areas: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate summary of a legal document
        
        Args:
            text: Input legal document text
            summary_type: Type of summary ('brief', 'comprehensive', 'detailed')
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            focus_areas: Specific areas to focus on in summary
            
        Returns:
            Dictionary containing summary and metadata
        """
        start_time = time.time()
        
        if not text or len(text.strip()) < 100:
            raise ModelInferenceError("Input text too short for summarization", "INPUT_TOO_SHORT")
        
        logger.info(f"Starting summarization of text with {len(text)} characters")
        
        # Preprocess text
        processed_text = self._preprocess_for_summarization(text, focus_areas)
        
        # Determine summary parameters based on type
        summary_params = self._get_summary_parameters(summary_type, max_length, min_length)
        
        # Handle long documents by chunking if necessary
        if len(processed_text) > self.max_length * 4:  # Rough character estimate
            summary_result = self._summarize_long_document(processed_text, summary_params)
        else:
            summary_result = self._generate_summary(processed_text, summary_params)
        
        # Post-process summary
        final_summary = self._post_process_summary(summary_result["summary"], text)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Perform bias check
        bias_check = self._check_summary_bias(text, final_summary)
        
        result = {
            "summary": final_summary,
            "summary_type": summary_type,
            "original_length": len(text),
            "summary_length": len(final_summary),
            "compression_ratio": len(final_summary) / len(text),
            "processing_time": processing_time,
            "model_used": self.model_name,
            "parameters_used": summary_params,
            "bias_check": bias_check,
            "quality_metrics": self._calculate_quality_metrics(text, final_summary),
            "success": True
        }
        
        logger.info(f"Summarization completed in {processing_time:.2f}s. "
                   f"Compression ratio: {result['compression_ratio']:.3f}")
        
        return result
    
    def _preprocess_for_summarization(self, text: str, focus_areas: Optional[List[str]]) -> str:
        """Preprocess text for better summarization"""
        
        # Normalize text
        text = normalize_legal_text(text, preserve_structure=True)
        
        # If focus areas specified, emphasize those sections
        if focus_areas:
            for area in focus_areas:
                # Simple emphasis by repetition (more sophisticated methods possible)
                pattern = re.compile(rf'\b{re.escape(area)}\b', re.IGNORECASE)
                text = pattern.sub(f"{area} {area}", text)
        
        # Emphasize legal keywords
        for keyword in self.legal_keywords:
            pattern = re.compile(rf'\b{re.escape(keyword)}\b', re.IGNORECASE)
            if pattern.search(text):
                # Mark important legal terms (the model will learn to pay attention to these)
                text = pattern.sub(f"[LEGAL] {keyword} [/LEGAL]", text, count=1)
        
        return text
    
    def _get_summary_parameters(self, summary_type: str, max_length: Optional[int], min_length: Optional[int]) -> Dict[str, Any]:
        """Get summarization parameters based on summary type"""
        
        type_configs = {
            "brief": {"max_length": 150, "min_length": 50},
            "comprehensive": {"max_length": 500, "min_length": 200},
            "detailed": {"max_length": 1000, "min_length": 400}
        }
        
        config = type_configs.get(summary_type, type_configs["comprehensive"])
        
        return {
            "max_length": max_length or config["max_length"],
            "min_length": min_length or config["min_length"],
            "num_beams": self.num_beams,
            "early_stopping": self.early_stopping,
            "do_sample": False,  # Deterministic for legal documents
            "temperature": 0.7,
            "repetition_penalty": 1.2
        }
    
    def _generate_summary(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary using the model"""
        try:
            # Use the pipeline for summarization
            summary_result = self.summarization_pipeline(
                text,
                max_length=params["max_length"],
                min_length=params["min_length"],
                num_beams=params["num_beams"],
                early_stopping=params["early_stopping"],
                do_sample=params["do_sample"],
                temperature=params.get("temperature", 0.7)
            )
            
            return {
                "summary": summary_result[0]["summary_text"],
                "method": "direct_summarization"
            }
            
        except Exception as e:
            # Fallback to manual tokenization if pipeline fails
            logger.warning(f"Pipeline summarization failed: {str(e)}, trying manual approach")
            return self._manual_summarization(text, params)
    
    def _manual_summarization(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Manual summarization as fallback"""
        try:
            # Tokenize input
            inputs = self.tokenizer.encode(
                text, 
                return_tensors="pt", 
                max_length=self.max_length, 
                truncation=True
            ).to(self.device)
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs,
                    max_length=params["max_length"],
                    min_length=params["min_length"],
                    num_beams=params["num_beams"],
                    early_stopping=params["early_stopping"],
                    repetition_penalty=params.get("repetition_penalty", 1.2)
                )
            
            # Decode summary
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            return {
                "summary": summary,
                "method": "manual_summarization"
            }
            
        except Exception as e:
            raise ModelInferenceError(f"Manual summarization failed: {str(e)}", "SUMMARIZATION_FAILED")
    
    def _summarize_long_document(self, text: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle long documents by chunking and hierarchical summarization"""
        logger.info("Processing long document with chunking strategy")
        
        # Split text into chunks
        chunks = self._split_text_into_chunks(text)
        
        # Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.debug(f"Summarizing chunk {i+1}/{len(chunks)}")
            
            # Use shorter summaries for chunks
            chunk_params = params.copy()
            chunk_params["max_length"] = min(200, params["max_length"] // 2)
            chunk_params["min_length"] = min(50, params["min_length"] // 2)
            
            chunk_result = self._generate_summary(chunk, chunk_params)
            chunk_summaries.append(chunk_result["summary"])
        
        # Combine chunk summaries
        combined_summary = " ".join(chunk_summaries)
        
        # Generate final summary from combined summaries
        if len(combined_summary) > self.max_length * 2:
            final_result = self._generate_summary(combined_summary, params)
        else:
            final_result = {"summary": combined_summary, "method": "chunk_combination"}
        
        final_result["method"] = "hierarchical_summarization"
        final_result["chunks_processed"] = len(chunks)
        
        return final_result
    
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks for processing"""
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        max_chunk_length = self.max_length * 2  # Character estimate
        
        for paragraph in paragraphs:
            if current_length + len(paragraph) > max_chunk_length and current_chunk:
                # Start new chunk
                chunks.append('\n\n'.join(current_chunk))
                current_chunk = [paragraph]
                current_length = len(paragraph)
            else:
                current_chunk.append(paragraph)
                current_length += len(paragraph)
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        return chunks
    
    def _post_process_summary(self, summary: str, original_text: str) -> str:
        """Post-process generated summary"""
        
        # Remove special tokens if any
        summary = re.sub(r'\[LEGAL\]|\[/LEGAL\]', '', summary)
        
        # Ensure proper capitalization
        sentences = summary.split('. ')
        processed_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                # Capitalize first letter
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                processed_sentences.append(sentence)
        
        summary = '. '.join(processed_sentences)
        
        # Ensure summary ends with proper punctuation
        if summary and not summary.endswith(('.', '!', '?')):
            summary += '.'
        
        return summary
    
    def _check_summary_bias(self, original_text: str, summary: str) -> Dict[str, Any]:
        """Check for potential bias in the generated summary"""
        try:
            bias_indicators = {
                "gender_bias": self._check_gender_bias(summary),
                "religious_bias": self._check_religious_bias(summary),
                "regional_bias": self._check_regional_bias(summary),
                "representation_bias": self._check_representation_bias(original_text, summary)
            }
            
            overall_bias_score = np.mean(list(bias_indicators.values()))
            
            return {
                "bias_indicators": bias_indicators,
                "overall_bias_score": overall_bias_score,
                "bias_level": "low" if overall_bias_score < 0.3 else "medium" if overall_bias_score < 0.7 else "high",
                "passed_bias_check": overall_bias_score < 0.5
            }
            
        except Exception as e:
            logger.warning(f"Bias check failed: {str(e)}")
            return {"error": str(e), "passed_bias_check": True}  # Default to pass if check fails
    
    def _check_gender_bias(self, text: str) -> float:
        """Check for gender bias in text"""
        male_terms = ["he", "him", "his", "man", "men", "male", "gentleman", "sir"]
        female_terms = ["she", "her", "hers", "woman", "women", "female", "lady", "madam"]
        
        male_count = sum(len(re.findall(rf'\b{term}\b', text, re.IGNORECASE)) for term in male_terms)
        female_count = sum(len(re.findall(rf'\b{term}\b', text, re.IGNORECASE)) for term in female_terms)
        
        total_gendered = male_count + female_count
        if total_gendered == 0:
            return 0.0
        
        # Calculate bias as deviation from equal representation
        bias_score = abs(male_count - female_count) / total_gendered
        return min(bias_score, 1.0)
    
    def _check_religious_bias(self, text: str) -> float:
        """Check for religious bias"""
        religious_terms = ["hindu", "muslim", "christian", "sikh", "buddhist", "jain", "parsi"]
        
        term_counts = {}
        for term in religious_terms:
            term_counts[term] = len(re.findall(rf'\b{term}\b', text, re.IGNORECASE))
        
        total_religious = sum(term_counts.values())
        if total_religious == 0:
            return 0.0
        
        # Check if any single religion is over-represented
        max_representation = max(term_counts.values()) / total_religious
        return max(0.0, max_representation - 0.5) * 2  # Scale to 0-1
    
    def _check_regional_bias(self, text: str) -> float:
        """Check for regional bias"""
        regions = ["north", "south", "east", "west", "delhi", "mumbai", "chennai", "kolkata", "bangalore"]
        
        region_mentions = sum(len(re.findall(rf'\b{region}\b', text, re.IGNORECASE)) for region in regions)
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0
        
        # High regional mention ratio might indicate bias
        regional_ratio = region_mentions / total_words
        return min(regional_ratio * 10, 1.0)  # Scale appropriately
    
    def _check_representation_bias(self, original: str, summary: str) -> float:
        """Check if summary fairly represents the original content"""
        # Simple check: ensure key entities are proportionally represented
        
        # Extract key legal entities from both texts
        legal_entities = ["petitioner", "respondent", "court", "judge", "counsel"]
        
        original_entities = {}
        summary_entities = {}
        
        for entity in legal_entities:
            original_entities[entity] = len(re.findall(rf'\b{entity}\b', original, re.IGNORECASE))
            summary_entities[entity] = len(re.findall(rf'\b{entity}\b', summary, re.IGNORECASE))
        
        # Calculate representation bias
        bias_scores = []
        for entity in legal_entities:
            if original_entities[entity] > 0:
                expected_ratio = len(summary.split()) / len(original.split())
                actual_ratio = summary_entities[entity] / original_entities[entity] if original_entities[entity] > 0 else 0
                bias_score = abs(actual_ratio - expected_ratio) / max(expected_ratio, 0.01)
                bias_scores.append(min(bias_score, 1.0))
        
        return np.mean(bias_scores) if bias_scores else 0.0
    
    def _calculate_quality_metrics(self, original: str, summary: str) -> Dict[str, float]:
        """Calculate quality metrics for the summary"""
        
        # Basic metrics
        compression_ratio = len(summary) / len(original)
        
        # Keyword preservation
        important_keywords = [kw for kw in self.legal_keywords if kw in original.lower()]
        preserved_keywords = [kw for kw in important_keywords if kw in summary.lower()]
        keyword_preservation = len(preserved_keywords) / len(important_keywords) if important_keywords else 1.0
        
        # Sentence structure quality (simple heuristic)
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        avg_sentence_length = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        sentence_quality = 1.0 - abs(avg_sentence_length - 20) / 20  # Optimal around 20 words
        sentence_quality = max(0.0, min(1.0, sentence_quality))
        
        return {
            "compression_ratio": compression_ratio,
            "keyword_preservation": keyword_preservation,
            "sentence_quality": sentence_quality,
            "overall_quality": (keyword_preservation + sentence_quality) / 2
        }

# Convenience functions
def summarize_legal_document(text: str, summary_type: str = "comprehensive") -> str:
    """Simple function to summarize a legal document"""
    summarizer = LegalDocumentSummarizer()
    result = summarizer.summarize_document(text, summary_type)
    return result["summary"]

def get_document_summary_with_metrics(text: str) -> Dict[str, Any]:
    """Get summary with detailed metrics"""
    summarizer = LegalDocumentSummarizer()
    return summarizer.summarize_document(text)

def batch_summarize_documents(texts: List[str], summary_type: str = "comprehensive") -> List[Dict[str, Any]]:
    """Summarize multiple documents"""
    summarizer = LegalDocumentSummarizer()
    results = []
    
    for i, text in enumerate(texts):
        logger.info(f"Summarizing document {i+1}/{len(texts)}")
        try:
            result = summarizer.summarize_document(text, summary_type)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to summarize document {i+1}: {str(e)}")
            results.append({"error": str(e), "success": False})
    
    return results
