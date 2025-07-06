"""
Text normalization and preprocessing utilities for Indian legal documents
Handles multilingual content, inconsistent formatting, and data augmentation
"""

import re
import unicodedata
from typing import List, Dict, Any, Optional
import string
from utils.logger import logger
from utils.error_handler import handle_exceptions, DocumentProcessingError

class IndianLegalTextNormalizer:
    """
    Specialized text normalizer for Indian legal documents
    Handles multilingual content, legal terminology, and formatting inconsistencies
    """
    
    def __init__(self):
        self.legal_abbreviations = self._load_legal_abbreviations()
        self.court_names = self._load_court_names()
        self.legal_terms = self._load_legal_terms()
        self.citation_patterns = self._compile_citation_patterns()
        
    def _load_legal_abbreviations(self) -> Dict[str, str]:
        """Load common legal abbreviations and their expansions"""
        return {
            "SC": "Supreme Court",
            "HC": "High Court",
            "CrPC": "Criminal Procedure Code",
            "CPC": "Civil Procedure Code",
            "IPC": "Indian Penal Code",
            "AIR": "All India Reporter",
            "SCC": "Supreme Court Cases",
            "PLR": "Punjab Law Reporter",
            "MLJ": "Madras Law Journal",
            "Cal": "Calcutta",
            "Bom": "Bombay",
            "Mad": "Madras",
            "Del": "Delhi",
            "Ker": "Kerala",
            "Guj": "Gujarat",
            "MP": "Madhya Pradesh",
            "UP": "Uttar Pradesh",
            "WB": "West Bengal",
            "TN": "Tamil Nadu",
            "AP": "Andhra Pradesh",
            "Raj": "Rajasthan",
            "Har": "Haryana",
            "Pun": "Punjab",
            "Ori": "Orissa",
            "Ass": "Assam",
            "J&K": "Jammu and Kashmir",
            "HP": "Himachal Pradesh",
            "Goa": "Goa",
            "Mani": "Manipur",
            "Tri": "Tripura",
            "Meg": "Meghalaya",
            "Nag": "Nagaland",
            "Miz": "Mizoram",
            "Aru": "Arunachal Pradesh",
            "Sik": "Sikkim"
        }
    
    def _load_court_names(self) -> List[str]:
        """Load standardized court names"""
        return [
            "Supreme Court of India",
            "Delhi High Court",
            "Bombay High Court",
            "Calcutta High Court",
            "Madras High Court",
            "Karnataka High Court",
            "Kerala High Court",
            "Gujarat High Court",
            "Rajasthan High Court",
            "Madhya Pradesh High Court",
            "Chhattisgarh High Court",
            "Patna High Court",
            "Jharkhand High Court",
            "Orissa High Court",
            "Andhra Pradesh High Court",
            "Telangana High Court",
            "Punjab and Haryana High Court",
            "Himachal Pradesh High Court",
            "Uttarakhand High Court",
            "Jammu and Kashmir High Court",
            "Gauhati High Court",
            "Sikkim High Court",
            "Tripura High Court",
            "Manipur High Court",
            "Meghalaya High Court"
        ]
    
    def _load_legal_terms(self) -> List[str]:
        """Load common legal terms for standardization"""
        return [
            "petitioner", "respondent", "appellant", "appellee",
            "plaintiff", "defendant", "accused", "complainant",
            "judgment", "judgement", "order", "decree", "writ",
            "mandamus", "certiorari", "prohibition", "quo-warranto",
            "habeas corpus", "injunction", "stay", "interim",
            "ex-parte", "inter-partes", "suo moto", "prima facie",
            "res judicata", "sub judice", "locus standi", "ratio decidendi",
            "obiter dicta", "stare decisis", "ultra vires", "intra vires"
        ]
    
    def _compile_citation_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for legal citations"""
        patterns = [
            # AIR citations: AIR 1950 SC 124
            re.compile(r'AIR\s+(\d{4})\s+([A-Z]{2,4})\s+(\d+)', re.IGNORECASE),
            # SCC citations: (1950) 1 SCC 124
            re.compile(r'\((\d{4})\)\s+(\d+)\s+SCC\s+(\d+)', re.IGNORECASE),
            # Year and volume citations: 2020 (1) SCC 123
            re.compile(r'(\d{4})\s+\((\d+)\)\s+([A-Z]{2,4})\s+(\d+)', re.IGNORECASE),
            # Simple year citations: 2020 SC 123
            re.compile(r'(\d{4})\s+([A-Z]{2,4})\s+(\d+)', re.IGNORECASE),
        ]
        return patterns
    
    @handle_exceptions(DocumentProcessingError, "Text normalization failed")
    def normalize_text(self, text: str, preserve_structure: bool = True) -> str:
        """
        Main text normalization function
        
        Args:
            text: Raw text to normalize
            preserve_structure: Whether to preserve paragraph structure
            
        Returns:
            Normalized text
        """
        if not text or not isinstance(text, str):
            raise DocumentProcessingError("Invalid input text", "INVALID_TEXT_INPUT")
        
        logger.debug(f"Starting text normalization for text of length {len(text)}")
        
        # Step 1: Unicode normalization
        text = self._normalize_unicode(text)
        
        # Step 2: Clean whitespace and formatting
        text = self._clean_whitespace(text, preserve_structure)
        
        # Step 3: Normalize legal abbreviations
        text = self._normalize_abbreviations(text)
        
        # Step 4: Standardize citations
        text = self._standardize_citations(text)
        
        # Step 5: Fix common OCR errors
        text = self._fix_ocr_errors(text)
        
        # Step 6: Normalize legal terminology
        text = self._normalize_legal_terms(text)
        
        # Step 7: Handle multilingual content
        text = self._handle_multilingual_content(text)
        
        logger.debug(f"Text normalization completed. Output length: {len(text)}")
        return text
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters"""
        # Normalize to NFC form
        text = unicodedata.normalize('NFC', text)
        
        # Replace common Unicode variations
        replacements = {
            '\u2018': "'",  # Left single quotation mark
            '\u2019': "'",  # Right single quotation mark
            '\u201C': '"',  # Left double quotation mark
            '\u201D': '"',  # Right double quotation mark
            '\u2013': '-',  # En dash
            '\u2014': '--', # Em dash
            '\u2026': '...', # Horizontal ellipsis
            '\u00A0': ' ',  # Non-breaking space
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def _clean_whitespace(self, text: str, preserve_structure: bool) -> str:
        """Clean and normalize whitespace"""
        if preserve_structure:
            # Preserve paragraph breaks but clean other whitespace
            paragraphs = text.split('\n\n')
            cleaned_paragraphs = []
            
            for para in paragraphs:
                # Clean within paragraph
                para = re.sub(r'\s+', ' ', para.strip())
                if para:
                    cleaned_paragraphs.append(para)
            
            return '\n\n'.join(cleaned_paragraphs)
        else:
            # Aggressive whitespace cleaning
            return re.sub(r'\s+', ' ', text.strip())
    
    def _normalize_abbreviations(self, text: str) -> str:
        """Expand legal abbreviations for consistency"""
        for abbrev, expansion in self.legal_abbreviations.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text
    
    def _standardize_citations(self, text: str) -> str:
        """Standardize legal citation formats"""
        for pattern in self.citation_patterns:
            matches = pattern.finditer(text)
            for match in reversed(list(matches)):  # Reverse to maintain positions
                original = match.group(0)
                standardized = self._format_citation(match)
                text = text[:match.start()] + standardized + text[match.end():]
        
        return text
    
    def _format_citation(self, match: re.Match) -> str:
        """Format a citation match into standard format"""
        groups = match.groups()
        
        if len(groups) == 3:  # AIR format or simple format
            year, court, number = groups
            return f"AIR {year} {court.upper()} {number}"
        elif len(groups) == 4:  # SCC format with volume
            year, volume, court, number = groups
            return f"({year}) {volume} {court.upper()} {number}"
        
        return match.group(0)  # Return original if can't format
    
    def _fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in legal documents"""
        ocr_fixes = {
            # Common character misrecognitions
            r'\b0\b': 'O',  # Zero to O
            r'\bl\b': 'I',  # lowercase l to I
            r'rn': 'm',     # rn to m
            r'vv': 'w',     # vv to w
            r'\bI\b(?=\s+[a-z])': 'I',  # Standalone I
            
            # Legal document specific fixes
            r'judgrnent': 'judgment',
            r'judgement': 'judgment',  # Standardize spelling
            r'petiti0ner': 'petitioner',
            r'resp0ndent': 'respondent',
            r'c0urt': 'court',
            r'0rder': 'order',
            
            # Date format fixes
            r'(\d{1,2})[.,](\d{1,2})[.,](\d{4})': r'\1/\2/\3',
        }
        
        for pattern, replacement in ocr_fixes.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _normalize_legal_terms(self, text: str) -> str:
        """Normalize legal terminology for consistency"""
        # Standardize case for legal terms
        for term in self.legal_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            text = re.sub(pattern, term.lower(), text, flags=re.IGNORECASE)
        
        # Standardize judgment vs judgement
        text = re.sub(r'\bjudgement\b', 'judgment', text, flags=re.IGNORECASE)
        
        return text
    
    def _handle_multilingual_content(self, text: str) -> str:
        """Handle multilingual content in Indian legal documents"""
        # Detect and preserve Devanagari script content
        devanagari_pattern = r'[\u0900-\u097F]+'
        
        # For now, preserve multilingual content as-is
        # In a production system, this could include:
        # - Script detection
        # - Transliteration
        # - Translation for specific terms
        
        return text
    
    def extract_key_sections(self, text: str) -> Dict[str, str]:
        """Extract key sections from legal documents"""
        sections = {}
        
        # Define section patterns
        section_patterns = {
            'case_title': r'(?:IN THE MATTER OF|BETWEEN|VERSUS|V/S|VS\.?)\s*:?\s*(.+?)(?:\n|CORAM|BEFORE)',
            'court_name': r'(?:IN THE|BEFORE THE)\s+(.+?COURT.+?)(?:\n|AT)',
            'judges': r'(?:CORAM|BEFORE|HON\'BLE)\s*:?\s*(.+?)(?:\n\n|JUDGMENT|ORDER)',
            'date': r'(?:DATED?|DECIDED ON|PRONOUNCED ON)\s*:?\s*(.+?)(?:\n|$)',
            'citation': r'(?:CITATION|REPORTED IN)\s*:?\s*(.+?)(?:\n|$)',
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        return sections
    
    def augment_text_data(self, text: str, augmentation_factor: int = 3) -> List[str]:
        """
        Generate augmented versions of text for training data enhancement
        Useful for low-resource Indian legal language processing
        """
        augmented_texts = [text]  # Original text
        
        for _ in range(augmentation_factor - 1):
            augmented = text
            
            # Synonym replacement for legal terms
            augmented = self._replace_synonyms(augmented)
            
            # Paraphrasing (simple version)
            augmented = self._simple_paraphrase(augmented)
            
            # Add slight variations
            augmented = self._add_variations(augmented)
            
            augmented_texts.append(augmented)
        
        return augmented_texts
    
    def _replace_synonyms(self, text: str) -> str:
        """Replace words with legal synonyms"""
        synonyms = {
            'judgment': 'decision',
            'petitioner': 'applicant',
            'respondent': 'defendant',
            'court': 'tribunal',
            'order': 'directive',
        }
        
        for original, synonym in synonyms.items():
            if original in text.lower():
                # Randomly replace some occurrences
                import random
                if random.random() < 0.3:  # 30% chance
                    text = re.sub(r'\b' + original + r'\b', synonym, text, count=1, flags=re.IGNORECASE)
        
        return text
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple paraphrasing techniques"""
        # This is a simplified version - in production, use more sophisticated NLP
        paraphrases = {
            'it is held that': 'the court holds that',
            'it is observed that': 'the court observes that',
            'in the present case': 'in this matter',
            'the learned counsel': 'counsel',
        }
        
        for original, paraphrase in paraphrases.items():
            text = re.sub(original, paraphrase, text, flags=re.IGNORECASE)
        
        return text
    
    def _add_variations(self, text: str) -> str:
        """Add minor variations to text"""
        # Add/remove articles occasionally
        import random
        
        if random.random() < 0.2:  # 20% chance
            text = re.sub(r'\bthe\s+(?=court|petitioner|respondent)', '', text, count=1, flags=re.IGNORECASE)
        
        return text

# Convenience functions
def normalize_legal_text(text: str, preserve_structure: bool = True) -> str:
    """Convenience function for text normalization"""
    normalizer = IndianLegalTextNormalizer()
    return normalizer.normalize_text(text, preserve_structure)

def extract_document_sections(text: str) -> Dict[str, str]:
    """Convenience function for section extraction"""
    normalizer = IndianLegalTextNormalizer()
    return normalizer.extract_key_sections(text)

def augment_legal_text(text: str, factor: int = 3) -> List[str]:
    """Convenience function for text augmentation"""
    normalizer = IndianLegalTextNormalizer()
    return normalizer.augment_text_data(text, factor)
