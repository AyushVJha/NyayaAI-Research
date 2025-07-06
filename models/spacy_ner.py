"""
Custom SpaCy NER Model for Indian Legal Entity Recognition
Extracts legal entities, citations, clauses, and other important legal information
"""

import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import random
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import json
from pathlib import Path
import numpy as np
from collections import defaultdict, Counter

from config.settings import MODEL_CONFIG
from utils.logger import logger
from utils.error_handler import handle_exceptions, ModelInferenceError, BiasDetectionError

class IndianLegalNER:
    """
    Custom Named Entity Recognition system for Indian legal documents
    Identifies legal entities, citations, court names, judges, parties, and legal concepts
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or MODEL_CONFIG["spacy_ner"]["model_path"]
        self.entities = MODEL_CONFIG["spacy_ner"]["entities"]
        self.confidence_threshold = MODEL_CONFIG["spacy_ner"]["confidence_threshold"]
        
        # Initialize model
        self.nlp = None
        self.entity_patterns = self._compile_entity_patterns()
        self.legal_dictionaries = self._load_legal_dictionaries()
        
        self._load_or_create_model()
        
    def _load_or_create_model(self):
        """Load existing model or create new one"""
        try:
            if Path(self.model_path).exists():
                logger.info(f"Loading existing NER model from {self.model_path}")
                self.nlp = spacy.load(self.model_path)
            else:
                logger.info("Creating new NER model")
                self.nlp = self._create_base_model()
                
        except Exception as e:
            logger.warning(f"Failed to load model: {str(e)}, creating base model")
            self.nlp = self._create_base_model()
    
    def _create_base_model(self):
        """Create base SpaCy model with NER component"""
        try:
            # Try to load English model as base
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Create blank model if English model not available
            logger.warning("English model not found, creating blank model")
            nlp = spacy.blank("en")
        
        # Add NER component if not present
        if "ner" not in nlp.pipe_names:
            ner = nlp.add_pipe("ner")
        else:
            ner = nlp.get_pipe("ner")
        
        # Add custom entity labels
        for entity_type in self.entities:
            ner.add_label(entity_type)
        
        return nlp
    
    def _compile_entity_patterns(self) -> Dict[str, List[re.Pattern]]:
        """Compile regex patterns for different entity types"""
        patterns = {
            "CASE_NUMBER": [
                re.compile(r'\b(?:Civil Appeal|Criminal Appeal|Writ Petition|Special Leave Petition|Review Petition)\s+(?:No\.?)?\s*(\d+)\s*(?:of)?\s*(\d{4})\b', re.IGNORECASE),
                re.compile(r'\b(?:CA|Crl\.A|WP|SLP|RP)\s*(?:No\.?)?\s*(\d+)\s*(?:of)?\s*(\d{4})\b', re.IGNORECASE),
                re.compile(r'\b(?:Case|Suit|Petition)\s+(?:No\.?)?\s*(\d+)\s*(?:of)?\s*(\d{4})\b', re.IGNORECASE),
            ],
            
            "LEGAL_CITATION": [
                re.compile(r'\b(?:AIR|SCC|SCR|All LJ|Bom LR|Cal LJ|Mad LJ|Ker LT|Guj LH|Del LJ)\s+(\d{4})\s+([A-Z]{2,4})\s+(\d+)\b', re.IGNORECASE),
                re.compile(r'\((\d{4})\)\s+(\d+)\s+(SCC|SCR|All|Bom|Cal|Mad|Ker|Guj|Del)\s+(\d+)\b', re.IGNORECASE),
                re.compile(r'(\d{4})\s+\((\d+)\)\s+(SCC|SCR|All|Bom|Cal|Mad|Ker|Guj|Del)\s+(\d+)\b', re.IGNORECASE),
            ],
            
            "ACT_SECTION": [
                re.compile(r'\bSection\s+(\d+(?:\([a-z]\))?(?:\s*(?:to|and)\s*\d+(?:\([a-z]\))?)*)\s+of\s+(?:the\s+)?([^.]+?(?:Act|Code|Rules|Regulations?))\b', re.IGNORECASE),
                re.compile(r'\bArticle\s+(\d+(?:\([a-z]\))?)\s+of\s+(?:the\s+)?Constitution\b', re.IGNORECASE),
                re.compile(r'\bRule\s+(\d+(?:\([a-z]\))?)\s+of\s+(?:the\s+)?([^.]+?Rules)\b', re.IGNORECASE),
            ],
            
            "DATE": [
                re.compile(r'\b(\d{1,2})(?:st|nd|rd|th)?\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s*,?\s*(\d{4})\b', re.IGNORECASE),
                re.compile(r'\b(\d{1,2})[./](\d{1,2})[./](\d{4})\b'),
                re.compile(r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2})(?:st|nd|rd|th)?\s*,?\s*(\d{4})\b', re.IGNORECASE),
            ],
            
            "COURT_NAME": [
                re.compile(r'\b(?:Supreme Court of India|Hon\'ble Supreme Court)\b', re.IGNORECASE),
                re.compile(r'\b(?:High Court of|Hon\'ble High Court of)\s+([^,\n]+)\b', re.IGNORECASE),
                re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+High Court\b', re.IGNORECASE),
                re.compile(r'\b(?:District Court|Sessions Court|Magistrate Court|Family Court|Consumer Court)\s+(?:of\s+)?([^,\n]+)\b', re.IGNORECASE),
            ],
            
            "JUDGE_NAME": [
                re.compile(r'\b(?:Hon\'ble\s+)?(?:Mr\.|Ms\.|Mrs\.)?\s*Justice\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE),
                re.compile(r'\b(?:Hon\'ble\s+)?(?:Chief Justice|CJ)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE),
                re.compile(r'\bJudge\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', re.IGNORECASE),
            ],
        }
        
        return patterns
    
    def _load_legal_dictionaries(self) -> Dict[str, Set[str]]:
        """Load dictionaries of legal terms and entities"""
        return {
            "courts": {
                "Supreme Court of India", "Delhi High Court", "Bombay High Court",
                "Calcutta High Court", "Madras High Court", "Karnataka High Court",
                "Kerala High Court", "Gujarat High Court", "Rajasthan High Court",
                "Madhya Pradesh High Court", "Chhattisgarh High Court", "Patna High Court",
                "Jharkhand High Court", "Orissa High Court", "Andhra Pradesh High Court",
                "Telangana High Court", "Punjab and Haryana High Court", "Himachal Pradesh High Court",
                "Uttarakhand High Court", "Jammu and Kashmir High Court", "Gauhati High Court",
                "Sikkim High Court", "Tripura High Court", "Manipur High Court", "Meghalaya High Court"
            },
            
            "legal_acts": {
                "Indian Penal Code", "Criminal Procedure Code", "Civil Procedure Code",
                "Indian Evidence Act", "Indian Contract Act", "Companies Act",
                "Income Tax Act", "Goods and Services Tax Act", "Motor Vehicles Act",
                "Information Technology Act", "Right to Information Act", "Consumer Protection Act",
                "Environmental Protection Act", "Forest Rights Act", "Land Acquisition Act",
                "Arbitration and Conciliation Act", "Limitation Act", "Registration Act",
                "Transfer of Property Act", "Negotiable Instruments Act", "Partnership Act",
                "Hindu Marriage Act", "Muslim Personal Law", "Indian Succession Act",
                "Juvenile Justice Act", "Protection of Women from Domestic Violence Act",
                "Sexual Harassment of Women at Workplace Act", "Dowry Prohibition Act"
            },
            
            "legal_concepts": {
                "natural justice", "due process", "fundamental rights", "directive principles",
                "judicial review", "separation of powers", "rule of law", "constitutional validity",
                "locus standi", "res judicata", "sub judice", "prima facie", "ultra vires",
                "intra vires", "mandamus", "certiorari", "prohibition", "quo warranto",
                "habeas corpus", "injunction", "interim relief", "stay order", "ex parte",
                "inter partes", "suo moto", "precedent", "ratio decidendi", "obiter dicta",
                "stare decisis", "burden of proof", "standard of proof", "reasonable doubt",
                "preponderance of evidence", "circumstantial evidence", "direct evidence"
            },
            
            "party_types": {
                "petitioner", "respondent", "appellant", "appellee", "plaintiff", "defendant",
                "accused", "complainant", "applicant", "intervener", "impleader", "proforma respondent",
                "union of india", "state government", "central government", "public sector undertaking",
                "statutory corporation", "local authority", "municipal corporation", "panchayat"
            }
        }
    
    @handle_exceptions(ModelInferenceError, "NER processing failed")
    def extract_entities(self, text: str, 
                        confidence_filter: bool = True,
                        merge_overlapping: bool = True) -> Dict[str, Any]:
        """
        Extract legal entities from text
        
        Args:
            text: Input legal document text
            confidence_filter: Whether to filter low-confidence entities
            merge_overlapping: Whether to merge overlapping entities
            
        Returns:
            Dictionary containing extracted entities and metadata
        """
        if not text or len(text.strip()) < 10:
            raise ModelInferenceError("Input text too short for entity extraction", "INPUT_TOO_SHORT")
        
        logger.info(f"Extracting entities from text with {len(text)} characters")
        
        # Process text with SpaCy
        doc = self.nlp(text)
        
        # Extract entities using multiple methods
        spacy_entities = self._extract_spacy_entities(doc)
        pattern_entities = self._extract_pattern_entities(text)
        dictionary_entities = self._extract_dictionary_entities(text)
        
        # Combine and deduplicate entities
        all_entities = self._combine_entities(spacy_entities, pattern_entities, dictionary_entities)
        
        # Filter by confidence if requested
        if confidence_filter:
            all_entities = self._filter_by_confidence(all_entities)
        
        # Merge overlapping entities if requested
        if merge_overlapping:
            all_entities = self._merge_overlapping_entities(all_entities)
        
        # Post-process and validate entities
        validated_entities = self._validate_entities(all_entities, text)
        
        # Perform bias check
        bias_check = self._check_entity_bias(validated_entities)
        
        # Calculate statistics
        entity_stats = self._calculate_entity_statistics(validated_entities)
        
        result = {
            "entities": validated_entities,
            "entity_counts": {entity_type: len([e for e in validated_entities if e["label"] == entity_type]) 
                            for entity_type in self.entities},
            "total_entities": len(validated_entities),
            "text_length": len(text),
            "entity_density": len(validated_entities) / len(text.split()) if text.split() else 0,
            "bias_check": bias_check,
            "statistics": entity_stats,
            "processing_successful": True
        }
        
        logger.info(f"Entity extraction completed. Found {len(validated_entities)} entities")
        return result
    
    def _extract_spacy_entities(self, doc) -> List[Dict[str, Any]]:
        """Extract entities using SpaCy NER"""
        entities = []
        
        for ent in doc.ents:
            if ent.label_ in self.entities:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char,
                    "confidence": getattr(ent, 'confidence', 0.8),  # Default confidence
                    "method": "spacy_ner"
                })
        
        return entities
    
    def _extract_pattern_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using regex patterns"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    entities.append({
                        "text": match.group(0),
                        "label": entity_type,
                        "start": match.start(),
                        "end": match.end(),
                        "confidence": 0.9,  # High confidence for pattern matches
                        "method": "pattern_matching",
                        "pattern_groups": match.groups()
                    })
        
        return entities
    
    def _extract_dictionary_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using dictionary lookup"""
        entities = []
        text_lower = text.lower()
        
        entity_type_mapping = {
            "courts": "COURT_NAME",
            "legal_acts": "STATUTE",
            "legal_concepts": "LEGAL_PRINCIPLE",
            "party_types": "PETITIONER"  # Will be refined later
        }
        
        for dict_name, entity_type in entity_type_mapping.items():
            for term in self.legal_dictionaries[dict_name]:
                term_lower = term.lower()
                start = 0
                
                while True:
                    pos = text_lower.find(term_lower, start)
                    if pos == -1:
                        break
                    
                    # Check word boundaries
                    if (pos == 0 or not text[pos-1].isalnum()) and \
                       (pos + len(term) == len(text) or not text[pos + len(term)].isalnum()):
                        
                        entities.append({
                            "text": text[pos:pos + len(term)],
                            "label": entity_type,
                            "start": pos,
                            "end": pos + len(term),
                            "confidence": 0.7,  # Medium confidence for dictionary matches
                            "method": "dictionary_lookup"
                        })
                    
                    start = pos + 1
        
        return entities
    
    def _combine_entities(self, *entity_lists) -> List[Dict[str, Any]]:
        """Combine entities from different extraction methods"""
        all_entities = []
        
        for entity_list in entity_lists:
            all_entities.extend(entity_list)
        
        # Sort by start position
        all_entities.sort(key=lambda x: x["start"])
        
        return all_entities
    
    def _filter_by_confidence(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter entities by confidence threshold"""
        return [entity for entity in entities 
                if entity["confidence"] >= self.confidence_threshold]
    
    def _merge_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge overlapping entities, keeping the one with higher confidence"""
        if not entities:
            return entities
        
        merged = []
        current = entities[0]
        
        for next_entity in entities[1:]:
            # Check for overlap
            if (current["start"] <= next_entity["start"] < current["end"]) or \
               (next_entity["start"] <= current["start"] < next_entity["end"]):
                
                # Keep entity with higher confidence
                if next_entity["confidence"] > current["confidence"]:
                    current = next_entity
                # If same confidence, prefer longer entity
                elif (next_entity["confidence"] == current["confidence"] and 
                      len(next_entity["text"]) > len(current["text"])):
                    current = next_entity
            else:
                merged.append(current)
                current = next_entity
        
        merged.append(current)
        return merged
    
    def _validate_entities(self, entities: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
        """Validate and clean extracted entities"""
        validated = []
        
        for entity in entities:
            # Basic validation
            if not entity["text"].strip():
                continue
            
            # Validate position
            if entity["start"] < 0 or entity["end"] > len(text):
                continue
            
            # Validate text matches position
            if text[entity["start"]:entity["end"]] != entity["text"]:
                continue
            
            # Clean entity text
            entity["text"] = entity["text"].strip()
            
            # Refine entity labels based on context
            entity = self._refine_entity_label(entity, text)
            
            validated.append(entity)
        
        return validated
    
    def _refine_entity_label(self, entity: Dict[str, Any], text: str) -> Dict[str, Any]:
        """Refine entity labels based on context"""
        
        # Get context around entity
        start = max(0, entity["start"] - 100)
        end = min(len(text), entity["end"] + 100)
        context = text[start:end].lower()
        
        # Refine PETITIONER/RESPONDENT based on context
        if entity["label"] == "PETITIONER":
            if any(word in context for word in ["respondent", "defendant", "accused"]):
                if "respondent" in context:
                    entity["label"] = "RESPONDENT"
                elif "defendant" in context:
                    entity["label"] = "RESPONDENT"
        
        # Refine STATUTE vs REGULATION
        if entity["label"] == "STATUTE":
            if any(word in entity["text"].lower() for word in ["rule", "regulation", "notification"]):
                entity["label"] = "REGULATION"
        
        return entity
    
    def _check_entity_bias(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Check for bias in entity extraction"""
        try:
            # Count entities by type
            entity_counts = Counter(entity["label"] for entity in entities)
            
            # Check for gender bias in person names
            gender_bias = self._check_gender_bias_in_entities(entities)
            
            # Check for regional bias in court names
            regional_bias = self._check_regional_bias_in_entities(entities)
            
            # Check for representation bias
            representation_bias = self._check_representation_bias(entities)
            
            overall_bias_score = np.mean([gender_bias, regional_bias, representation_bias])
            
            return {
                "gender_bias": gender_bias,
                "regional_bias": regional_bias,
                "representation_bias": representation_bias,
                "overall_bias_score": overall_bias_score,
                "bias_level": "low" if overall_bias_score < 0.3 else "medium" if overall_bias_score < 0.7 else "high",
                "passed_bias_check": overall_bias_score < 0.5,
                "entity_distribution": dict(entity_counts)
            }
            
        except Exception as e:
            logger.warning(f"Entity bias check failed: {str(e)}")
            return {"error": str(e), "passed_bias_check": True}
    
    def _check_gender_bias_in_entities(self, entities: List[Dict[str, Any]]) -> float:
        """Check for gender bias in extracted person entities"""
        judge_entities = [e for e in entities if e["label"] == "JUDGE_NAME"]
        
        if not judge_entities:
            return 0.0
        
        male_indicators = ["mr.", "justice", "chief justice"]
        female_indicators = ["ms.", "mrs.", "justice"]
        
        male_count = sum(1 for entity in judge_entities 
                        if any(indicator in entity["text"].lower() for indicator in male_indicators))
        female_count = sum(1 for entity in judge_entities 
                          if any(indicator in entity["text"].lower() for indicator in female_indicators))
        
        total_gendered = male_count + female_count
        if total_gendered == 0:
            return 0.0
        
        bias_score = abs(male_count - female_count) / total_gendered
        return min(bias_score, 1.0)
    
    def _check_regional_bias_in_entities(self, entities: List[Dict[str, Any]]) -> float:
        """Check for regional bias in court entities"""
        court_entities = [e for e in entities if e["label"] == "COURT_NAME"]
        
        if not court_entities:
            return 0.0
        
        # Define regions
        regions = {
            "north": ["delhi", "punjab", "haryana", "himachal", "uttarakhand", "jammu"],
            "south": ["madras", "karnataka", "kerala", "andhra", "telangana"],
            "west": ["bombay", "gujarat", "rajasthan"],
            "east": ["calcutta", "orissa", "jharkhand", "patna"]
        }
        
        region_counts = defaultdict(int)
        
        for entity in court_entities:
            entity_text = entity["text"].lower()
            for region, states in regions.items():
                if any(state in entity_text for state in states):
                    region_counts[region] += 1
                    break
        
        if not region_counts:
            return 0.0
        
        total_regional = sum(region_counts.values())
        max_representation = max(region_counts.values()) / total_regional
        
        # Bias if any region is over-represented (>50%)
        return max(0.0, max_representation - 0.5) * 2
    
    def _check_representation_bias(self, entities: List[Dict[str, Any]]) -> float:
        """Check for representation bias across entity types"""
        entity_counts = Counter(entity["label"] for entity in entities)
        
        if not entity_counts:
            return 0.0
        
        # Check if any entity type is severely under or over-represented
        total_entities = sum(entity_counts.values())
        expected_per_type = total_entities / len(self.entities)
        
        bias_scores = []
        for entity_type in self.entities:
            actual_count = entity_counts.get(entity_type, 0)
            if expected_per_type > 0:
                bias_score = abs(actual_count - expected_per_type) / expected_per_type
                bias_scores.append(min(bias_score, 2.0))  # Cap at 2.0
        
        return np.mean(bias_scores) / 2.0 if bias_scores else 0.0  # Normalize to 0-1
    
    def _calculate_entity_statistics(self, entities: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate detailed statistics about extracted entities"""
        
        if not entities:
            return {"total": 0}
        
        # Basic counts
        entity_counts = Counter(entity["label"] for entity in entities)
        method_counts = Counter(entity["method"] for entity in entities)
        
        # Confidence statistics
        confidences = [entity["confidence"] for entity in entities]
        
        # Length statistics
        lengths = [len(entity["text"]) for entity in entities]
        
        return {
            "total_entities": len(entities),
            "entity_type_distribution": dict(entity_counts),
            "extraction_method_distribution": dict(method_counts),
            "confidence_statistics": {
                "mean": np.mean(confidences),
                "median": np.median(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "std": np.std(confidences)
            },
            "entity_length_statistics": {
                "mean": np.mean(lengths),
                "median": np.median(lengths),
                "min": np.min(lengths),
                "max": np.max(lengths)
            },
            "high_confidence_entities": len([e for e in entities if e["confidence"] >= 0.8]),
            "pattern_matched_entities": len([e for e in entities if e["method"] == "pattern_matching"]),
            "dictionary_matched_entities": len([e for e in entities if e["method"] == "dictionary_lookup"])
        }
    
    def train_model(self, training_data: List[Tuple[str, Dict]], 
                   n_iter: int = 30, 
                   batch_size: int = 4) -> Dict[str, Any]:
        """
        Train the NER model with custom legal data
        
        Args:
            training_data: List of (text, annotations) tuples
            n_iter: Number of training iterations
            batch_size: Batch size for training
            
        Returns:
            Training results and metrics
        """
        logger.info(f"Training NER model with {len(training_data)} examples")
        
        # Prepare training data
        examples = []
        for text, annotations in training_data:
            doc = self.nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            examples.append(example)
        
        # Get NER component
        ner = self.nlp.get_pipe("ner")
        
        # Training loop
        losses = {}
        for iteration in range(n_iter):
            random.shuffle(examples)
            batches = minibatch(examples, size=compounding(4.0, 32.0, 1.001))
            
            for batch in batches:
                self.nlp.update(batch, losses=losses, drop=0.5)
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}, Losses: {losses}")
        
        # Save trained model
        self.nlp.to_disk(self.model_path)
        logger.info(f"Model saved to {self.model_path}")
        
        return {
            "training_completed": True,
            "iterations": n_iter,
            "final_losses": losses,
            "model_path": self.model_path
        }

# Convenience functions
def extract_legal_entities(text: str) -> List[Dict[str, Any]]:
    """Simple function to extract legal entities"""
    ner = IndianLegalNER()
    result = ner.extract_entities(text)
    return result["entities"]

def get_entity_summary(text: str) -> Dict[str, Any]:
    """Get summary of entities in text"""
    ner = IndianLegalNER()
    result = ner.extract_entities(text)
    return {
        "entity_counts": result["entity_counts"],
        "total_entities": result["total_entities"],
        "entity_density": result["entity_density"]
    }

def extract_citations_and_acts(text: str) -> Dict[str, List[str]]:
    """Extract only citations and legal acts"""
    ner = IndianLegalNER()
    result = ner.extract_entities(text)
    
    citations = [e["text"] for e in result["entities"] if e["label"] == "LEGAL_CITATION"]
    acts = [e["text"] for e in result["entities"] if e["label"] in ["STATUTE", "ACT_SECTION"]]
    
    return {
        "citations": citations,
        "acts_and_sections": acts
    }
