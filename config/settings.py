"""
Configuration settings for AI-Powered Legal Document Intelligence Platform
Centralized configuration for the Indian legal system platform
"""

import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models" / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# OCR Configuration
OCR_CONFIG = {
    "languages": ["eng", "hin", "tam", "tel", "ben", "guj", "kan", "mal", "mar", "ori", "pan", "urd"],
    "tesseract_cmd": os.environ.get("TESSERACT_CMD", "/usr/bin/tesseract"),
    "confidence_threshold": 60,
    "preprocessing": {
        "denoise": True,
        "contrast_enhancement": True,
        "deskew": True
    }
}

# Model Configuration
MODEL_CONFIG = {
    "longformer": {
        "model_name": "allenai/longformer-base-4096",
        "max_length": 4096,
        "min_length": 100,
        "num_beams": 4,
        "early_stopping": True,
        "cache_dir": str(MODELS_DIR / "longformer_cache")
    },
    "spacy_ner": {
        "model_path": str(MODELS_DIR / "legal_ner_model"),
        "entities": [
            "CASE_NUMBER", "COURT_NAME", "JUDGE_NAME", "PETITIONER", 
            "RESPONDENT", "LEGAL_CITATION", "ACT_SECTION", "DATE", 
            "LEGAL_PRINCIPLE", "PRECEDENT", "STATUTE", "REGULATION"
        ],
        "confidence_threshold": 0.7
    }
}

# Document Processing Configuration
DOCUMENT_CONFIG = {
    "supported_formats": [".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg", ".tiff"],
    "max_file_size_mb": 50,
    "chunk_size": 1000,
    "overlap": 200
}

# Streamlit Configuration
STREAMLIT_CONFIG = {
    "page_title": "AI Legal Document Intelligence - Indian Legal System",
    "page_icon": "⚖️",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
    "theme": {
        "primaryColor": "#1f4e79",
        "backgroundColor": "#ffffff",
        "secondaryBackgroundColor": "#f0f2f6",
        "textColor": "#262730"
    }
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": os.environ.get("LOG_LEVEL", "INFO"),
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": str(LOGS_DIR / "legal_platform.log"),
    "max_bytes": 10485760,  # 10MB
    "backup_count": 5
}

# Bias Mitigation Configuration
BIAS_MITIGATION = {
    "enable_fairness_checks": True,
    "demographic_parity": True,
    "equalized_odds": True,
    "confidence_calibration": True,
    "explanation_requirements": True
}

# API Configuration (for future extensions)
API_CONFIG = {
    "rate_limit": 100,  # requests per minute
    "timeout": 300,  # seconds
    "max_concurrent_requests": 10
}

# Environment-specific settings
ENVIRONMENT = os.environ.get("ENVIRONMENT", "development")

if ENVIRONMENT == "production":
    LOGGING_CONFIG["level"] = "WARNING"
    MODEL_CONFIG["longformer"]["cache_dir"] = "/app/models/cache"
    STREAMLIT_CONFIG["theme"]["primaryColor"] = "#2e7d32"
