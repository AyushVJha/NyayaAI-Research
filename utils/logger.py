"""
Centralized logging utility for the Legal Document Intelligence Platform
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from config.settings import LOGGING_CONFIG, LOGS_DIR

class LegalPlatformLogger:
    """Custom logger for the legal platform with enhanced features"""
    
    def __init__(self, name: str = "legal_platform"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, LOGGING_CONFIG["level"]))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup file and console handlers with proper formatting"""
        formatter = logging.Formatter(LOGGING_CONFIG["format"])
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            LOGGING_CONFIG["file_path"],
            maxBytes=LOGGING_CONFIG["max_bytes"],
            backupCount=LOGGING_CONFIG["backup_count"]
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Error file handler
        error_handler = logging.handlers.RotatingFileHandler(
            str(LOGS_DIR / "errors.log"),
            maxBytes=LOGGING_CONFIG["max_bytes"],
            backupCount=LOGGING_CONFIG["backup_count"]
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
    
    def log_document_processing(self, filename: str, stage: str, status: str, details: str = ""):
        """Log document processing events with structured format"""
        self.logger.info(f"DOCUMENT_PROCESSING | File: {filename} | Stage: {stage} | Status: {status} | Details: {details}")
    
    def log_model_inference(self, model_name: str, input_size: int, processing_time: float, status: str):
        """Log model inference events"""
        self.logger.info(f"MODEL_INFERENCE | Model: {model_name} | Input_Size: {input_size} | Time: {processing_time:.2f}s | Status: {status}")
    
    def log_user_action(self, action: str, user_id: str = "anonymous", details: str = ""):
        """Log user interactions for analytics and debugging"""
        self.logger.info(f"USER_ACTION | Action: {action} | User: {user_id} | Details: {details}")
    
    def log_bias_check(self, model_name: str, metric: str, value: float, threshold: float, passed: bool):
        """Log bias mitigation checks"""
        status = "PASSED" if passed else "FAILED"
        self.logger.warning(f"BIAS_CHECK | Model: {model_name} | Metric: {metric} | Value: {value:.3f} | Threshold: {threshold:.3f} | Status: {status}")
    
    def get_logger(self):
        """Return the configured logger instance"""
        return self.logger

# Global logger instance
logger_instance = LegalPlatformLogger()
logger = logger_instance.get_logger()

# Convenience functions
def log_info(message: str):
    logger.info(message)

def log_error(message: str, exc_info: bool = True):
    logger.error(message, exc_info=exc_info)

def log_warning(message: str):
    logger.warning(message)

def log_debug(message: str):
    logger.debug(message)
