"""
Centralized error handling utilities for the Legal Document Intelligence Platform
"""

import traceback
from typing import Optional, Dict, Any
from functools import wraps
from utils.logger import logger

class LegalPlatformError(Exception):
    """Base exception class for the legal platform"""
    
    def __init__(self, message: str, error_code: str = "GENERAL_ERROR", details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class DocumentProcessingError(LegalPlatformError):
    """Exception for document processing errors"""
    pass

class OCRError(LegalPlatformError):
    """Exception for OCR processing errors"""
    pass

class ModelInferenceError(LegalPlatformError):
    """Exception for model inference errors"""
    pass

class ValidationError(LegalPlatformError):
    """Exception for input validation errors"""
    pass

class BiasDetectionError(LegalPlatformError):
    """Exception for bias detection and mitigation errors"""
    pass

def handle_exceptions(error_type: type = LegalPlatformError, 
                     default_message: str = "An unexpected error occurred",
                     log_error: bool = True):
    """Decorator for handling exceptions in functions"""
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_error:
                    logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
                
                if isinstance(e, LegalPlatformError):
                    raise e
                else:
                    raise error_type(
                        message=f"{default_message}: {str(e)}",
                        error_code="UNEXPECTED_ERROR",
                        details={"function": func.__name__, "traceback": traceback.format_exc()}
                    )
        return wrapper
    return decorator

def validate_file_upload(file_path: str, max_size_mb: int = 50, allowed_extensions: list = None) -> bool:
    """Validate uploaded file"""
    import os
    from pathlib import Path
    
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.docx', '.doc', '.txt', '.png', '.jpg', '.jpeg', '.tiff']
    
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise ValidationError("File does not exist", "FILE_NOT_FOUND")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > max_size_mb:
            raise ValidationError(
                f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)",
                "FILE_TOO_LARGE"
            )
        
        # Check file extension
        file_extension = Path(file_path).suffix.lower()
        if file_extension not in allowed_extensions:
            raise ValidationError(
                f"File type '{file_extension}' not supported. Allowed types: {allowed_extensions}",
                "UNSUPPORTED_FILE_TYPE"
            )
        
        return True
        
    except Exception as e:
        if isinstance(e, ValidationError):
            raise e
        else:
            raise ValidationError(f"File validation failed: {str(e)}", "VALIDATION_ERROR")

def format_error_for_user(error: Exception) -> Dict[str, Any]:
    """Format error for user-friendly display"""
    
    if isinstance(error, LegalPlatformError):
        return {
            "error": True,
            "message": error.message,
            "error_code": error.error_code,
            "user_message": _get_user_friendly_message(error.error_code),
            "details": error.details
        }
    else:
        return {
            "error": True,
            "message": "An unexpected error occurred. Please try again.",
            "error_code": "UNEXPECTED_ERROR",
            "user_message": "Something went wrong. Please check your input and try again.",
            "details": {}
        }

def _get_user_friendly_message(error_code: str) -> str:
    """Get user-friendly error messages"""
    
    messages = {
        "FILE_NOT_FOUND": "The uploaded file could not be found. Please try uploading again.",
        "FILE_TOO_LARGE": "The file is too large. Please upload a file smaller than 50MB.",
        "UNSUPPORTED_FILE_TYPE": "This file type is not supported. Please upload a PDF, Word document, or image file.",
        "OCR_FAILED": "Could not extract text from the document. Please ensure the image is clear and readable.",
        "MODEL_INFERENCE_FAILED": "AI processing failed. Please try again with a different document.",
        "VALIDATION_ERROR": "There was an issue with your input. Please check and try again.",
        "BIAS_DETECTION_FAILED": "Bias detection check failed. The result may not be reliable.",
        "NETWORK_ERROR": "Network connection issue. Please check your internet connection and try again.",
        "TIMEOUT_ERROR": "Processing took too long. Please try with a smaller document.",
        "GENERAL_ERROR": "An unexpected error occurred. Please contact support if the issue persists."
    }
    
    return messages.get(error_code, messages["GENERAL_ERROR"])

def log_and_raise_error(error_type: type, message: str, error_code: str, details: Dict[str, Any] = None):
    """Log an error and raise the appropriate exception"""
    
    logger.error(f"Error [{error_code}]: {message}", extra={"details": details})
    raise error_type(message=message, error_code=error_code, details=details)

def safe_execute(func, *args, **kwargs):
    """Safely execute a function and return result or error info"""
    
    try:
        result = func(*args, **kwargs)
        return {"success": True, "result": result, "error": None}
    except Exception as e:
        error_info = format_error_for_user(e)
        return {"success": False, "result": None, "error": error_info}

class ErrorContext:
    """Context manager for error handling with automatic logging"""
    
    def __init__(self, operation_name: str, error_type: type = LegalPlatformError):
        self.operation_name = operation_name
        self.error_type = error_type
    
    def __enter__(self):
        logger.debug(f"Starting operation: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            logger.error(f"Operation '{self.operation_name}' failed: {exc_val}", exc_info=True)
            if not isinstance(exc_val, LegalPlatformError):
                # Convert generic exceptions to platform-specific ones
                raise self.error_type(
                    message=f"Operation '{self.operation_name}' failed: {str(exc_val)}",
                    error_code="OPERATION_FAILED",
                    details={"operation": self.operation_name, "original_error": str(exc_val)}
                )
        else:
            logger.debug(f"Operation completed successfully: {self.operation_name}")
        
        return False  # Don't suppress exceptions
