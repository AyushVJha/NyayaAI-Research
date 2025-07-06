# AI-Powered Legal Document Intelligence Platform - Dockerfile
# Multi-stage build for optimized production deployment

# Stage 1: Base image with system dependencies
FROM python:3.9-slim as base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Kolkata

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Tesseract OCR and language packs
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin \
    tesseract-ocr-tam \
    tesseract-ocr-tel \
    tesseract-ocr-ben \
    tesseract-ocr-guj \
    tesseract-ocr-kan \
    tesseract-ocr-mal \
    tesseract-ocr-mar \
    tesseract-ocr-ori \
    tesseract-ocr-pan \
    tesseract-ocr-urd \
    # Image processing libraries
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Document processing dependencies
    poppler-utils \
    antiword \
    unrtf \
    # Build tools
    gcc \
    g++ \
    make \
    # Utilities
    curl \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Stage 2: Python dependencies
FROM base as dependencies

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Download SpaCy models
RUN python -m spacy download en_core_web_sm

# Stage 3: Application
FROM dependencies as application

# Create non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/models/saved_models && \
    chown -R appuser:appuser /app

# Copy application code
COPY --chown=appuser:appuser . /app/

# Set proper permissions
RUN chmod +x /app/app/streamlit_app.py

# Switch to non-root user
USER appuser

# Create directories for user
RUN mkdir -p /app/data /app/logs /app/models/saved_models

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Expose Streamlit port
EXPOSE 8501

# Set working directory
WORKDIR /app

# Default command
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

# Alternative commands for different deployment scenarios
# For development:
# CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]

# For production with custom config:
# CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.enableCORS=false", "--server.enableXsrfProtection=false", "--server.maxUploadSize=50"]

# Build instructions:
# docker build -t legal-ai-platform .
# docker run -p 8501:8501 legal-ai-platform

# For development with volume mounting:
# docker run -p 8501:8501 -v $(pwd):/app legal-ai-platform

# For production deployment:
# docker run -d --name legal-ai-platform -p 8501:8501 --restart unless-stopped legal-ai-platform

# Environment variables that can be set at runtime:
# ENVIRONMENT=production
# LOG_LEVEL=INFO
# TESSERACT_CMD=/usr/bin/tesseract
# MODEL_CACHE_DIR=/app/models/cache
