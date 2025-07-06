#!/usr/bin/env python3
"""
AI-Powered Legal Document Intelligence Platform - Launcher Script
Simple script to start the Streamlit application with proper configuration
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'transformers',
        'spacy',
        'pytesseract',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Install missing packages with:")
        print("   pip install -r requirements.txt")
        return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract OCR is installed")
            return True
        else:
            print("⚠️  Tesseract OCR not found")
            return False
    except FileNotFoundError:
        print("⚠️  Tesseract OCR not found")
        print("💡 Install Tesseract OCR:")
        print("   Ubuntu/Debian: sudo apt-get install tesseract-ocr")
        print("   macOS: brew install tesseract")
        print("   Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def setup_environment():
    """Setup environment variables and directories"""
    # Create necessary directories
    directories = [
        'data',
        'logs',
        'models/saved_models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Set environment variables if not already set
    env_vars = {
        'PYTHONPATH': str(Path.cwd()),
        'STREAMLIT_SERVER_HEADLESS': 'true',
        'STREAMLIT_SERVER_ENABLE_CORS': 'false',
        'STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION': 'false'
    }
    
    for key, value in env_vars.items():
        if key not in os.environ:
            os.environ[key] = value
    
    print("✅ Environment setup complete")

def download_models():
    """Download required SpaCy models"""
    try:
        print("📥 Downloading SpaCy models...")
        subprocess.run([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'], 
                      check=True)
        print("✅ SpaCy models downloaded")
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Failed to download SpaCy models")
        print("💡 Try manually: python -m spacy download en_core_web_sm")
        return False

def run_application(port=8501, host='localhost', debug=False):
    """Run the Streamlit application"""
    app_path = Path('app/streamlit_app.py')
    
    if not app_path.exists():
        print(f"❌ Application file not found: {app_path}")
        return False
    
    print(f"🚀 Starting AI Legal Document Intelligence Platform...")
    print(f"📍 URL: http://{host}:{port}")
    print("⏹️  Press Ctrl+C to stop the application")
    
    # Streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run',
        str(app_path),
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'true' if not debug else 'false',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--server.maxUploadSize', '50'
    ]
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start application: {str(e)}")
        return False

def run_tests():
    """Run the test suite"""
    print("🧪 Running test suite...")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pytest', 'tests/', '-v'], 
                              check=True)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError:
        print("❌ Some tests failed")
        return False
    except FileNotFoundError:
        print("⚠️  pytest not found. Install with: pip install pytest")
        return False

def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='AI-Powered Legal Document Intelligence Platform Launcher'
    )
    
    parser.add_argument('--port', '-p', type=int, default=8501,
                       help='Port to run the application on (default: 8501)')
    parser.add_argument('--host', default='localhost',
                       help='Host to run the application on (default: localhost)')
    parser.add_argument('--debug', action='store_true',
                       help='Run in debug mode')
    parser.add_argument('--test', action='store_true',
                       help='Run tests instead of starting the application')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup environment and download models')
    parser.add_argument('--skip-checks', action='store_true',
                       help='Skip dependency and system checks')
    
    args = parser.parse_args()
    
    print("🏛️  AI-Powered Legal Document Intelligence Platform")
    print("=" * 60)
    
    # Setup environment
    setup_environment()
    
    if not args.skip_checks:
        # Check dependencies
        print("\n🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        # Check Tesseract (optional)
        check_tesseract()
        
        # Download models
        download_models()
    
    if args.setup_only:
        print("\n✅ Setup complete!")
        return
    
    if args.test:
        # Run tests
        success = run_tests()
        sys.exit(0 if success else 1)
    
    # Run application
    print("\n" + "=" * 60)
    success = run_application(args.port, args.host, args.debug)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
