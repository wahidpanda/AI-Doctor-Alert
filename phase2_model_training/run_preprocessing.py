#!/usr/bin/env python3
"""
Medical Dataset Preprocessing Runner
Run this script to prepare your dataset for LLM training
"""

import os
import subprocess
import sys

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False
    return True

def download_spacy_model():
    """Download required spaCy model"""
    print("🔄 Downloading spaCy model...")
    try:
        # Try to download the small English model
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✅ spaCy model downloaded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download spaCy model: {e}")
        print("💡 You can manually install with: python -m spacy download en_core_web_sm")
        return False
    return True

def main():
    """Main runner function"""
    print("🚀 Medical Dataset Preprocessing Runner")
    print("=" * 50)
    
    # Check if we're in the right directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"📁 Current directory: {current_dir}")
    
    # Install requirements
    if not install_requirements():
        print("⚠️ Continuing without installing requirements...")
    
    # Download spaCy model
    download_spacy_model()
    
    # Run the preprocessor
    print("\n🎯 Starting dataset preprocessing...")
    try:
        from prepare_llm_dataset import MedicalDatasetPreprocessor
        preprocessor = MedicalDatasetPreprocessor()
        preprocessor.run()
    except Exception as e:
        print(f"❌ Error running preprocessor: {e}")
        print("💡 Make sure all requirements are installed and dataset path is correct")

if __name__ == "__main__":
    main()