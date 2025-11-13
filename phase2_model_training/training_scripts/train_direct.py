#!/usr/bin/env python3
"""
Direct Training Runner - No complex dependencies
"""

import os
import sys

# Add the training scripts to path
current_dir = os.path.dirname(__file__)
training_scripts_dir = os.path.join(current_dir, 'training_scripts')
sys.path.insert(0, training_scripts_dir)

# Import and run directly
from medical_bert_training import run_training

# Your data path
data_path = os.path.join(current_dir, 'data', 'llm_medical_training_dataset.json')

if __name__ == "__main__":
    print("🚀 Direct Medical BERT Training")
    print("=" * 50)
    print(f"📁 Data path: {data_path}")
    
    if os.path.exists(data_path):
        run_training(data_path)
    else:
        print(f"❌ Data file not found: {data_path}")
        print("💡 Please ensure the data file exists")