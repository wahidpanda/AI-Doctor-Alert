#!/usr/bin/env python3
"""
Fixed Medical BERT Training Runner
"""

import os
import sys

def main():
    print("🚀 Fixed Medical BERT Training")
    print("=" * 50)
    
    # Add training_scripts to path
    current_dir = os.path.dirname(__file__)
    training_scripts_dir = os.path.join(current_dir, 'training_scripts')
    sys.path.insert(0, training_scripts_dir)
    
    # Data path
    data_path = 'D:\Project\Assesment AI Task\phase2_model_training\data\llm_medical_training_dataset.json'
    
    if not os.path.exists(data_path):
        print(f"❌ Data not found at: {data_path}")
        print("💡 Please check if the data file exists")
        return
    
    print(f"📁 Using data from: {data_path}")
    
    try:
        from medical_bert_training import run_training
        run_training(data_path)
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()