#!/usr/bin/env python3
"""
Quick script to run the data processing pipeline
"""

import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import main

if __name__ == "__main__":
    print("Medical Voice-to-Text System - Phase 1: Data Processing")
    print("Starting pipeline...")
    main()