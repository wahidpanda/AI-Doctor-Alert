import os
from pathlib import Path

# Base paths
PHASE1_ROOT = Path(__file__).parent.parent
PROJECT_ROOT = PHASE1_ROOT.parent

# Data paths
DATA_PATHS = {
    'raw_audio': PHASE1_ROOT / 'data' / 'raw_audio',
    'processed_audio': PHASE1_ROOT / 'data' / 'processed_audio',
    'transcripts': PHASE1_ROOT / 'data' / 'transcripts',
    'metadata': PHASE1_ROOT / 'data' / 'metadata',
    'outputs': PHASE1_ROOT / 'outputs',
    'llm_training': PHASE1_ROOT / 'outputs' / 'llm_training_data',
    'analysis_reports': PHASE1_ROOT / 'outputs' / 'analysis_reports',
    'logs': PHASE1_ROOT / 'logs',
    'kaggle_download': PHASE1_ROOT / 'data' / 'kaggle_download'
}

# Create directories
for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Processing configuration
PROCESSING_CONFIG = {
    'target_sample_rate': 16000,
    'whisper_model': 'base',
    'audio_formats': ['.mp3', '.wav', '.m4a', '.flac', '.aac'],
    'medical_specialties': {
        'CAR': 'Cardiology',
        'DER': 'Dermatology', 
        'GAS': 'Gastroenterology',
        'GEN': 'General Medicine',
        'MSK': 'Musculoskeletal',
        'RES': 'Respiratory'
    },
    'kaggle_dataset': "islamwahid/audio-data"
}