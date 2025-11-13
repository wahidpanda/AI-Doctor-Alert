import os
import glob
import logging
from pathlib import Path
from config.paths import DATA_PATHS, PROCESSING_CONFIG

logger = logging.getLogger(__name__)

class AudioDataCollector:
    def __init__(self, raw_audio_path=None):
        self.raw_audio_path = raw_audio_path or DATA_PATHS['raw_audio']
        self.audio_files = []
        self.placeholder_files = []
        self.medical_specialties = PROCESSING_CONFIG['medical_specialties']
        
    def discover_audio_files(self):
        """Discover all audio files and placeholder files"""
        logger.info("Discovering audio files...")
        
        audio_extensions = PROCESSING_CONFIG['audio_formats']
        
        # Find actual audio files
        for extension in audio_extensions:
            pattern = os.path.join(self.raw_audio_path, f"*{extension}")
            self.audio_files.extend(glob.glob(pattern))
        
        # Also check all files in directory
        for file_path in self.raw_audio_path.iterdir():
            if file_path.is_file():
                file_str = str(file_path)
                if any(file_str.lower().endswith(ext) for ext in audio_extensions):
                    if file_str not in self.audio_files:
                        self.audio_files.append(file_str)
                elif file_str.endswith('.txt'):
                    # This is a placeholder file
                    self.placeholder_files.append(file_str)
        
        logger.info(f"Found {len(self.audio_files)} audio files and {len(self.placeholder_files)} placeholder files")
        
        # Log file details
        for file_path in self.audio_files:
            filename = os.path.basename(file_path)
            specialty = self.get_medical_specialty(filename)
            logger.debug(f"  - {filename} ({specialty}) [AUDIO]")
        
        for file_path in self.placeholder_files:
            filename = os.path.basename(file_path)
            specialty = self.get_medical_specialty(filename.replace('.txt', ''))
            logger.debug(f"  - {filename} ({specialty}) [PLACEHOLDER]")
        
        return self.audio_files + self.placeholder_files
    
    def get_medical_specialty(self, filename):
        """Infer medical specialty from filename prefix"""
        # Remove extension for specialty detection
        name_without_ext = filename.split('.')[0]
        prefix = name_without_ext[:3]
        return self.medical_specialties.get(prefix, 'Unknown')
    
    def validate_dataset(self):
        """Validate the dataset structure and files"""
        logger.info("Validating dataset...")
        
        if not self.raw_audio_path.exists():
            logger.error(f"Raw audio path does not exist: {self.raw_audio_path}")
            return False
        
        all_files = self.audio_files + self.placeholder_files
        if not all_files:
            logger.warning("No audio or placeholder files found in the dataset")
            return False
        
        # Check file accessibility
        accessible_files = []
        for file_path in all_files:
            if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
                accessible_files.append(file_path)
            else:
                logger.warning(f"File not accessible: {file_path}")
        
        self.audio_files = [f for f in accessible_files if not f.endswith('.txt')]
        self.placeholder_files = [f for f in accessible_files if f.endswith('.txt')]
        
        logger.info(f"Accessible files: {len(self.audio_files)} audio, {len(self.placeholder_files)} placeholder")
        
        return len(accessible_files) > 0
    
    def get_dataset_summary(self):
        """Get summary of the dataset"""
        if not self.audio_files and not self.placeholder_files:
            self.discover_audio_files()
        
        specialty_count = {}
        file_types = {
            'audio': len(self.audio_files),
            'placeholder': len(self.placeholder_files)
        }
        
        # Count by specialty for audio files
        for file_path in self.audio_files:
            filename = os.path.basename(file_path)
            specialty = self.get_medical_specialty(filename)
            specialty_count[specialty] = specialty_count.get(specialty, 0) + 1
        
        # Count by specialty for placeholder files  
        for file_path in self.placeholder_files:
            filename = os.path.basename(file_path)
            specialty = self.get_medical_specialty(filename.replace('.txt', ''))
            specialty_count[specialty] = specialty_count.get(specialty, 0) + 1
        
        return {
            'total_files': len(self.audio_files) + len(self.placeholder_files),
            'file_types': file_types,
            'specialty_distribution': specialty_count,
            'file_formats': list(set(Path(f).suffix for f in self.audio_files))
        }
    
    def is_placeholder_file(self, file_path):
        """Check if file is a placeholder"""
        return str(file_path).endswith('.txt')
    
    def read_placeholder_metadata(self, file_path):
        """Read metadata from placeholder file"""
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Parse the placeholder file format
            metadata = {}
            for line in content.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            
            return metadata
        except Exception as e:
            logger.warning(f"Error reading placeholder file {file_path}: {e}")
            return {}