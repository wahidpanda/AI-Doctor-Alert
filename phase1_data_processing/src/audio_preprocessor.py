import os
import librosa
import soundfile as sf
import numpy as np
import logging
from pathlib import Path
from config.paths import DATA_PATHS, PROCESSING_CONFIG

logger = logging.getLogger(__name__)

class AudioPreprocessor:
    def __init__(self, target_sr=None):
        self.target_sr = target_sr or PROCESSING_CONFIG['target_sample_rate']
        self.target_channels = 1  # mono
        
    def load_audio(self, file_path):
        """Load audio file using librosa"""
        try:
            logger.debug(f"Loading: {os.path.basename(file_path)}")
            audio, sr = librosa.load(file_path, sr=self.target_sr, mono=True)
            logger.debug(f"  Original SR: {sr}, Length: {len(audio)/sr:.2f}s")
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            return None, None
    
    def convert_to_mono(self, audio):
        """Convert stereo to mono if needed"""
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        return audio
    
    def resample_audio(self, audio, original_sr):
        """Resample audio to target sample rate"""
        if original_sr != self.target_sr:
            audio = librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        return audio
    
    def noise_reduction(self, audio):
        """Advanced noise reduction using spectral gating"""
        try:
            stft = librosa.stft(audio)
            magnitude = np.abs(stft)
            phase = np.angle(stft)
            
            # Calculate noise threshold
            noise_frames = min(10, magnitude.shape[1])
            if noise_frames > 0:
                noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
                threshold = noise_profile * 2.0
                mask = magnitude / (magnitude + threshold + 1e-8)
                magnitude_reduced = magnitude * mask
                stft_reduced = magnitude_reduced * np.exp(1j * phase)
                audio_reduced = librosa.istft(stft_reduced)
                return audio_reduced
        except Exception as e:
            logger.warning(f"Noise reduction failed: {e}")
        
        return audio
    
    def normalize_audio(self, audio):
        """Normalize audio to prevent clipping"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.9
        return audio
    
    def trim_silence(self, audio, top_db=20):
        """Trim leading and trailing silence"""
        try:
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return audio_trimmed
        except Exception as e:
            logger.warning(f"Silence trimming failed: {e}")
            return audio
    
    def preprocess_audio_file(self, input_path, output_path):
        """Complete preprocessing pipeline for an audio file"""
        logger.info(f"Preprocessing: {os.path.basename(input_path)}")
        
        # Load audio
        audio, sr = self.load_audio(input_path)
        if audio is None:
            return False
        
        # Apply preprocessing steps
        original_length = len(audio)
        audio = self.convert_to_mono(audio)
        audio = self.resample_audio(audio, sr)
        audio = self.trim_silence(audio)
        audio = self.noise_reduction(audio)
        audio = self.normalize_audio(audio)
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save processed audio as WAV
        sf.write(output_path, audio, self.target_sr, subtype='PCM_16')
        
        logger.info(f"✓ Saved: {os.path.basename(output_path)}")
        logger.debug(f"  Duration: {original_length/sr:.2f}s → {len(audio)/self.target_sr:.2f}s")
        
        return True
    
    def batch_preprocess(self, audio_files):
        """Preprocess multiple audio files"""
        processed_files = []
        
        for i, audio_file in enumerate(audio_files, 1):
            filename = Path(audio_file).name
            output_file = DATA_PATHS['processed_audio'] / f"processed_{filename.replace('.mp3', '.wav').replace('.m4a', '.wav').replace('.flac', '.wav').replace('.aac', '.wav')}"
            
            logger.info(f"[{i}/{len(audio_files)}] Processing {filename}")
            
            if self.preprocess_audio_file(audio_file, output_file):
                processed_files.append({
                    'original_file': audio_file,
                    'processed_file': str(output_file),
                    'filename': filename
                })
        
        return processed_files