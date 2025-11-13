import whisper
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path
from config.paths import DATA_PATHS, PROCESSING_CONFIG

logger = logging.getLogger(__name__)

class AudioTranscriber:
    def __init__(self, model_size=None):
        self.model_size = model_size or PROCESSING_CONFIG['whisper_model']
        self.model = None
        if model_size != "placeholder":  # Only load Whisper if not in placeholder mode
            self._load_model()
    
    def _load_model(self):
        """Initialize Whisper model"""
        logger.info(f"Loading Whisper model ({self.model_size})...")
        try:
            self.model = whisper.load_model(self.model_size)
            logger.info("✓ Whisper model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            # Don't raise, we can use placeholder mode
    
    def transcribe_audio(self, audio_path, is_placeholder=False):
        """Transcribe audio file or read placeholder"""
        try:
            if is_placeholder:
                return self._transcribe_placeholder(audio_path)
            else:
                return self._transcribe_with_whisper(audio_path)
        except Exception as e:
            logger.error(f"Error transcribing {audio_path}: {e}")
            return None
    
    def _transcribe_with_whisper(self, audio_path):
        """Transcribe actual audio file with Whisper"""
        logger.info(f"Transcribing with Whisper: {Path(audio_path).name}")
        
        if self.model is None:
            logger.error("Whisper model not loaded")
            return None
        
        result = self.model.transcribe(
            audio_path,
            fp16=False,
            language='en',
            no_speech_threshold=0.6
        )
        
        transcription = result["text"].strip()
        
        # Calculate average confidence
        segments = result.get("segments", [])
        if segments:
            confidence = np.mean([seg.get("confidence", 0.5) for seg in segments])
        else:
            confidence = 0.5
        
        logger.info(f"✓ Transcription complete: {transcription[:100]}...")
        logger.debug(f"  Confidence: {confidence:.3f}")
        
        return {
            "text": transcription,
            "confidence": float(confidence),
            "language": "en",
            "timestamp": datetime.now().isoformat(),
            "segments": segments,
            "source": "whisper"
        }
    
    def _transcribe_placeholder(self, placeholder_path):
        """Extract transcription from placeholder file"""
        logger.info(f"Reading placeholder: {Path(placeholder_path).name}")
        
        from src.audio_collector import AudioDataCollector
        collector = AudioDataCollector()
        metadata = collector.read_placeholder_metadata(placeholder_path)
        
        transcription = metadata.get('Transcription', 'No transcription available')
        
        logger.info(f"✓ Placeholder processed: {transcription[:100]}...")
        
        return {
            "text": transcription,
            "confidence": 1.0,  # Placeholder has perfect confidence
            "language": "en",
            "timestamp": datetime.now().isoformat(),
            "segments": [],
            "source": "placeholder",
            "metadata": metadata
        }
    
    def batch_transcribe(self, processed_files, collector):
        """Transcribe multiple files (audio or placeholder)"""
        transcripts = []
        
        for i, file_info in enumerate(processed_files, 1):
            file_path = file_info['original_file']
            is_placeholder = collector.is_placeholder_file(file_path)
            
            logger.info(f"[{i}/{len(processed_files)}] Processing {Path(file_path).name}")
            
            transcription_result = self.transcribe_audio(file_path, is_placeholder)
            
            if transcription_result:
                transcript_entry = {
                    **file_info,
                    "transcription": transcription_result["text"],
                    "transcription_confidence": transcription_result["confidence"],
                    "transcription_timestamp": transcription_result["timestamp"],
                    "segments": transcription_result.get("segments", []),
                    "source": transcription_result.get("source", "unknown"),
                    "is_placeholder": is_placeholder
                }
                
                # Add placeholder metadata if available
                if is_placeholder and 'metadata' in transcription_result:
                    transcript_entry['placeholder_metadata'] = transcription_result['metadata']
                
                transcripts.append(transcript_entry)
                
                # Save individual transcript
                transcript_file = DATA_PATHS['transcripts'] / f"transcript_{file_info['filename'].replace('.mp3', '.json').replace('.m4a', '.json').replace('.flac', '.json').replace('.aac', '.json').replace('.txt', '.json')}"
                
                with open(transcript_file, 'w') as f:
                    json.dump(transcript_entry, f, indent=2)
                
                logger.debug(f"  Saved transcript: {transcript_file.name}")
        
        return transcripts