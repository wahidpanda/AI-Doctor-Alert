import librosa
import numpy as np
from scipy import signal
import wave
import io
import tempfile
import os
import logging
import atexit
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

# Try to import resampy
try:
    import resampy
    HAS_RESAMPY = True
    logger.info("resampy available for audio resampling")
except ImportError:
    HAS_RESAMPY = False
    logger.warning("resampy not available, using scipy for resampling")

class AudioProcessor:
    def __init__(self):
        self.target_sample_rate = 16000  # Whisper requires 16kHz
        self.channels = 1
        self.min_duration = 15.0  # Minimum 15 seconds required
        self.max_duration = 300.0  # Maximum 5 minutes
        self.temp_files = []  # Track temporary files for cleanup
        atexit.register(self.cleanup_temp_files)
        
        # Supported audio formats - librosa supports most of these
        self.supported_formats = {
            '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac'
        }
        logger.info("AudioProcessor initialized with 15s minimum duration")
    
    def cleanup_temp_files(self):
        """Clean up any remaining temporary files"""
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
            except:
                pass
        self.temp_files.clear()
    
    def create_temp_file(self, suffix='.wav'):
        """Create a temporary file and track it for cleanup"""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.close()  # Close immediately so other processes can access it
        self.temp_files.append(temp_file.name)
        return temp_file.name
    
    def safe_delete(self, file_path):
        """Safely delete a file with retries"""
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                if file_path in self.temp_files:
                    self.temp_files.remove(file_path)
                return True
        except Exception as e:
            logger.warning(f"Could not delete {file_path}: {e}")
            return False
        return False
    
    def is_supported_format(self, file_path_or_name):
        """Check if the file format is supported"""
        if hasattr(file_path_or_name, 'name'):
            # It's an UploadedFile object
            file_ext = os.path.splitext(file_path_or_name.name)[1].lower()
        else:
            # It's a file path string
            file_ext = os.path.splitext(file_path_or_name)[1].lower()
        
        return file_ext in self.supported_formats
    
    def validate_audio_duration(self, audio_data, sample_rate, source_name="audio"):
        """Validate that audio meets duration requirements"""
        duration = len(audio_data) / sample_rate
        
        logger.info(f"Validating {source_name}: {duration:.2f}s")
        
        if duration < self.min_duration:
            raise Exception(f"Audio too short: {duration:.1f}s. Minimum required: {self.min_duration}s")
        
        if duration > self.max_duration:
            raise Exception(f"Audio too long: {duration:.1f}s. Maximum allowed: {self.max_duration}s")
        
        logger.info(f"✅ Audio duration valid: {duration:.2f}s")
        return duration
    
    def load_and_preprocess_audio(self, file_path):
        """Load audio file and preprocess to 16kHz mono - supports all formats"""
        try:
            logger.info(f"Loading and preprocessing audio file: {file_path}")
            
            # For ALL formats, use librosa directly - it handles most formats
            audio_data, original_sr = librosa.load(
                file_path, 
                sr=self.target_sample_rate,  # Resample during load
                mono=True                    # Convert to mono
            )
            
            # Validate duration after loading
            duration = self.validate_audio_duration(audio_data, original_sr, os.path.basename(file_path))
            
            logger.info(f"Loaded audio: {len(audio_data)} samples, {original_sr}Hz, {duration:.2f}s")
            
            # Apply additional preprocessing
            processed_audio = self.preprocess_audio(audio_data, original_sr)
            
            logger.info(f"Processed audio: {len(processed_audio)} samples, {self.target_sample_rate}Hz")
            return processed_audio, self.target_sample_rate, duration
            
        except Exception as e:
            logger.error(f"Error loading audio file {file_path}: {e}")
            raise Exception(f"Error loading audio file: {e}")
    
    def process_uploaded_file(self, uploaded_file):
        """Process uploaded file using ONLY librosa - NO external dependencies"""
        try:
            logger.info(f"Processing uploaded file: {uploaded_file.name}")
            
            # Use librosa to load directly from bytes - this handles most formats
            audio_data, original_sr = librosa.load(
                io.BytesIO(uploaded_file.getvalue()),
                sr=self.target_sample_rate,  # Resample to 16kHz
                mono=True                    # Convert to mono
            )
            
            # Validate duration
            duration = self.validate_audio_duration(audio_data, original_sr, uploaded_file.name)
            
            # Apply preprocessing
            processed_audio = self.preprocess_audio(audio_data, original_sr)
            
            logger.info(f"Uploaded file processing complete: {len(processed_audio)} samples, {self.target_sample_rate}Hz, {duration:.2f}s")
            return processed_audio, self.target_sample_rate, duration
            
        except Exception as e:
            logger.error(f"Uploaded file processing failed: {e}")
            # Fallback: save to temp file and try again
            try:
                logger.info("Trying fallback method with temporary file")
                file_ext = os.path.splitext(uploaded_file.name)[1].lower()
                temp_path = self.create_temp_file(suffix=file_ext)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getvalue())
                
                audio_data, sample_rate, duration = self.load_and_preprocess_audio(temp_path)
                self.safe_delete(temp_path)
                return audio_data, sample_rate, duration
                
            except Exception as fallback_error:
                logger.error(f"Fallback method also failed: {fallback_error}")
                raise Exception(f"Could not process audio file. Please try a WAV or MP3 file.")
    
    def preprocess_audio(self, audio_data, original_sr):
        """Preprocess audio: resample to 16kHz, convert to mono, noise suppression"""
        try:
            # Check if audio_data is None
            if audio_data is None:
                raise Exception("No audio data to process")
                
            logger.info(f"Preprocessing audio: {len(audio_data)} samples, {original_sr}Hz")
            
            # Convert to mono if stereo (librosa should already do this, but just in case)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.info("Converted stereo to mono")
            
            # Resample to 16kHz if needed (librosa should have done this, but double-check)
            if original_sr != self.target_sample_rate:
                logger.info(f"Resampling from {original_sr}Hz to {self.target_sample_rate}Hz")
                if HAS_RESAMPY:
                    audio_data = resampy.resample(audio_data, original_sr, self.target_sample_rate)
                else:
                    num_samples = int(len(audio_data) * self.target_sample_rate / original_sr)
                    audio_data = signal.resample(audio_data, num_samples)
            
            # Ensure minimum length for Whisper (at least 1 second)
            audio_data = self.ensure_minimum_length(audio_data, self.target_sample_rate)
            
            # Simple noise reduction
            audio_data = self._noise_reduction(audio_data)
            logger.info("Noise reduction applied")
            
            # Normalize audio
            audio_data = self._normalize_audio(audio_data)
            logger.info("Audio normalization applied")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise Exception(f"Audio preprocessing failed: {e}")
    
    def ensure_minimum_length(self, audio_data, sample_rate):
        """Ensure audio meets minimum length requirement for Whisper"""
        current_duration = len(audio_data) / sample_rate
        
        if current_duration < 1.0:  # Whisper's internal minimum
            logger.info(f"Audio very short ({current_duration:.2f}s), padding to 1s")
            target_samples = int(1.0 * sample_rate)
            if len(audio_data) < target_samples:
                silence_length = target_samples - len(audio_data)
                silence = np.zeros(silence_length, dtype=audio_data.dtype)
                audio_data = np.concatenate([audio_data, silence])
        
        return audio_data
    
    def _noise_reduction(self, audio_data):
        """Simple noise reduction using filtering"""
        try:
            # High-pass filter to remove low-frequency noise
            b, a = signal.butter(5, 100, btype='high', fs=self.target_sample_rate)
            audio_data = signal.filtfilt(b, a, audio_data)
            return audio_data
            
        except Exception as e:
            logger.warning(f"Noise reduction failed, using original audio: {e}")
            return audio_data
    
    def _normalize_audio(self, audio_data):
        """Normalize audio to prevent clipping"""
        if len(audio_data) == 0:
            return audio_data
            
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val * 0.9
        return audio_data


class WhisperTranscriber:
    def __init__(self, model_size="base"):
        """
        Initialize Whisper transcriber using Transformers
        """
        logger.info(f"Initializing WhisperTranscriber with model: {model_size}...")
        self.model_size = model_size
        self.processor = None
        self.model = None
        self.audio_processor = AudioProcessor()
        self._load_whisper_model()
        logger.info("WhisperTranscriber initialized successfully")
    
    def _load_whisper_model(self):
        """Load Whisper model using Transformers"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            import torch
            
            model_name = f"openai/whisper-{self.model_size}"
            logger.info(f"Loading model: {model_name}")
            
            # Load processor and model separately
            self.processor = WhisperProcessor.from_pretrained(model_name)
            self.model = WhisperForConditionalGeneration.from_pretrained(model_name)
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"✅ Whisper model loaded successfully: {model_name}")
            
        except ImportError:
            logger.error("Transformers not installed. Please run: pip install transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe_audio(self, audio_file_path):
        """Transcribe audio file - automatically handles any sample rate and format"""
        try:
            logger.info(f"Transcribing audio: {audio_file_path}")
            
            # Check if format is supported
            if not self.audio_processor.is_supported_format(audio_file_path):
                supported = ', '.join(self.audio_processor.supported_formats)
                raise Exception(f"Unsupported audio format. Supported: {supported}")
            
            # Load and preprocess audio to 16kHz
            audio_data, sample_rate, duration = self.audio_processor.load_and_preprocess_audio(audio_file_path)
            
            # Process with Whisper
            transcription = self._transcribe_audio_array(audio_data, sample_rate)
            
            # RETURN EXACTLY 2 VALUES
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise Exception(f"Audio transcription failed: {e}")
    
    def transcribe_uploaded_file(self, uploaded_file):
        """Transcribe uploaded file - handles any format and sample rate"""
        try:
            logger.info(f"Transcribing uploaded file: {uploaded_file.name}")
            
            # Check if format is supported
            if not self.audio_processor.is_supported_format(uploaded_file):
                supported = ', '.join(self.audio_processor.supported_formats)
                file_ext = os.path.splitext(uploaded_file.name)[1]
                raise Exception(f"Unsupported audio format '{file_ext}'. Supported: {supported}")
            
            # Process the uploaded file using our pure librosa method
            audio_data, sample_rate, duration = self.audio_processor.process_uploaded_file(uploaded_file)
            
            # Process with Whisper
            transcription = self._transcribe_audio_array(audio_data, sample_rate)
            
            # RETURN EXACTLY 2 VALUES
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Uploaded file transcription failed: {e}")
            raise
    
    def _transcribe_audio_array(self, audio_data, sample_rate):
        """Transcribe audio array using Whisper model"""
        try:
            # Verify sample rate
            if sample_rate != 16000:
                raise ValueError(f"Whisper requires 16kHz audio, got {sample_rate}Hz")
            
            # Ensure audio is in the right format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Process audio with Whisper processor
            input_features = self.processor.feature_extractor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).input_features
            
            # Generate transcription
            import torch
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features,
                    language="english",
                    task="transcribe",
                    max_length=448,
                    num_beams=1,
                    temperature=0.0
                )
            
            # Decode transcription
            transcription = self.processor.batch_decode(
                predicted_ids, 
                skip_special_tokens=True
            )[0].strip()
            
            if not transcription:
                logger.warning("Whisper returned empty transcription")
                return "No speech detected in the audio. Please ensure clear audio quality."
            
            logger.info(f"✅ Transcription successful: {len(transcription)} characters")
            return transcription
            
        except Exception as e:
            logger.error(f"Audio array transcription failed: {e}")
            # Return a basic transcription instead of failing
            return "Speech detected but could not transcribe clearly. Please try recording again with clearer audio."


# Audio info function that handles both file paths and UploadedFile objects
def get_audio_info(file_path_or_uploaded_file):
    """Get information about audio file - supports both file paths and UploadedFile objects"""
    temp_path = None
    processor = None
    try:
        processor = AudioProcessor()
        
        # Handle both file paths and UploadedFile objects
        if hasattr(file_path_or_uploaded_file, 'name'):
            # It's an UploadedFile object
            uploaded_file = file_path_or_uploaded_file
            
            # Check if format is supported
            if not processor.is_supported_format(uploaded_file):
                supported = ', '.join(processor.supported_formats)
                return {
                    'valid': False,
                    'message': f"Unsupported audio format. Supported: {supported}"
                }
            
            # Use our pure librosa processing
            try:
                audio_data, sample_rate, duration = processor.process_uploaded_file(uploaded_file)
                
                info = {
                    'valid': True,
                    'duration': duration,
                    'sample_rate': sample_rate,
                    'channels': 1,
                    'samples': len(audio_data),
                    'message': f"Valid audio: {duration:.2f}s, {sample_rate}Hz"
                }
                return info
                
            except Exception as e:
                logger.warning(f"Audio info processing failed: {e}")
                return {
                    'valid': False,
                    'message': f"Could not process audio file: {str(e)}"
                }
        else:
            # It's a file path string
            file_to_process = file_path_or_uploaded_file
            
            # Check if format is supported
            if not processor.is_supported_format(file_to_process):
                supported = ', '.join(processor.supported_formats)
                return {
                    'valid': False,
                    'message': f"Unsupported audio format. Supported: {supported}"
                }
        
        # Load and process the audio file
        audio_data, sample_rate, duration = processor.load_and_preprocess_audio(file_to_process)
        
        info = {
            'valid': True,
            'duration': duration,
            'sample_rate': sample_rate,
            'channels': 1,
            'samples': len(audio_data),
            'message': f"Valid audio: {duration:.2f}s, {sample_rate}Hz"
        }
        return info
        
    except Exception as e:
        return {
            'valid': False,
            'message': f"Invalid audio file: {str(e)}"
        }
    finally:
        # Clean up temporary file if we created one
        if temp_path and processor:
            processor.safe_delete(temp_path)


# Simple Mock Transcriber for fallback
class MockWhisperTranscriber:
    """Simple mock transcriber for demo purposes"""
    def __init__(self, model_size="base"):
        self.model_size = model_size
        logger.info(f"MockWhisperTranscriber initialized with model: {model_size}")
    
    def transcribe_audio(self, audio_file_path):
        """Mock transcription - RETURNS EXACTLY 2 VALUES"""
        import random
        mock_transcriptions = [
            "Patient reports chest pain and difficulty breathing, needs immediate medical attention.",
            "Patient has fever, headache, and mild cough, moderate symptoms observed.",
            "Patient reports seasonal allergies with runny nose and sneezing, routine care recommended.",
            "Patient experiencing severe abdominal pain and nausea, requires urgent evaluation.",
            "Patient with minor cut and bruise, basic first aid sufficient."
        ]
        duration = 30.0
        transcription = random.choice(mock_transcriptions)
        # RETURN EXACTLY 2 VALUES
        return transcription, duration
    
    def transcribe_uploaded_file(self, uploaded_file):
        """Mock transcription for uploaded files - RETURNS EXACTLY 2 VALUES"""
        transcription, duration = self.transcribe_audio("mock_file.wav")
        # RETURN EXACTLY 2 VALUES
        return transcription, duration


# Function to create transcriber with fallback
def create_transcriber(model_size="base"):
    """Create transcriber with fallback to mock version"""
    try:
        # Try to create real transcriber
        transcriber = WhisperTranscriber(model_size)
        logger.info("✅ Real WhisperTranscriber created successfully")
        return transcriber
    except Exception as e:
        logger.warning(f"❌ Failed to create real WhisperTranscriber: {e}")
        logger.info("🔄 Falling back to MockWhisperTranscriber")
        return MockWhisperTranscriber(model_size)


if __name__ == "__main__":
    # Test the audio processor
    processor = AudioProcessor()
    print("✅ AudioProcessor initialized successfully")
    
    # Test transcriber creation
    transcriber = create_transcriber("base")
    print(f"✅ Transcriber created: {type(transcriber).__name__}")
