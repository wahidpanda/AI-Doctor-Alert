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

# Try to import pydub
try:
    from pydub import AudioSegment
    HAS_PYDUB = True
    logger.info("pydub available for audio conversion")
except ImportError:
    HAS_PYDUB = False
    logger.warning("pydub not available, using alternative methods")

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
        
        # Supported audio formats
        self.supported_formats = {
            '.wav', '.mp3', '.m4a', '.flac', '.ogg', '.aac', 
            '.mp4', '.webm', '.3gp'
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
    
    def convert_to_wav_using_librosa(self, input_path, output_path):
        """Convert audio to WAV using librosa (works for most formats)"""
        try:
            logger.info(f"Converting {input_path} to WAV using librosa")
            
            # Load audio with librosa
            audio_data, sr = librosa.load(input_path, sr=None, mono=True)
            
            # Validate duration before processing
            self.validate_audio_duration(audio_data, sr, os.path.basename(input_path))
            
            # Resample to 16kHz if needed
            if sr != self.target_sample_rate:
                if HAS_RESAMPY:
                    audio_data = resampy.resample(audio_data, sr, self.target_sample_rate)
                else:
                    num_samples = int(len(audio_data) * self.target_sample_rate / sr)
                    audio_data = signal.resample(audio_data, num_samples)
            
            # Save as WAV
            audio_data_int16 = np.int16(audio_data * 32767)
            
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.target_sample_rate)
                wf.writeframes(audio_data_int16.tobytes())
            
            logger.info(f"✅ Successfully converted to WAV: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Librosa conversion failed: {e}")
            raise Exception(f"Could not process audio file: {str(e)}")
    
    def load_and_preprocess_audio(self, file_path):
        """Load audio file and preprocess to 16kHz mono - supports all formats"""
        try:
            logger.info(f"Loading and preprocessing audio file: {file_path}")
            
            # Check file extension
            file_ext = os.path.splitext(file_path)[1].lower()
            
            if file_ext != '.wav':
                # Convert non-WAV files to WAV first
                logger.info(f"Converting {file_ext} file to WAV format")
                wav_temp_path = self.create_temp_file(suffix='.wav')
                self.convert_to_wav_using_librosa(file_path, wav_temp_path)
                
                # Load the converted WAV file
                audio_data, original_sr = librosa.load(
                    wav_temp_path, 
                    sr=self.target_sample_rate,
                    mono=True
                )
                
                # Clean up temporary WAV file
                self.safe_delete(wav_temp_path)
            else:
                # Directly load WAV files
                audio_data, original_sr = librosa.load(
                    file_path, 
                    sr=self.target_sample_rate,
                    mono=True
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
    
    def preprocess_audio(self, audio_data, original_sr):
        """Preprocess audio: resample to 16kHz, convert to mono, noise suppression"""
        try:
            # Check if audio_data is None
            if audio_data is None:
                raise Exception("No audio data to process")
                
            logger.info(f"Preprocessing audio: {original_sr}Hz -> {self.target_sample_rate}Hz")
            
            # Convert to mono if stereo
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.info("Converted stereo to mono")
            
            # Always resample to 16kHz for Whisper compatibility
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
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Whisper transcription failed: {e}")
            raise Exception(f"Audio transcription failed: {e}")
    
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
            inputs = self.processor(
                audio_data, 
                sampling_rate=sample_rate, 
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=480000
            )
            
            # Generate transcription
            import torch
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs.input_features,
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
            raise Exception(f"Audio processing failed: {e}")
    
    def transcribe_uploaded_file(self, uploaded_file):
        """Transcribe uploaded file - handles any format and sample rate"""
        input_path = None
        
        try:
            logger.info(f"Transcribing uploaded file: {uploaded_file.name}")
            
            # Check if format is supported
            if not self.audio_processor.is_supported_format(uploaded_file):
                supported = ', '.join(self.audio_processor.supported_formats)
                file_ext = os.path.splitext(uploaded_file.name)[1]
                raise Exception(f"Unsupported audio format '{file_ext}'. Supported: {supported}")
            
            # Create temporary input file
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            input_path = self.audio_processor.create_temp_file(suffix=file_extension)
            
            # Save uploaded file
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Transcribe the file and get duration
            transcription, duration = self.transcribe_audio(input_path)
            
            return transcription, duration
            
        except Exception as e:
            logger.error(f"Uploaded file transcription failed: {e}")
            raise
        finally:
            # Clean up temporary files
            if input_path:
                self.audio_processor.safe_delete(input_path)


# Updated Audio info function that handles both file paths and UploadedFile objects
def get_audio_info(file_path_or_uploaded_file):
    """Get information about audio file - supports both file paths and UploadedFile objects"""
    temp_path = None
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
            
            # Create temporary file
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            temp_path = processor.create_temp_file(suffix=file_extension)
            
            # Save uploaded file to temporary path
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            file_to_process = temp_path
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
