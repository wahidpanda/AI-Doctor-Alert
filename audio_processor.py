import librosa
import numpy as np
from scipy import signal
import wave
import io
import tempfile
import os
import logging
import atexit
from pydub import AudioSegment
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

class AudioProcessor:
    def __init__(self):
        self.target_sample_rate = 16000  # Whisper requires 16kHz
        self.channels = 1
        self.temp_files = []  # Track temporary files for cleanup
        atexit.register(self.cleanup_temp_files)
        logger.info("AudioProcessor initialized")
    
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
            # Don't remove from temp_files list so we can try again later
            return False
        return False
        
    def record_audio(self, duration=10, sample_rate=16000):
        """Record audio using Streamlit's native audio input"""
        try:
            logger.info(f"Waiting for audio recording...")
            
            # Use Streamlit's built-in audio recorder
            st.info("🎤 Click the microphone button below to start recording...")
            audio_bytes = st.audio_input("Speak now...", key=f"recorder_{duration}")
            
            if audio_bytes:
                logger.info("Audio recording received")
                
                # Convert bytes to AudioSegment
                audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # Convert to numpy array
                samples = np.array(audio.get_array_of_samples())
                
                # Convert to float32 for processing
                samples = samples.astype(np.float32) / 32768.0  # Normalize to [-1, 1]
                
                logger.info(f"Recording finished: {len(samples)} samples, {audio.frame_rate}Hz")
                return samples, audio.frame_rate
            else:
                logger.warning("No audio recorded - user didn't record anything")
                return None, sample_rate
            
        except Exception as e:
            logger.error(f"Recording failed: {e}")
            st.error(f"Audio recording failed: {e}")
            return None, sample_rate
    
    def save_audio_file(self, audio_data, file_path):
        """Save audio data to WAV file at 16kHz - FIXED to handle None"""
        try:
            logger.info(f"Saving audio to {file_path}")
            
            # Check if audio_data is None
            if audio_data is None:
                raise Exception("No audio data to save - recording failed or no audio recorded")
                
            if len(audio_data) == 0:
                raise Exception("Empty audio data")
                
            # Ensure audio data is properly scaled for 16-bit PCM
            audio_data_int16 = np.int16(audio_data * 32767)
            
            with wave.open(file_path, 'wb') as wf:
                wf.setnchannels(self.channels)
                wf.setsampwidth(2)
                wf.setframerate(self.target_sample_rate)
                wf.writeframes(audio_data_int16.tobytes())
            
            logger.info(f"Audio saved successfully to {file_path} at {self.target_sample_rate}Hz")
            return True
            
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            raise Exception(f"Error saving audio file: {e}")
    
    def preprocess_audio(self, audio_data, original_sr):
        """Preprocess audio: resample to 16kHz, convert to mono, noise suppression"""
        try:
            # Check if audio_data is None
            if audio_data is None:
                raise Exception("No audio data to process")
                
            logger.info(f"Preprocessing audio: {original_sr}Hz -> {self.target_sample_rate}Hz")
            
            # Convert to mono if stereo (though Streamlit audio should already be mono)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
                logger.info("Converted stereo to mono")
            
            # Always resample to 16kHz for Whisper compatibility
            if original_sr != self.target_sample_rate:
                logger.info(f"Resampling from {original_sr}Hz to {self.target_sample_rate}Hz")
                audio_data = librosa.resample(
                    audio_data, 
                    orig_sr=original_sr, 
                    target_sr=self.target_sample_rate,
                    res_type='kaiser_fast'
                )
            
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
    
    def load_and_preprocess_audio(self, file_path):
        """Load audio file and preprocess to 16kHz mono"""
        try:
            logger.info(f"Loading and preprocessing audio file: {file_path}")
            
            # Load audio with librosa
            audio_data, original_sr = librosa.load(
                file_path, 
                sr=None,
                mono=True,
                res_type='kaiser_fast'
            )
            
            logger.info(f"Original audio: {len(audio_data)} samples, {original_sr}Hz")
            
            # Preprocess to 16kHz
            processed_audio = self.preprocess_audio(audio_data, original_sr)
            
            logger.info(f"Processed audio: {len(processed_audio)} samples, {self.target_sample_rate}Hz")
            return processed_audio, self.target_sample_rate
            
        except Exception as e:
            logger.error(f"Error loading audio file: {e}")
            raise Exception(f"Error loading audio file: {e}")
    
    def convert_audio_to_16khz(self, input_path, output_path):
        """Convert any audio file to 16kHz WAV format"""
        try:
            logger.info(f"Converting {input_path} to 16kHz WAV: {output_path}")
            
            # Load and preprocess audio
            audio_data, sr = self.load_and_preprocess_audio(input_path)
            
            # Save as 16kHz WAV
            self.save_audio_file(audio_data, output_path)
            
            logger.info(f"✅ Audio converted to 16kHz: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise Exception(f"Audio conversion failed: {e}")


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
        """Transcribe audio file - automatically handles any sample rate"""
        try:
            logger.info(f"Transcribing audio: {audio_file_path}")
            
            # Load and preprocess audio to 16kHz
            audio_data, sample_rate = self.audio_processor.load_and_preprocess_audio(audio_file_path)
            
            # Verify sample rate is 16kHz
            if sample_rate != 16000:
                logger.warning(f"Sample rate is {sample_rate}Hz, expected 16000Hz. Resampling...")
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            
            # Process with Whisper
            transcription = self._transcribe_audio_array(audio_data, sample_rate)
            
            return transcription
            
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
                padding=True
            )
            
            # Generate transcription
            import torch
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    inputs.input_features,
                    language="english",
                    task="transcribe"
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
        output_path = None
        
        try:
            logger.info(f"Transcribing uploaded file: {uploaded_file.name}")
            
            # Create temporary input file
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            input_path = self.audio_processor.create_temp_file(suffix=file_extension)
            
            with open(input_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Create temporary output file (16kHz WAV)
            output_path = self.audio_processor.create_temp_file(suffix='.wav')
            
            # Convert to 16kHz WAV first
            self.audio_processor.convert_audio_to_16khz(input_path, output_path)
            
            # Transcribe the converted file
            transcription = self.transcribe_audio(output_path)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Uploaded file transcription failed: {e}")
            raise
        finally:
            # Clean up temporary files
            if input_path:
                self.audio_processor.safe_delete(input_path)
            if output_path:
                self.audio_processor.safe_delete(output_path)


# Audio info function
def get_audio_info(file_path):
    """Get information about audio file"""
    try:
        processor = AudioProcessor()
        audio_data, sample_rate = processor.load_and_preprocess_audio(file_path)
        duration = len(audio_data) / sample_rate
        
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
