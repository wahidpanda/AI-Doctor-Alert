import os
import zipfile
import streamlit as st
from pathlib import Path
import requests
import time

class ModelDownloader:
    def __init__(self):
        self.models_dir = "models"
        self.drive_base_url = "https://drive.google.com/uc?id="
        self.model_files = {
            "whisper": {
                "url": "https://drive.google.com/file/d/1-EXAMPLE-WHISPER-ID/view?usp=sharing",
                "file_id": "1-EXAMPLE-WHISPER-ID",  # Replace with actual file ID
                "filename": "whisper_model.zip",
                "extract_to": "whisper_models"
            },
            "medical_bert": {
                "url": "https://drive.google.com/file/d/1-EXAMPLE-BERT-ID/view?usp=sharing",
                "file_id": "1-EXAMPLE-BERT-ID",  # Replace with actual file ID
                "filename": "medical_bert_model.zip",
                "extract_to": "medical_bert_model"
            },
            "audio_models": {
                "url": "https://drive.google.com/file/d/1-EXAMPLE-AUDIO-ID/view?usp=sharing",
                "file_id": "1-EXAMPLE-AUDIO-ID",  # Replace with actual file ID
                "filename": "audio_models.zip",
                "extract_to": "audio_models"
            }
        }
    
    def get_file_id_from_url(self, url):
        """Extract file ID from Google Drive URL"""
        if 'drive.google.com' in url:
            if '/file/d/' in url:
                file_id = url.split('/file/d/')[1].split('/')[0]
            elif 'id=' in url:
                file_id = url.split('id=')[1].split('&')[0]
            else:
                file_id = url.split('/')[-2]
            return file_id
        return None
    
    def download_file_from_google_drive(self, file_id, destination):
        """Download file from Google Drive"""
        URL = "https://drive.google.com/uc?id="
        
        session = requests.Session()
        response = session.get(URL + file_id, stream=True)
        
        # Get confirmation token
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                token = value
                break
        else:
            token = None
        
        if token:
            params = {'id': file_id, 'confirm': token}
            response = session.get(URL, params=params, stream=True)
        
        # Save content
        CHUNK_SIZE = 32768
        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
    
    def extract_zip(self, zip_path, extract_to):
        """Extract zip file"""
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    def download_models(self, drive_link=None, specific_model=None):
        """Download all models or specific model"""
        os.makedirs(self.models_dir, exist_ok=True)
        
        # If specific drive link provided
        if drive_link:
            st.info(f"Downloading models from: {drive_link}")
            file_id = self.get_file_id_from_url(drive_link)
            if file_id:
                zip_path = os.path.join(self.models_dir, "all_models.zip")
                self.download_file_from_google_drive(file_id, zip_path)
                
                # Extract all models
                self.extract_zip(zip_path, self.models_dir)
                st.success("All models downloaded and extracted successfully!")
                return True
            else:
                st.error("Invalid Google Drive link")
                return False
        
        # Download specific model
        if specific_model and specific_model in self.model_files:
            model_info = self.model_files[specific_model]
            file_id = model_info["file_id"]
            zip_path = os.path.join(self.models_dir, model_info["filename"])
            extract_path = os.path.join(self.models_dir, model_info["extract_to"])
            
            st.info(f"Downloading {specific_model} model...")
            self.download_file_from_google_drive(file_id, zip_path)
            self.extract_zip(zip_path, extract_path)
            st.success(f"{specific_model} model downloaded successfully!")
            return True
        
        return False
    
    def check_models_exist(self):
        """Check if models are already downloaded"""
        required_paths = [
            "models/whisper_models",
            "models/medical_bert_model",
            "models/audio_models"
        ]
        
        for path in required_paths:
            if not os.path.exists(path):
                return False
        return True
    
    def get_model_path(self, model_name):
        """Get path to specific model"""
        model_paths = {
            "whisper": "models/whisper_models",
            "medical_bert": "models/medical_bert_model",
            "audio_processor": "models/audio_models"
        }
        return model_paths.get(model_name)

def download_models_interface():
    """Streamlit interface for model download"""
    st.title("🔧 Model Download Center")
    st.markdown("### Download required AI models for the application")
    
    downloader = ModelDownloader()
    
    # Check if models already exist
    if downloader.check_models_exist():
        st.success("✅ All models are already downloaded and ready!")
        return True
    
    st.warning("⚠️ Required models are not downloaded yet.")
    
    # Option 1: Download all models from a single link
    st.subheader("Option 1: Download All Models")
    st.markdown("""
    **Instructions:**
    1. Upload all your model files to a folder on Google Drive
    2. Zip the folder (name it `medical_models.zip`)
    3. Share the zip file publicly and paste the link below
    """)
    
    drive_link = st.text_input(
        "Google Drive Link to Models Zip File:",
        placeholder="https://drive.google.com/file/d/your-file-id/view?usp=sharing"
    )
    
    if st.button("Download All Models", type="primary"):
        if drive_link:
            with st.spinner("Downloading models... This may take a few minutes."):
                success = downloader.download_models(drive_link=drive_link)
                if success:
                    st.balloons()
                    st.success("All models downloaded successfully! You can now use the application.")
                    return True
                else:
                    st.error("Failed to download models. Please check the link.")
        else:
            st.error("Please provide a Google Drive link")
    
    st.markdown("---")
    
    # Option 2: Download individual models (if you have separate links)
    st.subheader("Option 2: Download Individual Models")
    st.markdown("If you have separate links for each model, you can download them individually.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Download Whisper Model"):
            with st.spinner("Downloading Whisper model..."):
                downloader.download_models(specific_model="whisper")
    
    with col2:
        if st.button("Download Medical BERT Model"):
            with st.spinner("Downloading Medical BERT model..."):
                downloader.download_models(specific_model="medical_bert")
    
    with col3:
        if st.button("Download Audio Models"):
            with st.spinner("Downloading audio models..."):
                downloader.download_models(specific_model="audio_models")
    
    return False