import streamlit as st
import sqlite3
import os
import tempfile
from datetime import datetime
import sys
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(__file__))

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

try:
    from audio_processor import AudioProcessor, WhisperTranscriber, get_audio_info
    from model_predictor import MedicalBERTPredictor
    from database import DatabaseManager
    import torch
except ImportError as e:
    st.error(f"❌ Missing dependency: {e}")
    st.info("Please run: pip install torch transformers librosa plotly")

# Email Configuration
EMAIL_CONFIG = {
    'sender_email': 'pagoldr01@gmail.com',
    'sender_password': 'vcahdtpodoqbwalw',  # Google App Password
    'doctor_email': 'islamoahidul12@gmail.com',
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587
}

class EmailNotifier:
    def __init__(self, config):
        self.config = config
    
    def send_urgency_alert(self, patient_info, analysis_result):
        """Send email alert to doctor for high urgency cases"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config['sender_email']
            msg['To'] = self.config['doctor_email']
            msg['Subject'] = f"🚨 URGENT MEDICAL ALERT - High Urgency Case Detected"
            
            # Create email body
            body = f"""
            <html>
            <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
                <div style="background-color: white; padding: 30px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                    <h2 style="color: #ff4b4b; margin-bottom: 20px;">🚨 MEDICAL URGENCY ALERT</h2>
                    
                    <div style="background-color: #fff4f4; padding: 15px; border-radius: 5px; margin-bottom: 20px;">
                        <h3 style="color: #ff4b4b; margin-top: 0;">URGENT ATTENTION REQUIRED</h3>
                        <p style="font-size: 16px; margin: 10px 0;">
                            A <strong>high urgency medical case</strong> has been detected through the Medical Urgency Analysis System.
                        </p>
                    </div>
                    
                    <h3 style="color: #333;">Patient Analysis Details:</h3>
                    <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Patient Status</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{analysis_result.get('patient_status', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Urgency Level</td>
                            <td style="padding: 10px; border: 1px solid #ddd; color: #ff4b4b; font-weight: bold;">{analysis_result.get('urgency_level', 'N/A')}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Confidence Score</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{analysis_result.get('confidence_score', 0):.2f}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Transcribed Text</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{analysis_result.get('transcribed_text', 'N/A')}</td>
                        </tr>
                        <tr style="background-color: #f8f9fa;">
                            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">Analysis Time</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{analysis_result.get('created_at', 'N/A')}</td>
                        </tr>
                        <tr>
                            <td style="padding: 10px; border: 1px solid #ddd; font-weight: bold;">User</td>
                            <td style="padding: 10px; border: 1px solid #ddd;">{patient_info.get('full_name', 'N/A')} ({patient_info.get('username', 'N/A')})</td>
                        </tr>
                    </table>
                    
                    <div style="background-color: #e8f4fd; padding: 15px; border-radius: 5px; border-left: 4px solid #2196F3;">
                        <h4 style="color: #2196F3; margin-top: 0;">Recommended Action:</h4>
                        <ul style="margin: 10px 0; padding-left: 20px;">
                            <li>Review the patient's case immediately</li>
                            <li>Contact the patient for follow-up assessment</li>
                            <li>Consider emergency intervention if necessary</li>
                            <li>Document the case in patient records</li>
                        </ul>
                    </div>
                    
                    <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                    <p style="color: #666; font-size: 12px;">
                        This alert was automatically generated by the Medical Urgency Analysis System.<br>
                        Please do not reply to this automated message.
                    </p>
                </div>
            </body>
            </html>
            """
            
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config['smtp_server'], self.config['smtp_port'])
            server.starttls()
            server.login(self.config['sender_email'], self.config['sender_password'])
            text = msg.as_string()
            server.sendmail(self.config['sender_email'], self.config['doctor_email'], text)
            server.quit()
            
            return True, "Alert email sent successfully to doctor"
            
        except Exception as e:
            return False, f"Failed to send email: {str(e)}"

# Initialize database and email notifier
db = DatabaseManager()
email_notifier = EmailNotifier(EMAIL_CONFIG)

# Initialize components
@st.cache_resource
def load_ai_components():
    """Load AI models"""
    try:
        # Initialize audio processor
        audio_processor = AudioProcessor()
        
        # Initialize Whisper transcriber
        transcriber = WhisperTranscriber(model_size="base")
        
        # Initialize medical BERT model
        model_path = "D:\Project\Assesment AI Task\phase3_webapp\models\medical_bert_model"
        model_predictor = MedicalBERTPredictor(model_path)
        
        return audio_processor, transcriber, model_predictor
        
    except Exception as e:
        st.error(f"❌ Failed to load AI components: {e}")
        return None, None, None

def safe_delete_file(file_path, max_retries=3, delay=0.1):
    """Safely delete a file with retries"""
    for attempt in range(max_retries):
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
                return True
        except PermissionError:
            if attempt < max_retries - 1:
                time.sleep(delay)
                continue
        except Exception:
            pass
    return False

def main():
    st.set_page_config(
        page_title="Medical Urgency Analyzer",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.recorded_audio_path = None
        st.session_state.current_analysis_result = None
        st.session_state.analysis_complete = False
        st.session_state.email_sent = False
    
    # Show login/register if not authenticated
    if not st.session_state.authenticated:
        show_auth_interface()
    else:
        show_main_app()

def show_auth_interface():
    """Show authentication interface (login/register)"""
    st.title("🏥 Medical Urgency Analysis System")
    st.markdown("### Please login or register to continue")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_btn = st.form_submit_button("Login")
            
            if login_btn:
                if username and password:
                    user = db.authenticate_user(username, password)
                    if user:
                        st.session_state.authenticated = True
                        st.session_state.user = user
                        st.success(f"Welcome back, {user['full_name']}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
    
    with tab2:
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            with col1:
                full_name = st.text_input("Full Name")
                username = st.text_input("Username")
            with col2:
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
            
            register_btn = st.form_submit_button("Register")
            
            if register_btn:
                if all([full_name, username, email, password]):
                    success, message = db.create_user(username, email, password, full_name)
                    if success:
                        st.success("Registration successful! Please login.")
                    else:
                        st.error(message)
                else:
                    st.error("Please fill all fields")

def show_main_app():
    """Show main application after authentication"""
    # Load AI components
    with st.spinner("Loading AI models..."):
        components = load_ai_components()
    
    if components[2] is None:
        st.error("Failed to load AI models. Please check the console.")
        return
    
    audio_processor, transcriber, model_predictor = components
    
    # Sidebar navigation and user info
    with st.sidebar:
        st.title(f"👋 Welcome, {st.session_state.user['full_name']}!")
        st.write(f"**Username:** {st.session_state.user['username']}")
        st.write(f"**Email:** {st.session_state.user['email']}")
        
        # User stats
        user_stats = db.get_user_stats(st.session_state.user['id'])
        if user_stats:
            st.metric("Total Analyses", user_stats['total_analyses'])
        
        st.markdown("---")
        
        # Navigation
        menu_options = ["Dashboard", "Analysis History", "Alerts", "Account", "Email Settings"]
        choice = st.selectbox("Navigation", menu_options)
        
        st.markdown("---")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.session_state.current_analysis_result = None
            st.session_state.analysis_complete = False
            st.session_state.email_sent = False
            st.rerun()
    
    # Main content area
    if choice == "Dashboard":
        show_dashboard(audio_processor, transcriber, model_predictor)
    elif choice == "Analysis History":
        show_analysis_history()
    elif choice == "Alerts":
        show_alerts()
    elif choice == "Account":
        show_account_info()
    elif choice == "Email Settings":
        show_email_settings()

def show_dashboard(audio_processor, transcriber, model_predictor):
    """Show enhanced dashboard with audio analysis and statistics"""
    st.title("🏥 Medical Urgency Analysis Dashboard")
    
    # Show success message if analysis was just completed
    if st.session_state.get('analysis_complete'):
        st.success("✅ Analysis completed and saved successfully!")
        st.session_state.analysis_complete = False
    
    # Show email notification status
    if st.session_state.get('email_sent'):
        st.success("📧 Alert email sent to doctor successfully!")
        st.session_state.email_sent = False
    
    # Two main columns: Analysis on left, Stats on right
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🎤 Quick Audio Analysis")
        show_quick_analysis_interface(audio_processor, transcriber, model_predictor)
        
        # Show current analysis result if exists
        if st.session_state.get('current_analysis_result'):
            st.markdown("---")
            st.subheader("📊 Latest Analysis Result")
            display_analysis_result(st.session_state.current_analysis_result)
    
    with col2:
        st.subheader("📈 Your Statistics")
        show_dashboard_stats()
        
        st.markdown("---")
        st.subheader("🚨 Recent Alerts")
        show_recent_alerts()

def show_quick_analysis_interface(audio_processor, transcriber, model_predictor):
    """Show quick analysis interface in dashboard"""
    tab1, tab2 = st.tabs(["Upload Audio", "Record Audio"])
    
    with tab1:
        show_upload_interface(audio_processor, transcriber, model_predictor)
    
    with tab2:
        show_recording_interface(audio_processor, transcriber, model_predictor)

def show_upload_interface(audio_processor, transcriber, model_predictor):
    """Show upload interface"""
    uploaded_file = st.file_uploader(
        "Choose an audio file", 
        type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'aac', 'webm'],
        help="Upload audio files up to 25MB"
    )
    
    if uploaded_file is not None:
        # Create temporary file for preview
        temp_file_path = None
        try:
            temp_file_path = audio_processor.create_temp_file(
                suffix=os.path.splitext(uploaded_file.name)[1]
            )
            with open(temp_file_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            
            # Display audio player
            st.audio(temp_file_path, format='audio/wav')
            
            # Show file info
            audio_info = get_audio_info(temp_file_path)
            if audio_info['valid']:
                st.info(f"📊 File Info: {audio_info['message']}")
                
        finally:
            if temp_file_path:
                safe_delete_file(temp_file_path)
        
        if st.button("Analyze Audio", type="primary", key="analyze_upload"):
            with st.spinner("🔍 Processing audio..."):
                result = process_uploaded_file(uploaded_file, transcriber, model_predictor, audio_processor)
                if result['success']:
                    st.session_state.current_analysis_result = result
                    st.session_state.analysis_complete = True
                    
                    # Send email if high urgency
                    if result['urgency_level'] == "High":
                        send_doctor_alert(result)
                    
                    st.rerun()
                else:
                    st.error(f"❌ Analysis failed: {result['error']}")


def show_recording_interface(audio_processor, transcriber, model_predictor):
    """Show recording interface using Streamlit's native audio input"""
    st.info("""
    **Recording Instructions:**
    - Click the microphone button below to record
    - Ensure you're in a quiet environment
    - Speak clearly and describe symptoms in detail
    - Recommended duration: 10-30 seconds
    - After recording, click 'Analyze Recording' to process
    """)
    
    # Use Streamlit's native audio input - this works on Streamlit Cloud
    audio_bytes = st.audio_input("Click the microphone to record audio", key="audio_recorder")
    
    if audio_bytes is not None:
        # Show the recorded audio
        st.audio(audio_bytes, format="audio/wav")
        st.success("✅ Recording completed! Click 'Analyze Recording' below.")
        
        if st.button("Analyze Recording", type="primary", key="analyze_record"):
            with st.spinner("🔍 Processing recording..."):
                try:
                    # Save temporary file directly from bytes
                    temp_path = audio_processor.create_temp_file(suffix='.wav')
                    with open(temp_path, 'wb') as f:
                        f.write(audio_bytes)
                    
                    # Process the recording
                    result = process_recorded_audio(
                        temp_path, 
                        transcriber, 
                        model_predictor
                    )
                    
                    if result['success']:
                        st.session_state.current_analysis_result = result
                        st.session_state.analysis_complete = True
                        
                        # Send email if high urgency
                        if result['urgency_level'] == "High":
                            send_doctor_alert(result)
                        
                        # Clean up
                        safe_delete_file(temp_path)
                        
                        st.rerun()
                    else:
                        st.error(f"❌ Analysis failed: {result['error']}")
                        
                except Exception as e:
                    st.error(f"❌ Processing error: {e}")
    else:
        st.info("🎤 Click the microphone button above to start recording")

def send_doctor_alert(analysis_result):
    """Send email alert to doctor for high urgency cases"""
    try:
        success, message = email_notifier.send_urgency_alert(st.session_state.user, analysis_result)
        if success:
            st.session_state.email_sent = True
            # Update alarm status in database
            db.update_alarm_status(analysis_result.get('record_id'), "Notified to Dr")
            st.success("🚨 High urgency detected! Alert email sent to doctor.")
        else:
            st.warning(f"⚠️ Analysis completed but email notification failed: {message}")
    except Exception as e:
        st.error(f"❌ Error sending email alert: {e}")

def process_uploaded_file(uploaded_file, transcriber, model_predictor, audio_processor):
    """Process uploaded file and save to database"""
    try:
        # Get file info
        temp_info_path = None
        original_sample_rate = 0
        file_size = len(uploaded_file.getvalue())
        duration = 0
        
        try:
            temp_info_path = audio_processor.create_temp_file(
                suffix=os.path.splitext(uploaded_file.name)[1]
            )
            with open(temp_info_path, 'wb') as f:
                f.write(uploaded_file.getvalue())
            audio_info = get_audio_info(temp_info_path)
            if audio_info['valid']:
                original_sample_rate = audio_info.get('sample_rate', 0)
                duration = audio_info.get('duration', 0)
        finally:
            if temp_info_path:
                safe_delete_file(temp_info_path)
        
        # Transcribe
        transcribed_text = transcriber.transcribe_uploaded_file(uploaded_file)
        
        # Analyze with confidence score
        urgency_level, patient_status, alarm_status, confidence_score = model_predictor.predict_urgency_with_confidence(transcribed_text)
        
        # Save to database
        record_id = db.save_audio_record(
            st.session_state.user['id'],
            uploaded_file.name,
            original_sample_rate,
            file_size,
            duration,
            transcribed_text,
            urgency_level,
            patient_status,
            alarm_status,
            confidence_score
        )
        
        # Create alert if high urgency
        if urgency_level == "High":
            db.create_alert(
                st.session_state.user['id'],
                "High Urgency Case",
                f"High urgency detected in analysis of {uploaded_file.name}. Patient status: {patient_status}",
                "High"
            )
        
        return {
            'success': True,
            'transcribed_text': transcribed_text,
            'urgency_level': urgency_level,
            'patient_status': patient_status,
            'alarm_status': alarm_status,
            'confidence_score': confidence_score,
            'filename': uploaded_file.name,
            'record_id': record_id,
            'duration': duration,
            'sample_rate': original_sample_rate,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def process_recorded_audio(audio_path, transcriber, model_predictor):
    """Process recorded audio and save to database"""
    try:
        # Get file info
        audio_info = get_audio_info(audio_path)
        duration = audio_info.get('duration', 0) if audio_info['valid'] else 0
        file_size = os.path.getsize(audio_path)
        
        # Transcribe
        transcribed_text = transcriber.transcribe_audio(audio_path)
        
        # Analyze with confidence score
        urgency_level, patient_status, alarm_status, confidence_score = model_predictor.predict_urgency_with_confidence(transcribed_text)
        
        # Save to database
        record_id = db.save_audio_record(
            st.session_state.user['id'],
            "recording.wav",
            16000,
            file_size,
            duration,
            transcribed_text,
            urgency_level,
            patient_status,
            alarm_status,
            confidence_score
        )
        
        # Create alert if high urgency
        if urgency_level == "High":
            db.create_alert(
                st.session_state.user['id'],
                "High Urgency Case",
                f"High urgency detected in recorded audio analysis. Patient status: {patient_status}",
                "High"
            )
        
        return {
            'success': True,
            'transcribed_text': transcribed_text,
            'urgency_level': urgency_level,
            'patient_status': patient_status,
            'alarm_status': alarm_status,
            'confidence_score': confidence_score,
            'filename': "recording.wav",
            'record_id': record_id,
            'duration': duration,
            'sample_rate': 16000,
            'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def display_analysis_result(result):
    """Display analysis results - simple version without nested columns"""
    if result['success']:
        # Confidence score
        confidence = result.get('confidence_score', 0.5)
        st.metric("Confidence Score", f"{confidence:.2f}")
        
        # Transcribed Text
        st.subheader("📝 Transcribed Text")
        st.write(result['transcribed_text'])
        
        # File Information
        st.subheader("📊 File Information")
        st.write(f"**Duration:** {result.get('duration', 0):.1f}s")
        st.write(f"**Sample Rate:** {result.get('sample_rate', 0)}Hz") 
        st.write(f"**File:** {result['filename']}")
        
        # Medical Analysis
        st.subheader("🔍 Medical Analysis")
        
        # Urgency level
        urgency = result['urgency_level']
        if urgency == "High":
            st.error(f"🚨 **Urgency Level:** {urgency}")
        elif urgency == "Medium":
            st.warning(f"⚠️ **Urgency Level:** {urgency}")
        else:
            st.success(f"✅ **Urgency Level:** {urgency}")
        
        st.info(f"**Patient Status:** {result['patient_status']}")
        st.write(f"**Alarm Status:** {result['alarm_status']}")
        
        # Email notification status
        if urgency == "High" and result.get('alarm_status') == "Notified to Dr":
            st.success("📧 Alert email sent to doctor")
        
        # Record ID
        if result.get('record_id'):
            st.success(f"✅ Analysis saved (Record ID: {result['record_id']})")

def show_dashboard_stats():
    """Show dashboard statistics"""
    user_stats = db.get_user_stats(st.session_state.user['id'])
    recent_records = db.get_user_audio_records(st.session_state.user['id'], limit=5)
    
    if not user_stats or user_stats['total_analyses'] == 0:
        st.info("🎯 Get started by analyzing your first audio file!")
        return
    
    # Key metrics
    st.metric("Total Analyses", user_stats['total_analyses'])
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("High Urgency", user_stats['high_urgency_count'], delta_color="inverse")
    with col2:
        st.metric("Medium Urgency", user_stats['medium_urgency_count'])
    with col3:
        st.metric("Low Urgency", user_stats['low_urgency_count'])
    
    # Urgency distribution pie chart
    if user_stats['total_analyses'] > 0:
        fig_pie = go.Figure(data=[go.Pie(
            labels=['High', 'Medium', 'Low'],
            values=[user_stats['high_urgency_count'], 
                   user_stats['medium_urgency_count'], 
                   user_stats['low_urgency_count']],
            hole=.4,
            marker_colors=['#FF4B4B', '#FFA500', '#00D4AA']
        )])
        fig_pie.update_layout(
            title_text="Urgency Distribution",
            height=200,
            margin=dict(t=30, b=0, l=0, r=0)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

def show_recent_alerts():
    """Show recent alerts in dashboard"""
    user_alerts = db.get_user_alerts(st.session_state.user['id'], limit=3)
    
    if not user_alerts:
        st.info("No recent alerts")
        return
    
    for alert in user_alerts:
        alert_color = "🔴" if alert['urgency'] == "High" else "🟡"
        st.write(f"{alert_color} **{alert['type']}**")
        st.write(f"_{alert['created_at'][:16]}_")
        st.write(alert['message'][:50] + "..." if len(alert['message']) > 50 else alert['message'])
        st.markdown("---")

def show_analysis_history():
    """Show user's analysis history with detailed records"""
    st.title("📋 Analysis History")
    
    # Force refresh of data
    records = db.get_user_audio_records(st.session_state.user['id'])
    
    if not records:
        st.info("No analysis history found. Start by analyzing some audio file in the Dashboard!")
        # Show quick analysis option
        if st.button("Go to Dashboard for Analysis"):
            st.session_state.current_analysis_result = None
            st.rerun()
        return
    
    # Summary statistics
    total = len(records)
    high_count = sum(1 for r in records if r['urgency_type'] == 'High')
    medium_count = sum(1 for r in records if r['urgency_type'] == 'Medium')
    low_count = sum(1 for r in records if r['urgency_type'] == 'Low')
    
    st.subheader("📊 Summary Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Analyses", total)
    col2.metric("High Urgency", high_count)
    col3.metric("Medium Urgency", medium_count)
    col4.metric("Low Urgency", low_count)
    
    st.markdown("---")
    st.subheader("📝 Detailed Records")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        urgency_filter = st.selectbox("Filter by Urgency", ["All", "High", "Medium", "Low"])
    with col2:
        sort_by = st.selectbox("Sort by", ["Newest First", "Oldest First", "Confidence Score"])
    with col3:
        items_per_page = st.selectbox("Items per page", [10, 20, 50])
    
    # Apply filters
    filtered_records = records
    if urgency_filter != "All":
        filtered_records = [r for r in filtered_records if r['urgency_type'] == urgency_filter]
    
    # Apply sorting
    if sort_by == "Newest First":
        filtered_records.sort(key=lambda x: x['created_at'], reverse=True)
    elif sort_by == "Oldest First":
        filtered_records.sort(key=lambda x: x['created_at'])
    elif sort_by == "Confidence Score":
        filtered_records.sort(key=lambda x: x['confidence_score'], reverse=True)
    
    # Pagination
    total_pages = max(1, (len(filtered_records) + items_per_page - 1) // items_per_page)
    page_number = st.number_input("Page", min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page_number - 1) * items_per_page
    end_idx = min(start_idx + items_per_page, len(filtered_records))
    page_records = filtered_records[start_idx:end_idx]
    
    # Display records
    for record in page_records:
        urgency_color = {
            "High": "🔴",
            "Medium": "🟡",
            "Low": "🟢"
        }.get(record['urgency_type'], "⚪")
        
        with st.expander(f"{urgency_color} {record['filename']} - {record['created_at'][:16]}", expanded=False):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.write("**Transcribed Text:**")
                st.write(record['transcribed_text'])
                
                st.write("**File Information:**")
                col1a, col1b, col1c = st.columns(3)
                with col1a:
                    st.write(f"**Duration:** {record['duration']:.1f}s")
                with col1b:
                    st.write(f"**Sample Rate:** {record['sample_rate']}Hz")
                with col1c:
                    st.write(f"**File:** {record['filename']}")
            
            with col2:
                st.write("**Analysis Results:**")
                
                # Urgency badge
                urgency_style = {
                    "High": "background-color: #ff4b4b; color: white; padding: 5px; border-radius: 5px;",
                    "Medium": "background-color: #ffa500; color: white; padding: 5px; border-radius: 5px;",
                    "Low": "background-color: #00d4aa; color: white; padding: 5px; border-radius: 5px;"
                }
                st.markdown(f"<div style='{urgency_style.get(record['urgency_type'], '')}'>"
                           f"<strong>Urgency:</strong> {record['urgency_type']}</div>", 
                           unsafe_allow_html=True)
                
                st.write(f"**Patient Status:** {record['patient_status']}")
                st.write(f"**Alarm Status:** {record['alarm_status']}")
                
                # Confidence score with color
                confidence = record['confidence_score']
                confidence_color = "#ff4b4b" if confidence < 0.6 else "#ffa500" if confidence < 0.8 else "#00d4aa"
                st.markdown(f"**Confidence Score:** <span style='color: {confidence_color};'>{confidence:.2f}</span>", 
                           unsafe_allow_html=True)
    
    # Pagination info
    st.caption(f"Showing {start_idx + 1}-{end_idx} of {len(filtered_records)} records (Page {page_number} of {total_pages})")

def show_alerts():
    """Show user alerts"""
    st.title("🚨 Alerts & Notifications")
    
    alerts = db.get_user_alerts(st.session_state.user['id'])
    
    if not alerts:
        st.info("No alerts at this time. Alerts are generated for high urgency cases.")
        return
    
    unread_count = sum(1 for alert in alerts if not alert['is_read'])
    if unread_count > 0:
        st.warning(f"You have {unread_count} unread alert(s)")
    
    # Alert filters
    col1, col2 = st.columns(2)
    with col1:
        urgency_filter = st.selectbox("Filter by Urgency", ["All", "High", "Medium", "Low"])
    with col2:
        read_filter = st.selectbox("Filter by Status", ["All", "Unread", "Read"])
    
    # Apply filters
    filtered_alerts = alerts
    if urgency_filter != "All":
        filtered_alerts = [a for a in filtered_alerts if a['urgency'] == urgency_filter]
    if read_filter == "Unread":
        filtered_alerts = [a for a in filtered_alerts if not a['is_read']]
    elif read_filter == "Read":
        filtered_alerts = [a for a in filtered_alerts if a['is_read']]
    
    for alert in filtered_alerts:
        alert_color = "🔴" if alert['urgency'] == "High" else "🟡"
        background_color = "#FFE6E6" if alert['urgency'] == "High" else "#FFF4E6"
        border_color = "#FF4B4B" if alert['urgency'] == "High" else "#FFA500"
        
        with st.container():
            st.markdown(
                f"""
                <div style='background-color: {background_color}; padding: 15px; border-radius: 10px; margin: 10px 0; border-left: 5px solid {border_color};'>
                    <h4>{alert_color} {alert['type']}</h4>
                    <p style='margin: 10px 0;'>{alert['message']}</p>
                    <small><i>🕒 {alert['created_at'][:16]}</i></small>
                    {'<br><strong>📬 UNREAD</strong>' if not alert['is_read'] else '<br><strong>✅ READ</strong>'}
                </div>
                """, 
                unsafe_allow_html=True
            )

def show_account_info():
    """Show user account information"""
    st.title("👤 Account Information")
    
    user = st.session_state.user
    user_stats = db.get_user_stats(user['id'])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Personal Information")
        st.write(f"**Full Name:** {user['full_name']}")
        st.write(f"**Username:** {user['username']}")
        st.write(f"**Email:** {user['email']}")
    
    with col2:
        st.subheader("Usage Statistics")
        if user_stats:
            st.write(f"**Total Analyses:** {user_stats['total_analyses']}")
            st.write(f"**High Urgency Cases:** {user_stats['high_urgency_count']}")
            st.write(f"**Medium Urgency Cases:** {user_stats['medium_urgency_count']}")
            st.write(f"**Low Urgency Cases:** {user_stats['low_urgency_count']}")
            st.write(f"**Last Analysis:** {user_stats['last_analysis_date'] or 'Never'}")
        else:
            st.write("No analysis data available")
    
    st.markdown("---")
    st.subheader("Account Actions")
    
    if st.button("Delete All My Data", type="secondary"):
        st.warning("This will permanently delete all your analysis history and cannot be undone!")
        
        if st.button("Confirm Deletion", type="primary"):
            st.error("Data deletion feature not implemented in demo")
#
def show_email_settings():
    """Show email configuration settings"""
    st.title("📧 Email Notification Settings")
    
    st.info("""
    **Email Configuration:**
    - System will automatically send alerts to the doctor for high urgency cases
    - Notifications include detailed analysis results and patient information
    - Emails are sent securely using Gmail SMTP
    """)
    
    # Display current email configuration
    st.subheader("Current Configuration")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Sender Email:**")
        st.code(EMAIL_CONFIG['sender_email'])
        
        st.write("**SMTP Server:**")
        st.code(f"{EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}")
    
    with col2:
        st.write("**Doctor's Email:**")
        st.code(EMAIL_CONFIG['doctor_email'])
        
        st.write("**Status:**")
        st.success("✅ Email system is active")
    
    # Test email functionality
    st.markdown("---")
    st.subheader("Test Email System")
    
    if st.button("Send Test Email", type="primary"):
        with st.spinner("Sending test email..."):
            try:
                # Create test analysis result
                test_result = {
                    'urgency_level': 'High',
                    'patient_status': 'Test Patient - Critical Condition',
                    'confidence_score': 0.95,
                    'transcribed_text': 'This is a test message to verify email functionality.',
                    'created_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                success, message = email_notifier.send_urgency_alert(st.session_state.user, test_result)
                if success:
                    st.success("✅ Test email sent successfully to doctor!")
                else:
                    st.error(f"❌ Failed to send test email: {message}")
            except Exception as e:
                st.error(f"❌ Error sending test email: {e}")

if __name__ == "__main__":
    main()
