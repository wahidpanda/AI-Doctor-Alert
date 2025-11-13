import smtplib
import sqlite3
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

class AlertSystem:
    def __init__(self, db_path='app.db'):
        self.db_path = db_path
        self.init_alert_settings()
    
    def init_alert_settings(self):
        """Initialize alert settings table"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            CREATE TABLE IF NOT EXISTS alert_settings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                doctor_email TEXT,
                enable_email_alerts BOOLEAN DEFAULT 1,
                enable_sms_alerts BOOLEAN DEFAULT 0,
                high_urgency_alert BOOLEAN DEFAULT 1,
                medium_urgency_alert BOOLEAN DEFAULT 0,
                low_urgency_alert BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_smtp_config(self):
        """Get SMTP configuration from environment variables"""
        return {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'sender_email': os.getenv('SENDER_EMAIL', ''),
            'sender_password': os.getenv('SENDER_PASSWORD', ''),
            'use_tls': os.getenv('USE_TLS', 'True').lower() == 'true'
        }
    
    def save_alert_settings(self, user_id, doctor_email=None, enable_email=True, 
                           enable_sms=False, high_urgency=True, medium_urgency=False, low_urgency=False):
        """Save user alert settings"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # Check if settings exist
            c.execute('SELECT id FROM alert_settings WHERE user_id = ?', (user_id,))
            existing = c.fetchone()
            
            if existing:
                # Update existing settings
                c.execute('''
                    UPDATE alert_settings 
                    SET doctor_email = ?, enable_email_alerts = ?, enable_sms_alerts = ?,
                        high_urgency_alert = ?, medium_urgency_alert = ?, low_urgency_alert = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                ''', (doctor_email, enable_email, enable_sms, high_urgency, medium_urgency, low_urgency, user_id))
            else:
                # Insert new settings
                c.execute('''
                    INSERT INTO alert_settings 
                    (user_id, doctor_email, enable_email_alerts, enable_sms_alerts,
                     high_urgency_alert, medium_urgency_alert, low_urgency_alert)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (user_id, doctor_email, enable_email, enable_sms, high_urgency, medium_urgency, low_urgency))
            
            conn.commit()
            return True, "Alert settings saved successfully"
            
        except Exception as e:
            return False, f"Error saving alert settings: {str(e)}"
        finally:
            conn.close()
    
    def get_alert_settings(self, user_id):
        """Get user alert settings"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT doctor_email, enable_email_alerts, enable_sms_alerts,
                   high_urgency_alert, medium_urgency_alert, low_urgency_alert
            FROM alert_settings 
            WHERE user_id = ?
        ''', (user_id,))
        
        settings = c.fetchone()
        conn.close()
        
        if settings:
            return {
                'doctor_email': settings[0],
                'enable_email': bool(settings[1]),
                'enable_sms': bool(settings[2]),
                'high_urgency': bool(settings[3]),
                'medium_urgency': bool(settings[4]),
                'low_urgency': bool(settings[5])
            }
        return None
    
    def send_email_alert(self, doctor_email, patient_name, urgency_level, transcribed_text, confidence_score):
        """Send email alert to doctor"""
        try:
            smtp_config = self.get_smtp_config()
            
            if not smtp_config['sender_email'] or not smtp_config['sender_password']:
                logger.warning("SMTP credentials not configured")
                return False, "Email configuration missing"
            
            # Create message
            subject = f"🚨 Medical Urgency Alert - {urgency_level} Priority"
            
            body = f"""
            MEDICAL URGENCY ALERT
            
            Priority Level: {urgency_level}
            Patient: {patient_name}
            Confidence Score: {confidence_score:.2f}
            Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            TRANSCRIBED SYMPTOMS:
            {transcribed_text}
            
            ACTION REQUIRED:
            - Please review this case immediately
            - Contact patient if necessary
            - Update patient records
            
            This is an automated alert from Medical Urgency Analysis System.
            """
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config['sender_email']
            msg['To'] = doctor_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                if smtp_config['use_tls']:
                    server.starttls()
                server.login(smtp_config['sender_email'], smtp_config['sender_password'])
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {doctor_email}")
            return True, "Email alert sent successfully"
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False, f"Failed to send email: {str(e)}"
    
    def trigger_alert(self, user_id, urgency_level, transcribed_text, confidence_score, patient_name="Patient"):
        """Trigger alerts based on urgency level and user settings"""
        settings = self.get_alert_settings(user_id)
        
        if not settings:
            logger.info("No alert settings configured for user")
            return False, "No alert settings configured"
        
        # Check if alert should be sent for this urgency level
        if urgency_level == "High" and not settings['high_urgency']:
            return False, "High urgency alerts disabled"
        elif urgency_level == "Medium" and not settings['medium_urgency']:
            return False, "Medium urgency alerts disabled"
        elif urgency_level == "Low" and not settings['low_urgency']:
            return False, "Low urgency alerts disabled"
        
        results = []
        
        # Send email alert
        if settings['enable_email'] and settings['doctor_email']:
            success, message = self.send_email_alert(
                settings['doctor_email'],
                patient_name,
                urgency_level,
                transcribed_text,
                confidence_score
            )
            results.append(f"Email: {'✅' if success else '❌'} {message}")
        
        # SMS alerts (placeholder for future implementation)
        if settings['enable_sms']:
            results.append("SMS: ⚠️ SMS alerts not yet implemented")
        
        return True, " | ".join(results)
    
    def test_email_configuration(self, test_email):
        """Test email configuration"""
        try:
            smtp_config = self.get_smtp_config()
            
            if not smtp_config['sender_email'] or not smtp_config['sender_password']:
                return False, "SMTP credentials not configured"
            
            # Create test message
            subject = "Medical Alert System - Test Email"
            body = "This is a test email from the Medical Urgency Analysis System. Your email configuration is working correctly."
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config['sender_email']
            msg['To'] = test_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Send test email
            with smtplib.SMTP(smtp_config['smtp_server'], smtp_config['smtp_port']) as server:
                if smtp_config['use_tls']:
                    server.starttls()
                server.login(smtp_config['sender_email'], smtp_config['sender_password'])
                server.send_message(msg)
            
            return True, "Test email sent successfully"
            
        except Exception as e:
            return False, f"Test failed: {str(e)}"