import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EmailSender:
    def __init__(self):
        # Your email configuration
        self.sender_email = "pagoldr01@gmail.com"
        self.sender_password = "vcahdtpodoqbwalw"  # Google App Password
        self.doctor_email = "transformereroded88@gmail.com"
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
    
    def send_urgency_alert(self, patient_info, transcribed_text, urgency_level, confidence_score):
        """
        Send urgency alert email to doctor - AUTOMATIC DIRECT ACTION
        """
        try:
            # Create message with alert emoji in subject
            subject = f"🚨 URGENCY ALERT: {urgency_level} Priority Case - Immediate Attention Required!"
            
            # Create HTML email body
            body = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; }}
                    .alert {{ color: #ff0000; font-weight: bold; font-size: 18px; }}
                    .info {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
                    .urgent {{ background-color: #ffcccc; padding: 15px; border-radius: 5px; border-left: 5px solid red; }}
                </style>
            </head>
            <body>
                <div class="urgent">
                    <h2>🚨 MEDICAL URGENCY ALERT 🚨</h2>
                    <p class="alert">URGENCY LEVEL: {urgency_level.upper()}</p>
                </div>
                
                <div class="info">
                    <h3>Patient Case Details:</h3>
                    <p><strong>Timestamp:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                    <p><strong>User:</strong> {patient_info.get('full_name', 'N/A')}</p>
                    <p><strong>Username:</strong> {patient_info.get('username', 'N/A')}</p>
                    <p><strong>Email:</strong> {patient_info.get('email', 'N/A')}</p>
                    <p><strong>Confidence Score:</strong> {confidence_score:.2%}</p>
                </div>
                
                <div style="margin-top: 20px;">
                    <h3>📝 Transcribed Medical Description:</h3>
                    <div style="background-color: #fff; padding: 15px; border: 1px solid #ddd; border-radius: 5px;">
                        {transcribed_text}
                    </div>
                </div>
                
                <div style="margin-top: 20px; padding: 10px; background-color: #fffacd; border-radius: 5px;">
                    <h4>⚠️ Action Required:</h4>
                    <p>This case has been flagged as <strong>{urgency_level} urgency</strong>. Please review the patient's condition and take appropriate medical action.</p>
                    <p><strong>Immediate attention is recommended.</strong></p>
                </div>
                
                <hr>
                <p style="color: #666; font-size: 12px;">
                    This is an automated alert from the Medical Urgency Analysis System. 
                    Please do not reply to this email.
                </p>
            </body>
            </html>
            """
            
            # Create message container - FIXED IMPORT NAMES
            msg = MimeMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.sender_email
            msg['To'] = self.doctor_email
            
            # Attach HTML body
            html_part = MimeText(body, 'html')
            msg.attach(html_part)
            
            # Send email directly
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.sender_email, self.sender_password)
            server.send_message(msg)
            server.quit()
            
            print(f"✅ Urgency alert email sent successfully to {self.doctor_email}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to send email: {str(e)}")
            return False

# Global email sender instance
email_sender = EmailSender()



# import smtplib
# smtplibObj= smtp.SMTP('smtp.gmail.com',587)
# smtpObj.ehlo()
# smtpObj.starttls()
# smtpObj.login("pagoldr01@gmail.com",'vcahdtpodoqbwalw')
# smtpObj.sendemail('islamoahidul12@gmail.com','SUbject:.....')
# smtplibObj.quit()