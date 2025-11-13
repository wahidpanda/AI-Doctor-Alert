import sqlite3
import hashlib
import secrets
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path='app.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with all required tables"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Users table with salt
        c.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                full_name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Audio records table with user relationship
        c.execute('''
            CREATE TABLE IF NOT EXISTS audio_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_sample_rate INTEGER,
                file_size INTEGER,
                duration REAL,
                transcribed_text TEXT,
                urgency_type TEXT,
                patient_status TEXT,
                alarm_status TEXT,
                confidence_score REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # User alerts/notifications table
        c.execute('''
            CREATE TABLE IF NOT EXISTS user_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                alert_type TEXT NOT NULL,
                alert_message TEXT NOT NULL,
                urgency_level TEXT,
                is_read BOOLEAN DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Analysis statistics table
        c.execute('''
            CREATE TABLE IF NOT EXISTS analysis_stats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                total_analyses INTEGER DEFAULT 0,
                high_urgency_count INTEGER DEFAULT 0,
                medium_urgency_count INTEGER DEFAULT 0,
                low_urgency_count INTEGER DEFAULT 0,
                last_analysis_date TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    
    def hash_password(self, password, salt=None):
        """Hash password using SHA-256 with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        password_hash = hashlib.sha256((password + salt).encode()).hexdigest()
        return password_hash, salt
    
    def verify_password(self, password, password_hash, salt):
        """Verify password against hash using the stored salt"""
        test_hash, _ = self.hash_password(password, salt)
        return test_hash == password_hash
    
    def create_user(self, username, email, password, full_name):
        """Create new user"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            password_hash, salt = self.hash_password(password)
            
            c.execute('''
                INSERT INTO users (username, email, password_hash, salt, full_name)
                VALUES (?, ?, ?, ?, ?)
            ''', (username, email, password_hash, salt, full_name))
            
            user_id = c.lastrowid
            
            # Initialize user stats - FIXED: Ensure stats record is created
            c.execute('''
                INSERT INTO analysis_stats (user_id, total_analyses, high_urgency_count, medium_urgency_count, low_urgency_count) 
                VALUES (?, 0, 0, 0, 0)
            ''', (user_id,))
            
            conn.commit()
            logger.info(f"User created successfully: {username}")
            return True, "User created successfully"
            
        except sqlite3.IntegrityError as e:
            error_msg = str(e)
            if "username" in error_msg:
                return False, "Username already exists"
            elif "email" in error_msg:
                return False, "Email already exists"
            else:
                return False, "User creation failed"
        except Exception as e:
            logger.error(f"Error creating user: {e}")
            return False, f"Error creating user: {str(e)}"
        finally:
            conn.close()
    
    def authenticate_user(self, username, password):
        """Authenticate user"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT id, username, email, password_hash, salt, full_name 
            FROM users 
            WHERE username = ? AND is_active = 1
        ''', (username,))
        
        user = c.fetchone()
        conn.close()
        
        if user and self.verify_password(password, user[3], user[4]):
            # Update last login
            self.update_last_login(user[0])
            return {
                'id': user[0],
                'username': user[1],
                'email': user[2],
                'full_name': user[5]
            }
        return None
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute('''
            UPDATE users 
            SET last_login = CURRENT_TIMESTAMP 
            WHERE id = ?
        ''', (user_id,))
        conn.commit()
        conn.close()
    
    def save_audio_record(self, user_id, filename, original_sample_rate, file_size, 
                         duration, transcribed_text, urgency_type, patient_status, 
                         alarm_status, confidence_score):
        """Save audio analysis record"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            c.execute('''
                INSERT INTO audio_records 
                (user_id, filename, original_sample_rate, file_size, duration,
                 transcribed_text, urgency_type, patient_status, alarm_status, confidence_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, filename, original_sample_rate, file_size, duration,
                  transcribed_text, urgency_type, patient_status, alarm_status, confidence_score))
            
            record_id = c.lastrowid
            
            # Update user statistics
            self.update_user_stats(user_id, urgency_type)
            
            # Create alert if high urgency
            if urgency_type == "High":
                self.create_alert(
                    user_id, 
                    "High Urgency Alert", 
                    f"High urgency case detected in analysis: {filename}",
                    "High"
                )
            
            conn.commit()
            logger.info(f"Audio record saved: {filename}")
            return record_id
            
        except Exception as e:
            logger.error(f"Error saving audio record: {e}")
            return None
        finally:
            conn.close()
    
    def update_user_stats(self, user_id, urgency_type):
        """Update user analysis statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # First, ensure stats record exists
        c.execute('SELECT id FROM analysis_stats WHERE user_id = ?', (user_id,))
        if not c.fetchone():
            # Create stats record if it doesn't exist
            c.execute('''
                INSERT INTO analysis_stats (user_id, total_analyses, high_urgency_count, medium_urgency_count, low_urgency_count)
                VALUES (?, 0, 0, 0, 0)
            ''', (user_id,))
        
        # Increment total analyses
        c.execute('''
            UPDATE analysis_stats 
            SET total_analyses = total_analyses + 1,
                last_analysis_date = CURRENT_TIMESTAMP
            WHERE user_id = ?
        ''', (user_id,))
        
        # Increment specific urgency count
        if urgency_type == "High":
            c.execute('''
                UPDATE analysis_stats 
                SET high_urgency_count = high_urgency_count + 1 
                WHERE user_id = ?
            ''', (user_id,))
        elif urgency_type == "Medium":
            c.execute('''
                UPDATE analysis_stats 
                SET medium_urgency_count = medium_urgency_count + 1 
                WHERE user_id = ?
            ''', (user_id,))
        else:
            c.execute('''
                UPDATE analysis_stats 
                SET low_urgency_count = low_urgency_count + 1 
                WHERE user_id = ?
            ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def create_alert(self, user_id, alert_type, alert_message, urgency_level=None):
        """Create user alert"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO user_alerts (user_id, alert_type, alert_message, urgency_level)
            VALUES (?, ?, ?, ?)
        ''', (user_id, alert_type, alert_message, urgency_level))
        
        conn.commit()
        conn.close()
    
    def get_user_stats(self, user_id):
        """Get user analysis statistics"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT total_analyses, high_urgency_count, medium_urgency_count, 
                   low_urgency_count, last_analysis_date
            FROM analysis_stats 
            WHERE user_id = ?
        ''', (user_id,))
        
        stats = c.fetchone()
        conn.close()
        
        if stats:
            return {
                'total_analyses': stats[0],
                'high_urgency_count': stats[1],
                'medium_urgency_count': stats[2],
                'low_urgency_count': stats[3],
                'last_analysis_date': stats[4]
            }
        else:
            # Return default stats if no record exists
            return {
                'total_analyses': 0,
                'high_urgency_count': 0,
                'medium_urgency_count': 0,
                'low_urgency_count': 0,
                'last_analysis_date': None
            }
    
    def get_user_alerts(self, user_id, limit=10):
        """Get user alerts"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT alert_type, alert_message, urgency_level, is_read, created_at
            FROM user_alerts 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        alerts = c.fetchall()
        conn.close()
        
        return [{
            'type': alert[0],
            'message': alert[1],
            'urgency': alert[2],
            'is_read': bool(alert[3]),
            'created_at': alert[4]
        } for alert in alerts]
    
    def get_user_audio_records(self, user_id, limit=20):
        """Get user's audio analysis history"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            SELECT filename, original_sample_rate, duration, transcribed_text,
                   urgency_type, patient_status, alarm_status, confidence_score, created_at
            FROM audio_records 
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        ''', (user_id, limit))
        
        records = c.fetchall()
        conn.close()
        
        return [{
            'filename': record[0],
            'sample_rate': record[1],
            'duration': record[2],
            'transcribed_text': record[3],
            'urgency_type': record[4],
            'patient_status': record[5],
            'alarm_status': record[6],
            'confidence_score': record[7],
            'created_at': record[8]
        } for record in records]
    
    def mark_alert_as_read(self, alert_id):
        """Mark alert as read"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            UPDATE user_alerts 
            SET is_read = 1 
            WHERE id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()