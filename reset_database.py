import os
import sqlite3
from database import DatabaseManager

def reset_database_completely():
    """Completely reset the database with new schema"""
    db_path = 'medical_app.db'
    
    # Delete existing database file
    if os.path.exists(db_path):
        os.remove(db_path)
        print("🗑️  Old database deleted")
    
    # Create new database with updated schema
    db = DatabaseManager(db_path)
    
    # Test the schema by creating a test user and record
    try:
        # Create test user
        success, message = db.create_user("testuser", "test@example.com", "password", "Test User")
        if success:
            print("✅ Test user created successfully")
            
            # Get user ID
            user = db.authenticate_user("testuser", "password")
            if user:
                print("✅ User authentication working")
                
                # Create test audio record with all columns
                record_id = db.save_audio_record(
                    user_id=user['id'],
                    filename="test_audio.wav",
                    original_sample_rate=44100,
                    file_size=1024000,
                    duration=30.5,
                    transcribed_text="Test transcription text",
                    urgency_type="Medium",
                    patient_status="Test patient status",
                    alarm_status="Not Notified",
                    confidence_score=0.85
                )
                
                if record_id:
                    print("✅ Audio record creation working")
                    
                    # Test getting records
                    records = db.get_user_audio_records(user['id'])
                    if records:
                        print("✅ Record retrieval working")
                        print(f"📊 Found {len(records)} records")
                    else:
                        print("❌ Record retrieval failed")
                else:
                    print("❌ Audio record creation failed")
            else:
                print("❌ User authentication failed")
        else:
            print(f"❌ Test user creation failed: {message}")
            
    except Exception as e:
        print(f"❌ Database test failed: {e}")
    
    print("🎉 Database reset completed successfully!")

if __name__ == "__main__":
    print("🚀 Starting complete database reset...")
    reset_database_completely()