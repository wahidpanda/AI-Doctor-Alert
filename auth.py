import os
import sqlite3
from database import DatabaseManager

def fix_authentication():
    print("🛠️ Fixing Authentication System")
    print("=" * 50)
    
    # Step 1: Completely reset database
    print("1. Resetting database...")
    if os.path.exists('app.db'):
        os.remove('app.db')
        print("   ✅ Old database deleted")
    
    # Step 2: Create new database
    db = DatabaseManager()
    print("   ✅ New database created")
    
    # Step 3: Create test user with exact credentials you're using
    print("2. Creating test user with your credentials...")
    success, message = db.create_user("ll321", "l@gmail.com", "ll321", "Test User")
    
    if success:
        print("   ✅ User created: ll321 / ll321")
        
        # Test that user stats were created
        user = db.authenticate_user("ll321", "ll321")
        if user:
            user_stats = db.get_user_stats(user['id'])
            print(f"   ✅ User stats initialized: {user_stats}")
    else:
        print(f"   ❌ Failed to create user: {message}")
        return
    
    # Step 4: Test authentication
    print("3. Testing authentication...")
    user = db.authenticate_user("ll321", "ll321")
    
    if user:
        print("   ✅ Authentication SUCCESSFUL!")
        print(f"   User data: {user}")
    else:
        print("   ❌ Authentication FAILED!")
        
        # Debug why it failed
        conn = sqlite3.connect('app.db')
        c = conn.cursor()
        c.execute("SELECT username, password_hash, salt FROM users WHERE username = 'll321'")
        db_user = c.fetchone()
        conn.close()
        
        if db_user:
            print(f"   Database shows - Username: '{db_user[0]}', Password Hash: '{db_user[1]}', Salt: '{db_user[2]}'")
            print(f"   You entered - Username: 'll321', Password: 'll321'")
    
    print("\n🎉 Fix completed!")

if __name__ == "__main__":
    fix_authentication()