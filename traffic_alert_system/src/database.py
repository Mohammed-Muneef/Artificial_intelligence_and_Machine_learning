import sqlite3
import time
from src import config

def init_db():
    """Initialize the SQLite database and create tables if they don't exist."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    # Create table for traffic logs
    # We log: ID, Timestamp, Vehicle Type, Vehicle ID (tracker ID), Confidence
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL,
            vehicle_type TEXT,
            vehicle_id INTEGER,
            confidence REAL
        )
    ''')
    
    conn.commit()
    conn.close()

def log_vehicle(vehicle_type, vehicle_id, confidence):
    """Log a detected vehicle crossing the line."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    timestamp = time.time()
    
    cursor.execute('''
        INSERT INTO traffic_logs (timestamp, vehicle_type, vehicle_id, confidence)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, vehicle_type, vehicle_id, confidence))
    
    conn.commit()
    conn.close()

def get_recent_counts(minutes=1):
    """Get number of vehicles detected in the last N minutes."""
    conn = sqlite3.connect(config.DB_PATH)
    cursor = conn.cursor()
    
    time_threshold = time.time() - (minutes * 60)
    
    cursor.execute('''
        SELECT COUNT(*) FROM traffic_logs
        WHERE timestamp > ?
    ''', (time_threshold,))
    
    count = cursor.fetchone()[0]
    conn.close()
    return count
