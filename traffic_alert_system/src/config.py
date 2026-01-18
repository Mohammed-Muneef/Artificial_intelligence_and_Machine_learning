import os

# ==========================================
# FILE PATHS & SYSTEM SETTINGS
# ==========================================
# Video Source: '0' for webcam, or path to video file
VIDEO_SOURCE = '15300538-hd_1920_1080_60fps.mp4' 
MODEL_PATH = 'yolov8n.pt'
DB_PATH = 'traffic_data.db'

# ==========================================
# DETECTION SETTINGS
# ==========================================
CONFIDENCE_THRESHOLD = 0.5
# COCO IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
VEHICLE_CLASSES = [2, 3, 5, 7] 

# Class ID to Name mapping for display
CLASS_NAMES = {
    2: "Car",
    3: "Motorcycle",
    5: "Bus",
    7: "Truck"
}

# ==========================================
# TRACKING & COUNTING SETTINGS
# ==========================================
# Position of the counting line (0.0 to 1.0 relative to frame height)
# e.g., 0.6 means 60% down from the top
LINE_POSITION = 0.6 
# Offset to consider a valid cross (in pixels)
LINE_OFFSET = 30

# ==========================================
# ALERT SETTINGS
# ==========================================
TRAFFIC_ALERT_THRESHOLD = 10     # Vehicles crossing per minute to trigger alert
ALERT_COOLDOWN_SECONDS = 10 
