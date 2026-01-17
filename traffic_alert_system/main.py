import cv2
import pyttsx3
import time
import threading
from ultralytics import YOLO

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_SOURCE = '15300538-hd_1920_1080_60fps.mp4'  # 0 for webcam, or path to video file (e.g., 'traffic.mp4')
CONFIDENCE_THRESHOLD = 0.5
VEHICLE_CLASSES = [2, 3, 5, 7]  # COCO IDs: 2=Car, 3=Motorcycle, 5=Bus, 7=Truck
TRAFFIC_ALERT_THRESHOLD = 10     # Number of vehicles to trigger "Heavy Traffic"
ALERT_COOLDOWN_SECONDS = 5     # How often to speak the alert

# Initialize Text-to-Speech Engine
try:
    tts_engine = pyttsx3.init()
    # Optional: Set properties (rate, volume)
    tts_engine.setProperty('rate', 150)
except Exception as e:
    print(f"Warning: TTS Engine could not be initialized. Audio alerts will not work. Error: {e}")
    tts_engine = None

last_alert_time = 0

def speak_alert(message):
    """
    Function to run TTS in a separate thread so it doesn't block the video loop.
    """
    if tts_engine:
        def _speak():
            try:
                # Re-initializing engine inside thread sometimes helps with thread safety depending on OS
                # But typically sharing the engine or a lock is needed. 
                # pyttsx3 has known threading issues. A simple lock or queue is better, 
                # but for simplicity we'll try running the runAndWait inside the thread.
                # NOTE: On some volatile environments, pyttsx3 might need to be run in the main thread.
                # We will use a trusted pattern: queueing doesn't work out of the box without a loop.
                # We'll use a simple blocking call in a thread for now.
                tts_engine.say(message)
                tts_engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")
        
        t = threading.Thread(target=_speak)
        t.daemon = True
        t.start()

def main():
    global last_alert_time
    
    print("Loading YOLOv8 model...")
    # Using 'yolov8n.pt' (Nano) for speed. It will download automatically if not present.
    model = YOLO('yolov8n.pt') 

    print(f"Opening video source: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)

    if not cap.isOpened():
        print(f"Error: Could not open video source {VIDEO_SOURCE}")
        return

    print("System detection started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break

        # 1. Run YOLO Inference
        results = model(frame, stream=True, verbose=False)

        vehicle_count = 0
        
        # 2. Process Results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if cls_id in VEHICLE_CLASSES and conf > CONFIDENCE_THRESHOLD:
                    vehicle_count += 1
                    
                    # Draw Bounding Box
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label_name = model.names[cls_id]
                    
                    # Color for vehicle (Green)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label_name} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 3. Traffic Analysis & Alert Logic
        status_text = f"Vehicles: {vehicle_count}"
        color_status = (0, 255, 0) # Green

        if vehicle_count >= TRAFFIC_ALERT_THRESHOLD:
            status_text += " | HEAVY TRAFFIC DETECTED!"
            color_status = (0, 0, 255) # Red
            
            # Check cooldown
            current_time = time.time()
            if (current_time - last_alert_time) > ALERT_COOLDOWN_SECONDS:
                print(f"[ALERT] Heavy Traffic! Count: {vehicle_count}")
                speak_alert(f"Warning. Heavy traffic detected. {vehicle_count} vehicles observed.")
                last_alert_time = current_time

        # 4. Display Dashboard
        cv2.putText(frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_status, 2)

        cv2.imshow("Smart City Traffic Alert (YOLOv8 + TTS)", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if tts_engine:
        tts_engine.stop()

if __name__ == "__main__":
    main()
