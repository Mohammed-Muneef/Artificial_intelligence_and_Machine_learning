import cv2
import time
from ultralytics import YOLO
from src import config, database, tracker

def main():
    # 1. Initialize Database
    print("Initializing Database...")
    database.init_db()

    # 2. Load Model
    print(f"Loading YOLOv8 Model: {config.MODEL_PATH}")
    model = YOLO(config.MODEL_PATH)

    # 3. Open Video Source
    print(f"Opening Video Source: {config.VIDEO_SOURCE}")
    cap = cv2.VideoCapture(config.VIDEO_SOURCE)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {config.VIDEO_SOURCE}")
        return

    # 4. Initialize Tracker
    traffic_tracker = tracker.TrafficTracker()

    print("System Started. Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream.")
            break
        
        # Resize for consistent processing speed if needed (Optional)
        # frame = cv2.resize(frame, (1280, 720))

        # 5. Run Tracking
        # persist=True is crucial for ID tracking across frames
        results = model.track(frame, persist=True, verbose=False)
        
        # 6. Process & Count
        frame = traffic_tracker.process_frame(frame, results)

        # 7. Check Traffic Levels (Query DB for recent counts)
        recent_count = database.get_recent_counts(minutes=1)
        
        # Status Display
        status_text = f"Vehicles (Last 1m): {recent_count}"
        color = (0, 255, 0) # Green

        if recent_count >= config.TRAFFIC_ALERT_THRESHOLD:
            status_text += " | HEAVY TRAFFIC!"
            color = (0, 0, 255) # Red
            # Here we could trigger the audio alert logic again if desired
        
        cv2.rectangle(frame, (0, 0), (500, 60), (0, 0, 0), -1)
        cv2.putText(frame, status_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Traffic Analytics System", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
