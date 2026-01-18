import cv2
from src import config, database

class TrafficTracker:
    def __init__(self):
        # Store the center points of previous frames objects to track movement direction
        # Format: {vehicle_id: (x, y)}
        self.verify_map = {} 
        self.counted_ids = set()

    def process_frame(self, frame, tracks):
        """
        Process trackers and count vehicles crossing the line.
        
        Args:
            frame: The video frame.
            tracks: Output from YOLOv8 model.track()
        
        Returns:
            frame: Annotated frame
            count: Total count of vehicles crossing the line in this session (or relevant metric)
        """
        h, w, _ = frame.shape
        line_y = int(h * config.LINE_POSITION)
        
        # Draw the counting line
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)
        cv2.putText(frame, "Detection Line", (10, line_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if tracks[0].boxes.id is not None:
            boxes = tracks[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = tracks[0].boxes.id.cpu().numpy().astype(int)
            clss = tracks[0].boxes.cls.cpu().numpy().astype(int)
            confs = tracks[0].boxes.conf.cpu().numpy().astype(float)

            for box, id, cls, conf in zip(boxes, ids, clss, confs):
                
                # Filter by class
                if int(cls) not in config.VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                
                vehicle_name = config.CLASS_NAMES.get(cls, "Vehicle")
                
                # Draw Box and ID
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID:{id} {vehicle_name}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)

                # Checking line crossing (Simple Directional Logic)
                # We check if the object crosses from ABOVE to BELOW the line
                # Ideally, we track history. For now, we check if it's within OFFSET range of line.
                
                if (line_y - config.LINE_OFFSET) < cy < (line_y + config.LINE_OFFSET):
                    if id not in self.counted_ids:
                        self.counted_ids.add(id)
                        # Log to DB
                        database.log_vehicle(vehicle_name, int(id), float(conf))
                        # Visually indicate count
                        cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1)

        return frame
