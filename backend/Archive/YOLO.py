from ultralytics import YOLO
import cv2

# Load the smallest YOLOv8 face model (Nano)
# Model size: ~6.5 MB
detector = YOLO('yolov8n-face.pt') 

def run_gatekeeper(frame):
    results = detector(frame, conf=0.5)[0]
    boxes = results.boxes
    
    if len(boxes) > 1:
        cv2.putText(frame, "ONLY ONE PERSON ALLOWED", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    elif len(boxes) == 1:
        # Proceed to recognition stage
        pass