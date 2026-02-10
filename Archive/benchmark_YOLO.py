import cv2
import time
import psutil
import os
from ultralytics import YOLO
import config 
# Load model and force CPU for consistency
model = YOLO('yolov8n.pt').to('cpu')
process = psutil.Process(os.getpid()) # Get current process for memory tracking

cap = cv2.VideoCapture(config.CAMERA_INDEX)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # 1. Simulate Edge Resolution (Standard VGA)
    frame = cv2.resize(frame, (640, 480))

    # 2. Benchmarking Inference Time
    start_inference = time.perf_counter()
    results = model(frame, classes=[0], conf=0.5, verbose=False)
    inf_time = (time.perf_counter() - start_inference) * 1000 # Convert to ms

    # 3. Benchmarking Memory Usage (Resident Set Size in MB)
    mem_usage = process.memory_info().rss / (1024 * 1024)

    # Display Metrics on Screen
    cv2.putText(frame, f"Time: {inf_time:.1f}ms", (10, 30), 2, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"RAM: {mem_usage:.1f}MB", (10, 60), 2, 0.7, (255, 0, 0), 2)

    cv2.imshow("Benchmarking YOLOv8n", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()