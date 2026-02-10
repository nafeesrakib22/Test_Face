import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import psutil
import os

# 1. Setup MediaPipe Task
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

process = psutil.Process(os.getpid())
cap = cv2.VideoCapture(2)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Standardize to Edge Resolution
    frame = cv2.resize(frame, (640, 480))
    # Convert for MediaPipe (Must be mp.Image)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # 2. Benchmark Inference
    start_time = time.perf_counter()
    detection_result = detector.detect(mp_image)
    inf_time = (time.perf_counter() - start_time) * 1000

    # 3. Benchmark Memory
    mem_usage = process.memory_info().rss / (1024 * 1024)

    # Visualization
    if detection_result.detections:
        for detection in detection_result.detections:
            bbox = detection.bounding_box
            cv2.rectangle(frame, (bbox.origin_x, bbox.origin_y), 
                          (bbox.origin_x + bbox.width, bbox.origin_y + bbox.height), (0, 255, 0), 2)

    msg = f"Inf: {inf_time:.1f}ms | RAM: {mem_usage:.1f}MB"
    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("MediaPipe Tasks Benchmark", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()