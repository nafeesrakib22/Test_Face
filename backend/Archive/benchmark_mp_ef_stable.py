import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import time
import os
import psutil
import csv
import datetime
from backbones import get_model
from torchvision import transforms
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

# --- CONFIGURATION ---
DB_PATH = 'data/face_db'
THRESHOLD = 0.50
STABILITY_FRAMES = 10
CSV_FILE = "benchmark_xs_gamma_06.csv"

# --- MODEL SETUP ---
device = torch.device('cpu')
model_name = "edgeface_xs_gamma_06" # Explicitly requesting the Quantized architecture

print(f"Loading {model_name}...")
# Direct loading as requested
model = get_model(model_name)
model.load_state_dict(torch.load(f'checkpoints/{model_name}.pt', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# --- MEDIAPIPE SETUP ---
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# --- CSV LOGGING SETUP ---
f = open(CSV_FILE, 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Timestamp", "Mode", "Total_Latency_ms", "Recog_Time_ms", "RAM_MB", "Detected_User"])
print(f"Logging benchmark data to: {os.path.abspath(CSV_FILE)}")

# --- HELPER FUNCTIONS ---
def get_embedding(img_bgr):
    img_resized = cv2.resize(img_bgr, (112, 112))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return F.normalize(embedding)

def load_database(path):
    print(f"Loading Face Database from {path}...")
    db = {}
    if not os.path.exists(path):
        os.makedirs(path)
        return db
    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0]
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                db[name] = get_embedding(img)
    print(f"Database Ready: {len(db)} users loaded.")
    return db

# --- MAIN EXECUTION ---
process = psutil.Process(os.getpid())
known_faces = load_database(DB_PATH)
cap = cv2.VideoCapture(config.CAMERA_INDEX)
stability_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_start = time.perf_counter()
    
    frame = cv2.resize(frame, (640, 480))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    results = detector.detect(mp_image)
    
    # State Variables
    mode = "Idle"
    recog_time = 0.0
    detected_user = "None"
    msg, color = "Searching...", (255, 255, 255)
    
    if results.detections:
        num_faces = len(results.detections)
        
        if num_faces > 1:
            stability_counter = 0
            mode = "Warning"
            msg, color = f"Access Denied: {num_faces} People", (0, 0, 255)
        else:
            stability_counter += 1
            mode = "Stabilizing"
            
            det = results.detections[0]
            b = det.bounding_box
            x1, y1 = max(0, b.origin_x), max(0, b.origin_y)
            x2, y2 = min(640, b.origin_x + b.width), min(480, b.origin_y + b.height)
            
            if stability_counter < STABILITY_FRAMES:
                msg = f"Hold Still... {stability_counter}/{STABILITY_FRAMES}"
                color = (0, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                mode = "Recognizing"
                recog_start = time.perf_counter()
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    curr_emb = get_embedding(face_crop)
                    best_score = -1.0
                    
                    for name, db_emb in known_faces.items():
                        score = torch.mm(curr_emb, db_emb.t()).item()
                        if score > best_score:
                            best_score = score
                            detected_user = name
                    
                    if best_score > THRESHOLD:
                        msg = f"{detected_user} ({best_score:.2f})"
                        color = (0, 255, 0)
                    else:
                        detected_user = "Unknown"
                        msg = f"Unknown ({best_score:.2f})"
                        color = (0, 0, 255)

                recog_time = (time.perf_counter() - recog_start) * 1000
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        stability_counter = 0

    # Metrics
    total_latency = (time.perf_counter() - frame_start) * 1000
    memory_mb = process.memory_info().rss / (1024 * 1024)
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Log
    writer.writerow([current_time, mode, f"{total_latency:.2f}", f"{recog_time:.2f}", f"{memory_mb:.2f}", detected_user])

    # HUD
    perf_text = f"Lat: {total_latency:.1f}ms | RAM: {memory_mb:.1f}MB | Model: {model_name}"
    cv2.rectangle(frame, (5, 45), (500, 75), (0,0,0), -1) 
    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, perf_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow(f"Benchmark: {model_name}", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

f.close()
cap.release()
cv2.destroyAllWindows()
print(f"Benchmark finished. Data saved to {CSV_FILE}")