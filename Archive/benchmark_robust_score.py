import cv2
import mediapipe as mp
import numpy as np
import time
import os
import psutil
import csv
import datetime
import onnxruntime as ort
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

# --- CONFIGURATION ---
ONNX_MODEL_PATH = "edgeface_xs_gamma_06.onnx"
DB_PATH = 'data/face_db'
CSV_FILE = "benchmark_robust_score.csv"
THRESHOLD = 0.60
STABILITY_FRAMES = 10

# 1. Initialize ONNX Runtime
print(f"Loading ONNX Model: {ONNX_MODEL_PATH}")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

# 2. Initialize MediaPipe
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# --- CSV SETUP ---
f = open(CSV_FILE, 'w', newline='')
writer = csv.writer(f)
# [UPDATED] Added "Confidence" to header
writer.writerow(["Timestamp", "Mode", "Total_Latency_ms", "Recog_Time_ms", "RAM_MB", "Detected_User", "Confidence"])
print(f"Logging benchmark data to: {os.path.abspath(CSV_FILE)}")

# --- HELPER FUNCTIONS ---
def get_embedding_onnx(img_bgr):
    img_resized = cv2.resize(img_bgr, (112, 112))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    embeddings = ort_session.run(None, {input_name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)

def load_database(path):
    print(f"Loading Face Database from {path}...")
    db = {}
    if not os.path.exists(path): return db
    
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        name = os.path.splitext(filename)[0]
        
        if filename.endswith('.npy'):
            try:
                db[name] = np.load(file_path)
                print(f"Loaded Robust ID (Centroid): {name}")
            except: pass
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if name not in db:
                img = cv2.imread(file_path)
                if img is not None:
                    db[name] = get_embedding_onnx(img)
                    print(f"Loaded Standard ID (Image): {name}")
    return db

# --- MAIN LOOP ---
known_faces = load_database(DB_PATH)
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
process = psutil.Process(os.getpid())
stability_counter = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_start = time.perf_counter()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    # Init variables for this frame
    mode = "Idle"
    recog_time = 0.0
    detected_user = "None"
    current_confidence = 0.0 # [NEW] Default 0.0 if no face
    msg, color = "Searching...", (255, 255, 255)
    
    if results.detections:
        if len(results.detections) > 1:
            stability_counter = 0
            mode = "Warning"
            msg, color = "Warning: Multiple Faces", (0, 0, 255)
        else:
            stability_counter += 1
            mode = "Stabilizing"
            
            det = results.detections[0]
            b = det.bounding_box
            x1, y1 = max(0, b.origin_x), max(0, b.origin_y)
            x2, y2 = min(640, b.origin_x + b.width), min(480, b.origin_y + b.height)
            
            if stability_counter < STABILITY_FRAMES:
                msg, color = f"Stabilizing {stability_counter}/{STABILITY_FRAMES}", (0, 255, 255)
            else:
                mode = "Recognizing"
                recog_start = time.perf_counter()
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    curr_emb = get_embedding_onnx(face_crop)
                    best_score = -1.0
                    
                    for name, db_emb in known_faces.items():
                        score = np.dot(curr_emb, db_emb.T).item()
                        if score > best_score:
                            best_score = score
                            detected_user = name
                    
                    # [UPDATED] Capture the best score for CSV
                    current_confidence = best_score
                    
                    if best_score > THRESHOLD:
                        msg, color = f"{detected_user} ({best_score:.2f})", (0, 255, 0)
                    else:
                        detected_user = "Unknown"
                        msg, color = f"Unknown ({best_score:.2f})", (0, 0, 255)
                
                recog_time = (time.perf_counter() - recog_start) * 1000
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        stability_counter = 0

    # Metrics
    total_latency = (time.perf_counter() - frame_start) * 1000
    memory_mb = process.memory_info().rss / (1024 * 1024)
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]

    # Log with Confidence column
    writer.writerow([current_time, mode, f"{total_latency:.2f}", f"{recog_time:.2f}", f"{memory_mb:.2f}", detected_user, f"{current_confidence:.4f}"])

    # HUD
    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"RAM: {memory_mb:.1f}MB | Lat: {total_latency:.1f}ms", (10, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Final Robust System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

f.close()
cap.release()
cv2.destroyAllWindows()
print(f"Benchmark finished. Data with scores saved to {CSV_FILE}")