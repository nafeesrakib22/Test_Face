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
from collections import deque, Counter
import config

# --- ENHANCED CONFIGURATION ---
ONNX_MODEL_PATH = "edgeface_xs_gamma_06.onnx"
DB_PATH = 'data/face_db'
CSV_FILE = "benchmark_masked_improved.csv"

# [OPTIMIZED VALUES FOR HIGH SEPARATION]
THRESHOLD = 0.68              # Primary gate for the model
MARGIN_THRESHOLD = 0.10       # Require winner to beat runner-up by 0.10
STABILITY_FRAMES = 5          # Initial detection stability
BUFFER_SIZE = 12              # Increased temporal memory for smoothing

# --- CLASS: IDENTITY STABILIZER (80% Majority Logic) ---
class IdentityStabilizer:
    def __init__(self, maxlen=12):
        self.history = deque(maxlen=maxlen)
        self.current_display_name = "Unknown"
        
    def update(self, name):
        self.history.append(name)
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        
        # Require 80% of the buffer (approx 10/12 frames) to agree before switching
        if count >= int(self.history.maxlen * 0.8):
            self.current_display_name = most_common
            
        return self.current_display_name

# 1. Initialize Engines
print(f"Loading ONNX Model: {ONNX_MODEL_PATH}")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 2. Init Stabilizer
stabilizer = IdentityStabilizer(maxlen=BUFFER_SIZE)

# 3. CSV Setup
f = open(CSV_FILE, 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Timestamp", "Mode", "Raw_User", "Display_User", "Confidence", 
                 "Second_Best", "Margin", "Decision_Factor", "RAM_MB"])

# --- HELPER FUNCTIONS ---
def get_embedding_onnx(img_bgr):
    """Extract embedding with elliptical masking to remove background"""
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    # APPLY ELLIPTICAL MASK
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (45, 58), 0, 0, 360, 255, -1)
    img_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
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
            db[name] = np.load(file_path)
            print(f"✓ Loaded Masked ID: {name}")
    return db

# --- MAIN LOOP ---
known_faces = load_database(DB_PATH)
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
process = psutil.Process(os.getpid())
stability_counter = 0

print(f"\n{'='*70}")
print(f"FINAL STABILIZED SYSTEM STARTING...")
print(f"Threshold: {THRESHOLD} | Margin: {MARGIN_THRESHOLD} | Buffer: {BUFFER_SIZE}")
print(f"{'='*70}\n")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame_start = time.perf_counter()
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    mode = "Idle"
    raw_user = "None"
    display_user = "None"
    confidence = 0.0
    second_best_score = 0.0
    margin = 0.0
    decision_factor = "N/A"
    msg, color = "Searching...", (255, 255, 255)
    
    if results.detections:
        stability_counter += 1
        mode = "Stabilizing"
        
        det = results.detections[0]
        b = det.bounding_box
        x1, y1 = max(0, b.origin_x), max(0, b.origin_y)
        x2, y2 = min(640, b.origin_x + b.width), min(480, b.origin_y + b.height)
        
        if stability_counter < STABILITY_FRAMES:
            msg, color = f"Stabilizing...", (0, 255, 255)
        else:
            mode = "Recognizing"
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                curr_emb = get_embedding_onnx(face_crop)
                
                # --- CALCULATE ALL SCORES ---
                scores_list = []
                for name, db_emb in known_faces.items():
                    score = np.dot(curr_emb, db_emb.T).item()
                    scores_list.append((score, name))
                
                scores_list.sort(reverse=True, key=lambda x: x[0])
                
                best_score, best_match_name = scores_list[0]
                second_best_score = scores_list[1][0] if len(scores_list) > 1 else 0.0
                
                confidence = best_score
                margin = best_score - second_best_score
                
                # --- CONFIDENCE-WEIGHTED MARGIN LOGIC ---
                # We require a stricter margin (0.15) if the confidence is borderline.
                required_margin = MARGIN_THRESHOLD if best_score > 0.80 else MARGIN_THRESHOLD * 1.5
                
                if best_score < THRESHOLD:
                    raw_user = "Unknown"
                    decision_factor = f"Low_Conf({best_score:.2f})"
                elif margin < required_margin:
                    raw_user = "Unknown"
                    decision_factor = f"Low_Margin({margin:.3f})"
                else:
                    raw_user = best_match_name
                    decision_factor = "Pass"
                
                # --- SMOOTHING ---
                display_user = stabilizer.update(raw_user)
                
                if display_user != "Unknown":
                    msg, color = f"{display_user} ({confidence:.2f})", (0, 255, 0)
                else:
                    msg, color = f"Unknown ({confidence:.2f})", (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        stability_counter = 0

    # Metrics & Logging
    total_latency = (time.perf_counter() - frame_start) * 1000
    memory_mb = process.memory_info().rss / (1024 * 1024)
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    writer.writerow([current_time, mode, raw_user, display_user, f"{confidence:.4f}", 
                     f"{second_best_score:.4f}", f"{margin:.4f}", decision_factor, f"{memory_mb:.2f}"])

    # HUD
    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Margin: {margin:.3f} | RAM: {memory_mb:.0f}MB", (10, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow("Final Stabilized Benchmarking", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

f.close()
cap.release()
cv2.destroyAllWindows()