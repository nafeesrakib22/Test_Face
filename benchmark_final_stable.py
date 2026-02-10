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

# --- CONFIGURATION ---
ONNX_MODEL_PATH = "edgeface_xs_gamma_06.onnx"
DB_PATH = 'data/face_db'
CSV_FILE = "benchmark_stable.csv"
THRESHOLD = 0.65  
STABILITY_FRAMES = 10  # Wait  before even trying to recognize
BUFFER_SIZE = 8       # [NEW] Number of frames to vote on (Smoothing)

#  CLASS: IDENTITY STABILIZER  to prevent flickering
class IdentityStabilizer:
    def __init__(self, maxlen=10):
        self.history = deque(maxlen=maxlen)
        self.current_display_name = "Unknown"
        self.locked_frames = 0
        
    def update(self, name, score):
        # Add new prediction to history
        self.history.append(name)
        
        # Count frequency in the buffer
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        
        # HYSTERESIS LOGIC:
        # Only switch if the new name appears in >60% of the buffer
        if count > (self.history.maxlen * 0.6):
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
writer.writerow(["Timestamp", "Mode", "Raw_User", "Display_User", "Confidence", "RAM_MB"])

# --- HELPER FUNCTIONS ---
def get_embedding_onnx(img_bgr):
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (45, 58), 0, 0, 360, 255, -1)
    img_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask)

    # For ensuring face is properly detected
    cv2.imshow("What the AI Sees", img_resized) 
    # Preprocessing 

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
                print(f"Loaded Robust ID: {name}")
            except: pass
        elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            if name not in db:
                img = cv2.imread(file_path)
                if img is not None:
                    db[name] = get_embedding_onnx(img)
                    print(f"Loaded Standard ID: {name}")
    return db

# --- MAIN LOOP ---
known_faces = load_database(DB_PATH)
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
process = psutil.Process(os.getpid())
stability_counter = 0


while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    mode = "Idle"
    raw_user = "None"     # What the model saw this specific frame
    display_user = "None" # What we show to the user (Smoothed)
    confidence = 0.0
    msg, color = "Searching...", (255, 255, 255)
    
    if results.detections:
        if len(results.detections) > 1:
            stability_counter = 0
            mode = "Warning"
            msg, color = "Multiple Faces", (0, 0, 255)
        else:
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
                    best_score = -1.0
                    raw_user = "Unknown"
                    
                    # 1. Find Best Match
                    for name, db_emb in known_faces.items():
                        score = np.dot(curr_emb, db_emb.T).item()
                        if score > best_score:
                            best_score = score
                            best_match_name = name
                    
                    confidence = best_score
                    
                    # 2. Threshold Check
                    if best_score > THRESHOLD:
                        raw_user = best_match_name
                    else:
                        raw_user = "Unknown"
                        
                    # 3. SMOOTHING 
                    display_user = stabilizer.update(raw_user, confidence)
                    
                    # 4. Display Logic
                    if display_user != "Unknown":
                        msg = f"{display_user} ({confidence:.2f})"
                        color = (0, 255, 0)
                    else:
                        msg = f"Unknown ({confidence:.2f})"
                        color = (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        stability_counter = 0
        # Reset stabilizer history if nobody is there? 
        # Optional: stabilizer.history.clear()

    # Metrics
    memory_mb = process.memory_info().rss / (1024 * 1024)
    current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    writer.writerow([current_time, mode, raw_user, display_user, f"{confidence:.4f}", f"{memory_mb:.2f}"])

    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Stabilized System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

f.close()
cap.release()
cv2.destroyAllWindows()