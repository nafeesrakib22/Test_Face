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
ONNX_MODEL_PATH = 'models/edgeface_xs_gamma_06.onnx'
DETECTOR_MODEL_PATH = 'models/blaze_face_short_range.tflite'
DB_PATH = 'data/face_db'
CSV_FILE = 'models/benchmark_multi_profile.csv'

# Updated for 5-Phase robustness
THRESHOLD = 0.65              # Slightly raised for safety with more profiles
STABILITY_FRAMES = 8          
BUFFER_SIZE = 12              
SMOOTHING_FACTOR = 0.20       
PADDING = 0.25  

# --- CLASS: IDENTITY STABILIZER ---
class IdentityStabilizer:
    def __init__(self, maxlen=12):
        self.history = deque(maxlen=maxlen)
        self.current_display_name = "Unknown"
        
    def update(self, name):
        self.history.append(name)
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        if count >= int(self.history.maxlen * 0.7):
            self.current_display_name = most_common
        return self.current_display_name

# 1. Initialize Engines
print(f"Loading ONNX Model: {ONNX_MODEL_PATH}")
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

stabilizer = IdentityStabilizer(maxlen=BUFFER_SIZE)

# 2. Optimized Helper: Fast Quality + Masking
def get_embedding_onnx(img_bgr):
    img_resized = cv2.resize(img_bgr, (112, 112))
    mask = np.zeros((112, 112), dtype=np.uint8)
    # Match the enrollment ellipse exactly
    cv2.ellipse(mask, (56, 56), (int(112*0.42), int(112*0.52)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)

    img_rgb = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
    img_input = np.transpose((img_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    
    embeddings = ort_session.run(None, {input_name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)

def load_database(path):
    print(f"Loading 5-Phase Database from {path}...")
    db = {}
    if not os.path.exists(path): return db
    
    for filename in os.listdir(path):
        if filename.endswith('.npy'):
            identity_name = filename.split('_')[0]
            vector = np.load(os.path.join(path, filename))
            if identity_name not in db:
                db[identity_name] = []
            db[identity_name].append(vector)
            
    for name in db:
        print(f"✓ {name}: {len(db[name])} profiles (Front, L, R, Up, Down)")
    return db

# --- MAIN LOOP ---
known_faces = load_database(DB_PATH)
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

f = open(CSV_FILE, 'w', newline='')
writer = csv.writer(f)
writer.writerow(["Timestamp", "Mode", "Raw_User", "Display_User", "Confidence", "RAM_MB"])

process = psutil.Process(os.getpid())
stability_counter = 0
prev_box = None 

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    mode, raw_user, confidence = "Idle", "None", 0.0
    msg, color = "Searching...", (255, 255, 255)
    
    if results.detections:
        if len(results.detections) > 1:
            stability_counter = 0
            msg, color = "⚠️ MULTIPLE FACES", (0, 0, 255)
        else:
            stability_counter += 1
            
            # 1. Padded Bounding Box (Synced with 5-Phase Enrollment)
            det = results.detections[0].bounding_box
            curr_box = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
            if prev_box is None: prev_box = curr_box
            prev_box = (prev_box * (1.0 - SMOOTHING_FACTOR)) + (curr_box * SMOOTHING_FACTOR)
            
            sx, sy, sw, sh = prev_box
            cx, cy = sx + (sw / 2), sy + (sh / 2)
            pw, ph = sw * (1 + PADDING), sh * (1 + PADDING)
            x1, y1 = int(max(0, cx - pw/2)), int(max(0, cy - ph/2))
            x2, y2 = int(min(w, x1 + pw)), int(min(h, y1 + ph))
            
            if stability_counter < STABILITY_FRAMES:
                msg, color = f"Stabilizing...", (0, 255, 255)
            else:
                mode = "Recognizing"
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    curr_emb = get_embedding_onnx(face_crop)
                    best_score, best_name = -1.0, "Unknown"
                    
                    # 2. 5-Phase Max-Pooling
                    for name, profiles in known_faces.items():
                        for profile_vec in profiles:
                            score = np.dot(curr_emb.flatten(), profile_vec.flatten()).item()
                            if score > best_score:
                                best_score, best_name = score, name
                    
                    confidence = best_score
                    raw_user = best_name if confidence > THRESHOLD else "Unknown"
                    display_user = stabilizer.update(raw_user)
                    
                    color = (0, 255, 0) if display_user != "Unknown" else (0, 0, 255)
                    msg = f"{display_user} ({confidence:.2f})"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        stability_counter = 0
        prev_box = None 

    # Metrics
    memory_mb = process.memory_info().rss / (1024 * 1024)
    writer.writerow([datetime.datetime.now().strftime("%H:%M:%S.%f"), mode, raw_user, stabilizer.current_display_name, f"{confidence:.4f}", f"{memory_mb:.2f}"])

    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Multi-Profile Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

f.close()
cap.release()
cv2.destroyAllWindows()