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

# --- PRO CONFIGURATION ---
ONNX_MODEL_PATH = "edgeface_xs_gamma_06.onnx"
DB_PATH = 'data/face_db'
PRIMARY_USER = "Nafees"        # [NEW] Give this user a small mathematical advantage
USER_BIAS = 0.05               # [NEW] The "Owner Bonus"
THRESHOLD = 0.70               
MARGIN_THRESHOLD = 0.08        
BUFFER_SIZE = 12               

# --- IDENTITY STABILIZER ---
class IdentityStabilizer:
    def __init__(self, maxlen=12):
        self.history = deque(maxlen=maxlen)
        self.current_display_name = "Unknown"
        
    def update(self, name):
        self.history.append(name)
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        if count >= int(self.history.maxlen * 0.75):
            self.current_display_name = most_common
        return self.current_display_name

# 1. Initialize Engines
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 2
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

stabilizer = IdentityStabilizer(maxlen=BUFFER_SIZE)

# --- THE FIX: PREPROCESSING WITH CLAHE ---
def get_embedding_onnx(img_bgr):
    # 1. Resize
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    # 2. [NEW] CLAHE Lighting Correction
    # This makes the AI ignore room lighting and focus on bone structure
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    img_resized = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # 3. Apply Elliptical Mask
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (45, 58), 0, 0, 360, 255, -1)
    img_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    # [DEBUG] Optional: cv2.imshow("AI_INPUT", img_resized)

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    
    embeddings = ort_session.run(None, {input_name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)

def load_database(path):
    db = {}
    for filename in os.listdir(path):
        if filename.endswith('.npy'):
            db[os.path.splitext(filename)[0]] = np.load(os.path.join(path, filename))
    return db

# --- MAIN LOOP ---
known_faces = load_database(DB_PATH)
cap = cv2.VideoCapture(config.CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    msg, color = "Searching...", (255, 255, 255)
    
    if results.detections:
        det = results.detections[0]
        b = det.bounding_box
        x1, y1, x2, y2 = max(0, b.origin_x), max(0, b.origin_y), min(640, b.origin_x + b.width), min(480, b.origin_y + b.height)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size > 0:
            curr_emb = get_embedding_onnx(face_crop)
            
            scores = []
            for name, db_emb in known_faces.items():
                score = np.dot(curr_emb, db_emb.T).item()
                
                # [NEW] Apply User Bias
                if name == PRIMARY_USER:
                    score += USER_BIAS
                
                scores.append((score, name))
            
            scores.sort(reverse=True, key=lambda x: x[0])
            best_score, best_name = scores[0]
            margin = best_score - scores[1][0] if len(scores) > 1 else 1.0
            
            raw_user = "Unknown"
            if best_score > THRESHOLD and margin > MARGIN_THRESHOLD:
                raw_user = best_name
            
            display_user = stabilizer.update(raw_user)
            
            if display_user != "Unknown":
                msg, color = f"{display_user} ({best_score:.2f})", (0, 255, 0)
            else:
                msg, color = f"Unknown ({best_score:.2f})", (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Final Production Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()