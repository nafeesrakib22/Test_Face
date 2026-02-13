import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from skimage import transform as trans
from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CAMERA_INDEX = 3 
ONNX_MODEL_PATH = 'models/edgeface_xs_gamma_06.onnx'
DETECTOR_MODEL_PATH = 'models/blaze_face_short_range.tflite'
DB_PATH = 'data/face_db'

THRESHOLD = 0.65             
STABILITY_FRAMES = 8          
BUFFER_SIZE = 12              
SMOOTHING_FACTOR = 0.20       

# EXPERIMENTAL PARAMS (Synced with enrollment/benchmark scripts)
PADDING = 0.25  
# ---------------------

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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

# --- GLOBAL STATE ---
known_faces = {}
ort_session = None
detector = None
stabilizer = IdentityStabilizer(maxlen=BUFFER_SIZE)
prev_box = None
stability_counter = 0

def load_resources():
    global ort_session, known_faces, detector
    print("⏳ Synchronizing AI Engines with Experimental Logic...")
    
    # 1. ONNX Inference
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # 2. Multi-Profile Database
    known_faces = {} # Reset to ensure no stale data
    if os.path.exists(DB_PATH):
        for filename in os.listdir(DB_PATH):
            if filename.endswith('.npy'):
                identity_name = filename.split('_')[0]
                vector = np.load(os.path.join(DB_PATH, filename))
                if identity_name not in known_faces:
                    known_faces[identity_name] = []
                known_faces[identity_name].append(vector)
        print(f"✅ Loaded {len(known_faces)} identities with new profiles.")

    # 3. Detector
    base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

load_resources()

def get_embedding_onnx(img_bgr):
    """EXPERIMENTAL: Dynamic Elliptical Masking with Gaussian Blur"""
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    # Create Dynamic Soft Mask
    mask = np.zeros((112, 112), dtype=np.uint8)
    axes = (int(112 * 0.42), int(112 * 0.52))
    cv2.ellipse(mask, (56, 56), axes, 0, 0, 360, 255, -1)
    
    # Gaussian Blur softens the black boundary to reduce AI edge-noise
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    img_rgb = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    
    embeddings = ort_session.run(None, {ort_session.get_inputs()[0].name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)

def generate_frames():
    global prev_box, stability_counter
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        
        msg, color = "Scanning...", (255, 255, 255)
        
        if results.detections:
            if len(results.detections) > 1:
                stability_counter = 0
                msg, color = "⚠️ MULTIPLE FACES", (0, 0, 255)
                for det in results.detections:
                    b = det.bounding_box
                    cv2.rectangle(frame, (b.origin_x, b.origin_y), (b.origin_x+b.width, b.origin_y+b.height), (0,0,255), 2)
            else:
                stability_counter += 1
                
                # 1. Smoothed Bounding Box
                det = results.detections[0].bounding_box
                curr_box = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
                if prev_box is None: prev_box = curr_box
                prev_box = (prev_box * (1.0 - SMOOTHING_FACTOR)) + (curr_box * SMOOTHING_FACTOR)
                
                # 2. EXPERIMENTAL: Center-Anchored Padding
                sx, sy, sw, sh = prev_box
                center_x, center_y = sx + (sw / 2), sy + (sh / 2)
                padded_w, padded_h = sw * (1 + PADDING), sh * (1 + PADDING)

                x1 = int(max(0, center_x - (padded_w / 2)))
                y1 = int(max(0, center_y - (padded_h / 2)))
                x2 = int(min(w, x1 + padded_w))
                y2 = int(min(h, y1 + padded_h))
                
                if stability_counter < STABILITY_FRAMES:
                    msg, color = "Stabilizing...", (0, 255, 255)
                else:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        curr_emb = get_embedding_onnx(face_crop)
                        best_overall_score = -1.0
                        best_match_name = "Unknown"
                        
                        # Multi-Profile Max-Pooling
                        for name, profile_list in known_faces.items():
                            for profile_vec in profile_list:
                                score = np.dot(curr_emb.flatten(), profile_vec.flatten()).item()
                                if score > best_overall_score:
                                    best_overall_score = score
                                    best_match_name = name
                        
                        raw_user = best_match_name if best_overall_score > THRESHOLD else "Unknown"
                        display_user = stabilizer.update(raw_user)
                        
                        color = (0, 255, 0) if display_user != "Unknown" else (0, 0, 255)
                        msg = f"{display_user} ({best_overall_score:.2f})"
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            stability_counter = 0
            prev_box = None

        cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/")
def home():
    return {"status": "Experimental Engine Online", "identities": list(known_faces.keys())}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")