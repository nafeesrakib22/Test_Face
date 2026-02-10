import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from skimage import transform as trans

# --- NEW IMPORTS FOR TASKS API ---
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CAMERA_INDEX = 3  
CONFIDENCE_THRESHOLD = 0.6
FACE_DB_PATH = "data/face_db"
EDGEFACE_MODEL_PATH = "models/edgeface_xs_gamma_06.onnx"
DETECTOR_MODEL_PATH = "models/blaze_face_short_range.tflite"
# ---------------------

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- GLOBAL VARIABLES ---
face_db = {}
ort_session = None
detector = None

# --- ALIGNMENT MATRIX ---
src1 = np.array([
    [51.642, 50.115], [57.617, 49.990], [35.740, 69.007],
    [51.157, 89.050], [57.025, 89.702]], dtype=np.float32)

def estimate_norm(lmk):
    tform = trans.SimilarityTransform()
    tform.estimate(lmk, src1)
    return tform.params[0:2, :]

def load_resources():
    global ort_session, face_db, detector
    print("⏳ Loading AI Models...")
    
    # 1. Load EdgeFace
    try:
        ort_session = ort.InferenceSession(EDGEFACE_MODEL_PATH)
        print("✅ EdgeFace Model Loaded")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: EdgeFace failed to load: {e}")

    # 2. Load Database
    if os.path.exists(FACE_DB_PATH):
        count = 0
        for file in os.listdir(FACE_DB_PATH):
            if file.endswith(".npy"):
                name = os.path.splitext(file)[0]
                embedding = np.load(os.path.join(FACE_DB_PATH, file))
                face_db[name] = embedding
                count += 1
        print(f"✅ Loaded {count} users from database.")
    
    # 3. Load MediaPipe Tasks
    try:
        if not os.path.exists(DETECTOR_MODEL_PATH):
             print(f"❌ CRITICAL ERROR: .tflite model not found at {DETECTOR_MODEL_PATH}")
        
        base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
        options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.5)
        detector = vision.FaceDetector.create_from_options(options)
        print("✅ MediaPipe Tasks Detector Loaded")
    except Exception as e:
        print(f"❌ CRITICAL ERROR: MediaPipe failed to load: {e}")

load_resources()

def get_embedding(face_img):
    face_img = (face_img / 255.0 - 0.5) / 0.5
    face_img = face_img.transpose(2, 0, 1).astype(np.float32)
    face_img = np.expand_dims(face_img, axis=0)
    inputs = {ort_session.get_inputs()[0].name: face_img}
    return ort_session.run(None, inputs)[0][0]

def generate_frames():
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("❌ Camera failed to open inside generator!")
    
    print("📷 Camera Stream Started...")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("❌ Frame read failed.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        try:
            detection_result = detector.detect(mp_image)
            
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                w_box = bbox.width
                h_box = bbox.height

                # Draw raw detection box (Blue)
                cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), (255, 0, 0), 2)

                kps = detection.keypoints
                landmarks = np.array([[kp.x * frame.shape[1], kp.y * frame.shape[0]] for kp in kps])

                if len(landmarks) >= 5:
                    M = estimate_norm(landmarks[:5])
                    aligned_face = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
                    
                    # --- THE FIX IS HERE ---
                    emb = get_embedding(aligned_face).flatten() # Force to 1D array
                    
                    identity = "Unknown"
                    max_score = 0
                    
                    for name, db_emb in face_db.items():
                        db_emb_flat = db_emb.flatten() # Force DB to 1D array
                        
                        # Cosine Similarity
                        score = np.dot(emb, db_emb_flat) / (np.linalg.norm(emb) * np.linalg.norm(db_emb_flat))
                        
                        if score > max_score:
                            max_score = score
                            identity = name
                    
                    # -----------------------
                    
                    color = (0, 255, 0) if max_score > CONFIDENCE_THRESHOLD else (0, 0, 255)
                    label = f"{identity} ({max_score:.2f})"
                    
                    cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
        except Exception as e:
            print(f"❌ Processing Error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
@app.get("/")
def home():
    return {"status": "AI System Online", "users": list(face_db.keys())}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")