import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os
import time
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CAMERA_INDEX = 1  # 0 is default, 3 was your previous setup
ONNX_MODEL_PATH = 'models/edgeface_xs_gamma_06.onnx'
DETECTOR_MODEL_PATH = 'models/blaze_face_short_range.tflite'
DB_PATH = 'data/face_db'

# Recognition & Stability Logic
THRESHOLD = 0.65             
STABILITY_FRAMES = 8          
BUFFER_SIZE = 12              
SMOOTHING_FACTOR = 0.20       
PADDING = 0.25  

# Enrollment Quality Gates
BLUR_THRESHOLD = 80
LIGHTING_RANGE = (70, 210)
SAMPLES_PER_PHASE = 25
# ---------------------

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- GLOBAL STATE ---
known_faces = {}
ort_session = None
detector = None
enrollment_status = {"name": None, "progress": 0, "complete": False}
prev_box = None
stability_counter = 0

# SINGLE GLOBAL CAMERA OBJECT (Solves the Black Screen Lock)
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

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

stabilizer = IdentityStabilizer(maxlen=BUFFER_SIZE)

def load_resources():
    global ort_session, known_faces, detector
    print("⏳ Initializing AI Models...")
    
    # 1. EdgeFace ONNX
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    # 2. Database loading
    known_faces = {}
    if os.path.exists(DB_PATH):
        for filename in os.listdir(DB_PATH):
            if filename.endswith('.npy'):
                identity_name = filename.split('_')[0]
                vector = np.load(os.path.join(DB_PATH, filename))
                if identity_name not in known_faces:
                    known_faces[identity_name] = []
                known_faces[identity_name].append(vector)
    
    # 3. MediaPipe Detector
    base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    print("✅ System Ready.")

load_resources()

# --- HELPER LOGIC ---

def is_well_lit(image):
    if image.size == 0: return False
    small = cv2.resize(image, (32, 32))
    avg_luma = np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    return LIGHTING_RANGE[0] < avg_luma < LIGHTING_RANGE[1]

def is_sharp(image):
    if image.size == 0: return 0
    small = cv2.resize(image, (100, 100))
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def get_embedding_onnx(img_bgr):
    img_resized = cv2.resize(img_bgr, (112, 112))
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (int(112*0.42), int(112*0.52)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    img_float = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :].astype(np.float32)
    embeddings = ort_session.run(None, {ort_session.get_inputs()[0].name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)



def detect_pose(keypoints, target_pose):
    p_left_eye, p_right_eye, nose, mouth = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    
    # Mirror mapping
    s_left_eye = p_left_eye if p_left_eye.x < p_right_eye.x else p_right_eye
    s_right_eye = p_right_eye if p_left_eye.x < p_right_eye.x else p_left_eye

    # Horizontal (Yaw)
    eye_dist = abs(s_right_eye.x - s_left_eye.x)
    yaw_ratio = (nose.x - s_left_eye.x) / eye_dist if eye_dist > 0.01 else 0.5
    
    # Vertical (Pitch)
    eye_y_avg = (s_left_eye.y + s_right_eye.y) / 2
    total_v_dist = abs(mouth.y - eye_y_avg)
    pitch_ratio = abs(nose.y - eye_y_avg) / total_v_dist if total_v_dist > 0.01 else 0.5

    if target_pose == "FRONTAL":
        if 0.42 <= yaw_ratio <= 0.58 and 0.40 <= pitch_ratio <= 0.60:
            return "FRONTAL", "Perfect. Look straight."
        return "TRANSITIONING", "Look back to center."

    elif target_pose == "LEFT_PROFILE":
        if yaw_ratio > 0.58:
            if yaw_ratio > 0.82: return "OVER", "TOO FAR! Turn back right."
            if yaw_ratio > 0.70: return "LEFT_PROFILE", "STOP! Hold this angle."
            return "LEFT_PROFILE", "Good, slowly turn more LEFT..."
        return "TRANSITIONING", "Turn slowly to your LEFT..."

    elif target_pose == "RIGHT_PROFILE":
        if yaw_ratio < 0.42:
            if yaw_ratio < 0.18: return "OVER", "TOO FAR! Turn back left."
            if yaw_ratio < 0.30: return "RIGHT_PROFILE", "STOP! Hold this angle."
            return "RIGHT_PROFILE", "Good, slowly turn more RIGHT..."
        return "TRANSITIONING", "Turn slowly to your RIGHT..."

    elif target_pose == "LOOK_UP":
        if pitch_ratio < 0.40:
            if pitch_ratio < 0.20: return "OVER", "TOO FAR! Tilt chin down."
            if pitch_ratio < 0.30: return "LOOK_UP", "STOP! Hold this angle."
            return "LOOK_UP", "Good, tilt chin MORE UP..."
        return "TRANSITIONING", "Slowly tilt chin UP..."

    elif target_pose == "LOOK_DOWN":
        if pitch_ratio > 0.60:
            if pitch_ratio > 0.85: return "OVER", "TOO FAR! Tilt chin up."
            if pitch_ratio > 0.75: return "LOOK_DOWN", "STOP! Hold this angle."
            return "LOOK_DOWN", "Good, tilt chin MORE DOWN..."
        return "TRANSITIONING", "Slowly tilt chin DOWN..."

    return "UNKNOWN", "Adjusting..."

# --- STREAM GENERATORS ---

def generate_recognition_frames():
    global prev_box, stability_counter
    while True:
        success, frame = cap.read()
        if not success: 
            time.sleep(0.1)
            continue
            
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        msg, color = "Scanning...", (255, 255, 255)
        
        if results.detections:
            if len(results.detections) > 1:
                stability_counter = 0
                msg, color = "⚠️ MULTIPLE FACES", (0, 0, 255)
            else:
                stability_counter += 1
                det = results.detections[0].bounding_box
                curr_box = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
                if prev_box is None: prev_box = curr_box
                prev_box = (prev_box * (1.0 - SMOOTHING_FACTOR)) + (curr_box * SMOOTHING_FACTOR)
                
                sx, sy, sw, sh = prev_box
                cx, cy = sx + (sw/2), sy + (sh/2)
                pw, ph = sw*(1+PADDING), sh*(1+PADDING)
                x1, y1, x2, y2 = int(max(0, cx-pw/2)), int(max(0, cy-ph/2)), int(min(w, cx+pw/2)), int(min(h, cy+ph/2))
                
                if stability_counter < STABILITY_FRAMES:
                    msg, color = "Stabilizing...", (0, 255, 255)
                else:
                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        curr_emb = get_embedding_onnx(face_crop)
                        best_score, best_name = -1.0, "Unknown"
                        for name, profiles in known_faces.items():
                            for p_vec in profiles:
                                score = np.dot(curr_emb.flatten(), p_vec.flatten()).item()
                                if score > best_score:
                                    best_score, best_name = score, name
                        
                        raw_user = best_name if best_score > THRESHOLD else "Unknown"
                        display_user = stabilizer.update(raw_user)
                        color = (0, 255, 0) if display_user != "Unknown" else (0, 0, 255)
                        msg = f"{display_user} ({best_score:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            stability_counter, prev_box = 0, None

        cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_enrollment_frames(name: str):
    global enrollment_status
    enrollment_status = {"name": name, "progress": 0, "complete": False}
    
    phases = ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE", "LOOK_UP", "LOOK_DOWN"]
    phase_idx, all_embeddings, prev_box_enroll = 0, [], None

    while phase_idx < len(phases):
        success, frame = cap.read()
        if not success: 
            time.sleep(0.1)
            continue

        h, w, _ = frame.shape
        target_pose = phases[phase_idx]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = detector.detect(mp_image)
        msg, color = "Searching...", (255, 255, 255)

        if results.detections:
            det = results.detections[0].bounding_box
            keypoints = results.detections[0].keypoints
            cur_pose, instruction = detect_pose(keypoints, target_pose)
            
            curr_box = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
            if prev_box_enroll is None: prev_box_enroll = curr_box
            prev_box_enroll = (prev_box_enroll * (1.0 - SMOOTHING_FACTOR)) + (curr_box * SMOOTHING_FACTOR)
            
            sx, sy, sw, sh = prev_box_enroll
            cx, cy = sx + (sw/2), sy + (sh/2)
            pw, ph = sw*(1+PADDING), sh*(1+PADDING)
            x1, y1, x2, y2 = int(max(0, cx-pw/2)), int(max(0, cy-ph/2)), int(min(w, cx+pw/2)), int(min(h, cy+ph/2))
            
            face_crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if cur_pose == target_pose:
                if is_well_lit(face_crop) and is_sharp(face_crop) > BLUR_THRESHOLD:
                    all_embeddings.append(get_embedding_onnx(face_crop))
                    color, msg = (0, 255, 0), instruction
                    enrollment_status["progress"] = int((len(all_embeddings) / 125) * 100)
                    
                    if len(all_embeddings) >= (phase_idx + 1) * SAMPLES_PER_PHASE:
                        phase_idx += 1
                else:
                    color, msg = (0, 0, 255), "QUALITY LOW: Hold Still"
            else:
                color = (0, 0, 255) if "TOO FAR" in instruction else (0, 255, 255)
                msg = instruction

        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Phase {phase_idx+1}/5 | Total: {len(all_embeddings)}/125", (20, 70), 0, 0.6, (255,255,255), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    # Final Save
    if len(all_embeddings) == 125:
        p_names = ["frontal", "left", "right", "up", "down"]
        for i, p in enumerate(p_names):
            chunk = all_embeddings[i*25 : (i+1)*25]
            centroid = np.mean(np.array(chunk), axis=0)
            np.save(os.path.join(DB_PATH, f"{name}_{p}.npy"), centroid / np.linalg.norm(centroid))
        
        load_resources()
        enrollment_status["complete"] = True

# --- FASTAPI ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "5-Phase Engine Online", "identities": list(known_faces.keys())}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_recognition_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/enroll")
def enroll_user(name: str = Query(..., min_length=1)):
    return StreamingResponse(generate_enrollment_frames(name), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/enroll_status")
def get_enroll_status():
    return enrollment_status

@app.get("/reset_session")
def reset_session():
    global enrollment_status, stability_counter, prev_box
    enrollment_status = {"name": None, "progress": 0, "complete": False}
    stability_counter = 0
    prev_box = None
    return {"status": "Session Reset"}


import atexit
@atexit.register
def shutdown():
    if cap.isOpened():
        cap.release()