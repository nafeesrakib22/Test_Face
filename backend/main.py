import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os
import time
import atexit
import threading
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIGURATION ---
CAMERA_INDEX = 0 
ONNX_MODEL_PATH = 'models/edgeface_xs_gamma_06.onnx'
DETECTOR_MODEL_PATH = 'models/blaze_face_short_range.tflite'
DB_PATH = 'data/face_db'

THRESHOLD = 0.65             
STABILITY_FRAMES = 8          
BUFFER_SIZE = 12              
SMOOTHING_FACTOR = 0.20       
PADDING = 0.25  

BLUR_THRESHOLD = 80
LIGHTING_RANGE = (70, 210)
SAMPLES_PER_PHASE = 25
# ---------------------

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# --- THREADED CAMERA CLASS ---
class VideoStream:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.update, args=(), daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                continue
            with self.lock:
                self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.stopped = True
        self.cap.release()

# Global State
vs = VideoStream(CAMERA_INDEX).start()
known_faces = {}
ort_session = None
detector = None
enrollment_status = {"name": None, "progress": 0, "complete": False}
stabilizer = None # Initialized in load_resources

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

def load_resources():
    global ort_session, known_faces, detector, stabilizer
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
    
    known_faces = {}
    if os.path.exists(DB_PATH):
        for filename in os.listdir(DB_PATH):
            if filename.endswith('.npy'):
                name = filename.split('_')[0]
                vec = np.load(os.path.join(DB_PATH, filename))
                if name not in known_faces: known_faces[name] = []
                known_faces[name].append(vec)
    
    base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    stabilizer = IdentityStabilizer(maxlen=BUFFER_SIZE)

load_resources()

# --- HELPERS ---
def is_well_lit(image):
    if image.size == 0: return False
    small = cv2.resize(image, (32, 32))
    return LIGHTING_RANGE[0] < np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)) < LIGHTING_RANGE[1]

def is_sharp(image):
    if image.size == 0: return 0
    small = cv2.resize(image, (100, 100))
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def get_embedding_onnx(img_bgr):
    img = cv2.resize(img_bgr, (112, 112))
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (47, 58), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    img = cv2.bitwise_and(img, img, mask=mask)
    
    # Standard EdgeFace preprocessing: BGR to RGB, [0,1], then Normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    
    emb = ort_session.run(None, {ort_session.get_inputs()[0].name: img})[0]
    # CRITICAL: L2 Normalization
    return emb / np.linalg.norm(emb)

def detect_pose(keypoints, target):
    l_eye, r_eye, nose, mouth = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    sl_eye = l_eye if l_eye.x < r_eye.x else r_eye
    sr_eye = r_eye if l_eye.x < r_eye.x else l_eye
    yaw = (nose.x - sl_eye.x) / abs(sr_eye.x - sl_eye.x) if abs(sr_eye.x - sl_eye.x) > 0.01 else 0.5
    pitch = abs(nose.y - (sl_eye.y + sr_eye.y)/2) / abs(mouth.y - (sl_eye.y + sr_eye.y)/2) if abs(mouth.y - (sl_eye.y + sr_eye.y)/2) > 0.01 else 0.5

    if target == "FRONTAL":
        return ("FRONTAL", "Perfect. Look straight.") if 0.42 <= yaw <= 0.58 and 0.40 <= pitch <= 0.60 else ("TRANS", "Look back to center.")
    elif target == "LEFT_PROFILE":
        if yaw > 0.82: return "OVER", "TOO FAR! Turn back right."
        return ("LEFT_PROFILE", "Good, keep turning LEFT.") if yaw > 0.58 else ("TRANS", "Turn slowly left...")
    elif target == "RIGHT_PROFILE":
        if yaw < 0.18: return "OVER", "TOO FAR! Turn back left."
        return ("RIGHT_PROFILE", "Good, keep turning RIGHT.") if yaw < 0.42 else ("TRANS", "Turn slowly right...")
    elif target == "LOOK_UP":
        if pitch < 0.20: return "OVER", "TOO FAR! Chin down."
        return ("LOOK_UP", "Good, tilt chin UP.") if pitch < 0.40 else ("TRANS", "Tilt chin up...")
    elif target == "LOOK_DOWN":
        if pitch > 0.85: return "OVER", "TOO FAR! Chin up."
        return ("LOOK_DOWN", "Good, tilt chin DOWN.") if pitch > 0.60 else ("TRANS", "Tilt chin down...")
    return "UNK", "Adjusting..."

# --- GENERATORS ---
def generate_recognition_frames():
    prev_box, stability = None, 0
    while True:
        ret, frame = vs.read()
        if not ret: continue
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        msg, color = "Scanning...", (255, 255, 255)
        
        if res.detections:
            if len(res.detections) > 1: msg, color = "⚠️ MULTIPLE FACES", (0, 0, 255)
            else:
                stability += 1
                det = res.detections[0].bounding_box
                curr = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
                prev_box = curr if prev_box is None else (prev_box * 0.8) + (curr * 0.2)
                sx, sy, sw, sh = prev_box
                cx, cy = sx + sw/2, sy + sh/2
                x1, y1, x2, y2 = int(max(0, cx-sw*0.625)), int(max(0, cy-sh*0.625)), int(min(w, cx+sw*0.625)), int(min(h, cy+sh*0.625))
                
                if stability >= STABILITY_FRAMES:
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        emb = get_embedding_onnx(crop)
                        best_s, best_n = -1.0, "Unknown"
                        for n, p_list in known_faces.items():
                            for p in p_list:
                                s = np.dot(emb.flatten(), p.flatten()).item()
                                if s > best_s: best_s, best_n = s, n
                        raw = best_n if best_s > THRESHOLD else "Unknown"
                        display = stabilizer.update(raw)
                        color = (0, 255, 0) if display != "Unknown" else (0, 0, 255)
                        msg = f"{display} ({best_s:.2f})"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else: stability, prev_box = 0, None

        cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

def generate_enrollment_frames(name):
    global enrollment_status
    enrollment_status = {"name": name, "progress": 0, "complete": False}
    phases = ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE", "LOOK_UP", "LOOK_DOWN"]
    p_idx, embs, prev_e = 0, [], None

    while p_idx < len(phases):
        ret, frame = vs.read()
        if not ret: continue
        h, w, target = frame.shape[0], frame.shape[1], phases[p_idx]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp_image)
        msg, color = "Searching...", (255, 255, 255)

        if res.detections:
            det, kps = res.detections[0].bounding_box, res.detections[0].keypoints
            cur_p, instr = detect_pose(kps, target)
            curr = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
            prev_e = curr if prev_e is None else (prev_e * 0.8) + (curr * 0.2)
            sx, sy, sw, sh = prev_e
            cx, cy = sx + sw/2, sy + sh/2
            x1, y1, x2, y2 = int(max(0, cx-sw*0.625)), int(max(0, cy-sh*0.625)), int(min(w, cx+sw*0.625)), int(min(h, cy+sh*0.625))
            crop = frame[y1:y2, x1:x2]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

            if cur_p == target:
                if is_well_lit(crop) and is_sharp(crop) > BLUR_THRESHOLD:
                    embs.append(get_embedding_onnx(crop))
                    color, msg = (0, 255, 0), instr
                    enrollment_status["progress"] = int((len(embs) / 125) * 100)
                    if len(embs) >= (p_idx + 1) * 25: p_idx += 1
                else: color, msg = (0, 0, 255), "QUALITY LOW: Hold Still"
            else: color, msg = ((0, 0, 255) if "TOO FAR" in instr else (0, 255, 255)), instr

        cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Phase {p_idx+1}/5 | {enrollment_status['progress']}%", (20, 70), 0, 0.6, (255,255,255), 1)
        _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

    if len(embs) == 125:
        p_names = ["frontal", "left", "right", "up", "down"]
        for i, p in enumerate(p_names):
            chunk = np.array(embs[i*25 : (i+1)*25])
            # Average the 25 frames for this pose
            centroid = np.mean(chunk, axis=0)
            # CRITICAL: Re-normalize the centroid
            normalized_centroid = centroid / np.linalg.norm(centroid)
            
            np.save(os.path.join(DB_PATH, f"{name}_{p}.npy"), normalized_centroid)
        
        load_resources() # Reloads database into memory
        enrollment_status["complete"] = True

# --- ENDPOINTS ---
@app.get("/video_feed")
def video_feed(): return StreamingResponse(generate_recognition_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/enroll")
def enroll_user(name: str): return StreamingResponse(generate_enrollment_frames(name), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/enroll_status")
def get_enroll_status(): return enrollment_status

@app.get("/reset_session")
def reset_session():
    global enrollment_status; enrollment_status = {"name": None, "progress": 0, "complete": False}
    return {"status": "Reset"}

@atexit.register
def shutdown(): vs.stop()