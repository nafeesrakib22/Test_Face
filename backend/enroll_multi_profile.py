import cv2
import numpy as np
import os
import onnxruntime as ort
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

# --- CONFIGURATION ---
ONNX_MODEL_PATH = "models/edgeface_xs_gamma_06.onnx"
DETECTOR_MODEL_PATH = 'models/blaze_face_short_range.tflite'
DB_PATH = 'data/face_db'
SAMPLES_PER_PHASE = 25

# Quality Thresholds
BLUR_THRESHOLD = 80
LIGHTING_RANGE = (70, 210)
PADDING = 0.25
SMOOTHING_FACTOR = 0.20

# 1. Setup Engines
print("Initializing Automated Interactive Enrollment Engine...")
sess_options = ort.SessionOptions()
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 2. Optimized Helpers
def is_well_lit(image):
    if image.size == 0: return False
    small = cv2.resize(image, (32, 32))
    avg_luma = np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY))
    return LIGHTING_RANGE[0] < avg_luma < LIGHTING_RANGE[1]

def is_sharp(image):
    if image.size == 0: return 0
    small = cv2.resize(image, (100, 100))
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()

def detect_pose(keypoints, target_pose):
    """
    Returns: (Current_Pose_Name, Instruction_Msg)
    Uses the Nose-to-Eye horizontal ratio to provide real-time guidance.
    """
    l_eye, r_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_dist = abs(r_eye.x - l_eye.x)
    if eye_dist < 0.02: return "UNKNOWN", "Angle too steep! Turn back."
    
    ratio = (nose.x - l_eye.x) / eye_dist
    
    # FRONTAL LOGIC
    if 0.42 <= ratio <= 0.58:
        return "FRONTAL", "Perfect. Look straight."
    
    # LEFT PROFILE LOGIC
    elif 0.15 <= ratio < 0.42:
        if target_pose == "LEFT_PROFILE":
            if ratio > 0.30: return "LEFT_PROFILE", "Good, turn MORE left..."
            if ratio < 0.22: return "LEFT_PROFILE", "Stop! Hold this angle."
            return "LEFT_PROFILE", "Great, keep turning slowly."
        return "LEFT_PROFILE", "Target is different. Turn back."
            
    # RIGHT PROFILE LOGIC
    elif 0.58 < ratio <= 0.85:
        if target_pose == "RIGHT_PROFILE":
            if ratio < 0.70: return "RIGHT_PROFILE", "Good, turn MORE right..."
            if ratio > 0.78: return "RIGHT_PROFILE", "Stop! Hold this angle."
            return "RIGHT_PROFILE", "Great, keep turning slowly."
        return "RIGHT_PROFILE", "Target is different. Turn back."
    
    return "TRANSITIONING", "Turn slowly towards target..."

def get_embedding_onnx(img_bgr):
    img_resized = cv2.resize(img_bgr, (112, 112))
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (int(112*0.42), int(112*0.52)), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    img_float = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))[np.newaxis, :]
    
    emb = ort_session.run(None, {input_name: img_input})[0]
    return emb / np.linalg.norm(emb)

# 3. Main Loop
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
name = input("\nEnter Name: ").strip()
if not name: exit()

cap = cv2.VideoCapture(config.CAMERA_INDEX)
phases = ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE"]
phase_idx = 0
all_embeddings = []
prev_box = None

print(f"\n--- AI Automated Interactive Enrollment for {name} ---")



while phase_idx < len(phases):
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    target_pose = phases[phase_idx]
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    msg, color = f"Searching...", (255, 255, 255)

    if results.detections:
        det = results.detections[0].bounding_box
        keypoints = results.detections[0].keypoints
        current_pose, instruction = detect_pose(keypoints, target_pose)
        
        # Bounding Box Logic
        curr_box = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
        if prev_box is None: prev_box = curr_box
        prev_box = (prev_box * (1.0 - SMOOTHING_FACTOR)) + (curr_box * SMOOTHING_FACTOR)
        
        sx, sy, sw, sh = prev_box
        cx, cy = sx + (sw/2), sy + (sh/2)
        pw, ph = sw*(1+PADDING), sh*(1+PADDING)
        x1, y1 = int(max(0, cx-pw/2)), int(max(0, cy-ph/2))
        x2, y2 = int(min(w, x1+pw)), int(min(h, y1+ph))
        
        face_crop = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # Automated Capture Logic with Interactive Guidance
        if current_pose == target_pose:
            if is_well_lit(face_crop) and is_sharp(face_crop) > BLUR_THRESHOLD:
                all_embeddings.append(get_embedding_onnx(face_crop))
                color = (0, 255, 0) # Green
                msg = instruction
                
                if len(all_embeddings) >= (phase_idx + 1) * SAMPLES_PER_PHASE:
                    phase_idx += 1
                    print(f"✓ {target_pose} complete.")
            else:
                color = (0, 0, 255) # Red
                msg = "QUALITY LOW: Check Light/Hold Still"
        else:
            color = (0, 255, 255) # Yellow
            msg = f"{target_pose} -> {instruction}"

    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Progress: {len(all_embeddings)}/75", (20, 70), 0, 0.6, (255,255,255), 1)
    cv2.imshow("Automated Enrollment", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()

# 4. Save
if len(all_embeddings) == 75:
    for i, p in enumerate(["frontal", "left", "right"]):
        chunk = all_embeddings[i*25 : (i+1)*25]
        centroid = np.mean(np.array(chunk), axis=0)
        np.save(os.path.join(DB_PATH, f"{name}_{p}.npy"), centroid / np.linalg.norm(centroid))
    print("\nEnrollment Successful!")