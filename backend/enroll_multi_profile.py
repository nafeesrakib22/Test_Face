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
print("Initializing Automated Enrollment Engine...")
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

def detect_pose(keypoints):
    """
    Keypoint Indices: 0:L Eye, 1:R Eye, 2:Nose, 3:Mouth, 4:L Ear, 5:R Ear
    Uses the Nose-to-Eye horizontal ratio to determine head yaw.
    """
    l_eye, r_eye, nose = keypoints[0], keypoints[1], keypoints[2]
    eye_dist = abs(r_eye.x - l_eye.x)
    if eye_dist == 0: return "UNKNOWN"
    
    # Ratio of nose position between eyes (0.5 is centered)
    ratio = (nose.x - l_eye.x) / eye_dist
    
    if 0.38 <= ratio <= 0.62: return "FRONTAL"
    elif ratio < 0.38: return "LEFT_PROFILE"
    elif ratio > 0.62: return "RIGHT_PROFILE"
    return "TRANSITIONING"

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

print(f"\n--- AI Automated Enrollment for {name} ---")



while phase_idx < len(phases):
    ret, frame = cap.read()
    if not ret: break
    h, w, _ = frame.shape
    target_pose = phases[phase_idx]
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    msg, color = f"Target: {target_pose}", (0, 255, 255)

    if results.detections:
        # A. Bounding Box & Pose Detection
        det = results.detections[0].bounding_box
        keypoints = results.detections[0].keypoints
        current_pose = detect_pose(keypoints)
        
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

        # B. Automated Capture Logic
        if current_pose == target_pose:
            if is_well_lit(face_crop) and is_sharp(face_crop) > BLUR_THRESHOLD:
                all_embeddings.append(get_embedding_onnx(face_crop))
                color = (0, 255, 0)
                msg = f"CAPTURING {target_pose}..."
                
                if len(all_embeddings) >= (phase_idx + 1) * SAMPLES_PER_PHASE:
                    phase_idx += 1
                    print(f"✓ {target_pose} complete.")
            else:
                color = (0, 0, 255)
                msg = "QUALITY LOW: Check Light/Hold Still"
        else:
            msg = f"PLEASE TURN TO: {target_pose}"

    cv2.putText(frame, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Total Progress: {len(all_embeddings)}/75", (20, 70), 0, 0.6, (255,255,255), 1)
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