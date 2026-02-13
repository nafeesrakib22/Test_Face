import cv2
import numpy as np
import os
import onnxruntime as ort
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config


ONNX_MODEL_PATH = "models/edgeface_xs_gamma_06.onnx"
DETECTOR_MODEL_PATH = 'models/blaze_face_short_range.tflite'
DB_PATH = 'data/face_db'
SAMPLES_PER_PHASE = 25
BLUR_THRESHOLD = 100
LIGHTING_RANGE = (60, 200)

# EXPERIMENTAL PARAMS
PADDING = 0.25           # Expands the box by 25%
SMOOTHING_FACTOR = 0.20  


# 1. Setup Engines
print("Initializing Engines...")
sess_options = ort.SessionOptions()
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

base_options = python.BaseOptions(model_asset_path=DETECTOR_MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 2. Quality Helpers
def is_sharp(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def is_well_lit(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return LIGHTING_RANGE[0] < np.mean(gray) < LIGHTING_RANGE[1]

def get_embedding_onnx(img_bgr):
    """Extracts embedding with Dynamic Elliptical Masking"""
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    # Dynamic Mask Calculation
    mask = np.zeros((112, 112), dtype=np.uint8)
    axes = (int(112 * 0.42), int(112 * 0.52))
    cv2.ellipse(mask, (56, 56), axes, 0, 0, 360, 255, -1)
    
    # Soften edges
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    
    # Debug View
    cv2.imshow("AI Input Crop", img_masked)

    img_rgb = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    
    embeddings = ort_session.run(None, {input_name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)

# 3. Main Logic
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)

name = input("\nEnter Name to Enroll: ").strip()
if not name: exit()

cap = cv2.VideoCapture(config.CAMERA_INDEX)
# Set high-speed MJPG if supported
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

phases = [
    {"name": "FRONTAL", "target": SAMPLES_PER_PHASE},
    {"name": "LEFT_PROFILE", "target": SAMPLES_PER_PHASE * 2},
    {"name": "RIGHT_PROFILE", "target": SAMPLES_PER_PHASE * 3}
]

phase_idx = 0
all_embeddings = []
capturing = False
prev_box = None

print(f"\n--- Starting Multi-Profile Enrollment for {name} ---")
print("Press 's' to START or CONTINUE a phase.")



while phase_idx < len(phases):
    ret, frame = cap.read()
    if not ret:
        break
    
    h, w, _ = frame.shape
    current_phase = phases[phase_idx]
    
    # 1. Detection
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    status_msg = f"Phase: {current_phase['name']}"
    color = (0, 255, 255) # Yellow (Waiting)

    if results.detections:
        # 2. Padded Bounding Box 
        det = results.detections[0].bounding_box
        curr_box = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)

        if prev_box is None: prev_box = curr_box
        prev_box = (prev_box * (1.0 - SMOOTHING_FACTOR)) + (curr_box * SMOOTHING_FACTOR)
        
        sx, sy, sw, sh = prev_box
        center_x, center_y = sx + (sw / 2), sy + (sh / 2)
        padded_w, padded_h = sw * (1 + PADDING), sh * (1 + PADDING)

        x1 = int(max(0, center_x - (padded_w / 2)))
        y1 = int(max(0, center_y - (padded_h / 2)))
        x2 = int(min(w, x1 + padded_w))
        y2 = int(min(h, y1 + padded_h))

        face_crop = frame[y1:y2, x1:x2]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

        # 3. Capture Logic
        if capturing:
            if face_crop.size > 0 and is_sharp(face_crop) > BLUR_THRESHOLD and is_well_lit(face_crop):
                all_embeddings.append(get_embedding_onnx(face_crop))
                color = (0, 255, 0) # Green (Good Capture)
                time.sleep(0.05) 
                
                if len(all_embeddings) >= current_phase['target']:
                    capturing = False
                    phase_idx += 1
                    print(f"✓ {current_phase['name']} complete.")
            else:
                color = (0, 0, 255) # Red (Quality Issue)
                status_msg += " (Hold Still / Check Lighting)"

    # 4. UI 
    if not capturing:
        status_msg += " | Press 's' to START"
    else:
        status_msg += f" | Progress: {len(all_embeddings)}/{current_phase['target']}"

    cv2.putText(frame, status_msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Multi-Profile Enrollment", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not capturing:
        capturing = True
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 5. Save Centroids
if len(all_embeddings) == SAMPLES_PER_PHASE * 3:
    print("\nProcessing Centroids...")
    profile_data = {
        "frontal": all_embeddings[0:25],
        "left": all_embeddings[25:50],
        "right": all_embeddings[50:75]
    }
    
    for p_name, embs in profile_data.items():
        centroid = np.mean(np.array(embs), axis=0)
        centroid /= np.linalg.norm(centroid)
        save_path = os.path.join(DB_PATH, f"{name}_{p_name}.npy")
        np.save(save_path, centroid)
        print(f"✅ Saved Profile: {save_path}")
    print("\nEnrollment Successful.")