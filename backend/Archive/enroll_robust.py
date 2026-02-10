import cv2
import numpy as np
import os
import onnxruntime as ort
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- CONFIG ---
ONNX_MODEL_PATH = "edgeface_xs_gamma_06.onnx"
DB_PATH = 'data/face_db'

# 1. Setup ONNX Engine (Recognition)
sess_options = ort.SessionOptions()
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

# 2. Setup MediaPipe (Detection)
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 3. Helper: Get Embedding
def get_embedding_onnx(img_bgr):
    # 1. Standard Resize
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    # --- NEW: APPLY OVAL MASK ---
    # Create a black image the same size as the face
    mask = np.zeros((112, 112), dtype=np.uint8)
    # Draw a white filled ellipse in the center (ignoring corners)
    # Center: (56,56), Axes: (45, 58), Angle: 0
    cv2.ellipse(mask, (56, 56), (45, 58), 0, 0, 360, 255, -1)
    # Apply the mask: only pixels inside the ellipse stay, others go black
    img_resized = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    # ----------------------------

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    
    embeddings = ort_session.run(None, {input_name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)

# 4. Main Enrollment Logic
if not os.path.exists(DB_PATH): os.makedirs(DB_PATH)
name = input("Enter User Name: ").strip()
cap = cv2.VideoCapture(0) # Adjust index to 2 if needed

print(f"--- AUTO-ENROLLMENT FOR: {name} ---")
print("1. Look at the camera.")
print("2. Press 's' ONCE to start capturing.")
print("3. Move your head slightly (left/right/tilt) during capture for better accuracy.")

embeddings = []
capturing = False

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Resize for detection speed
    frame_small = cv2.resize(frame, (640, 480))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    
    results = detector.detect(mp_image)
    
    face_crop = None
    
    if results.detections:
        det = results.detections[0]
        b = det.bounding_box
        
        # Draw Green Box
        x1, y1 = max(0, b.origin_x), max(0, b.origin_y)
        x2, y2 = min(640, b.origin_x + b.width), min(480, b.origin_y + b.height)
        cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        face_crop = frame_small[y1:y2, x1:x2]

    # Status Text
    if not capturing:
        msg = "Press 's' to Start"
        color = (0, 255, 255)
    else:
        msg = f"Capturing: {len(embeddings)}/50"
        color = (0, 0, 255) # Red for recording
        
        # AUTO CAPTURE LOGIC
        if face_crop is not None and face_crop.size > 0:
            emb = get_embedding_onnx(face_crop)
            embeddings.append(emb)
            time.sleep(0.05) # Small delay to avoid duplicates
            
            if len(embeddings) >= 50:
                print("Capture Complete!")
                break
    
    cv2.putText(frame_small, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("Auto Enrollment", frame_small)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not capturing:
        capturing = True
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 5. Save the Centroid
if len(embeddings) == 50:
    centroid = np.mean(np.array(embeddings), axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    
    save_path = os.path.join(DB_PATH, f"{name}.npy")
    np.save(save_path, centroid)
    print(f"\n[SUCCESS] Robust Identity saved: {save_path}")
    print("This file contains the mathematical average of 50 angles of your face.")
else:
    print("\n[FAIL] Enrollment cancelled.")