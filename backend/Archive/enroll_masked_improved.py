import cv2
import numpy as np
import os
import onnxruntime as ort
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config
# --- ENHANCED CONFIG ---
ONNX_MODEL_PATH = "edgeface_xs_gamma_06.onnx"
DB_PATH = 'data/face_db'
NUM_SAMPLES = 75              # Increased from 50
BLUR_THRESHOLD = 100          # Reject blurry frames
MIN_FACE_SIZE = 80            # Minimum face dimension
LIGHTING_RANGE = (60, 200)    # Acceptable brightness range

# 1. Setup ONNX Engine
sess_options = ort.SessionOptions()
ort_session = ort.InferenceSession(ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider'])
input_name = ort_session.get_inputs()[0].name

# 2. Setup MediaPipe
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 3. Quality Check Helpers
def is_sharp(image):
    """Check if image is sharp enough (not blurry)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def is_well_lit(image):
    """Check if image has good lighting"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return LIGHTING_RANGE[0] < mean_brightness < LIGHTING_RANGE[1]

# --- ADDED HELPERS & UPDATED EMBEDDING FUNCTION ---

def get_embedding_onnx(img_bgr):
    # 1. Resize
    img_resized = cv2.resize(img_bgr, (112, 112))
    
    # 2. Skip CLAHE 
    #  Elliptical Mask
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (45, 58), 0, 0, 360, 255, -1)
    img_masked = cv2.bitwise_and(img_resized, img_resized, mask=mask)
    cv2.imshow("What the AI Sees", img_masked)
    
    # 3. Standard normalization
    img_rgb = cv2.cvtColor(img_masked, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    img_norm = (img_float - 0.5) / 0.5
    img_input = np.transpose(img_norm, (2, 0, 1))
    img_input = np.expand_dims(img_input, axis=0).astype(np.float32)
    
    embeddings = ort_session.run(None, {input_name: img_input})[0]
    return embeddings / np.linalg.norm(embeddings)


# 4. Main Enrollment Logic
if not os.path.exists(DB_PATH): 
    os.makedirs(DB_PATH)

print("\n" + "="*70)
print("     ENHANCED ENROLLMENT: MASKING + QUALITY FILTERING")
print("="*70)
name = input("Enter User Name: ").strip()

if not name:
    print("❌ Error: Name cannot be empty!")
    exit(1)

# Check if user already exists
existing_file = os.path.join(DB_PATH, f"{name}.npy")
if os.path.exists(existing_file):
    print(f"\n⚠️  WARNING: {name} already enrolled!")
    response = input("Delete old enrollment and re-enroll? (yes/no): ").strip().lower()
    if response == 'yes':
        os.remove(existing_file)
        print(f"✓ Deleted old enrollment for {name}")
    else:
        print("❌ Enrollment cancelled")
        exit(0)

cap = cv2.VideoCapture(config.CAMERA_INDEX) 
# Force MJPEG codec for high-speed USB transfer
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print(f"\n📋 ENROLLMENT GUIDE FOR: {name}")
print("="*70)
print("BENEFITS OF THIS SYSTEM:")
print("  ✓ Elliptical masking = Background-independent recognition")
print("  ✓ Quality filtering = Only sharp, well-lit frames accepted")
print("  ✓ 75 samples = Better coverage of angles and expressions")
print("\nSTEPS:")
print("  1. Press 's' to start enrollment")
print("  2. During capture, slowly move your head:")
print("     • Frames 0-25: Face FORWARD")
print("     • Frames 25-45: Turn LEFT slowly")
print("     • Frames 45-65: Turn RIGHT slowly")
print("     • Frames 65-75: Look UP/DOWN slightly")
print("\n⚠️  CRITICAL REQUIREMENTS:")
print("  • Good, even lighting (no harsh shadows)")
print("  • Stay 0.5m - 1.5m from camera")
print("  • Move slowly and smoothly")
print("  • System auto-rejects blurry/dark frames")
print("="*70)

embeddings = []
capturing = False
rejected_frames = {'blur': 0, 'lighting': 0, 'size': 0}
total_attempts = 0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_small = cv2.resize(frame, (640, 480))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, 
                        data=cv2.cvtColor(frame_small, cv2.COLOR_BGR2RGB))
    results = detector.detect(mp_image)
    
    face_crop = None
    quality_status = ""
    status_color = (255, 255, 255)
    
    if results.detections:
        det = results.detections[0]
        b = det.bounding_box
        
        x1, y1 = max(0, b.origin_x), max(0, b.origin_y)
        x2, y2 = min(640, b.origin_x + b.width), min(480, b.origin_y + b.height)
        
        face_width = x2 - x1
        face_height = y2 - y1
        
        # Check minimum face size
        if face_width < MIN_FACE_SIZE or face_height < MIN_FACE_SIZE:
            quality_status = "⚠ Too far! Move closer"
            status_color = (0, 165, 255)
            cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 165, 255), 2)
            if capturing:
                rejected_frames['size'] += 1
                total_attempts += 1
        else:
            face_crop = frame_small[y1:y2, x1:x2]
            
            if capturing and face_crop.size > 0:
                total_attempts += 1
                
                # Quality Checks
                sharpness = is_sharp(face_crop)
                well_lit = is_well_lit(face_crop)
                
                if sharpness < BLUR_THRESHOLD:
                    quality_status = f"⚠ Blurry ({sharpness:.0f})"
                    status_color = (0, 165, 255)
                    rejected_frames['blur'] += 1
                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 165, 255), 2)
                elif not well_lit:
                    quality_status = "⚠ Poor lighting"
                    status_color = (0, 165, 255)
                    rejected_frames['lighting'] += 1
                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 165, 255), 2)
                else:
                    # ✓ GOOD QUALITY - CAPTURE!
                    emb = get_embedding_onnx(face_crop)
                    embeddings.append(emb)
                    quality_status = f"✓ Captured ({sharpness:.0f})"
                    status_color = (0, 255, 0)
                    cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    time.sleep(0.05)
                    
                    if len(embeddings) >= NUM_SAMPLES:
                        print("\n✓ Capture Complete!")
                        break
            else:
                cv2.rectangle(frame_small, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Status Display
    if not capturing:
        msg = "Press 's' to Start Enrollment"
        color = (0, 255, 255)
    else:
        progress = len(embeddings)
        msg = f"Capturing: {progress}/{NUM_SAMPLES}"
        color = (0, 0, 255)
        
        # Progress guidance
        if progress < 25:
            guide = "Look FORWARD at camera"
        elif progress < 45:
            guide = "Slowly turn LEFT"
        elif progress < 65:
            guide = "Slowly turn RIGHT"
        else:
            guide = "Look UP & DOWN slightly"
        
        cv2.putText(frame_small, guide, (20, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if quality_status:
            cv2.putText(frame_small, quality_status, (20, 120), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Acceptance rate
        if total_attempts > 0:
            accept_rate = (len(embeddings) / total_attempts) * 100
            cv2.putText(frame_small, f"Accept: {accept_rate:.0f}%", 
                       (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame_small, msg, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Progress bar
    if capturing:
        bar_width = 400
        bar_height = 20
        bar_x = (640 - bar_width) // 2
        bar_y = 450
        filled_width = int((progress / NUM_SAMPLES) * bar_width)
        
        cv2.rectangle(frame_small, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (255, 255, 255), 2)
        cv2.rectangle(frame_small, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                     (0, 255, 0), -1)
    
    cv2.imshow("Enhanced Enrollment (Masked)", frame_small)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not capturing:
        capturing = True
        print("\n📸 Starting capture with quality filtering...")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 5. Save the Robust Centroid
if len(embeddings) >= NUM_SAMPLES:
    print("\n" + "="*70)
    print("PROCESSING ENROLLMENT DATA")
    print("="*70)
    
    # Calculate centroid
    centroid = np.mean(np.array(embeddings), axis=0)
    centroid = centroid / np.linalg.norm(centroid)
    
    # Save
    save_path = os.path.join(DB_PATH, f"{name}.npy")
    np.save(save_path, centroid)
    
    total_rejected = sum(rejected_frames.values())
    acceptance_rate = (len(embeddings) / (len(embeddings) + total_rejected)) * 100 if total_rejected > 0 else 100
    
    print(f"✓ Quality samples captured: {len(embeddings)}")
    print(f"  - Rejected (blur): {rejected_frames['blur']}")
    print(f"  - Rejected (lighting): {rejected_frames['lighting']}")
    print(f"  - Rejected (size): {rejected_frames['size']}")
    print(f"✓ Overall acceptance rate: {acceptance_rate:.1f}%")
    print(f"\n✅ SUCCESS: Background-independent identity saved to:")
    print(f"   {save_path}")
    print(f"\n💡 KEY FEATURES:")
    print(f"   • Elliptical masking = Works in ANY location/background")
    print(f"   • {NUM_SAMPLES} quality samples = Robust to angles/lighting")
    print(f"   • Pure facial features = No environmental contamination")
    print("="*70)
    
    # Suggest next steps
    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Enroll other users with same process")
    print(f"   2. Run benchmark_masked_improved.py to test")
    print(f"   3. Try sitting in different locations to verify background independence!")
    
else:
    print("\n❌ ENROLLMENT CANCELLED")
    print(f"   Only {len(embeddings)}/{NUM_SAMPLES} samples captured")
    print("   Tip: Ensure good lighting and minimize motion blur")