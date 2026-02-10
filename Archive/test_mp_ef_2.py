import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import time
import os
from backbones import get_model
from torchvision import transforms
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import config

# --- CONFIGURATION ---
DB_PATH = 'data/face_db'
THRESHOLD = 0.50  # Similarity score needed to confirm identity

# 1. Initialize EdgeFace
device = torch.device('cpu')
model_name = "edgeface_xs_gamma_06"
model = get_model(model_name)
model.load_state_dict(torch.load(f'checkpoints/{model_name}.pt', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# 2. Initialize MediaPipe (Gatekeeper)
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# --- HELPER FUNCTIONS ---
def get_embedding(img_bgr):
    # Resize and normalize for EdgeFace
    img_resized = cv2.resize(img_bgr, (112, 112))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return F.normalize(embedding)

def load_database(path):
    print(f"Loading Face Database from {path}...")
    db = {}
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created empty folder: {path}. Please add photos.")
        return db

    for filename in os.listdir(path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            name = os.path.splitext(filename)[0] # "Moriarty.jpg" -> "Moriarty"
            img_path = os.path.join(path, filename)
            img = cv2.imread(img_path)
            if img is not None:
                db[name] = get_embedding(img)
                print(f"Loaded: {name}")
    print(f"Database Ready: {len(db)} users loaded.")
    return db

# --- MAIN EXECUTION ---
known_faces = load_database(DB_PATH)
cap = cv2.VideoCapture(config.CAMERA_INDEX)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Standardize Edge Resolution
    frame = cv2.resize(frame, (640, 480))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # 1. Detect Faces
    results = detector.detect(mp_image)
    
    if results.detections:
        num_faces = len(results.detections)
        
        # Gatekeeper Logic
        if num_faces > 1:
            msg, color = f"Access Denied: {num_faces} People", (0, 0, 255)
        else:
            # 2. Identify the Single User
            det = results.detections[0]
            b = det.bounding_box
            
            # Safe Crop
            h, w, _ = frame.shape
            x1 = max(0, b.origin_x)
            y1 = max(0, b.origin_y)
            x2 = min(w, b.origin_x + b.width)
            y2 = min(h, b.origin_y + b.height)
            
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                curr_emb = get_embedding(face_crop)
                
                best_score = -1.0
                best_name = "Unknown"
                
                # Compare against ALL loaded users
                for name, db_emb in known_faces.items():
                    score = torch.mm(curr_emb, db_emb.t()).item()
                    if score > best_score:
                        best_score = score
                        best_name = name
                
                # Check Threshold
                if best_score > THRESHOLD:
                    msg = f"Hello, {best_name} ({best_score:.2f})"
                    color = (0, 255, 0) # Green
                else:
                    msg = f"Unknown User ({best_score:.2f})"
                    color = (0, 0, 255) # Red
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        msg, color = "Searching...", (255, 255, 255)

    cv2.putText(frame, msg, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Multi-User Gatekeeper", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()