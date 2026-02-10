import cv2
import torch
import torch.nn.functional as F
import mediapipe as mp
import numpy as np
import time
from backbones import get_model
from torchvision import transforms
import config 

# 1. Initialize EdgeFace (Recognition)
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
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
base_options = python.BaseOptions(model_asset_path='blaze_face_short_range.tflite')
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 3. Enrollment (The 'Master' Identity)
def get_embedding(img_bgr):
    img_resized = cv2.resize(img_bgr, (112, 112))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(tensor)
    return F.normalize(embedding)

print("Loading Master Identity...")
master_img = cv2.imread('data/master_identity.jpg')
master_emb = get_embedding(master_img)

cap = cv2.VideoCapture(config.CAMERA_INDEX)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    frame = cv2.resize(frame, (640, 480))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    start_time = time.perf_counter()
    results = detector.detect(mp_image)
    
    if results.detections:
        num_faces = len(results.detections)
        
        if num_faces > 1:
            msg, color = f"ALERT: {num_faces} Users Detected", (0, 0, 255)
        else:
            # Single user: Extract and Recognize
            det = results.detections[0]
            b = det.bounding_box
            # Crop with safety margins
            y1, y2 = max(0, b.origin_y), min(480, b.origin_y + b.height)
            x1, x2 = max(0, b.origin_x), min(640, b.origin_x + b.width)
            face_crop = frame[y1:y2, x1:x2]
            
            if face_crop.size > 0:
                curr_emb = get_embedding(face_crop)
                # Cosine Similarity
                similarity = torch.mm(master_emb, curr_emb.t()).item()
                
                if similarity > 0.6: # Standard threshold for EdgeFace is 0.5
                    msg, color = f"Verified: {similarity:.2f}", (0, 255, 0)
                else:
                    msg, color = f"Unknown: {similarity:.2f}", (0, 0, 255)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    else:
        msg, color = "Searching for face...", (255, 255, 255)

    inf_time = (time.perf_counter() - start_time) * 1000
    cv2.putText(frame, f"{msg} ({inf_time:.1f}ms)", (10, 40), 2, 0.7, color, 2)
    cv2.imshow("EdgeFace Gatekeeper Benchmark", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()