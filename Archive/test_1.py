import cv2
from ultralytics import YOLO
import time

model = YOLO('yolov8n.pt').to('cpu') 
cap = cv2.VideoCapture(2)


stability_counter = 0
STABILITY_THRESHOLD = 10 # Number of frames to wait before Auth

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.resize(frame, (640, 480))

    results = model(frame, classes=[0], conf=0.5, verbose=False)
    
    for r in results:
        print(r.boxes.cls)
        num_people = len(r.boxes)
        
        if num_people > 1:
            stability_counter = 0 # Reset if multiple people appear
            msg, color = "ACCESS DENIED: Multiple Users", (0, 0, 255)
        elif num_people == 1:
            stability_counter += 1
            if stability_counter >= STABILITY_THRESHOLD:
                msg, color = "STABLE: Triggering Face Recognition...", (0, 255, 0)
                # call: authenticate_user(frame)
            else:
                msg, color = f"Hold still... {stability_counter}/{STABILITY_THRESHOLD}", (0, 255, 255)
        else:
            stability_counter = 0
            msg, color = "No user detected", (255, 255, 255)

        cv2.putText(frame, msg, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Company Gatekeeper Project", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()