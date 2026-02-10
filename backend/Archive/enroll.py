import cv2
import os
import re
import config 
# 1. Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Changing this to 'face_db' to match your multi-user structure
DB_DIR = os.path.join(BASE_DIR, 'data', 'face_db')

# Create 'face_db' folder if it doesn't exist
if not os.path.exists(DB_DIR):
    os.makedirs(DB_DIR)

# 2. Get User Name (Sanitized to be safe for filenames)
raw_name = input("Enter the name of the new user (e.g., 'John_Doe'): ").strip()
# Remove special characters to prevent file path errors
safe_name = re.sub(r'[^a-zA-Z0-9_\-]', '', raw_name)

if not safe_name:
    print("Error: Invalid name. Please try again.")
    exit()

print(f"--- Enrolling: {safe_name} ---")
print("Press 's' to capture and save.")
print("Press 'q' to quit.")

# 3. Start Camera (Index 2 for DroidCam/Phone), changed to 0 after system restart
cap = cv2.VideoCapture(config.CAMERA_INDEX)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame. Check DroidCam connection.")
        break
    
    # Mirror for display only (feels more natural)
    display_frame = cv2.flip(frame, 1)
    
    # Overlay instructions
    cv2.putText(display_frame, f"User: {safe_name}", (20, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.putText(display_frame, "Press 's' to Save", (20, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Enrollment Mode", display_frame)
    
    key = cv2.waitKey(3) & 0xFF
    if key == ord('s'):
        # Save the original frame (not mirrored)
        img_filename = f"{safe_name}.jpg"
        img_path = os.path.join(DB_DIR, img_filename)
        
        cv2.imwrite(img_path, frame)
        print(f"SUCCESS: Identity saved for '{safe_name}' at:")
        print(f" -> {img_path}")
        break
    elif key == ord('q'):
        print("Enrollment cancelled.")
        break

cap.release()
cv2.destroyAllWindows()