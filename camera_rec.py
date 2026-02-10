import cv2

def find_cameras():
    print("--- Scanning for Camera Indices ---")
    available_indices = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get the resolution to identify if it's the "High Quality" one
            w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Index {i}: [ACTIVE] - Default Resolution: {w}x{h}")
            available_indices.append(i)
            cap.release()
        else:
            cap.release()
    
    if not available_indices:
        print("No cameras found. Check USB connection.")
    return available_indices

if __name__ == "__main__":
    indices = find_cameras()
    if indices:
        print(f"\nRecommended: Try the highest index first (likely your USB webam): {indices[-1]}")