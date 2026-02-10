import numpy as np
import os

# Paths to the enrolled files you want to compare
user_1_path = 'data/face_db/Nafees_frontal.npy'
user_2_path = 'data/face_db/Tasdik_frontal.npy'

if os.path.exists(user_1_path) and os.path.exists(user_2_path):
    vec_n = np.load(user_1_path)
    vec_s = np.load(user_2_path)
    
    # Calculate Cosine Similarity
    similarity = np.dot(vec_n, vec_s.T).item()
    
    print(f"\n--- IDENTITY SEPARATION REPORT ---")
    print(f"Similarity Score: {similarity:.4f}")
    
    if similarity > 0.75:
        print("❌ CRITICAL OVERLAP: The model sees you as nearly identical.")
        print("Action: Try a higher-resolution backbone or change camera angle.")
    elif similarity > 0.60:
        print("⚠️  MODERATE OVERLAP: You share similar features.")
        print("Action: Ensure your Benchmark Threshold is set to 0.70+.")
    else:
        print("✅ CLEAR SEPARATION: The identities are mathematically distinct.")
else:
    print("Files not found. Please enroll both users first.")