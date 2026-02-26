# config.py
import os

# Base directory of the backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_INDEX = 2
ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'edgeface_xs_gamma_06.onnx')
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'blaze_face_short_range.tflite')
LANDMARKER_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'face_landmarker.task')
DB_PATH = os.path.join(BASE_DIR, 'data', 'face_db')

THRESHOLD = 0.65             
STABILITY_FRAMES = 8          
BUFFER_SIZE = 12              
SMOOTHING_FACTOR = 0.20       
PADDING = 0.25  

BLUR_THRESHOLD = 80
LIGHTING_RANGE = (70, 210)
SAMPLES_PER_PHASE = 25

# Liveness detection (blink challenge)
EAR_THRESHOLD       = 0.22   # EAR below this = eye closed
BLINK_CONSEC_FRAMES = 2      # frames eye must be closed to count as a blink
LIVENESS_TIMEOUT    = 150    # frames before challenge resets (~10s at 15fps)