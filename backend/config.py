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
STABILITY_FRAMES = 10
BUFFER_SIZE = 15              
SMOOTHING_FACTOR = 0.20       
PADDING = 0.25  

BLUR_THRESHOLD       = 80    # enrollment quality gate
FRAME_BLUR_THRESHOLD = 40    # live feed warning (lower — less strict than enrollment)
LIGHTING_RANGE = (70, 210)
SAMPLES_PER_PHASE = 40

# Liveness detection (blink challenge)
EAR_THRESHOLD       = 0.22   # EAR below this = eye closed
BLINK_CONSEC_FRAMES = 2      # frames eye must be closed to count as a blink
LIVENESS_TIMEOUT    = 225    # frames before challenge resets (~15s at 15fps)

# CPU throttling — run heavy inference every Nth post-liveness frame.
# Skipped frames return the cached previous result instantly.
# 3 = ~5fps effective inference, which is plenty for stable recognition.
INFERENCE_SKIP = 3

REQUIRED_BLINKS     = 3      # blinks needed to pass liveness

# Face alignment — ArcFace standard eye target positions (112×112 output)
ALIGN_LEFT_EYE  = (38.2946, 51.6963)
ALIGN_RIGHT_EYE = (73.5318, 51.6963)