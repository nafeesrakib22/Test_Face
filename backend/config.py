# config.py
import os

# Base directory of the backend folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

CAMERA_INDEX = 2
ONNX_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'edgeface_xs_gamma_06.onnx')
DETECTOR_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'blaze_face_short_range.tflite')
DB_PATH = os.path.join(BASE_DIR, 'data', 'face_db')

THRESHOLD = 0.65             
STABILITY_FRAMES = 8          
BUFFER_SIZE = 12              
SMOOTHING_FACTOR = 0.20       
PADDING = 0.25  

BLUR_THRESHOLD = 80
LIGHTING_RANGE = (70, 210)
SAMPLES_PER_PHASE = 25