import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import os
from collections import deque, Counter
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from backend import config

# Global Engine States
known_faces = {}
ort_session = None
detector = None
enrollment_status = {"name": None, "progress": 0, "complete": False}
stabilizer = None

class IdentityStabilizer:
    def __init__(self, maxlen=12):
        self.history = deque(maxlen=maxlen)
        self.current_display_name = "Unknown"

    def update(self, name):
        self.history.append(name)
        counts = Counter(self.history)
        most_common, count = counts.most_common(1)[0]
        if count >= int(self.history.maxlen * 0.7):
            self.current_display_name = most_common
        return self.current_display_name

    def reset(self):
        self.history.clear()
        self.current_display_name = "Unknown"


def load_resources():
    global ort_session, known_faces, detector, stabilizer
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(
        config.ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider']
    )

    known_faces.clear()
    if os.path.exists(config.DB_PATH):
        for filename in os.listdir(config.DB_PATH):
            if filename.endswith('.npy'):
                name = filename.split('_')[0]
                vec = np.load(os.path.join(config.DB_PATH, filename))
                if name not in known_faces:
                    known_faces[name] = []
                known_faces[name].append(vec)

    base_options = python.BaseOptions(model_asset_path=config.DETECTOR_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    stabilizer = IdentityStabilizer(maxlen=config.BUFFER_SIZE)


# ---------------------------------------------------------------------------
# AI Helper Functions
# ---------------------------------------------------------------------------

def is_well_lit(image):
    if image.size == 0:
        return False
    small = cv2.resize(image, (32, 32))
    return config.LIGHTING_RANGE[0] < np.mean(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)) < config.LIGHTING_RANGE[1]


def is_sharp(image):
    if image.size == 0:
        return 0
    small = cv2.resize(image, (100, 100))
    return cv2.Laplacian(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def get_embedding_onnx(img_bgr):
    img = cv2.resize(img_bgr, (112, 112))
    mask = np.zeros((112, 112), dtype=np.uint8)
    cv2.ellipse(mask, (56, 56), (47, 58), 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)
    img = cv2.bitwise_and(img, img, mask=mask)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5
    img = np.transpose(img, (2, 0, 1))[np.newaxis, :]
    emb = ort_session.run(None, {ort_session.get_inputs()[0].name: img})[0]
    return emb / np.linalg.norm(emb)


def detect_pose(keypoints, target):
    l_eye, r_eye, nose, mouth = keypoints[0], keypoints[1], keypoints[2], keypoints[3]
    sl_eye = l_eye if l_eye.x < r_eye.x else r_eye
    sr_eye = r_eye if l_eye.x < r_eye.x else l_eye
    eye_dist = abs(sr_eye.x - sl_eye.x)
    yaw = (nose.x - sl_eye.x) / eye_dist if eye_dist > 0.01 else 0.5
    v_dist = abs(mouth.y - (sl_eye.y + sr_eye.y) / 2)
    pitch = abs(nose.y - (sl_eye.y + sr_eye.y) / 2) / v_dist if v_dist > 0.01 else 0.5

    if target == "FRONTAL":
        return ("FRONTAL", "Perfect. Look straight.") if 0.42 <= yaw <= 0.58 and 0.40 <= pitch <= 0.60 else ("TRANS", "Look back to center.")
    elif target == "LEFT_PROFILE":
        if yaw > 0.82:
            return "OVER", "TOO FAR! Turn back right."
        return ("LEFT_PROFILE", "Good, keep turning LEFT.") if yaw > 0.58 else ("TRANS", "Turn slowly left...")
    elif target == "RIGHT_PROFILE":
        if yaw < 0.18:
            return "OVER", "TOO FAR! Turn back left."
        return ("RIGHT_PROFILE", "Good, keep turning RIGHT.") if yaw < 0.42 else ("TRANS", "Turn slowly right...")
    elif target == "LOOK_UP":
        if pitch < 0.20:
            return "OVER", "TOO FAR! Chin down."
        return ("LOOK_UP", "Good, tilt chin UP.") if pitch < 0.40 else ("TRANS", "Tilt chin up...")
    elif target == "LOOK_DOWN":
        if pitch > 0.85:
            return "OVER", "TOO FAR! Chin up."
        return ("LOOK_DOWN", "Good, tilt chin DOWN.") if pitch > 0.60 else ("TRANS", "Tilt chin down...")
    return "UNK", "Adjusting..."


# ---------------------------------------------------------------------------
# Frame Processing Functions (WebSocket-based, client sends JPEG bytes)
# ---------------------------------------------------------------------------

# Per-connection state for recognition (stored externally or passed in)
def _decode_frame(jpeg_bytes: bytes):
    """Decode raw JPEG bytes into a BGR numpy array."""
    arr = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def process_recognition_frame(jpeg_bytes: bytes, state: dict) -> dict:
    """
    Receive JPEG bytes from the client, run face detection + recognition,
    return a JSON-serialisable dict with detection results.

    `state` is a mutable dict kept alive per WebSocket connection:
        { "prev_box": None, "stability": 0 }
    """
    frame = _decode_frame(jpeg_bytes)
    if frame is None:
        return {"status": "error", "message": "Could not decode frame"}

    h, w = frame.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_image)

    if not res.detections:
        state["stability"] = 0
        state["prev_box"] = None
        stabilizer.reset()
        return {"status": "scanning", "name": None, "confidence": 0.0, "box": None}

    if len(res.detections) > 1:
        state["stability"] = 0
        state["prev_box"] = None
        return {"status": "multiple", "name": None, "confidence": 0.0, "box": None}

    # Single face detected
    state["stability"] += 1
    det = res.detections[0].bounding_box
    curr = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
    prev = state.get("prev_box")
    prev = curr if prev is None else (prev * 0.8) + (curr * 0.2)
    state["prev_box"] = prev

    sx, sy, sw, sh = prev
    cx, cy = sx + sw / 2, sy + sh / 2
    x1 = int(max(0, cx - sw * 0.625))
    y1 = int(max(0, cy - sh * 0.625))
    x2 = int(min(w, cx + sw * 0.625))
    y2 = int(min(h, cy + sh * 0.625))
    box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    if state["stability"] < config.STABILITY_FRAMES:
        return {"status": "stabilizing", "name": None, "confidence": 0.0, "box": box}

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return {"status": "scanning", "name": None, "confidence": 0.0, "box": box}

    emb = get_embedding_onnx(crop)
    best_s, best_n = -1.0, "Unknown"
    for n, p_list in known_faces.items():
        for p in p_list:
            s = np.dot(emb.flatten(), p.flatten()).item()
            if s > best_s:
                best_s, best_n = s, n

    raw = best_n if best_s > config.THRESHOLD else "Unknown"
    display = stabilizer.update(raw)

    return {
        "status": "identified" if display != "Unknown" else "unknown",
        "name": display,
        "confidence": round(float(best_s), 4),
        "box": box,
    }


# Per-enrollment-session state
_enrollment_state: dict = {
    "name": None,
    "phase_idx": 0,
    "embeddings": [],
    "prev_box": None,
}
PHASES = ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE", "LOOK_UP", "LOOK_DOWN"]
PHASE_NAMES = ["frontal", "left", "right", "up", "down"]
SAMPLES_PER_PHASE = 25
TOTAL_SAMPLES = len(PHASES) * SAMPLES_PER_PHASE


def reset_enrollment_state(name: str):
    _enrollment_state.update({
        "name": name,
        "phase_idx": 0,
        "embeddings": [],
        "prev_box": None,
    })
    enrollment_status.update({"name": name, "progress": 0, "complete": False})


def process_enrollment_frame(jpeg_bytes: bytes) -> dict:
    """
    Receive JPEG bytes during enrollment, guide the user through 5 poses,
    accumulate embeddings, save centroids when done.
    Returns a JSON-serialisable dict.
    """
    frame = _decode_frame(jpeg_bytes)
    if frame is None:
        return {"status": "error", "message": "Could not decode frame"}

    state = _enrollment_state
    if state["name"] is None:
        return {"status": "error", "message": "Enrollment not started"}

    h, w = frame.shape[:2]
    p_idx = state["phase_idx"]

    if p_idx >= len(PHASES):
        return {"status": "complete", "progress": 100, "phase": p_idx, "instruction": "Done!"}

    target = PHASES[p_idx]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_image)

    if not res.detections:
        return {
            "status": "searching",
            "phase": p_idx + 1,
            "phase_name": target,
            "progress": enrollment_status["progress"],
            "instruction": "Searching for face...",
            "quality_ok": False,
            "box": None,
            "complete": False,
        }

    det = res.detections[0].bounding_box
    kps = res.detections[0].keypoints
    cur_pose, instr = detect_pose(kps, target)

    curr = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
    prev = state["prev_box"]
    prev = curr if prev is None else (prev * 0.8) + (curr * 0.2)
    state["prev_box"] = prev

    sx, sy, sw, sh = prev
    cx, cy = sx + sw / 2, sy + sh / 2
    x1 = int(max(0, cx - sw * 0.625))
    y1 = int(max(0, cy - sh * 0.625))
    x2 = int(min(w, cx + sw * 0.625))
    y2 = int(min(h, cy + sh * 0.625))
    box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    quality_ok = False
    if cur_pose == target:
        crop = frame[y1:y2, x1:x2]
        if crop.size > 0 and is_well_lit(crop) and is_sharp(crop) > config.BLUR_THRESHOLD:
            quality_ok = True
            state["embeddings"].append(get_embedding_onnx(crop))
            total_embs = len(state["embeddings"])
            progress = int((total_embs / TOTAL_SAMPLES) * 100)
            enrollment_status["progress"] = progress

            if total_embs >= (p_idx + 1) * SAMPLES_PER_PHASE:
                state["phase_idx"] += 1
                p_idx = state["phase_idx"]

    progress = enrollment_status["progress"]

    # Check if all phases are done
    if state["phase_idx"] >= len(PHASES) and len(state["embeddings"]) == TOTAL_SAMPLES:
        _save_enrollment(state["name"], state["embeddings"])
        load_resources()
        enrollment_status["complete"] = True
        return {
            "status": "complete",
            "phase": len(PHASES),
            "phase_name": "DONE",
            "progress": 100,
            "instruction": "Enrollment complete! All 5 profiles saved.",
            "quality_ok": True,
            "box": box,
            "complete": True,
        }

    return {
        "status": "enrolling",
        "phase": min(p_idx + 1, len(PHASES)),
        "phase_name": PHASES[min(p_idx, len(PHASES) - 1)],
        "progress": progress,
        "instruction": instr,
        "quality_ok": quality_ok,
        "box": box,
        "complete": False,
    }


def _save_enrollment(name: str, embeddings: list):
    os.makedirs(config.DB_PATH, exist_ok=True)
    for i, p_name in enumerate(PHASE_NAMES):
        chunk = np.array(embeddings[i * SAMPLES_PER_PHASE: (i + 1) * SAMPLES_PER_PHASE])
        centroid = np.mean(chunk, axis=0)
        centroid /= np.linalg.norm(centroid)
        np.save(os.path.join(config.DB_PATH, f"{name}_{p_name}.npy"), centroid)
