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
landmarker = None
enrollment_status = {"name": None, "progress": 0, "complete": False}
stabilizer = None

# ---------------------------------------------------------------------------
# Identity Stabilizer
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Blink Detector (per-connection state)
# ---------------------------------------------------------------------------
# FaceMesh landmark indices for Eye Aspect Ratio
# Right eye: [p1, p2, p3, p4, p5, p6]
_RIGHT_EYE = [33, 160, 158, 133, 153, 144]
# Left eye:  [p1, p2, p3, p4, p5, p6]
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]


def _dist(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def _compute_ear(landmarks) -> float:
    """Compute average Eye Aspect Ratio for both eyes."""
    def ear(indices):
        p = [landmarks[i] for i in indices]
        vertical   = _dist(p[1], p[5]) + _dist(p[2], p[4])
        horizontal = _dist(p[0], p[3])
        return vertical / (2.0 * horizontal) if horizontal > 1e-6 else 0.3
    return (ear(_RIGHT_EYE) + ear(_LEFT_EYE)) / 2.0


class BlinkDetector:
    """Detects a single deliberate blink from a stream of EAR values.

    Uses a short rolling average to filter out single-frame EAR drops
    caused by webcam focus changes or landmark jitter.
    """
    def __init__(self, ear_smooth_len: int = 3):
        self.ear_history  = deque(maxlen=ear_smooth_len)
        self.consec_below = 0   # consecutive smoothed frames with EAR < threshold
        self.blink_count  = 0

    def update(self, ear: float) -> bool:
        """Returns True the frame the blink completes (eye reopens)."""
        self.ear_history.append(ear)
        smoothed = sum(self.ear_history) / len(self.ear_history)

        if smoothed < config.EAR_THRESHOLD:
            self.consec_below += 1
        else:
            if self.consec_below >= config.BLINK_CONSEC_FRAMES:
                self.blink_count += 1
                self.consec_below = 0
                return True
            self.consec_below = 0
        return False


    def reset(self):
        self.ear_history.clear()
        self.consec_below = 0
        self.blink_count  = 0


# ---------------------------------------------------------------------------
# Startup: load all models + face DB
# ---------------------------------------------------------------------------
def load_resources():
    global ort_session, known_faces, detector, landmarker, stabilizer

    # ONNX face recognition model
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 2
    ort_session = ort.InferenceSession(
        config.ONNX_MODEL_PATH, sess_options, providers=['CPUExecutionProvider']
    )

    # Face database (.npy centroids)
    known_faces.clear()
    if os.path.exists(config.DB_PATH):
        for filename in os.listdir(config.DB_PATH):
            if filename.endswith('.npy'):
                name = filename.split('_')[0]
                vec  = np.load(os.path.join(config.DB_PATH, filename))
                if name not in known_faces:
                    known_faces[name] = []
                known_faces[name].append(vec)

    # BlazeFace detector
    base_options = python.BaseOptions(model_asset_path=config.DETECTOR_MODEL_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # FaceMesh landmarker (for blink liveness)
    lm_options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=config.LANDMARKER_MODEL_PATH),
        num_faces=1,
    )
    landmarker = vision.FaceLandmarker.create_from_options(lm_options)

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
    yaw   = (nose.x - sl_eye.x) / eye_dist if eye_dist > 0.01 else 0.5
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
# Frame Processing — Recognition (WebSocket per-connection)
# ---------------------------------------------------------------------------

def _decode_frame(jpeg_bytes: bytes):
    """Decode raw JPEG bytes into a BGR numpy array."""
    arr   = np.frombuffer(jpeg_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def make_recognition_state() -> dict:
    """Return a fresh per-connection state dict for recognition."""
    return {
        "prev_box":          None,
        "stability":         0,
        "liveness_confirmed": False,
        "blink_detector":    BlinkDetector(),
        "liveness_timeout":  0,
    }


def process_recognition_frame(jpeg_bytes: bytes, state: dict) -> dict:
    """
    Three-stage pipeline per frame:
      1. BlazeFace  — detect face, smooth box, count stability frames
      2. FaceLandmarker — blink liveness challenge (until confirmed)
      3. EdgeFace   — identity embedding + cosine matching

    `state` is a mutable dict kept alive per WebSocket connection.
    Initialise with make_recognition_state().
    """
    frame = _decode_frame(jpeg_bytes)
    if frame is None:
        return {"status": "error", "message": "Could not decode frame"}

    h, w   = frame.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # ── Stage 1: BlazeFace detection ─────────────────────────────────────────
    res = detector.detect(mp_img)

    if not res.detections:
        # Face lost — full reset
        state["stability"]          = 0
        state["prev_box"]           = None
        state["liveness_confirmed"] = False
        state["liveness_timeout"]   = 0
        state["blink_detector"].reset()
        stabilizer.reset()
        return {"status": "scanning", "name": None, "confidence": 0.0, "box": None}

    if len(res.detections) > 1:
        state["stability"] = 0
        state["prev_box"]  = None
        return {"status": "multiple", "name": None, "confidence": 0.0, "box": None}

    # Single face — smooth bounding box
    state["stability"] += 1
    det  = res.detections[0].bounding_box
    curr = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
    prev = state.get("prev_box")
    prev = curr if prev is None else (prev * 0.9) + (curr * 0.1)
    state["prev_box"] = prev

    sx, sy, sw, sh = prev
    cx, cy = sx + sw / 2, sy + sh / 2
    x1 = int(max(0, cx - sw * 0.625))
    y1 = int(max(0, cy - sh * 0.625))
    x2 = int(min(w, cx + sw * 0.625))
    y2 = int(min(h, cy + sh * 0.625))
    box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}

    # Check blur on the face crop (used across all stages for the warning)
    crop_check = frame[y1:y2, x1:x2]
    blurry = bool(crop_check.size > 0 and is_sharp(crop_check) < config.FRAME_BLUR_THRESHOLD)

    if state["stability"] < config.STABILITY_FRAMES:
        return {"status": "stabilizing", "name": None, "confidence": 0.0, "box": box, "blurry": blurry}

    # ── Stage 2: Blink liveness challenge ────────────────────────────────────
    if not state["liveness_confirmed"]:
        # Skip EAR processing on blurry frames — pause timeout, return warning
        if blurry:
            return {
                "status": "blink_challenge",
                "name": None, "confidence": 0.0, "box": box,
                "blurry": True,
                "instruction": "⚠️ Blurry frame — hold still",
            }

        lm_result = landmarker.detect(mp_img)

        if lm_result.face_landmarks:
            landmarks = lm_result.face_landmarks[0]
            ear       = _compute_ear(landmarks)
            blinked   = state["blink_detector"].update(ear)
            blinks_so_far = state["blink_detector"].blink_count

            if blinks_so_far >= config.REQUIRED_BLINKS:
                state["liveness_confirmed"] = True
                state["liveness_timeout"]   = 0
                # Fall through immediately to Stage 3 this frame
            else:
                state["liveness_timeout"] += 1
                if state["liveness_timeout"] >= config.LIVENESS_TIMEOUT:
                    # Timeout — reset and ask again
                    state["liveness_timeout"] = 0
                    state["blink_detector"].reset()
                    return {
                        "status": "blink_challenge",
                        "name": None, "confidence": 0.0, "box": box,
                        "blurry": False,
                        "instruction": "No blink detected. Please blink naturally.",
                    }
                remaining = config.REQUIRED_BLINKS - blinks_so_far
                eye_icons = "👁" * remaining
                instr = f"Blink {remaining} more time{'s' if remaining > 1 else ''} {eye_icons}"
                return {
                    "status": "blink_challenge",
                    "name": None, "confidence": 0.0, "box": box,
                    "blurry": False,
                    "instruction": instr,
                }
        else:
            # Landmarker found no landmarks — keep waiting
            return {
                "status": "blink_challenge",
                "name": None, "confidence": 0.0, "box": box,
                "blurry": False,
                "instruction": f"Blink {config.REQUIRED_BLINKS} times to verify",
            }

    # ── Stage 3: EdgeFace identity verification ───────────────────────────────
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

    raw     = best_n if best_s > config.THRESHOLD else "Unknown"
    display = stabilizer.update(raw)

    return {
        "status":     "identified" if display != "Unknown" else "unknown",
        "name":       display,
        "confidence": round(float(best_s), 4),
        "box":        box,
        "blurry":     blurry,
    }


# ---------------------------------------------------------------------------
# Enrollment (unchanged)
# ---------------------------------------------------------------------------

_enrollment_state: dict = {
    "name":      None,
    "phase_idx": 0,
    "embeddings": [],
    "prev_box":  None,
}
PHASES        = ["FRONTAL", "LEFT_PROFILE", "RIGHT_PROFILE", "LOOK_UP", "LOOK_DOWN"]
PHASE_NAMES   = ["frontal", "left", "right", "up", "down"]
SAMPLES_PER_PHASE = 40
TOTAL_SAMPLES = len(PHASES) * SAMPLES_PER_PHASE


def reset_enrollment_state(name: str):
    _enrollment_state.update({
        "name":      name,
        "phase_idx": 0,
        "embeddings": [],
        "prev_box":  None,
    })
    enrollment_status.update({"name": name, "progress": 0, "complete": False})


def process_enrollment_frame(jpeg_bytes: bytes) -> dict:
    frame = _decode_frame(jpeg_bytes)
    if frame is None:
        return {"status": "error", "message": "Could not decode frame"}

    state = _enrollment_state
    if state["name"] is None:
        return {"status": "error", "message": "Enrollment not started"}

    h, w  = frame.shape[:2]
    p_idx = state["phase_idx"]

    if p_idx >= len(PHASES):
        return {"status": "complete", "progress": 100, "phase": p_idx, "instruction": "Done!"}

    target = PHASES[p_idx]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = detector.detect(mp_img)

    if not res.detections:
        return {
            "status":      "searching",
            "phase":       p_idx + 1,
            "phase_name":  target,
            "progress":    enrollment_status["progress"],
            "instruction": "Searching for face...",
            "quality_ok":  False,
            "box":         None,
            "complete":    False,
        }

    det     = res.detections[0].bounding_box
    kps     = res.detections[0].keypoints
    cur_pose, instr = detect_pose(kps, target)

    curr = np.array([det.origin_x, det.origin_y, det.width, det.height], dtype=float)
    prev = state["prev_box"]
    prev = curr if prev is None else (prev * 0.9) + (curr * 0.1)
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
            progress   = int((total_embs / TOTAL_SAMPLES) * 100)
            enrollment_status["progress"] = progress

            if total_embs >= (p_idx + 1) * SAMPLES_PER_PHASE:
                state["phase_idx"] += 1
                p_idx = state["phase_idx"]

    progress = enrollment_status["progress"]

    if state["phase_idx"] >= len(PHASES) and len(state["embeddings"]) == TOTAL_SAMPLES:
        _save_enrollment(state["name"], state["embeddings"])
        load_resources()
        enrollment_status["complete"] = True
        return {
            "status":      "complete",
            "phase":       len(PHASES),
            "phase_name":  "DONE",
            "progress":    100,
            "instruction": "Enrollment complete! All 5 profiles saved.",
            "quality_ok":  True,
            "box":         box,
            "complete":    True,
        }

    return {
        "status":      "enrolling",
        "phase":       min(p_idx + 1, len(PHASES)),
        "phase_name":  PHASES[min(p_idx, len(PHASES) - 1)],
        "progress":    progress,
        "instruction": instr,
        "quality_ok":  quality_ok,
        "box":         box,
        "complete":    False,
    }


def _save_enrollment(name: str, embeddings: list):
    os.makedirs(config.DB_PATH, exist_ok=True)
    for i, p_name in enumerate(PHASE_NAMES):
        chunk    = np.array(embeddings[i * SAMPLES_PER_PHASE: (i + 1) * SAMPLES_PER_PHASE])
        centroid = np.mean(chunk, axis=0)
        centroid /= np.linalg.norm(centroid)
        np.save(os.path.join(config.DB_PATH, f"{name}_{p_name}.npy"), centroid)
