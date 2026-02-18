import cv2
from fastapi import APIRouter
from backend.services.camera_services import switch_camera
from backend import config

router = APIRouter()


@router.get("/cameras")
def list_cameras():
    """Scan indices 0-9 and return a list of available cameras."""
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cameras.append({
                "index": i,
                "label": f"Camera {i}  ({w}×{h})"
            })
            cap.release()
        else:
            cap.release()
    return {"cameras": cameras, "current": config.CAMERA_INDEX}


@router.get("/set_camera")
def set_camera(camera_index: int):
    """Switch the active VideoStream to the given camera index."""
    switch_camera(camera_index)
    return {"status": "ok", "camera_index": camera_index}
