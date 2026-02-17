from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from backend.services.camera_services import generate_recognition_frames

router = APIRouter()

@router.get("/video_feed")
def video_feed():
    return StreamingResponse(
        generate_recognition_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )