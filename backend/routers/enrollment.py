from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
from backend.services.camera_services import generate_enrollment_frames, enrollment_status

router = APIRouter()

@router.get("/enroll")
def enroll_user(name: str = Query(..., min_length=1)):
    return StreamingResponse(
        generate_enrollment_frames(name), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@router.get("/enroll_status")
def get_enroll_status():
    return enrollment_status