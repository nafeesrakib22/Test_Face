from fastapi import APIRouter
from backend.services.camera_services import enrollment_status

router = APIRouter()

@router.get("/reset_session")
def reset_session():
    # Update in-place to maintain the reference across the package
    enrollment_status.update({"name": None, "progress": 0, "complete": False})
    return {"status": "Reset Successful"}