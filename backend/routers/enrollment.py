import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from backend.services.camera_services import (
    process_enrollment_frame,
    reset_enrollment_state,
    enrollment_status,
)

router = APIRouter()


@router.websocket("/ws/enroll")
async def ws_enroll(websocket: WebSocket, name: str = Query(..., min_length=1)):
    """
    WebSocket endpoint for guided 5-phase face enrollment.

    Protocol:
      Client → Server : raw JPEG bytes (one frame per message)
      Server → Client : JSON text message with pose guidance + progress
    """
    await websocket.accept()
    reset_enrollment_state(name)

    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            result = process_enrollment_frame(jpeg_bytes)
            await websocket.send_text(json.dumps(result))

            # Close the connection gracefully once enrollment is complete
            if result.get("complete"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()


@router.get("/enroll_status")
def get_enroll_status():
    """Polling fallback — returns current enrollment progress."""
    return enrollment_status
