import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.camera_services import process_recognition_frame

router = APIRouter()


@router.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    """
    WebSocket endpoint for real-time face recognition.

    Protocol:
      Client → Server : raw JPEG bytes (one frame per message)
      Server → Client : JSON text message with detection result
    """
    await websocket.accept()

    # Per-connection state (bounding-box smoothing + stability counter)
    state = {"prev_box": None, "stability": 0}

    try:
        while True:
            jpeg_bytes = await websocket.receive_bytes()
            result = process_recognition_frame(jpeg_bytes, state)
            await websocket.send_text(json.dumps(result))
    except WebSocketDisconnect:
        pass
