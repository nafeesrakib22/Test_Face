import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services.camera_services import process_recognition_frame, make_recognition_state

router = APIRouter()

# Single-worker executor keeps MediaPipe/ONNX calls serialised across
# all open connections (models are not thread-safe for concurrent calls).
_inference_executor = ThreadPoolExecutor(max_workers=1)


@router.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    """
    WebSocket endpoint for real-time face recognition.

    Protocol:
      Client → Server : raw JPEG bytes (one frame per message)
      Server → Client : JSON text message with detection result

    Architecture: a concurrent receiver task keeps latest_frame up to date
    while inference runs in a thread-pool executor.  The event loop is never
    blocked, so frames never queue up regardless of how long inference takes.
    """
    await websocket.accept()

    state        = make_recognition_state()
    loop         = asyncio.get_running_loop()
    latest_frame: bytes | None = None
    frame_ready  = asyncio.Event()

    async def receive_frames() -> None:
        """Continuously receive frames, always keeping only the newest."""
        nonlocal latest_frame
        try:
            while True:
                data         = await websocket.receive_bytes()
                latest_frame = data   # overwrite — only the newest frame matters
                frame_ready.set()
        except WebSocketDisconnect:
            latest_frame = None       # signal processor to exit cleanly
            frame_ready.set()

    recv_task = asyncio.create_task(receive_frames())

    try:
        while True:
            await frame_ready.wait()
            frame_ready.clear()

            frame        = latest_frame
            latest_frame = None

            if frame is None:
                break   # disconnected

            result = await loop.run_in_executor(
                _inference_executor,
                process_recognition_frame,
                frame,
                state,
            )
            try:
                await websocket.send_text(json.dumps(result))
            except Exception:
                break   # client disconnected between inference and send

    except WebSocketDisconnect:
        pass
    finally:
        recv_task.cancel()
