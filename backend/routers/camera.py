# camera.py
# This router is no longer used in the WebSocket architecture.
# Camera access is now handled by the client browser via getUserMedia().
# The router file is kept here to avoid import errors if referenced elsewhere,
# but it registers no routes.

from fastapi import APIRouter

router = APIRouter()
