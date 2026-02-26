from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import video, enrollment, session
from backend.services.camera_services import load_resources

# Load AI models and face database on startup
load_resources()

app = FastAPI(title="EdgeFace Modular Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(video.router)
app.include_router(enrollment.router)
app.include_router(session.router)


@app.get("/")
def health_check():
    return {"status": "online", "architecture": "websocket-modular"}
