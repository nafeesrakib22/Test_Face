from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import video, enrollment, session, camera
from backend import lifecycle

# Initialize hardware and models
lifecycle.startup()

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
app.include_router(camera.router)

@app.get("/")
def health_check():
    return {"status": "online", "architecture": "modular"}