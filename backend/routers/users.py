import os
from fastapi import APIRouter, HTTPException
from backend.services.camera_services import known_faces, load_resources
from backend import config

router = APIRouter(prefix="/users", tags=["users"])


@router.get("")
def list_users():
    """Return a sorted list of all enrolled user names."""
    return {"users": sorted(known_faces.keys())}


@router.delete("/{name}")
def delete_user(name: str):
    """
    Delete all .npy face profile files for the given user and
    reload the in-memory face database immediately.
    """
    if name not in known_faces:
        raise HTTPException(status_code=404, detail=f"User '{name}' not found.")

    # Remove every phase file belonging to this user
    removed = []
    if os.path.exists(config.DB_PATH):
        for filename in os.listdir(config.DB_PATH):
            if filename.startswith(f"{name}_") and filename.endswith(".npy"):
                os.remove(os.path.join(config.DB_PATH, filename))
                removed.append(filename)

    # Reload face DB into memory (and re-init models)
    load_resources()

    return {"status": "deleted", "user": name, "files_removed": removed}
