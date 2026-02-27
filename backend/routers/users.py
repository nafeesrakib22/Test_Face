import os
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException
from backend.services.camera_services import known_faces, last_seen, _reload_face_db, _save_last_seen
from backend import config

router = APIRouter(prefix="/users", tags=["users"])


@router.get("")
def list_users():
    """
    Return enrolled users with activity stats.

    Each entry contains:
      name             – user identifier
      enrolled_at      – ISO-8601 UTC timestamp from first .npy file mtime
      last_seen        – ISO-8601 UTC timestamp of last confident identification
                         (null if not seen since server start)
      recognition_count – total identifications since server start
    """
    result = []
    for name in sorted(known_faces.keys()):
        # Enrolled date: mtime of the first matching .npy file
        enrolled_at = None
        if os.path.exists(config.DB_PATH):
            for filename in sorted(os.listdir(config.DB_PATH)):
                if filename.startswith(f"{name}_") and filename.endswith(".npy"):
                    mtime = os.path.getmtime(os.path.join(config.DB_PATH, filename))
                    enrolled_at = datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat()
                    break

        result.append({
            "name":        name,
            "enrolled_at": enrolled_at,
            "last_seen":   last_seen.get(name),
        })

    return {"users": result}


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

    # Clear in-memory last-seen for this user
    last_seen.pop(name, None)
    _save_last_seen()

    # Reload face DB in-memory — model sessions are NOT recreated
    _reload_face_db()

    return {"status": "deleted", "user": name, "files_removed": removed}
