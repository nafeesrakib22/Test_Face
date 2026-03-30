import os
import json
from fastapi import APIRouter, HTTPException
from backend.services.camera_services import _UNKNOWN_EVENTS_PATH

router = APIRouter(prefix="/events", tags=["events"])


@router.get("/unknown")
def list_unknown_events():
    """List all unacknowledged unknown person events."""
    if not os.path.exists(_UNKNOWN_EVENTS_PATH):
        return {"events": []}
        
    try:
        with open(_UNKNOWN_EVENTS_PATH, 'r') as f:
            events = json.load(f)
        
        # Return only unacknowledged events
        unacked = [e for e in events if not e.get("acknowledged", False)]
        return {"events": unacked}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/unknown/{event_id}/ack")
def acknowledge_event(event_id: str):
    """Mark an unknown person event as acknowledged."""
    if not os.path.exists(_UNKNOWN_EVENTS_PATH):
        raise HTTPException(status_code=404, detail="No events found")
        
    try:
        with open(_UNKNOWN_EVENTS_PATH, 'r') as f:
            events = json.load(f)
            
        found = False
        for e in events:
            if e["id"] == event_id:
                e["acknowledged"] = True
                found = True
                break
                
        if not found:
            raise HTTPException(status_code=404, detail="Event ID not found")
            
        with open(_UNKNOWN_EVENTS_PATH, 'w') as f:
            json.dump(events, f, indent=2)
            
        return {"status": "acknowledged", "id": event_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
