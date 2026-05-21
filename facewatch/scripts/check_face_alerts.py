#!/usr/bin/env python3
import urllib.request
import json
import sys

API_URL = "http://localhost:8000/events/unknown"

def check_alerts():
    try:
        with urllib.request.urlopen(API_URL) as response:
            data = json.loads(response.read())
            events = data.get("events", [])
            
            unacknowledged = [e for e in events if not e.get("acknowledged")]
            
            if not unacknowledged:
                return None

            alert_msg = "⚠️ SECURITY ALERT: Unidentified face detected for over 10 seconds!"
            
            # Acknowledge them so we don't alert again
            for event in unacknowledged:
                eid = event["id"]
                req = urllib.request.Request(f"{API_URL}/{eid}/ack", method="POST")
                urllib.request.urlopen(req)
            
            return alert_msg
            
    except Exception as e:
        # Silently fail for heartbeat logs if server is down, 
        # but maybe print to stderr for debugging
        print(f"Error checking face alerts: {e}", file=sys.stderr)
        return None

if __name__ == "__main__":
    msg = check_alerts()
    if msg:
        print(msg)
    else:
        print("HEARTBEAT_OK")
