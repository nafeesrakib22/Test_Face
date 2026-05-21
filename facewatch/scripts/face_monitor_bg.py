#!/usr/bin/env python3
import urllib.request
import json
import time
import subprocess
import os

API_URL = "http://localhost:8000/events/unknown"
CHAT_ID = "510817528"
CHANNEL = "telegram"

def send_alert(msg):
    cmd = [
        "openclaw", "message", "send",
        "--channel", CHANNEL,
        "--target", CHAT_ID,
        "--message", msg
    ]
    subprocess.run(cmd)

def monitor():
    print("Face recognition monitor started (1 min interval)...")
    while True:
        try:
            with urllib.request.urlopen(API_URL) as response:
                data = json.loads(response.read())
                events = data.get("events", [])
                
                unacknowledged = [e for e in events if not e.get("acknowledged")]
                
                if unacknowledged:
                    alert_msg = f"⚠️ SECURITY ALERT: {len(unacknowledged)} unidentified face(s) detected for over 10 seconds!"
                    send_alert(alert_msg)
                    
                    # Acknowledge them
                    for event in unacknowledged:
                        eid = event["id"]
                        req = urllib.request.Request(f"{API_URL}/{eid}/ack", method="POST")
                        urllib.request.urlopen(req)
                        print(f"Acknowledged {eid}")
                        
        except Exception as e:
            print(f"Error in monitor: {e}")
            
        time.sleep(60)

if __name__ == "__main__":
    monitor()
