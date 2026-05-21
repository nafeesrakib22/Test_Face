---
name: facewatch
description: >
  Query the Face Recognition system to check when people were last seen
  or monitor for unknown person alerts.
---

## Instructions
The Face Recognition API is available at `http://localhost:8000`.

### Tool: Get Last Seen Data
**Endpoint:** `GET http://localhost:8000/users`
**Description:** Use this to answer questions like "When was [John] last seen?" or "Who is currently active?".
**Example Response:**
```json
{
  "users": [
    { "name": "Alice", "last_seen": "2024-03-30T10:00:00Z", "is_active": false }
  ]
}
```

### Heartbeat: Monitor Unknown Persons
Connect to the heartbeat mechanism to check for security alerts.
**Pulse Interval:** 30 seconds
**Step 1:** Call `GET http://localhost:8000/events/unknown`
**Step 2:** If the list is not empty, for each event:
   - Send a Telegram message: "⚠️ SECURITY ALERT: An unknown person has been detected for over 30 seconds!"
   - Call `POST http://localhost:8000/events/unknown/{event_id}/ack` to stop the alert.

### Real-time Monitoring
A background monitor is available to poll the system every minute for unidentified face detections.
**Script:** `scripts/face_monitor_bg.py`
**Description:** Polls `/events/unknown` and sends Telegram alerts if detection exceeds 10s.

### How to use this skill
- When asked "When was [Name] last seen?":
  1. Fetch `GET http://localhost:8000/users`.
  2. Find the user and report their `last_seen` time (convert to a friendly relative format like "10 minutes ago").
  3. If not found, report that the user is not enrolled.
- On each heartbeat:
  1. Check for unknown events and notify the user via Telegram if found.

