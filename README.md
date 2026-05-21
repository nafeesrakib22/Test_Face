# 🧠 Edge-Optimized Face Recognition System

A high-performance, background-agnostic face recognition pipeline optimized for **Edge devices**. This system utilizes the **EdgeFace-XS** architecture and **MediaPipe Tasks API** to provide robust identity verification with minimal resource overhead.

---

## 🚀 Key Technical Features

- **Elliptical Masking:** Mitigates background noise and environmental bias by focusing the model exclusively on the facial manifold.
- **Multi-Profile Identity (Pose Manifold Clustering):** Captures Frontal, Left, and Right pose-centroids to ensure accuracy remains high even during head movement.
- **Identity Stabilization:** Utilizes a 12-frame hysteresis buffer and temporal smoothing to prevent identity "flickering."
- **Single-Face Enforcement:** Security logic that denies access and provides visual warnings if multiple faces are detected in the frame.
- **Box Smoothing (Lerp):** Implements Linear Interpolation for bounding box coordinates to reduce visual jitter during detection.
- **3D Pose Estimation (solvePnP):** Uses MediaPipe FaceMesh and OpenCV's `solvePnP` for precise, degree-accurate head pose guidance during enrollment.
- **ArcFace Alignment:** Performs an Affine 2D transform based on eye landmarks to properly align the face before inference, maximizing recognition accuracy.
- **Event Logging & Security Alerts:** Persistently tracks "Last Seen" timestamps for known users and logs "Unknown Person" alerts if an unrecognized face loiters for >10 seconds.
- **Liveness Detection Infrastructure:** Built-in Eye Aspect Ratio (EAR) blink detection logic to prevent photo/video spoofing.

---

## 📂 Project Structure

```text
Test_Face/
├── backend/
│   ├── main.py                # FastAPI Backend + AI Inference
│   ├── models/                # ONNX and TFLite models
│   ├── services/              # AI Services (camera, recognition, liveness)
│   ├── routers/               # API Endpoints (video, events, users)
│   └── data/                  # Face database (.npy) and JSON event logs
├── frontend/
│   ├── src/                   # React Components & Logic
│   ├── public/
│   └── package.json           # Node.js dependencies
├── Dockerfile.backend         # Backend container build
├── Dockerfile.frontend        # Frontend container build (Nginx)
├── docker-compose.yml         # Wires both services together
├── nginx.conf                 # Nginx config (WS proxy + SPA routing)
└── face_edge_env/             # Python Virtual Environment (dev only)
```

---

## 🐳 Quick Start — Docker (Recommended)

> For anyone who just wants to **run the application** without setting up a Python or Node.js environment.

**Requirements:** [Docker](https://docs.docker.com/get-docker/) and [Git](https://git-scm.com/)

```bash
# 1. Clone the repository
git clone https://github.com/nafeesrakib22/Test_Face.git
cd Test_Face

# 2. Build the containers (first time only — takes a few minutes)
docker compose build

# 3. Start the app
docker compose up -d
```

Then open **`http://localhost:3000`** in your browser.

The face database is empty on first run — use **"Enroll New Face"** in the dashboard to register a face before recognition will work.

```bash
# To stop the app
docker compose down
```

---

## 🛠️ Developer Setup

> For contributors who want to run the app locally and make code changes.

### 1️⃣ Backend Environment (Python)

Requires **Python 3.12** and a virtual environment:

```bash
cd ~/Documents/Test_Face
python3 -m venv face_edge_env
source face_edge_env/bin/activate
pip install -r requirements.txt
```

### 2️⃣ Frontend Environment (Node.js)

Requires **Node.js v18+**:

```bash
cd frontend
npm install
```

---

## 💻 Running Locally (Dev Mode)

#### Terminal 1 — Start Backend (from project root)

```bash
source face_edge_env/bin/activate
uvicorn backend.main:app --reload
```

#### Terminal 2 — Start Frontend

```bash
cd frontend
npm run dev
```

Then open **`http://localhost:5173`** in your browser.

---

## 📊 Real-Time Benchmarking (CLI Mode)

If you prefer to run the standalone benchmark script without the web interface:

```bash
python backend/benchmark_multi_profile.py
```

---

## 🤖 Optional — Telegram Alerts via OpenClaw

> This is an **optional integration** for users who have [OpenClaw](https://openclaw.ai) installed and a Telegram bot configured.

This integration adds two capabilities:
- **On-demand query:** Ask your Telegram bot *"When was Alice last seen?"* and it responds with live data from the app.
- **Proactive alert:** If an unrecognized face is detected for more than 30 seconds, your bot automatically sends you a Telegram alert.

### Prerequisites
- OpenClaw installed and running locally
- A Telegram bot created via [@BotFather](https://t.me/botfather) and connected to OpenClaw
- The app running (Docker or local dev), backend accessible at `http://localhost:8000`

### Setup

The skill is included in this repo inside the `facewatch/` directory. Copy it into your OpenClaw skills folder:

```bash
cp -r facewatch ~/.openclaw/skills/facewatch
openclaw skills refresh
```

### 🔄 How It All Works

1. **Backend** writes an event to `backend/data/unknown_events.json` when an unknown face is seen for >30 continuous seconds.
2. **OpenClaw heartbeat** polls `GET /events/unknown` every 30 seconds.
3. **On alert found**, the skill sends a Telegram message and calls the `ack` endpoint to prevent repeat notifications.
4. **On-demand**, you can ask the bot *"Was Bob seen today?"* or *"Who has been seen this hour?"* at any time.

---