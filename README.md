# 🧠 Edge-Optimized Face Recognition System

A high-performance, background-agnostic face recognition pipeline optimized for **Edge devices**. This system utilizes the **EdgeFace-XS** architecture and **MediaPipe Tasks API** to provide robust identity verification with minimal resource overhead.

---

## 🚀 Key Technical Features

- **Elliptical Masking:** Mitigates background noise and environmental bias by focusing the model exclusively on the facial manifold.
- **Multi-Profile Identity (Pose Manifold Clustering):** Captures Frontal, Left, and Right pose-centroids to ensure accuracy remains high even during head movement.
- **Identity Stabilization:** Utilizes a 12-frame hysteresis buffer and temporal smoothing to prevent identity "flickering."
- **Single-Face Enforcement:** Security logic that denies access and provides visual warnings if multiple faces are detected in the frame.
- **Box Smoothing (Lerp):** Implements Linear Interpolation for bounding box coordinates to reduce visual jitter during detection.

---

## 📂 Project Structure

```text
Test_Face/
├── backend/
│   ├── main.py                # FastAPI Backend + AI Inference
│   ├── models/                # ONNX and TFLite models
│   └── data/face_db/          # Enrolled biometric templates (.npy)
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