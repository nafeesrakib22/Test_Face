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
└── face_edge_env/             # Python Virtual Environment
```

---

## 🛠️ Setup & Installation

### 1️⃣ Backend Environment (Python)

```bash
cd ~/Documents/Test_Face
source face_edge_env/bin/activate
pip install -r backend/requirements.txt
```

---

### 2️⃣ Frontend Environment (Node.js)

Ensure you have **Node.js (v18+)** installed.

```bash
cd frontend
npm install
```

---

## 💻 Usage Instructions

### ✅ Step 1: Multi-Phase Enrollment

Enroll your identity by capturing your face from three distinct angles. Hold still while enrolling every profile. 

```bash
python backend/enroll_multi_profile.py
```

---

### ✅ Step 2: Running the Full-Stack Application

To run the system with the Web Dashboard, start both the backend and frontend servers.

#### Terminal 1 — Start Backend (FastAPI)

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

#### Terminal 2 — Start Frontend (Vite)

```bash
cd frontend
npm run dev
```

---

### ✅ Step 3: Access the Dashboard

Open your browser and navigate to:

```
http://localhost:5173
```

You should see the **"Edge Face Recognition"** dashboard with a live AI-annotated video feed.

---

## 📊 Real-Time Benchmarking (CLI Mode)

If you prefer to run the standalone benchmark script without the web interface:

```bash
python backend/benchmark_multi_profile.py
```

---