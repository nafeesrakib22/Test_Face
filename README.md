# 🧠 Edge-Optimized Face Recognition System

A high-performance, background-agnostic face recognition pipeline
optimized for **Edge devices**. This system utilizes the **EdgeFace-XS**
architecture and **MediaPipe BlazeFace** to provide robust identity
verification with minimal resource overhead.

------------------------------------------------------------------------

## 🚀 Key Technical Features

-   **Elliptical Masking:** Mitigates background noise and environmental
    bias by focusing the model exclusively on the facial manifold.\
-   **Multi-Profile Identity (Pose Manifold Clustering):** Captures
    Frontal, Left, and Right pose-centroids to ensure accuracy remains
    high even during head movement.\
-   **Identity Stabilization:** Utilizes a 12-frame hysteresis buffer
    and temporal smoothing to prevent identity "flickering."\
-   **Box Smoothing (Lerp):** Implements Linear Interpolation for
    bounding box coordinates to reduce visual jitter during detection.

------------------------------------------------------------------------

## 🛠️ Setup & Installation

### 1. Environment Setup

Clone the repository and create a clean Python virtual environment:

``` bash
git clone https://github.com/YOUR_USERNAME/Test_Face.git
cd Test_Face
python -m venv face_edge
source face_edge/bin/activate  # On Windows: face_edge\Scripts\activate
```

------------------------------------------------------------------------

### 2. Install Dependencies

Ensure you are using the optimized versions for edge performance:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

### 3. Model Configuration

Create a `models/` directory and place your ONNX model and weight file
inside.

> **Note:** ONNX Runtime requires the `.data` file to be in the same
> folder as the `.onnx` file.

``` text
models/
├── edgeface_xs_gamma_06.onnx
└── edgeface_xs_gamma_06.onnx.data
```

------------------------------------------------------------------------

## 💻 Usage Instructions

### Step 1: Hardware Discovery

Identify the correct camera index for your webcam (crucial for external
USB devices):

``` bash
python camera_rec.py
```

Update the `CAMERA_INDEX` in `config.py` based on the script's output.

------------------------------------------------------------------------

### Step 2: Multi-Phase Enrollment

Enroll your identity by capturing your face from three distinct angles.
This prevents **centroid shift** and ensures accuracy regardless of head
pose.

``` bash
python enroll_multi_profile.py
```

**Enrollment Phases**

-   **Phase 1 (Frontal):** Look directly at the camera.\
-   **Phase 2 (Left):** Turn your head slightly to the left.\
-   **Phase 3 (Right):** Turn your head slightly to the right.

👉 Press **`s`** to start or advance each phase.

------------------------------------------------------------------------

### Step 3: Database Verification (Optional)

Check the mathematical separation between enrolled users to ensure
identities are distinct and the threshold is appropriate.

``` bash
python check_db.py
```

------------------------------------------------------------------------

### Step 4: Real-Time Benchmarking

Launch the primary inference engine to test the recognition system:

``` bash
python benchmark_multi_profile.py
```