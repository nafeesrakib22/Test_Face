Edge Face Recognition System: Setup & Usage

This project is a high-performance, background-agnostic face recognition pipeline optimized for Edge devices. It utilizes EdgeFace-XS and MediaPipe BlazeFace with custom elliptical masking and multi-profile pose clustering.
🛠️ Installation
1. Clone & Environment

Open your terminal and run the following commands to set up a dedicated virtual environment:
Bash

git clone https://github.com/YOUR_USERNAME/Test_Face.git
cd Test_Face
python -m venv face_edge_env
source face_edge_env/bin/activate  # Windows: face_edge_env\Scripts\activate

2. Install Dependencies
Bash

pip install -r requirements.txt

3. Model Setup

Create a models/ folder and place your ONNX model and its weights inside:
Plaintext

models/
├── edgeface_xs_gamma_06.onnx
└── edgeface_xs_gamma_06.onnx.data

💻 Usage Instructions
Step 1: Hardware Discovery

Identify the correct camera index for your webcam (especially if using an external USB camera).
Bash

python camera_rec.py

Update the CAMERA_INDEX in config.py based on the output.
Step 2: Multi-Phase Enrollment

Run the enrollment script to capture your facial features from multiple angles to handle head rotation.
Bash

python enroll_multi_profile.py

    Enter your name when prompted.

    The script will guide you through three phases: FRONTAL, LEFT PROFILE, and RIGHT PROFILE.

    Position your head accordingly and press 's' to begin capturing each phase.

    Once completed, the script saves three distinct .npy files in data/face_db/.

Step 3: Database Verification (Optional)

Check the mathematical separation between enrolled users:
Bash

python check_db.py

A score below 0.60 indicates a healthy separation between identities.
Step 4: Real-Time Benchmarking

Launch the main recognition engine:
Bash

python benchmark_multi_profile.py

    Stability: Uses Lerp (Linear Interpolation) to smooth the bounding box and a 12-frame buffer to stabilize identity.

    Security: Displays a warning and pauses if multiple faces are detected in the frame.

⌨️ Controls

    's': Start/Continue enrollment phases.

    'q': Quit any running script.
