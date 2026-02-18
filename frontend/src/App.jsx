import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [status, setStatus] = useState("Offline")
  const [isEnrolling, setIsEnrolling] = useState(false)
  const [enrollName, setEnrollName] = useState("")
  const [progress, setProgress] = useState(0)
  const [cameras, setCameras] = useState([])
  const [selectedCamera, setSelectedCamera] = useState(null)
  const [cameraLoading, setCameraLoading] = useState(false)
  // feedKey forces the <img> to re-request the stream when camera changes
  const [feedKey, setFeedKey] = useState(0)

  const BACKEND_URL = "http://127.0.0.1:8000"

  // 1. Initial Connection Check + Camera List Fetch
  useEffect(() => {
    fetch(`${BACKEND_URL}/`)
      .then(() => {
        setStatus("Online")
        return fetch(`${BACKEND_URL}/cameras`)
      })
      .then(res => res.json())
      .then(data => {
        setCameras(data.cameras)
        setSelectedCamera(data.current)
      })
      .catch(() => setStatus("Offline"))
  }, [])

  // 2. Polling for Enrollment Progress & Completion
  useEffect(() => {
    let interval;
    if (isEnrolling) {
      interval = setInterval(async () => {
        try {
          const res = await fetch(`${BACKEND_URL}/enroll_status`);
          const data = await res.json();
          setProgress(data.progress);
          if (data.complete) {
            handleEnrollmentComplete();
          }
        } catch (err) {
          console.error("Status check failed", err);
        }
      }, 800);
    }
    return () => clearInterval(interval);
  }, [isEnrolling]);

  const handleEnrollmentComplete = () => {
    setIsEnrolling(false);
    setEnrollName("");
    setProgress(0);
    alert("✅ Enrollment Successful! 5 Profiles Loaded into Database.");
  };

  const startEnrollment = () => {
    const name = prompt("Enter Name for Enrollment:");
    if (name && name.trim().length > 0) {
      setEnrollName(name.trim());
      setIsEnrolling(true);
    }
  };

  const cancelEnrollment = () => {
    fetch(`${BACKEND_URL}/reset_session`)
      .then(() => {
        setIsEnrolling(false);
        setEnrollName("");
        setProgress(0);
      });
  };

  const handleCameraChange = async (e) => {
    const newIndex = parseInt(e.target.value, 10);
    setSelectedCamera(newIndex);
    setCameraLoading(true);
    try {
      await fetch(`${BACKEND_URL}/set_camera?camera_index=${newIndex}`);
      // Give the OS a moment to release the old device before reconnecting the stream
      setTimeout(() => {
        setFeedKey(prev => prev + 1);
        setCameraLoading(false);
      }, 700);
    } catch (err) {
      console.error("Camera switch failed", err);
      setCameraLoading(false);
    }
  };

  const currentFeedSrc = isEnrolling
    ? `${BACKEND_URL}/enroll?name=${enrollName}`
    : `${BACKEND_URL}/video_feed`

  return (
    <div className="container">
      <header>
        <h1>🧠 Edge Face Recognition</h1>
        <div className={`status-badge ${isEnrolling ? 'enrolling' : status.toLowerCase()}`}>
          {isEnrolling ? `Enrolling: ${enrollName} (${progress}%)` : `System: ${status}`}
        </div>
      </header>

      <main>
        <div className="video-box">
          {status === "Online" ? (
            <>
              {cameraLoading ? (
                <div className="placeholder">
                  <p>🔄 Switching Camera...</p>
                </div>
              ) : (
                <img
                  key={`${isEnrolling ? 'enroll' : 'recognize'}-${feedKey}`}
                  src={currentFeedSrc}
                  alt="Live AI Feed"
                  className="live-feed"
                />
              )}

              {/* Progress Bar overlay during Enrollment */}
              {isEnrolling && (
                <div className="progress-container">
                  <div className="progress-bar" style={{ width: `${progress}%` }}></div>
                </div>
              )}
            </>
          ) : (
            <div className="placeholder">
              <p>🔌 Connecting to Python Backend...</p>
              <small>Ensure 'main.py' is running on port 8000</small>
            </div>
          )}
        </div>

        {/* Camera Selector */}
        {status === "Online" && cameras.length > 0 && (
          <div className="camera-selector">
            <label htmlFor="camera-select">📷 Camera:</label>
            <select
              id="camera-select"
              value={selectedCamera ?? ''}
              onChange={handleCameraChange}
              disabled={isEnrolling || cameraLoading}
            >
              {cameras.map(cam => (
                <option key={cam.index} value={cam.index}>
                  {cam.label}
                </option>
              ))}
            </select>
          </div>
        )}

        <div className="controls">
          {!isEnrolling ? (
            <>
              <button onClick={() => window.location.reload()}>🔄 Refresh</button>
              <button className="enroll-btn" onClick={startEnrollment} disabled={status !== "Online"}>
                👤 Enroll New Face
              </button>
            </>
          ) : (
            <button className="cancel-btn" onClick={cancelEnrollment}>❌ Cancel Enrollment</button>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
