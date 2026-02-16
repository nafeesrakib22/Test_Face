import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [status, setStatus] = useState("Offline")
  const [isEnrolling, setIsEnrolling] = useState(false)
  const [enrollName, setEnrollName] = useState("")
  const [progress, setProgress] = useState(0)

  const BACKEND_URL = "http://127.0.0.1:8000"

  // 1. Initial Connection Check
  useEffect(() => {
    fetch(`${BACKEND_URL}/`)
      .then(() => setStatus("Online"))
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
      }, 800); // Poll every 800ms
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
              <img 
                src={isEnrolling ? `${BACKEND_URL}/enroll?name=${enrollName}` : `${BACKEND_URL}/video_feed`} 
                alt="Live AI Feed" 
                className="live-feed" 
                key={isEnrolling ? 'enroll' : 'recognize'} // Force re-render when switching modes
              />
              
              {/* Overlay Progress Bar during Enrollment */}
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