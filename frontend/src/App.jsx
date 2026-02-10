import { useState, useEffect } from 'react'
import './App.css'

function App() {
  const [status, setStatus] = useState("Offline")
  const [streamUrl, setStreamUrl] = useState("")

  useEffect(() => {
    // Check if the backend is running on port 8000
    fetch("http://127.0.0.1:8000/")
      .then(() => {
        setStatus("Online")
        setStreamUrl("http://127.0.0.1:8000/video_feed")
      })
      .catch(() => setStatus("Offline"))
  }, [])

  return (
    <div className="container">
      <header>
        <h1>🧠 Edge Face Recognition</h1>
        {/* Dynamic Status Badge */}
        <div className={`status-badge ${status.toLowerCase()}`}>
          System: {status}
        </div>
      </header>

      <main>
        <div className="video-box">
          {status === "Online" ? (
            <img 
              src={streamUrl} 
              alt="Live AI Feed" 
              className="live-feed" 
            />
          ) : (
            <div className="placeholder">
              <p>🔌 Connecting to Python Backend...</p>
              <small>Ensure 'main.py' is running on port 8000</small>
            </div>
          )}
        </div>

        <div className="controls">
          <button onClick={() => window.location.reload()}>🔄 Refresh Stream</button>
        </div>
      </main>
    </div>
  )
}

export default App