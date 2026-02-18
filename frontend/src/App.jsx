import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const BACKEND_WS  = 'ws://127.0.0.1:8000'
const BACKEND_URL = 'http://127.0.0.1:8000'

// How many frames per second we send to the backend
const TARGET_FPS = 15

// Status → colour mapping for the bounding-box overlay
const STATUS_COLORS = {
  identified:  '#2ea043',   // green
  unknown:     '#da3633',   // red
  stabilizing: '#f1c40f',   // yellow
  scanning:    '#8b949e',   // grey
  multiple:    '#da3633',   // red
  searching:   '#8b949e',
  enrolling:   '#0576b9',   // blue
  complete:    '#2ea043',
}

export default function App() {
  const [status, setStatus]           = useState('Offline')
  const [isEnrolling, setIsEnrolling] = useState(false)
  const [enrollName, setEnrollName]   = useState('')
  const [progress, setProgress]       = useState(0)
  const [overlay, setOverlay]         = useState(null)   // last parsed WS message

  const videoRef      = useRef(null)   // <video> showing local camera
  const canvasRef     = useRef(null)   // hidden canvas for frame capture
  const overlayRef    = useRef(null)   // visible canvas for annotations
  const wsRef         = useRef(null)   // active WebSocket
  const intervalRef   = useRef(null)   // setInterval handle
  const streamRef     = useRef(null)   // MediaStream from getUserMedia

  // -------------------------------------------------------------------------
  // 1. Start local camera
  // -------------------------------------------------------------------------
  const startCamera = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' },
        audio: false,
      })
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
    } catch (err) {
      console.error('Camera access denied:', err)
      setStatus('Camera Denied')
    }
  }, [])

  // -------------------------------------------------------------------------
  // 2. Stop everything cleanly
  // -------------------------------------------------------------------------
  const stopAll = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
  }, [])

  // -------------------------------------------------------------------------
  // 3. Draw overlay annotations on the visible canvas
  // -------------------------------------------------------------------------
  const drawOverlay = useCallback((data) => {
    const canvas = overlayRef.current
    const video  = videoRef.current
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    canvas.width  = video.videoWidth  || 640
    canvas.height = video.videoHeight || 480
    ctx.clearRect(0, 0, canvas.width, canvas.height)

    if (!data) return

    const color = STATUS_COLORS[data.status] || '#8b949e'

    // Bounding box
    if (data.box) {
      const { x1, y1, x2, y2 } = data.box
      // Scale normalised coords if the backend returned them relative to the
      // original frame; since we send the full video resolution they are already
      // in pixel space.
      ctx.strokeStyle = color
      ctx.lineWidth   = 2
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      // Name + confidence label
      const label = data.name
        ? `${data.name}${data.confidence ? ` (${(data.confidence * 100).toFixed(0)}%)` : ''}`
        : data.status?.toUpperCase() ?? ''

      ctx.font         = 'bold 14px Segoe UI, sans-serif'
      ctx.fillStyle    = color
      const textW      = ctx.measureText(label).width
      const px         = x1
      const py         = y1 > 20 ? y1 - 6 : y2 + 18
      ctx.fillRect(px - 2, py - 14, textW + 8, 18)
      ctx.fillStyle = '#fff'
      ctx.fillText(label, px + 2, py)
    }

    // Status message (top-left)
    const statusMsg = buildStatusMsg(data)
    if (statusMsg) {
      ctx.font      = 'bold 15px Segoe UI, sans-serif'
      ctx.fillStyle = color
      ctx.fillText(statusMsg, 10, 28)
    }

    // Enrollment: phase indicator (bottom-left)
    if (data.phase !== undefined) {
      const phaseMsg = `Phase ${data.phase}/5: ${data.phase_name ?? ''} | ${data.progress ?? 0}%`
      ctx.font      = '13px Segoe UI, sans-serif'
      ctx.fillStyle = '#e6edf3'
      ctx.fillText(phaseMsg, 10, canvas.height - 12)
    }
  }, [])

  // -------------------------------------------------------------------------
  // 4. Open a WebSocket and start sending frames
  // -------------------------------------------------------------------------
  const openWebSocket = useCallback((endpoint) => {
    stopAll()

    const ws = new WebSocket(`${BACKEND_WS}${endpoint}`)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
      // Send frames at TARGET_FPS
      intervalRef.current = setInterval(() => {
        if (ws.readyState !== WebSocket.OPEN) return
        const video  = videoRef.current
        const canvas = canvasRef.current
        if (!video || !canvas || video.readyState < 2) return

        canvas.width  = video.videoWidth  || 640
        canvas.height = video.videoHeight || 480
        const ctx = canvas.getContext('2d')
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

        canvas.toBlob(
          (blob) => {
            if (!blob || ws.readyState !== WebSocket.OPEN) return
            blob.arrayBuffer().then((buf) => ws.send(buf))
          },
          'image/jpeg',
          0.7,   // JPEG quality
        )
      }, Math.round(1000 / TARGET_FPS))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        setOverlay(data)
        drawOverlay(data)

        if (data.progress !== undefined) setProgress(data.progress)

        if (data.complete) {
          stopAll()
          setIsEnrolling(false)
          setEnrollName('')
          setProgress(0)
          setOverlay(null)
          drawOverlay(null)
          alert('✅ Enrollment Successful! 5 Profiles Loaded into Database.')
          // Re-open recognition stream
          openRecognition()
        }
      } catch (e) {
        console.error('WS parse error', e)
      }
    }

    ws.onerror = (e) => console.error('WebSocket error', e)

    ws.onclose = () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current)
        intervalRef.current = null
      }
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [drawOverlay, stopAll])

  const openRecognition = useCallback(() => {
    openWebSocket('/ws/recognize')
  }, [openWebSocket])

  // -------------------------------------------------------------------------
  // 5. Mount: check backend, start camera, open recognition WS
  // -------------------------------------------------------------------------
  useEffect(() => {
    fetch(`${BACKEND_URL}/`)
      .then(() => {
        setStatus('Online')
        startCamera().then(() => {
          // Small delay so the video element has a chance to start
          setTimeout(openRecognition, 800)
        })
      })
      .catch(() => setStatus('Offline'))

    return () => stopAll()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Re-draw whenever overlay state changes (e.g. after enroll completes)
  useEffect(() => {
    drawOverlay(overlay)
  }, [overlay, drawOverlay])

  // -------------------------------------------------------------------------
  // 6. Enrollment controls
  // -------------------------------------------------------------------------
  const startEnrollment = () => {
    const name = prompt('Enter Name for Enrollment:')
    if (!name || !name.trim()) return
    const trimmed = name.trim()
    setEnrollName(trimmed)
    setIsEnrolling(true)
    setProgress(0)
    openWebSocket(`/ws/enroll?name=${encodeURIComponent(trimmed)}`)
  }

  const cancelEnrollment = () => {
    stopAll()
    fetch(`${BACKEND_URL}/reset_session`).catch(() => {})
    setIsEnrolling(false)
    setEnrollName('')
    setProgress(0)
    setOverlay(null)
    drawOverlay(null)
    openRecognition()
  }

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="container">
      <header>
        <h1>🧠 Edge Face Recognition</h1>
        <div className={`status-badge ${isEnrolling ? 'enrolling' : status.toLowerCase().replace(' ', '-')}`}>
          {isEnrolling
            ? `Enrolling: ${enrollName} (${progress}%)`
            : `System: ${status}`}
        </div>
      </header>

      <main>
        <div className="video-box">
          {status === 'Online' ? (
            <>
              {/* Local camera feed */}
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="live-feed"
              />

              {/* Annotation overlay canvas (sits on top of the video) */}
              <canvas ref={overlayRef} className="overlay-canvas" />

              {/* Hidden capture canvas (not rendered) */}
              <canvas ref={canvasRef} style={{ display: 'none' }} />

              {/* Progress bar during enrollment */}
              {isEnrolling && (
                <div className="progress-container">
                  <div className="progress-bar" style={{ width: `${progress}%` }} />
                </div>
              )}
            </>
          ) : (
            <div className="placeholder">
              <p>🔌 {status === 'Camera Denied'
                ? '🚫 Camera access was denied. Please allow camera permission and refresh.'
                : 'Connecting to Python Backend…'}
              </p>
              {status !== 'Camera Denied' && (
                <small>Ensure 'main.py' is running on port 8000</small>
              )}
            </div>
          )}
        </div>

        <div className="controls">
          {!isEnrolling ? (
            <>
              <button onClick={() => window.location.reload()}>🔄 Refresh</button>
              <button
                className="enroll-btn"
                onClick={startEnrollment}
                disabled={status !== 'Online'}
              >
                👤 Enroll New Face
              </button>
            </>
          ) : (
            <button className="cancel-btn" onClick={cancelEnrollment}>
              ❌ Cancel Enrollment
            </button>
          )}
        </div>
      </main>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function buildStatusMsg(data) {
  switch (data.status) {
    case 'scanning':    return 'Scanning...'
    case 'multiple':    return '⚠️ Multiple faces detected'
    case 'stabilizing': return 'Stabilizing...'
    case 'identified':  return null   // name shown on bbox label
    case 'unknown':     return null
    case 'searching':   return data.instruction ?? 'Searching...'
    case 'enrolling':   return data.instruction ?? 'Enrolling...'
    case 'complete':    return data.instruction ?? 'Complete!'
    default:            return null
  }
}
