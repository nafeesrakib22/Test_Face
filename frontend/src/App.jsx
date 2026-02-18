import { useState, useEffect, useRef, useCallback } from 'react'
import './App.css'

const BACKEND_WS  = 'ws://127.0.0.1:8000'
const BACKEND_URL = 'http://127.0.0.1:8000'

// How many frames per second we send to the backend
const TARGET_FPS = 15

// Status → colour mapping for the bounding-box overlay
const STATUS_COLORS = {
  identified:  '#2ea043',
  unknown:     '#da3633',
  stabilizing: '#f1c40f',
  scanning:    '#8b949e',
  multiple:    '#da3633',
  searching:   '#8b949e',
  enrolling:   '#0576b9',
  complete:    '#2ea043',
}

export default function App() {
  const [status, setStatus]             = useState('Offline')
  const [isEnrolling, setIsEnrolling]   = useState(false)
  const [enrollName, setEnrollName]     = useState('')
  const [progress, setProgress]         = useState(0)
  const [overlay, setOverlay]           = useState(null)

  // Camera selector state
  const [cameras, setCameras]               = useState([])        // [{ deviceId, label }]
  const [selectedDeviceId, setSelectedDeviceId] = useState(null)
  const [cameraLoading, setCameraLoading]   = useState(false)

  const videoRef    = useRef(null)
  const canvasRef   = useRef(null)   // hidden — frame capture
  const overlayRef  = useRef(null)   // visible — annotations
  const wsRef       = useRef(null)
  const intervalRef = useRef(null)
  const streamRef   = useRef(null)

  // -------------------------------------------------------------------------
  // Camera enumeration
  // -------------------------------------------------------------------------
  const enumerateCameras = useCallback(async () => {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const videoInputs = devices
        .filter(d => d.kind === 'videoinput')
        .map((d, i) => ({
          deviceId: d.deviceId,
          // Use the browser-provided label (only available after permission is
          // granted). Fall back to a generic name.
          label: d.label || `Camera ${i + 1}`,
        }))
      setCameras(videoInputs)
      return videoInputs
    } catch {
      return []
    }
  }, [])

  // -------------------------------------------------------------------------
  // 1. Start / switch camera
  // -------------------------------------------------------------------------
  const startCamera = useCallback(async (deviceId = null) => {
    // Stop any existing stream tracks first
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(t => t.stop())
      streamRef.current = null
    }

    const constraints = {
      video: {
        width:  { ideal: 640 },
        height: { ideal: 480 },
        ...(deviceId ? { deviceId: { exact: deviceId } } : { facingMode: 'user' }),
      },
      audio: false,
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
      }
      return stream
    } catch (err) {
      console.error('Camera access error:', err)
      setStatus('Camera Denied')
      return null
    }
  }, [])

  // -------------------------------------------------------------------------
  // 2. Stop WebSocket + frame loop (does NOT stop the camera)
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
  // 3. Draw overlay annotations
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

    // ------------------------------------------------------------------
    // Helper: draw text inside a dark pill with a coloured left accent bar
    // ------------------------------------------------------------------
    const drawTextPill = (text, font, x, y, accentColor) => {
      ctx.font = font
      const metrics  = ctx.measureText(text)
      const textW    = metrics.width
      const padH     = 10   // horizontal padding
      const padV     = 8    // vertical padding
      const accentW  = 4    // left colour bar width
      const radius   = 6

      const boxW = textW + padH * 2 + accentW
      const boxH = 20 + padV * 2        // approximate line height + padding
      const bx   = x
      const by   = y - boxH + padV

      // Dark translucent background
      ctx.fillStyle = 'rgba(0, 0, 0, 0.60)'
      ctx.beginPath()
      ctx.roundRect(bx, by, boxW, boxH, radius)
      ctx.fill()

      // Colour accent bar on the left
      ctx.fillStyle = accentColor
      ctx.beginPath()
      ctx.roundRect(bx, by, accentW, boxH, [radius, 0, 0, radius])
      ctx.fill()

      // White text
      ctx.fillStyle = '#ffffff'
      ctx.fillText(text, bx + accentW + padH, y)
    }

    // ------------------------------------------------------------------
    // Bounding box
    // ------------------------------------------------------------------
    if (data.box) {
      const { x1, y1, x2, y2 } = data.box

      // Box outline
      ctx.strokeStyle = color
      ctx.lineWidth   = 2.5
      ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

      // Name / status label above (or below if too close to top edge)
      const label = data.name
        ? `${data.name}${data.confidence ? ` (${(data.confidence * 100).toFixed(0)}%)` : ''}`
        : data.status?.toUpperCase() ?? ''

      ctx.font = 'bold 14px Segoe UI, sans-serif'
      const labelW = ctx.measureText(label).width
      const lx = x1
      const ly = y1 > 32 ? y1 - 6 : y2 + 28

      ctx.fillStyle = color
      ctx.fillRect(lx - 2, ly - 16, labelW + 10, 20)
      ctx.fillStyle = '#ffffff'
      ctx.fillText(label, lx + 3, ly)
    }

    // ------------------------------------------------------------------
    // Instruction / status message — top-left, large + pill background
    // ------------------------------------------------------------------
    const statusMsg = buildStatusMsg(data)
    if (statusMsg) {
      drawTextPill(
        statusMsg,
        'bold 20px Segoe UI, sans-serif',
        10,                // x
        36,                // y (baseline)
        color,
      )
    }

    // ------------------------------------------------------------------
    // Phase indicator — bottom-left, medium + pill background
    // ------------------------------------------------------------------
    if (data.phase !== undefined) {
      const phaseMsg = `Phase ${data.phase}/5: ${data.phase_name ?? ''}  |  ${data.progress ?? 0}%`
      drawTextPill(
        phaseMsg,
        'bold 15px Segoe UI, sans-serif',
        10,
        canvas.height - 10,
        color,
      )
    }
  }, [])

  // -------------------------------------------------------------------------
  // 4. Open WebSocket and start sending frames
  // -------------------------------------------------------------------------
  const openWebSocket = useCallback((endpoint) => {
    stopAll()

    const ws = new WebSocket(`${BACKEND_WS}${endpoint}`)
    ws.binaryType = 'arraybuffer'
    wsRef.current = ws

    ws.onopen = () => {
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
          0.7,
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
  // 5. Mount: health check → enumerate cameras → start camera → open WS
  // -------------------------------------------------------------------------
  useEffect(() => {
    fetch(`${BACKEND_URL}/`)
      .then(async () => {
        setStatus('Online')

        // Start with the default camera first so getUserMedia grants permission
        // (labels are only available after permission is granted)
        const stream = await startCamera(null)
        if (!stream) return

        // Now enumerate — labels will be populated since we have permission
        const cams = await enumerateCameras()
        if (cams.length > 0) {
          // Figure out which deviceId the current stream is using
          const activeTrack = stream.getVideoTracks()[0]
          const activeId    = activeTrack?.getSettings()?.deviceId ?? cams[0].deviceId
          setSelectedDeviceId(activeId)
        }

        setTimeout(openRecognition, 800)
      })
      .catch(() => setStatus('Offline'))

    // Listen for hot-plug / unplug events
    const handleDeviceChange = () => enumerateCameras()
    navigator.mediaDevices?.addEventListener('devicechange', handleDeviceChange)

    return () => {
      stopAll()
      navigator.mediaDevices?.removeEventListener('devicechange', handleDeviceChange)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => { drawOverlay(overlay) }, [overlay, drawOverlay])

  // -------------------------------------------------------------------------
  // 6. Camera switch handler
  // -------------------------------------------------------------------------
  const handleCameraChange = useCallback(async (e) => {
    const newDeviceId = e.target.value
    setSelectedDeviceId(newDeviceId)
    setCameraLoading(true)

    // Switch the physical camera; the WS frame loop keeps running and will
    // automatically pick up frames from the new stream.
    await startCamera(newDeviceId)

    setCameraLoading(false)
  }, [startCamera])

  // -------------------------------------------------------------------------
  // 7. Enrollment controls
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
              <video
                ref={videoRef}
                autoPlay
                playsInline
                muted
                className="live-feed"
              />
              <canvas ref={overlayRef} className="overlay-canvas" />
              <canvas ref={canvasRef} style={{ display: 'none' }} />

              {isEnrolling && (
                <div className="progress-container">
                  <div className="progress-bar" style={{ width: `${progress}%` }} />
                </div>
              )}
            </>
          ) : (
            <div className="placeholder">
              <p>
                {status === 'Camera Denied'
                  ? '🚫 Camera access was denied. Please allow camera permission and refresh.'
                  : '🔌 Connecting to Python Backend…'}
              </p>
              {status !== 'Camera Denied' && (
                <small>Ensure 'main.py' is running on port 8000</small>
              )}
            </div>
          )}
        </div>

        {/* ── Camera Selector — always visible when Online ── */}
        {status === 'Online' && cameras.length > 0 && (
          <div className="camera-selector">
            <label htmlFor="camera-select">📷 Camera:</label>
            <select
              id="camera-select"
              value={selectedDeviceId ?? ''}
              onChange={handleCameraChange}
              disabled={isEnrolling || cameraLoading}
            >
              {cameras.map((cam) => (
                <option key={cam.deviceId} value={cam.deviceId}>
                  {cam.label}
                </option>
              ))}
            </select>
            {cameraLoading && <span className="camera-loading">Switching…</span>}
          </div>
        )}

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
    case 'identified':  return null
    case 'unknown':     return null
    case 'searching':   return data.instruction ?? 'Searching...'
    case 'enrolling':   return data.instruction ?? 'Enrolling...'
    case 'complete':    return data.instruction ?? 'Complete!'
    default:            return null
  }
}
