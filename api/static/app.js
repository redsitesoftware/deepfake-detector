/* app.js — Deepfake Detector live test client */
'use strict';

// ── Config ────────────────────────────────────────────────────────────────────
const TARGET_FPS   = 15;          // frames/s sent to server
const JPEG_QUALITY = 0.72;        // webcam → server compression
const WS_BASE      = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}`;

// ── State ─────────────────────────────────────────────────────────────────────
let sessionId   = null;
let ws          = null;
let streaming   = false;
let captureLoop = null;
let frameId     = 0;
let lastFrameTs = 0;
let fpsSmooth   = 0;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const panelSetup      = document.getElementById('panel-setup');
const fileInput       = document.getElementById('file-input');
const uploadZone      = document.getElementById('upload-zone');
const uploadPreview   = document.getElementById('upload-preview');
const uploadPlaceholder = document.getElementById('upload-placeholder');
const uploadStatus    = document.getElementById('upload-status');
const btnStart        = document.getElementById('btn-start');

const header          = document.querySelector('header');
const sourceThumb     = document.getElementById('source-thumb');
const sourceName      = document.getElementById('source-name');
const wsStatus        = document.getElementById('ws-status');
const badgeMode       = document.getElementById('badge-mode');

const verdictBar      = document.getElementById('verdict-bar');
const verdictLabel    = document.getElementById('verdict-label');
const confBar         = document.getElementById('conf-bar');
const sigCnn          = document.getElementById('sig-cnn');
const sigTemporal     = document.getElementById('sig-temporal');
const sigLiveness     = document.getElementById('sig-liveness');
const fpsDisplay      = document.getElementById('fps-display');

const mainGrid        = document.getElementById('main-grid');
const videoRaw        = document.getElementById('video-raw');
const canvasRaw       = document.getElementById('canvas-raw');
const canvasSwap      = document.getElementById('canvas-swap');
const canvasOverlay   = document.getElementById('canvas-overlay');
const noFaceMsg       = document.getElementById('no-face-msg');
const panelOverlay    = document.getElementById('panel-overlay');

const btnTriple       = document.getElementById('btn-triple');
const btnResetTemporal = document.getElementById('btn-reset-temporal');
const btnChangeFace   = document.getElementById('btn-change-face');
const latencyDisplay  = document.getElementById('latency');

// ── Upload & session setup ────────────────────────────────────────────────────
let pendingFile = null;

fileInput.addEventListener('change', e => {
  const file = e.target.files?.[0];
  if (file) handleFileSelected(file);
});

uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('drag'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('drag'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('drag');
  const file = e.dataTransfer.files?.[0];
  if (file) handleFileSelected(file);
});

function handleFileSelected(file) {
  if (!file.type.startsWith('image/')) {
    setUploadStatus('⚠ Please select an image file', 'warn');
    return;
  }
  pendingFile = file;
  const url = URL.createObjectURL(file);
  uploadPreview.src = url;
  uploadPreview.style.display = 'block';
  uploadPlaceholder.style.display = 'none';
  btnStart.disabled = false;
  setUploadStatus(`✓ ${file.name}`, 'ok');
}

function setUploadStatus(msg, type = '') {
  uploadStatus.textContent = msg;
  uploadStatus.style.color = type === 'ok' ? 'var(--green)' : type === 'warn' ? 'var(--yellow)' : 'var(--red)';
}

btnStart.addEventListener('click', async () => {
  if (!pendingFile) return;
  btnStart.disabled = true;
  setUploadStatus('Uploading…', '');

  try {
    const fd = new FormData();
    fd.append('file', pendingFile);
    const res = await fetch('/api/source', { method: 'POST', body: fd });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    sessionId = data.session_id;

    if (!data.face_detected) {
      setUploadStatus('⚠ Face uncertain — try a clearer front-facing photo. Proceeding anyway.', 'warn');
    } else if (!data.dlc_available) {
      setUploadStatus('⚠ DLC unavailable — running detection only', 'warn');
    } else if (data.warning) {
      setUploadStatus(`⚠ ${data.warning}`, 'warn');
    } else {
      setUploadStatus('✓ Face locked', 'ok');
    }

    await startLiveTest(pendingFile.name);
  } catch (err) {
    setUploadStatus(`✗ ${err.message}`, 'error');
    btnStart.disabled = false;
  }
});

// ── Live test startup ─────────────────────────────────────────────────────────
async function startLiveTest(filename) {
  // Show source thumb in header
  sourceThumb.src = `/api/source/${sessionId}/thumb`;
  sourceThumb.style.display = 'block';
  sourceName.textContent = filename;

  // Hide setup overlay
  panelSetup.style.display = 'none';

  // Start webcam
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, frameRate: { ideal: 30 } },
      audio: false
    });
    videoRaw.srcObject = stream;
    await new Promise(r => videoRaw.addEventListener('loadedmetadata', r, { once: true }));
    videoRaw.play();
  } catch (err) {
    alert(`Camera error: ${err.message}`);
    return;
  }

  // Sync canvas sizes to video
  videoRaw.addEventListener('resize', syncCanvasSizes);
  syncCanvasSizes();

  connectWebSocket();
}

function syncCanvasSizes() {
  const w = videoRaw.videoWidth  || 640;
  const h = videoRaw.videoHeight || 480;
  [canvasRaw, canvasSwap, canvasOverlay].forEach(c => { c.width = w; c.height = h; });
}

// ── WebSocket ─────────────────────────────────────────────────────────────────
function connectWebSocket() {
  if (ws) ws.close();
  ws = new WebSocket(`${WS_BASE}/ws/live/${sessionId}`);
  ws.binaryType = 'blob';

  ws.onopen = () => {
    setWsStatus('connecting…', 'yellow');
    // Server sends {"type":"ready"} after accept — capture starts in onmessage
  };

  ws.onmessage = e => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'ready') {
      setWsStatus('⬤ connected', 'green');
      badgeMode.classList.add('live');
      badgeMode.textContent = 'LIVE';
      startCapture();
      return;
    }
    if (msg.type === 'error') {
      console.error('Server error:', msg.detail);
      return;
    }
    handleDetectionResult(msg);
  };

  ws.onclose = () => {
    setWsStatus('⬤ disconnected', '');
    stopCapture();
    badgeMode.classList.remove('live');
    badgeMode.textContent = 'LIVE TEST';
  };

  ws.onerror = err => {
    console.error('WS error', err);
    setWsStatus('⬤ error', 'red');
  };
}

function setWsStatus(text, color) {
  wsStatus.textContent = text;
  wsStatus.style.color = color === 'green' ? 'var(--green)'
    : color === 'yellow' ? 'var(--yellow)'
    : color === 'red'    ? 'var(--red)'
    : 'var(--muted)';
}

// ── Frame capture loop ────────────────────────────────────────────────────────
function startCapture() {
  streaming = true;
  const interval = 1000 / TARGET_FPS;
  captureLoop = setInterval(sendFrame, interval);
}

function stopCapture() {
  streaming = false;
  if (captureLoop) { clearInterval(captureLoop); captureLoop = null; }
}

const _captureCanvas = document.createElement('canvas');
const _captureCtx    = _captureCanvas.getContext('2d');

function sendFrame() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  if (!videoRaw.videoWidth) return;

  const w = videoRaw.videoWidth;
  const h = videoRaw.videoHeight;
  _captureCanvas.width  = w;
  _captureCanvas.height = h;
  _captureCtx.drawImage(videoRaw, 0, 0, w, h);

  const b64 = _captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY);

  // FPS tracking
  const now = performance.now();
  if (lastFrameTs) {
    const dt = (now - lastFrameTs) / 1000;
    fpsSmooth = fpsSmooth * 0.8 + (1 / dt) * 0.2;
    fpsDisplay.textContent = `FPS: ${fpsSmooth.toFixed(1)}`;
  }
  lastFrameTs = now;

  ws.send(JSON.stringify({ frame: b64, frame_id: frameId++ }));

  // Also draw raw cam to canvasRaw (with bbox if available)
  drawRawPanel(w, h);
}

// ── Detection result handler ──────────────────────────────────────────────────
let lastResult = null;

function handleDetectionResult(msg) {
  lastResult = msg;

  // Draw swapped frame
  if (msg.swapped) drawImageB64(canvasSwap, msg.swapped);

  // Draw overlay panel (if visible)
  if (panelOverlay.style.display !== 'none' && msg.swapped) {
    drawImageB64(canvasOverlay, msg.swapped, () => drawOverlayOnCanvas(canvasOverlay, msg));
  }

  // Update verdict bar
  updateVerdictBar(msg);

  // No-face warning
  noFaceMsg.style.display = msg.face_detected ? 'none' : 'block';

  // Latency
  latencyDisplay.textContent = `latency: ${msg.latency_ms}ms`;
}

function drawRawPanel(w, h) {
  const ctx = canvasRaw.getContext('2d');
  ctx.drawImage(videoRaw, 0, 0, w, h);

  // Draw face bbox if available
  if (lastResult?.face_bbox) {
    const [x, y, bw, bh] = lastResult.face_bbox;
    const isFake = lastResult.is_fake;
    ctx.strokeStyle = isFake ? '#ef4444' : '#22c55e';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);
  }
}

function drawImageB64(canvas, b64, callback) {
  const img = new Image();
  img.onload = () => {
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    if (callback) callback();
  };
  img.src = b64;
}

function drawOverlayOnCanvas(canvas, msg) {
  const ctx = canvas.getContext('2d');
  const w = canvas.width, h = canvas.height;

  // Semi-transparent overlay panel top-left
  ctx.fillStyle = 'rgba(0,0,0,0.55)';
  ctx.fillRect(8, 8, 220, 110);

  // Verdict text
  const isFake = msg.is_fake;
  ctx.font = 'bold 20px monospace';
  ctx.fillStyle = isFake ? '#ef4444' : '#22c55e';
  ctx.fillText(isFake ? 'FAKE' : 'REAL', 18, 36);

  // Confidence bar
  const conf = msg.confidence ?? 0;
  ctx.fillStyle = '#ffffff22';
  ctx.fillRect(18, 44, 190, 8);
  ctx.fillStyle = confColor(conf);
  ctx.fillRect(18, 44, Math.round(190 * conf), 8);

  // Signals
  const sigs = [
    ['CNN',      msg.signals?.cnn],
    ['Temporal', msg.signals?.temporal],
    ['Liveness', msg.signals?.liveness],
  ];
  ctx.font = '11px monospace';
  ctx.fillStyle = '#ffffffaa';
  sigs.forEach(([name, val], i) => {
    const text = val != null ? `${name}: ${val.toFixed(2)}` : `${name}: —`;
    ctx.fillText(text, 18, 70 + i * 17);
  });

  // Latency
  ctx.fillStyle = '#ffffff55';
  ctx.font = '10px monospace';
  ctx.fillText(`${msg.latency_ms}ms`, 18, 112);

  // Face bbox
  if (msg.face_bbox) {
    const [x, y, bw, bh] = msg.face_bbox;
    ctx.strokeStyle = isFake ? '#ef4444' : '#22c55e';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);
  }
}

// ── Verdict bar update ────────────────────────────────────────────────────────
function updateVerdictBar(msg) {
  const conf    = msg.confidence ?? 0;
  const isFake  = msg.is_fake;
  const hasData = msg.face_detected;

  // Label
  verdictLabel.textContent = hasData ? (isFake ? 'FAKE' : 'REAL') : '—';
  verdictLabel.className   = hasData ? (isFake ? 'fake' : 'real') : '';
  verdictBar.className     = hasData ? (isFake ? 'fake' : 'real') : '';

  // Confidence bar
  confBar.style.width = `${(conf * 100).toFixed(1)}%`;
  confBar.className   = 'bar-fill ' + (conf > 0.7 ? 'danger' : conf > 0.5 ? 'warn' : '');

  // Signals
  setSig(sigCnn,      msg.signals?.cnn);
  setSig(sigTemporal, msg.signals?.temporal);
  setSig(sigLiveness, msg.signals?.liveness);
}

function setSig(el, val) {
  if (val == null) {
    el.textContent = '—';
    el.className = 'sig-val null';
  } else {
    el.textContent = val.toFixed(2);
    el.className = 'sig-val' + (val > 0.65 ? ' danger' : '');
    el.style.color = val > 0.7 ? 'var(--red)' : val > 0.5 ? 'var(--yellow)' : 'var(--green)';
  }
}

function confColor(v) {
  if (v > 0.7) return '#ef4444';
  if (v > 0.5) return '#eab308';
  return '#22c55e';
}

// ── Controls ──────────────────────────────────────────────────────────────────
let tripleMode = false;
btnTriple.addEventListener('click', () => {
  tripleMode = !tripleMode;
  panelOverlay.style.display = tripleMode ? '' : 'none';
  mainGrid.className = tripleMode ? 'triple' : '';
  btnTriple.textContent = tripleMode ? '⊟ Dual View' : '⊞ Triple View';
  if (tripleMode && lastResult?.swapped) {
    drawImageB64(canvasOverlay, lastResult.swapped, () => drawOverlayOnCanvas(canvasOverlay, lastResult));
  }
});

btnResetTemporal.addEventListener('click', async () => {
  // Close current WS — server creates fresh Detector with clean temporal buffer on reconnect.
  // Session persists so the source face is retained.
  stopCapture();
  if (ws) { ws.close(); ws = null; }
  frameId   = 0;
  fpsSmooth = 0;
  lastResult = null;
  fpsDisplay.textContent = 'FPS: —';
  // Small delay to let server-side WS close cleanly, then reconnect
  setTimeout(() => {
    connectWebSocket();
  }, 500);
});

btnChangeFace.addEventListener('click', async () => {
  stopCapture();
  if (ws) ws.close();
  if (sessionId) {
    fetch(`/api/source/${sessionId}`, { method: 'DELETE' }).catch(() => {});
    sessionId = null;
  }

  // Stop webcam
  const stream = videoRaw.srcObject;
  if (stream) stream.getTracks().forEach(t => t.stop());
  videoRaw.srcObject = null;

  // Reset UI
  uploadPreview.style.display = 'none';
  uploadPlaceholder.style.display = '';
  uploadStatus.textContent = '';
  fileInput.value = '';
  pendingFile = null;
  btnStart.disabled = true;
  sourceThumb.style.display = 'none';
  sourceName.textContent = '';
  verdictLabel.textContent = '—';
  verdictLabel.className = '';
  verdictBar.className = '';
  confBar.style.width = '0%';
  [sigCnn, sigTemporal, sigLiveness].forEach(el => { el.textContent = '—'; el.className = 'sig-val null'; });
  lastResult = null;
  fpsDisplay.textContent = 'FPS: —';
  latencyDisplay.textContent = 'latency: —';
  [canvasRaw, canvasSwap, canvasOverlay].forEach(c => c.getContext('2d').clearRect(0,0,c.width,c.height));

  panelSetup.style.display = 'flex';
});

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.code === 'Space' && panelSetup.style.display === 'none') {
    e.preventDefault();
    btnResetTemporal.click();
  }
  if (e.code === 'KeyD' && panelSetup.style.display === 'none') btnTriple.click();
});
