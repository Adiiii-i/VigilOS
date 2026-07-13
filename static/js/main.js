/* ────────────────────────────────────────────────────────
   VigilOS — Client JS (Client-Side Streaming Mode)
   Retrieves local visitor camera and streams frames to Python.
   ──────────────────────────────────────────────────────── */

const SEV = {
  concealed: {label:'CONCEALED', color:'var(--alert)', soft:'var(--alert-soft)', title:'Concealed Person'},
  weapon:    {label:'WEAPON',    color:'var(--danger)', soft:'var(--danger-soft)', title:'Weapon Detected'},
  person:    {label:'TRACKED',   color:'var(--safe)',   soft:'var(--safe-soft)',   title:'Person Detected'},
};

// ── DOM Elements ───────────────────────────────────────
const feedWrap     = document.getElementById('feedWrap');
const videoFeed    = document.getElementById('videoFeed');
const toggleBtn    = document.getElementById('toggleBtn');
const btnIcon      = document.getElementById('btnIcon');
const btnLabel     = document.getElementById('btnLabel');
const statusPill   = document.getElementById('statusPill');
const statusLabel  = document.getElementById('statusLabel');
const fpsVal       = document.getElementById('fpsVal');
const logList      = document.getElementById('logList');
const eventCount   = document.getElementById('eventCount');
const signalBars   = document.getElementById('signalBars');
const signalChip   = document.getElementById('signalChip');
const feedBadges   = document.getElementById('feedBadges');
const feedCenterMsg= document.getElementById('feedCenterMsg');
const feedCenterSub= document.getElementById('feedCenterSub');

// Stats Counters
const statThreats   = document.getElementById('statThreats');
const statConcealed = document.getElementById('statConcealed');
const statConf      = document.getElementById('statConf');
const statUptime    = document.getElementById('statUptime');

// ── State variables ─────────────────────────────────────
let running = false;
let knownEventIds = new Set();
let statusInterval = null;
let logInterval = null;
let uptimeInterval = null;
let signalInterval = null;
let captureInterval = null;
let uptimeSec = 0;

// Local Web Media stream capture
let localStream = null;
let clientVideo = null;
let clientCanvas = null;
let canvasCtx = null;
let lastFrameTime = Date.now();

// ── Build Signal Bar Elements ────────────────────────────
signalBars.innerHTML = '';
for (let i = 0; i < 28; i++) {
  const bar = document.createElement('div');
  bar.style.height = '10%';
  bar.style.opacity = '0.25';
  signalBars.appendChild(bar);
}

// ── Uptime formatting ───────────────────────────────────
function pad(n) { return n.toString().padStart(2, '0'); }
function fmtUptime(s) {
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return `${pad(h)}:${pad(m)}:${pad(sec)}`;
}

// ── SVG Icon Generator ──────────────────────────────────
function svgIcon(kind) {
  if (kind === 'weapon') {
    return `<svg viewBox="0 0 24 24" fill="none"><path d="M12 2L4 6v6c0 5 3.4 9 8 10 4.6-1 8-5 8-10V6l-8-4z" stroke="currentColor" stroke-width="1.6"/><path d="M9 12l2 2 4-4" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
  }
  if (kind === 'concealed') {
    return `<svg viewBox="0 0 24 24" fill="none"><path d="M12 3l8 3.5V11c0 5-3.4 9.2-8 10.5C7.4 20.2 4 16 4 11V6.5L12 3z" stroke="currentColor" stroke-width="1.6"/><path d="M12 8v5M12 15.2v.1" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/></svg>`;
  }
  return `<svg viewBox="0 0 24 24" fill="none"><circle cx="12" cy="8" r="3.2" stroke="currentColor" stroke-width="1.6"/><path d="M5 20c1-4 4-6 7-6s6 2 7 6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/></svg>`;
}

// ── Map raw object to UI kind ────────────────────────────
function getEventKind(objectName) {
  const name = objectName.toLowerCase();
  if (['knife', 'fork', 'scissors', 'bottle', 'weapon'].some(w => name.includes(w))) {
    return 'weapon';
  }
  if (name.includes('conceal')) {
    return 'concealed';
  }
  return 'person';
}

// ── Live status checking ─────────────────────────────────
async function checkStatus() {
  try {
    const res = await fetch('/api/status');
    const data = await res.json();

    statThreats.textContent = data.total_detections;
    statConcealed.textContent = data.total_concealed;

    if (data.status === 'running') {
      if (!running) {
        setRunningState(true);
      }
    } else {
      if (running) {
        setRunningState(false);
      }
      if (data.status === 'error') {
        statusLabel.textContent = 'ERROR';
        statusPill.className = 'status-pill error';
        feedCenterMsg.textContent = 'SYSTEM ERROR';
        feedCenterSub.textContent = data.error || 'Server error detected';
      }
    }
  } catch (err) {
    console.warn('Status endpoint unavailable', err);
  }
}

// ── Poll logs for new events ─────────────────────────────
async function checkLogs() {
  try {
    const res = await fetch('/api/logs');
    const rows = await res.json();
    if (!Array.isArray(rows)) return;

    eventCount.textContent = `${rows.length} events`;

    // Extract new entries that we haven't seen yet
    let newEntries = rows.filter(r => !knownEventIds.has(r.id));

    if (newEntries.length > 0) {
      newEntries.forEach(row => {
        knownEventIds.add(row.id);
        const kind = getEventKind(row.object);
        const item = buildLogItem(row, kind);
        logList.insertBefore(item, logList.firstChild);

        // Flash alert border on new weapons/concealment
        if (kind === 'weapon' || kind === 'concealed') {
          flashAlert();
        }
      });
      
      // Update average confidence from the list of logs
      let sum = 0;
      rows.forEach(r => sum += (r.confidence || 0));
      const avg = rows.length ? Math.round((sum / rows.length) * 100) : 0;
      statConf.textContent = `${avg}%`;
    }

    // Clean up older items to save DOM resources
    while (logList.children.length > 40) {
      logList.removeChild(logList.lastChild);
    }
  } catch (err) {
    console.warn('Log endpoint unavailable', err);
  }
}

// ── Build a dynamic Log Item ─────────────────────────────
function buildLogItem(row, kind) {
  const sev = SEV[kind];
  const conf = Math.round((parseFloat(row.confidence) || 0) * 100);
  const timeStr = row.timestamp.split(' ')[1] || row.timestamp; // get HH:MM:SS part

  const item = document.createElement('div');
  item.className = 'log-item';
  item.style.setProperty('--sev-color', sev.color);
  item.style.setProperty('--sev-soft', sev.soft);
  item.innerHTML = `
    <div class="log-row">
      <div class="log-left">
        <span class="log-icon">${svgIcon(kind)}</span>
        <span class="log-title">${sev.title}</span>
      </div>
      <span class="log-tag">${sev.label}</span>
    </div>
    <div class="log-meta">
      <span class="conf">Confidence: ${conf}%</span>
      <span>${timeStr}</span>
    </div>
  `;
  return item;
}

// ── Visual Alert Flash ──────────────────────────────────
function flashAlert() {
  feedWrap.classList.add('alerting');
  clearTimeout(window._alertT);
  window._alertT = setTimeout(() => feedWrap.classList.remove('alerting'), 1500);
}

// ── Animate Signal bars ──────────────────────────────────
function tickSignal() {
  Array.from(signalBars.children).forEach(bar => {
    const h = running ? (8 + Math.random() * 92) : (4 + Math.random() * 6);
    bar.style.height = h + '%';
    bar.style.opacity = running ? (0.4 + Math.random() * 0.6) : 0.25;
  });
}

// ── Client-Side Frame Capturing & Posting ──────────────
function sendFrameToServer() {
  if (!running || !clientVideo || !canvasCtx) return;

  // Draw current webcam frame onto hidden canvas
  canvasCtx.drawImage(clientVideo, 0, 0, clientCanvas.width, clientCanvas.height);

  // Convert canvas frame to blob
  clientCanvas.toBlob(async (blob) => {
    if (!blob || !running) return;

    try {
      const res = await fetch('/api/process_frame', {
        method: 'POST',
        headers: { 'Content-Type': 'image/jpeg' },
        body: blob
      });

      if (!res.ok) return;

      const responseBlob = await res.blob();
      if (!running) return;

      const url = URL.createObjectURL(responseBlob);

      // Swap images and release memory to prevent memory leaks
      const oldUrl = videoFeed.src;
      videoFeed.src = url;
      if (oldUrl && oldUrl.startsWith('blob:')) {
        URL.revokeObjectURL(oldUrl);
      }

      // Calculate client-side process FPS
      const now = Date.now();
      const localFps = 1000 / (now - lastFrameTime);
      lastFrameTime = now;
      fpsVal.textContent = `${localFps.toFixed(1)} FPS`;

    } catch (err) {
      console.warn("Frame upload failed:", err);
    }
  }, 'image/jpeg', 0.65); // 0.65 quality compression keeps upload payload light
}

// ── Set the State of UI ──────────────────────────────────
async function setRunningState(shouldRun) {
  running = shouldRun;
  if (running) {
    feedWrap.classList.add('running');
    statusPill.className = 'status-pill live';
    statusLabel.textContent = 'LIVE';
    btnLabel.textContent = 'Stop';
    toggleBtn.classList.add('stop');
    btnIcon.innerHTML = '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>';
    signalChip.textContent = 'STREAMING';

    feedBadges.innerHTML = `<span class="feed-badge">INPUT: WEB CAMERA</span><span class="feed-badge">YOLOv8m + MEDIAPIPE</span>`;

    // ── Get local camera device access ──
    try {
      if (!clientVideo) {
        clientVideo = document.createElement('video');
        clientVideo.autoplay = true;
        clientVideo.playsInline = true;
        clientVideo.muted = true; // Crucial for browser camera playback permissions
        clientCanvas = document.createElement('canvas');
        clientCanvas.width = 640;
        clientCanvas.height = 480;
        canvasCtx = clientCanvas.getContext('2d');
      }

      localStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: "user"
        }
      });
      clientVideo.srcObject = localStream;
      
      // Explicitly trigger playback to resolve Chrome/Safari security delays
      try {
        await clientVideo.play();
      } catch (playErr) {
        console.warn("Video playback deferred:", playErr);
      }

      // Capture loop at ~150ms intervals (~7 FPS)
      captureInterval = setInterval(sendFrameToServer, 150);
    } catch (err) {
      console.error("Local webcam access failed:", err);
      statusLabel.textContent = 'ERROR';
      statusPill.className = 'status-pill error';
      feedCenterMsg.textContent = 'CAMERA BLOCKED';
      feedCenterSub.textContent = 'allow camera permission in browser';
      setRunningState(false);
      return;
    }

    // Start local timer loop for uptime
    if (!uptimeInterval) {
      uptimeInterval = setInterval(() => {
        uptimeSec++;
        statUptime.textContent = fmtUptime(uptimeSec);
      }, 1000);
    }
  } else {
    feedWrap.classList.remove('running');
    statusPill.className = 'status-pill';
    statusLabel.textContent = 'STOPPED';
    fpsVal.textContent = '0.0 FPS';
    btnLabel.textContent = 'Start';
    toggleBtn.classList.remove('stop');
    btnIcon.innerHTML = '<path d="M8 5v14l11-7z"/>';
    signalChip.textContent = 'IDLE';
    feedBadges.innerHTML = '';

    // Stop streams
    if (captureInterval) {
      clearInterval(captureInterval);
      captureInterval = null;
    }
    if (localStream) {
      localStream.getTracks().forEach(track => track.stop());
      localStream = null;
    }
    if (clientVideo) {
      clientVideo.srcObject = null;
    }
    
    // Revoke old blob to release browser memory
    if (videoFeed.src && videoFeed.src.startsWith('blob:')) {
      URL.revokeObjectURL(videoFeed.src);
    }
    videoFeed.src = ''; 

    if (uptimeInterval) {
      clearInterval(uptimeInterval);
      uptimeInterval = null;
    }
    uptimeSec = 0;
    statUptime.textContent = '00:00:00';
  }
}

// ── Control action handler ──────────────────────────────
async function toggleEngine() {
  const action = running ? 'stop' : 'start';
  try {
    const res = await fetch(`/api/control/${action}`);
    const data = await res.json();
    if (data.ok) {
      setRunningState(!running);
    }
  } catch (err) {
    console.error('Failed to trigger engine control', err);
    // Fallback toggle
    setRunningState(!running);
  }
}

// ── Toggle button click listener ────────────────────────
toggleBtn.addEventListener('click', toggleEngine);

// ── Bootstrap intervals ──────────────────────────────────
checkStatus();
checkLogs();
statusInterval = setInterval(checkStatus, 1500);
logInterval    = setInterval(checkLogs, 2000);
signalInterval = setInterval(tickSignal, 130);
