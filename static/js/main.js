/* ────────────────────────────────────────────────────────
   VigilOS — Client JS
   Links the redesigned UI to the live Python API.
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
let uptimeSec = 0;

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

    fpsVal.textContent = `${data.fps.toFixed(1)} FPS`;
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
        feedCenterMsg.textContent = 'CAMERA ERROR';
        feedCenterSub.textContent = data.error || 'Cannot open camera stream';
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

// ── Set the State of UI ──────────────────────────────────
function setRunningState(shouldRun) {
  running = shouldRun;
  if (running) {
    feedWrap.classList.add('running');
    statusPill.className = 'status-pill live';
    statusLabel.textContent = 'LIVE';
    btnLabel.textContent = 'Stop';
    toggleBtn.classList.add('stop');
    btnIcon.innerHTML = '<rect x="6" y="5" width="4" height="14" rx="1"/><rect x="14" y="5" width="4" height="14" rx="1"/>';
    signalChip.textContent = 'STREAMING';
    videoFeed.src = '/video_feed'; // Load the stream

    feedBadges.innerHTML = `<span class="feed-badge">MODEL: ATX-V4</span><span class="feed-badge">POSE + WEAPON NET</span>`;

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
    videoFeed.src = ''; // Clear image src to stop browser network load

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
