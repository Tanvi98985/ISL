/* ===== GLOBAL STATE ===== */
const API_BASE = 'http://127.0.0.1:7865';
const PREDICT_API = `${API_BASE}/predict_api`;
const HOLD_SECONDS = 2.0;
const MIN_CONF = 0.70;

let stream = null;
let detectionInterval = null;
let holdState = { pred: null, startedAt: null };
let wordText = '';
let isRunning = false;

/* ===== DOM HELPERS ===== */
const $ = id => document.getElementById(id);
const show = el => { if (el) el.style.display = el.dataset.display || 'flex'; };
const hide = el => { if (el) el.style.display = 'none'; };

/* ===== THEME ===== */
function initTheme() {
  const saved = localStorage.getItem('isl-theme') || 'dark';
  document.body.classList.toggle('light', saved === 'light');
  updateThemeBtn();
}

function toggleTheme() {
  document.body.classList.toggle('light');
  const isLight = document.body.classList.contains('light');
  localStorage.setItem('isl-theme', isLight ? 'light' : 'dark');
  updateThemeBtn();
}

function updateThemeBtn() {
  const btn = $('theme-toggle');
  if (!btn) return;
  const isLight = document.body.classList.contains('light');
  btn.textContent = isLight ? '🌙 Dark' : '☀️ Light';
}

/* ===== TAB SWITCHING ===== */
function initTabs() {
  document.querySelectorAll('[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => {
      const target = btn.dataset.tab;
      const group = btn.dataset.group || 'main';

      // Deactivate siblings
      document.querySelectorAll(`[data-group="${group}"]`).forEach(b => b.classList.remove('active'));
      document.querySelectorAll(`[data-panel-group="${group}"]`).forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const panel = document.querySelector(`[data-panel="${target}"][data-panel-group="${group}"]`);
      if (panel) panel.classList.add('active');

      // Stop camera when switching away from webcam tab
      if (group === 'detection' && target !== 'webcam' && isRunning) {
        stopCamera();
      }
    });
  });
}

/* ===== CAMERA ===== */
async function startCamera() {
  if (isRunning) return;

  const video = $('cam-video');
  const placeholder = $('cam-placeholder');
  const statusDot = $('status-dot');
  const statusText = $('status-text');

  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: 'user' } });
    video.srcObject = stream;
    await video.play();

    show(video);
    hide(placeholder);
    statusDot.classList.add('live');
    statusText.textContent = 'LIVE';
    isRunning = true;

    // Start sending frames
    detectionInterval = setInterval(captureAndPredict, 400);
    setStatus('Show your hand clearly in the frame…');
  } catch (err) {
    setStatus(`⚠️ Camera error: ${err.message}`);
  }
}

function stopCamera() {
  if (!isRunning) return;

  clearInterval(detectionInterval);
  detectionInterval = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }

  const video = $('cam-video');
  const placeholder = $('cam-placeholder');
  const statusDot = $('status-dot');
  const statusText = $('status-text');

  if (video) { video.srcObject = null; hide(video); }
  if (placeholder) show(placeholder);
  if (statusDot) statusDot.classList.remove('live');
  if (statusText) statusText.textContent = 'OFF';
  isRunning = false;

  holdState = { pred: null, startedAt: null };
  setStatus('Camera stopped.');
}

async function captureAndPredict() {
  const video = $('cam-video');
  if (!video || !video.videoWidth) return;

  // Draw to hidden canvas
  const canvas = $('cap-canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  const base64 = canvas.toDataURL('image/jpeg', 0.75).split(',')[1];

  try {
    const res = await fetch(PREDICT_API, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64 }),
    });
    const data = await res.json();
    handlePrediction(data);
  } catch (e) {
    // network error — silently skip
  }
}

function handlePrediction(data) {
  if (data.error) {
    setStatus(data.error);
    updateLetterDisplay(null, 0, []);
    holdState = { pred: null, startedAt: null };
    return;
  }

  const { letter, confidence, top5 } = data;
  updateLetterDisplay(letter, confidence, top5 || []);

  if (confidence >= MIN_CONF) {
    const now = Date.now() / 1000;
    if (letter !== holdState.pred) {
      holdState = { pred: letter, startedAt: now };
    }
    const elapsed = now - holdState.startedAt;
    const remaining = Math.max(0, HOLD_SECONDS - elapsed).toFixed(1);

    if (elapsed >= HOLD_SECONDS) {
      // Confirmed!
      appendLetter(letter);
      holdState = { pred: null, startedAt: null };
      setStatus(`✅ Confirmed: ${letter}`);
    } else {
      setStatus(`Hold steady… ${remaining}s  (${Math.round(confidence * 100)}%)`);
    }
  } else {
    holdState = { pred: null, startedAt: null };
    setStatus(`Show sign steadily (need ≥ ${Math.round(MIN_CONF * 100)}% confidence)`);
  }
}

function updateLetterDisplay(letter, confidence, top5) {
  const letterEl = $('detected-letter');
  const fillEl = $('conf-fill');
  const confPct = $('conf-pct');
  const list = $('candidates');

  if (letterEl) {
    letterEl.textContent = letter || '—';
    letterEl.className = 'letter-char' + (letter ? '' : ' empty');
  }
  if (fillEl) fillEl.style.width = `${Math.round(confidence * 100)}%`;
  if (confPct) confPct.textContent = `${Math.round(confidence * 100)}%`;

  if (list) {
    list.innerHTML = '';
    top5.slice(0, 5).forEach((item, i) => {
      const pct = Math.round(item[1] * 100);
      const li = document.createElement('li');
      li.className = 'candidate-item';
      li.innerHTML = `
        <span class="candidate-key">${item[0]}</span>
        <div class="candidate-bar"><div class="candidate-fill ${i === 0 ? 'top' : ''}" style="width:${pct}%"></div></div>
        <span class="candidate-pct">${pct}%</span>`;
      list.appendChild(li);
    });
  }
}

function setStatus(msg) {
  const el = $('status-msg');
  if (el) el.textContent = msg;
}

/* ===== WORD OUTPUT ===== */
function appendLetter(letter) {
  wordText += letter;
  renderWord();
}

function renderWord() {
  const el = $('word-output');
  if (!el) return;
  el.innerHTML = wordText + '<span class="cursor"></span>';
}

function clearWord() {
  wordText = '';
  renderWord();
  holdState = { pred: null, startedAt: null };
}

function addSpace() {
  wordText += ' ';
  renderWord();
}

function backspace() {
  wordText = wordText.slice(0, -1);
  renderWord();
}

/* ===== UPLOAD IMAGE ===== */
function initUpload() {
  const dropArea = $('drop-area');
  const fileInput = $('file-input');
  const preview = $('upload-preview');
  const previewImg = $('preview-img');

  if (!dropArea) return;

  dropArea.addEventListener('click', () => fileInput.click());

  dropArea.addEventListener('dragover', e => {
    e.preventDefault();
    dropArea.classList.add('drag-over');
  });

  dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag-over'));

  dropArea.addEventListener('drop', e => {
    e.preventDefault();
    dropArea.classList.remove('drag-over');
    handleFile(e.dataTransfer.files[0]);
  });

  fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));
}

async function handleFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = async (e) => {
    const base64 = e.target.result.split(',')[1];
    const previewImg = $('preview-img');
    const preview = $('upload-preview');
    const dropArea = $('drop-area');
    const resultBox = $('upload-result');
    const resultLetter = $('result-letter');
    const resultConf = $('result-conf');
    const topList = $('upload-top5');

    if (previewImg) previewImg.src = e.target.result;
    if (preview) { preview.style.display = 'flex'; }
    if (dropArea) hide(dropArea);
    if (resultBox) resultBox.innerHTML = '<div class="spinner"></div>';

    try {
      const res = await fetch(PREDICT_API, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 }),
      });
      const data = await res.json();

      if (data.error) {
        if (resultBox) resultBox.innerHTML = `<p style="color:var(--danger)">${data.error}</p>`;
        return;
      }

      const pct = Math.round(data.confidence * 100);
      let topHtml = (data.top5 || []).slice(0, 5).map(([k, v]) =>
        `<div class="candidate-item">
          <span class="candidate-key">${k}</span>
          <div class="candidate-bar"><div class="candidate-fill top" style="width:${Math.round(v*100)}%"></div></div>
          <span class="candidate-pct">${Math.round(v*100)}%</span>
        </div>`).join('');

      if (resultBox) {
        resultBox.innerHTML = `
          <div class="result-letter">${data.letter}</div>
          <div class="result-conf">${pct}% confidence</div>
          <div style="margin-top:1rem;">${topHtml}</div>`;
      }
    } catch (err) {
      if (resultBox) resultBox.innerHTML = `<p style="color:var(--danger)">API error — is the model running?</p>`;
    }
  };
  reader.readAsDataURL(file);
}

function resetUpload() {
  const dropArea = $('drop-area');
  const preview = $('upload-preview');
  const fileInput = $('file-input');
  if (dropArea) show(dropArea);
  if (preview) preview.style.display = 'none';
  if (fileInput) fileInput.value = '';
}

/* ===== INIT ===== */
document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  initTabs();
  initUpload();
  renderWord();

  // Button wiring
  const startBtn = $('start-btn');
  const stopBtn = $('stop-btn');
  if (startBtn) startBtn.addEventListener('click', startCamera);
  if (stopBtn) stopBtn.addEventListener('click', stopCamera);
  const clearBtn = $('clear-btn');
  const spaceBtn = $('space-btn');
  const bsBtn = $('backspace-btn');
  if (clearBtn) clearBtn.addEventListener('click', clearWord);
  if (spaceBtn) spaceBtn.addEventListener('click', addSpace);
  if (bsBtn) bsBtn.addEventListener('click', backspace);
  const themeBtn = $('theme-toggle');
  if (themeBtn) themeBtn.addEventListener('click', toggleTheme);
  const resetUploadBtn = $('reset-upload');
  if (resetUploadBtn) resetUploadBtn.addEventListener('click', resetUpload);
});
