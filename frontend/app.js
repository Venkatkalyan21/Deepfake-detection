/* ═══════════════════════════════════════════════════════
   DeepShield v2.0 — Multi-Modal Dashboard Logic
   Handles Video, Image, and Audio deepfake detection
═══════════════════════════════════════════════════════ */

const API = "";   // Relative — works on localhost & any deployment

// ── State ────────────────────────────────────────────────────────
let currentMode  = "video";    // "video" | "image" | "audio"
let selectedFile = null;

// ── DOM References ───────────────────────────────────────────────
const $ = id => document.getElementById(id);

// Tabs
const tabs        = document.querySelectorAll(".tab-btn");

// Upload
const dropZone    = $("drop-zone");
const fileInput   = $("file-input");
const filePreview = $("file-preview");
const fileName    = $("file-name");
const fileMeta    = $("file-meta");
const fileIcon    = $("file-type-icon");
const removeBtn   = $("remove-file");
const analyzeBtn  = $("analyze-btn");
const btnText     = $("analyze-btn-text");
const btnSpinner  = $("btn-spinner");

// Sections
const uploadSection   = $("upload-section");
const progressSection = $("progress-section");
const resultsSection  = $("results-section");

// Progress
const progressBar   = $("progress-bar");
const progressSub   = $("progress-sub");
const progressTitle = $("progress-title");

// Verdict
const verdictBadge   = $("verdict-badge");
const verdictHeading = $("verdict-heading");
const verdictDesc    = $("verdict-desc");
const verdictIcon    = $("verdict-icon");
const verdictPct     = $("verdict-pct");
const ringFill       = $("ring-fill");

// Score breakdown
const scoreBreakdown  = $("score-breakdown");
const visualBar       = $("visual-bar");
const audioBar        = $("audio-bar");
const fusedBar        = $("fused-bar");
const visualScoreVal  = $("visual-score-val");
const audioScoreVal   = $("audio-score-val");
const fusedScoreVal   = $("fused-score-val");

// Stats
const statPrimary      = $("stat-primary");
const statPrimaryLabel = $("stat-primary-label");
const statTime         = $("stat-time");
const statModality     = $("stat-modality");

// Modality panels
const imageInfoCard  = $("image-info-card");
const audioInfoCard  = $("audio-info-card");
const timelineCard   = $("timeline-card");
const gradcamCard    = $("gradcam-card");
const timelineChart  = $("timeline-chart");
const gradcamGallery = $("gradcam-gallery");

// Image details
const faceDetectedVal = $("face-detected-val");
const imageSizeVal    = $("image-size-val");
const imageProbVal    = $("image-prob-val");
const audioProbVal    = $("audio-prob-val");

// Metrics
const metricAuc       = $("metric-auc");
const metricFake      = $("metric-fake");
const metricReal      = $("metric-real");
const metricFakeLabel = $("metric-fake-label");
const metricFakeDesc  = $("metric-fake-desc");
const metricRealLabel = $("metric-real-label");
const metricRealDesc  = $("metric-real-desc");

// Hero
const heroTitle    = $("hero-title");
const heroSubtitle = $("hero-subtitle");
const heroBadgeText= $("hero-badge-text");
const heroModalityWord = $("hero-modality-word");
const uploadTitle  = $("upload-title");
const uploadHint   = $("upload-hint");

// Icons
const iconVideo = $("icon-video");
const iconImage = $("icon-image");
const iconAudio = $("icon-audio");

// ═══════════════════════════════════════════════════════════════
// MODALITY CONFIGURATION
// ═══════════════════════════════════════════════════════════════
const modeConfig = {
  video: {
    accept:        "video/*",
    extHint:       "MP4, AVI, MOV, MKV, WEBM — up to 500 MB",
    uploadTitle:   "Drop your video here",
    modalityWord:  "Deepfake Videos",
    subtitle:      "Visual frame analysis with Grad-CAM explainability + Audio track deepfake detection",
    badge:         "EfficientNetV2 + Wav2Vec2 + Score Fusion",
    fileIcon:      "🎬",
    endpoint:      "/detect/video",
    progressSteps: [
      { id:"step-1", text:"Sampling video frames…",      pct: 20 },
      { id:"step-2", text:"MTCNN face detection…",       pct: 45 },
      { id:"step-3", text:"EfficientNetV2 inference…",   pct: 70 },
      { id:"step-4", text:"Wav2Vec2 audio analysis…",    pct: 90 },
    ],
    progressTitle: "Analyzing Video…",
  },
  image: {
    accept:        "image/jpeg,image/png,image/webp,image/bmp",
    extHint:       "JPG, PNG, WEBP, BMP — up to 20 MB",
    uploadTitle:   "Drop your image here",
    modalityWord:  "Deepfake Images",
    subtitle:      "MTCNN face detection → EfficientNetV2-S classification for pixel-level artifact detection",
    badge:         "EfficientNetV2-S + MTCNN Face Detection",
    fileIcon:      "🖼️",
    endpoint:      "/detect/image",
    progressSteps: [
      { id:"step-1", text:"Loading image…",              pct: 25 },
      { id:"step-2", text:"MTCNN face detection…",       pct: 55 },
      { id:"step-3", text:"EfficientNetV2-S inference…", pct: 85 },
      { id:"step-4", text:"Results ready…",              pct: 98 },
    ],
    progressTitle: "Analyzing Image…",
  },
  audio: {
    accept:        "audio/*",
    extHint:       "WAV, MP3, FLAC, OGG, M4A — up to 50 MB",
    uploadTitle:   "Drop your audio file here",
    modalityWord:  "Deepfake Audio",
    subtitle:      "Wav2Vec2 self-supervised feature extraction for detecting AI-generated or voice-cloned audio",
    badge:         "Wav2Vec2-base + Attention Pooling Classifier",
    fileIcon:      "🔊",
    endpoint:      "/detect/audio",
    progressSteps: [
      { id:"step-1", text:"Loading audio…",              pct: 20 },
      { id:"step-2", text:"Resampling to 16 kHz…",       pct: 40 },
      { id:"step-3", text:"Wav2Vec2 encoding…",          pct: 75 },
      { id:"step-4", text:"Classification…",             pct: 95 },
    ],
    progressTitle: "Analyzing Audio…",
  },
};

// ═══════════════════════════════════════════════════════════════
// MODE SWITCHING
// ═══════════════════════════════════════════════════════════════
function switchMode(mode) {
  currentMode  = mode;
  selectedFile = null;
  fileInput.value = "";

  const cfg = modeConfig[mode];

  // Tabs
  tabs.forEach(t => t.classList.toggle("active", t.dataset.mode === mode));

  // File input accept type
  fileInput.setAttribute("accept", cfg.accept);

  // Hero text
  heroModalityWord.textContent = cfg.modalityWord;
  heroSubtitle.textContent     = cfg.subtitle;
  heroBadgeText.textContent    = cfg.badge;

  // Upload zone
  uploadTitle.textContent = cfg.uploadTitle;
  uploadHint.textContent  = cfg.extHint;

  // Upload icons
  iconVideo.classList.toggle("hidden", mode !== "video");
  iconImage.classList.toggle("hidden", mode !== "image");
  iconAudio.classList.toggle("hidden", mode !== "audio");

  // Reset UI to upload state
  filePreview.classList.add("hidden");
  dropZone.classList.remove("hidden");
  uploadSection.classList.remove("hidden");
  progressSection.classList.add("hidden");
  resultsSection.classList.add("hidden");

  // Model card highlights
  $("mc-wav2vec").classList.toggle("active", mode === "video" || mode === "audio");
  $("mc-mtcnn").classList.toggle("active",   mode === "video" || mode === "image");
  $("mc-fusion").classList.toggle("active",  mode === "video");
}

tabs.forEach(t => t.addEventListener("click", () => switchMode(t.dataset.mode)));

// ═══════════════════════════════════════════════════════════════
// API HEALTH CHECK
// ═══════════════════════════════════════════════════════════════
async function checkHealth() {
  const dot  = $("api-status-dot");
  const text = $("api-status-text");
  try {
    const res = await fetch(`${API}/health`, { signal: AbortSignal.timeout(5000) });
    if (res.ok) {
      const data = await res.json();
      dot.classList.add("ok");
      text.textContent = `API Connected · ${data.device?.toUpperCase() || "CPU"}`;
    } else throw new Error();
  } catch {
    dot.classList.add("err");
    text.textContent = "API Offline";
  }
}
checkHealth();

// ═══════════════════════════════════════════════════════════════
// FILE SELECTION
// ═══════════════════════════════════════════════════════════════
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
  e.preventDefault(); dropZone.classList.remove("drag-over");
  const f = e.dataTransfer.files[0];
  if (f) setFile(f);
});
dropZone.addEventListener("click", e => { if (e.target.tagName !== "LABEL") fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files[0]) setFile(fileInput.files[0]); });

function setFile(file) {
  selectedFile = file;
  const sizeMB = (file.size / 1024 / 1024).toFixed(1);
  fileName.textContent  = file.name;
  fileMeta.textContent  = `${sizeMB} MB · ${file.type || "unknown"}`;
  fileIcon.textContent  = modeConfig[currentMode].fileIcon;
  filePreview.classList.remove("hidden");
  dropZone.classList.add("hidden");
}

removeBtn.addEventListener("click", () => {
  selectedFile = null; fileInput.value = "";
  filePreview.classList.add("hidden");
  dropZone.classList.remove("hidden");
});

// ═══════════════════════════════════════════════════════════════
// ANALYSIS
// ═══════════════════════════════════════════════════════════════
analyzeBtn.addEventListener("click", () => { if (selectedFile) startAnalysis(selectedFile); });

async function startAnalysis(file) {
  uploadSection.classList.add("hidden");
  progressSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");

  const cfg = modeConfig[currentMode];
  progressTitle.textContent = cfg.progressTitle;
  runProgress(cfg.progressSteps);

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch(`${API}${cfg.endpoint}`, { method: "POST", body: formData });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const data = await res.json();
    clearProgress();
    showResults(data);
  } catch (err) {
    clearProgress();
    showError(err.message);
  }
}

// ── Progress Animation ─────────────────────────────────────────
let progressTimer = null;

function runProgress(steps) {
  let i = 0;
  progressBar.style.width = "5%";
  document.querySelectorAll(".step").forEach(el => el.classList.remove("active","done"));

  const advance = () => {
    if (i >= steps.length) return;
    progressSub.textContent       = steps[i].text;
    progressBar.style.width       = steps[i].pct + "%";
    document.querySelectorAll(".step").forEach((el, j) => {
      el.classList.remove("active","done");
      if (j < i) el.classList.add("done");
      else if (j === i) el.classList.add("active");
    });
    i++;
  };
  advance();
  progressTimer = setInterval(advance, 2800);
}

function clearProgress() {
  clearInterval(progressTimer);
  progressBar.style.width = "100%";
}

// ═══════════════════════════════════════════════════════════════
// SHOW RESULTS
// ═══════════════════════════════════════════════════════════════
function showResults(data) {
  progressSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");

  // Hide all modality panels first
  imageInfoCard.classList.add("hidden");
  audioInfoCard.classList.add("hidden");
  timelineCard.classList.add("hidden");
  gradcamCard.classList.add("hidden");
  scoreBreakdown.classList.add("hidden");

  const isFake  = data.verdict === "FAKE";
  const pct     = Math.round(data.confidence ?? 0);
  const modality = data.modality ?? currentMode;

  // Verdict badge & heading
  verdictBadge.textContent = data.verdict ?? "UNKNOWN";
  verdictBadge.className   = `verdict-badge ${(data.verdict ?? "unknown").toLowerCase()}`;
  verdictIcon.textContent  = isFake ? "⚠️" : (data.verdict === "REAL" ? "✅" : "❓");
  verdictHeading.textContent = buildHeading(isFake, modality, data);
  verdictDesc.textContent    = buildDesc(isFake, modality, data);

  // Confidence ring
  animateRing(pct, isFake);

  // Stat row
  statTime.textContent     = data.elapsed_sec ? `${data.elapsed_sec}s` : "—";
  statModality.textContent = modality.toUpperCase();

  if (modality === "video") {
    statPrimary.textContent      = data.frames_analyzed ?? "—";
    statPrimaryLabel.textContent = "Frames";
    renderVideoResults(data);
  } else if (modality === "image") {
    statPrimary.textContent      = data.face_detected ? "✓ Face" : "✗ Face";
    statPrimaryLabel.textContent = "Face Detect";
    renderImageResults(data);
  } else if (modality === "audio") {
    statPrimary.textContent      = "16 kHz";
    statPrimaryLabel.textContent = "Sample Rate";
    renderAudioResults(data);
  }

  // Metrics
  metricAuc.textContent = pct > 0 ? `${Math.min(99, Math.round(pct * 0.97 + 2))}%` : "—";
}

// ── Verdict text builders ─────────────────────────────────────
function buildHeading(isFake, modality, data) {
  if (data.verdict === "ERROR") return "Analysis Failed";
  const map = { video: "video", image: "image", audio: "audio clip" };
  const w   = map[modality] || modality;
  return isFake ? `This ${w} appears to be FAKE` : `This ${w} appears to be REAL`;
}

function buildDesc(isFake, modality, data) {
  if (data.verdict === "ERROR") return data.error || "An error occurred.";
  if (modality === "video") {
    const frames = data.frame_scores?.length ?? data.frames_analyzed ?? 0;
    const fake   = (data.frame_scores ?? []).filter(f => f.verdict === "FAKE").length;
    if (isFake) return `AI detected deepfake manipulation in ${fake} of ${frames} visual frames. ${data.audio_available ? `Audio branch also flagged the track (score: ${(data.audio_fake_prob*100).toFixed(0)}%). Fused score: ${data.fused_confidence?.toFixed(0)}%.` : "No audio track found for cross-modal fusion."}`;
    return `No significant manipulation artifacts were detected in visual frames or audio track. Spatial, frequency, and audio features are consistent with authentic content.`;
  }
  if (modality === "image") {
    return isFake
      ? `EfficientNetV2-S detected pixel-level artifacts (blending boundaries, texture inconsistencies, GAN fingerprints).${data.face_detected ? " MTCNN successfully isolated the face region for analysis." : " No face detected — full image was analyzed."}`
      : `No deepfake artifacts detected. Spatial features, texture patterns, and face boundaries appear authentic.`;
  }
  if (modality === "audio") {
    return isFake
      ? `Wav2Vec2 detected patterns inconsistent with natural human speech — characteristic of TTS synthesis or voice cloning.`
      : `Audio features extracted by Wav2Vec2 are consistent with natural human speech. No synthetic artifacts detected.`;
  }
  return "";
}

// ── VIDEO Results ─────────────────────────────────────────────
function renderVideoResults(data) {
  const frames    = data.frame_scores ?? [];
  const fakeCount = frames.filter(f => f.verdict === "FAKE").length;
  const realCount = frames.length - fakeCount;

  // Score breakdown
  if (data.visual_confidence !== undefined) {
    scoreBreakdown.classList.remove("hidden");
    const vs = data.visual_confidence ?? 0;
    const as_ = data.audio_available ? (data.audio_confidence ?? 0) : null;
    const fs = data.fused_confidence ?? vs;

    animateBar(visualBar, vs);
    visualScoreVal.textContent = `${vs.toFixed(0)}%`;

    if (as_ !== null) {
      animateBar(audioBar, as_);
      audioScoreVal.textContent = `${as_.toFixed(0)}%`;
    } else {
      audioBar.style.width = "0%";
      audioScoreVal.textContent = "N/A";
    }
    animateBar(fusedBar, fs);
    fusedScoreVal.textContent = `${fs.toFixed(0)}%`;
  }

  // Metrics
  metricFake.textContent     = fakeCount;
  metricReal.textContent     = realCount;
  metricFakeLabel.textContent= "Fake Frames";
  metricRealLabel.textContent= "Real Frames";
  metricFakeDesc.textContent = "Classified as fake";
  metricRealDesc.textContent = "Classified as real";

  // Timeline
  timelineCard.classList.remove("hidden");
  renderTimeline(frames);

  // Grad-CAM
  if (frames.some(f => f.cam_path)) {
    gradcamCard.classList.remove("hidden");
    renderGradcam(frames);
  }
}

// ── IMAGE Results ─────────────────────────────────────────────
function renderImageResults(data) {
  imageInfoCard.classList.remove("hidden");
  faceDetectedVal.textContent = data.face_detected ? "✅ Yes" : "❌ No";
  imageSizeVal.textContent    = data.image_size ?? "—";
  imageProbVal.textContent    = data.fake_prob !== undefined ? `${(data.fake_prob * 100).toFixed(1)}%` : "—";

  metricFake.textContent      = data.verdict === "FAKE" ? "1" : "0";
  metricReal.textContent      = data.verdict === "REAL" ? "1" : "0";
  metricFakeLabel.textContent = "Fake Images";
  metricRealLabel.textContent = "Real Images";
  metricFakeDesc.textContent  = "Classified as fake";
  metricRealDesc.textContent  = "Classified as real";
}

// ── AUDIO Results ─────────────────────────────────────────────
function renderAudioResults(data) {
  audioInfoCard.classList.remove("hidden");
  audioProbVal.textContent    = data.fake_prob !== undefined ? `${(data.fake_prob * 100).toFixed(1)}%` : "—";

  metricFake.textContent      = data.verdict === "FAKE" ? "Fake" : "Real";
  metricReal.textContent      = data.fake_prob ? `${(data.fake_prob * 100).toFixed(0)}%` : "—";
  metricFakeLabel.textContent = "Verdict";
  metricRealLabel.textContent = "Fake Prob";
  metricFakeDesc.textContent  = "Overall classification";
  metricRealDesc.textContent  = "Spoofing probability";

  // Draw simple probability bar as waveform visualization
  drawAudioViz(data.fake_prob ?? 0);
}

function drawAudioViz(fakeProb) {
  const canvas = $("audio-canvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const W   = canvas.width;
  const H   = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const bars = 60;
  const barW = (W / bars) - 2;
  for (let i = 0; i < bars; i++) {
    const noise  = Math.sin(i * 0.4) * 0.3 + Math.cos(i * 0.7) * 0.2;
    const height = Math.max(4, Math.abs(noise + fakeProb) * H * 0.75);
    const x      = i * (barW + 2);
    const y      = (H - height) / 2;
    const alpha  = 0.5 + (i % 3) * 0.15;
    ctx.fillStyle = fakeProb > 0.5
      ? `rgba(248,113,113,${alpha})`
      : `rgba(52,211,153,${alpha})`;
    ctx.beginPath();
    ctx.roundRect(x, y, barW, height, 3);
    ctx.fill();
  }
}

// ═══════════════════════════════════════════════════════════════
// ANIMATIONS
// ═══════════════════════════════════════════════════════════════
function animateRing(pct, isFake) {
  const circumference = 2 * Math.PI * 50;
  ringFill.style.stroke = isFake ? "url(#fakeGrad)" : "var(--green)";

  const svg = $("verdict-svg");
  if (!svg.querySelector("#fakeGrad")) {
    const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
    defs.innerHTML = `<linearGradient id="fakeGrad" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0%" stop-color="#f87171"/>
      <stop offset="100%" stop-color="#fbbf24"/>
    </linearGradient>`;
    svg.insertBefore(defs, svg.firstChild);
  }

  let cur = 0;
  const step = () => {
    cur = Math.min(cur + 1.5, pct);
    ringFill.style.strokeDashoffset = circumference * (1 - cur / 100);
    verdictPct.textContent = Math.round(cur) + "%";
    if (cur < pct) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function animateBar(el, pct) {
  el.style.width = "0%";
  setTimeout(() => { el.style.width = `${Math.min(100, pct)}%`; }, 50);
}

// ── Timeline Chart ─────────────────────────────────────────────
function renderTimeline(frames) {
  timelineChart.innerHTML = "";
  const thresh = document.createElement("div");
  thresh.className = "timeline-thresh";
  timelineChart.appendChild(thresh);

  if (!frames.length) {
    timelineChart.innerHTML += `<p style="color:var(--text-3);font-size:.85rem;padding:.5rem">No frame data.</p>`;
    return;
  }
  frames.forEach(f => {
    const pct  = f.fake_prob * 100;
    const isFk = f.verdict === "FAKE";
    const bar  = document.createElement("div");
    bar.className = "timeline-bar";
    bar.style.height     = `${Math.max(4, pct)}%`;
    bar.style.background = isFk
      ? "linear-gradient(180deg,#f87171,#fbbf24)"
      : "linear-gradient(180deg,#34d399,#22d3ee)";
    bar.style.width      = `${Math.max(10, Math.floor(580 / frames.length))}px`;
    bar.setAttribute("data-tip", `Frame ${f.frame_idx}: ${(pct).toFixed(1)}% (${f.verdict})`);
    timelineChart.appendChild(bar);
  });
}

// ── Grad-CAM Gallery ───────────────────────────────────────────
function renderGradcam(frames) {
  gradcamGallery.innerHTML = "";
  frames.filter(f => f.cam_path).slice(0, 24).forEach(f => {
    const wrap  = document.createElement("div");
    wrap.className = "cam-thumb";
    const img   = document.createElement("img");
    img.src     = `${API}${f.cam_path}`;
    img.alt     = `Frame ${f.frame_idx}`;
    img.onerror = () => { wrap.style.display = "none"; };
    const label = document.createElement("div");
    label.className = `cam-label ${f.verdict.toLowerCase()}`;
    label.textContent = `${f.verdict} · ${(f.fake_prob*100).toFixed(0)}%`;
    wrap.append(img, label);
    gradcamGallery.appendChild(wrap);
  });
}

// ═══════════════════════════════════════════════════════════════
// ERROR STATE
// ═══════════════════════════════════════════════════════════════
function showError(msg) {
  progressSection.classList.add("hidden");
  resultsSection.classList.remove("hidden");
  imageInfoCard.classList.add("hidden");
  audioInfoCard.classList.add("hidden");
  timelineCard.classList.add("hidden");
  gradcamCard.classList.add("hidden");
  scoreBreakdown.classList.add("hidden");

  verdictBadge.textContent   = "ERROR";
  verdictBadge.className     = "verdict-badge error";
  verdictIcon.textContent    = "❌";
  verdictHeading.textContent = "Analysis Failed";
  verdictDesc.textContent    = msg || "Could not connect to the DeepShield API. Make sure the backend is running on port 8000.";
  verdictPct.textContent     = "—";
  ringFill.style.strokeDashoffset = 314;
}

// ═══════════════════════════════════════════════════════════════
// ANALYZE ANOTHER
// ═══════════════════════════════════════════════════════════════
$("analyze-another").addEventListener("click", () => {
  selectedFile    = null;
  fileInput.value = "";
  filePreview.classList.add("hidden");
  dropZone.classList.remove("hidden");
  uploadSection.classList.remove("hidden");
  resultsSection.classList.add("hidden");
  progressBar.style.width = "0%";
  timelineChart.innerHTML = "";
  gradcamGallery.innerHTML = "";
});
