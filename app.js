/* =============================================================================
 * Frontfläche - live frontal-area estimator for alpine ski racing
 * -----------------------------------------------------------------------------
 * Goal: while a skier holds a tuck in front of the (front/selfie) camera, show
 * the projected frontal area in m^2, live and glanceable from a few metres away.
 *
 * Measurement principle
 * ---------------------
 *   1. The skier is a dark silhouette against bright snow -> detect by a
 *      luminance threshold, then keep only the LARGEST connected dark blob
 *      (this rejects shadows, spectators, distant dark objects, speckle).
 *   2. A neon ball of KNOWN diameter lies on the snow at foot level. It is the
 *      metric reference: its apparent pixel diameter gives the scale
 *      s = D_real / d_px  [metres per pixel].
 *   3. Frontal area  A = N_person * s^2   [m^2]
 *      where N_person is the dark-blob pixel count.
 *
 * Why this is resolution-independent (a nice property to rely on):
 *   If we change the processing resolution by a factor k, then
 *   N_person scales with k^2 and d_px scales with k, so s^2 scales with 1/k^2,
 *   and A = N_person * s^2 stays the same. Lowering the resolution only trades
 *   accuracy/jitter for speed, never the calibration.
 *
 * Scope (by design): this targets a SINGLE athlete comparing their own poses.
 * Absolute m^2 values carry biases (lens distortion, ground-level ski area,
 * body-vs-foot depth offset), but those biases stay roughly constant between
 * poses, so the RELATIVE change a skier sees while adjusting their tuck is what
 * matters and what this tool reports reliably.
 * ===========================================================================*/

'use strict';

// ---------------------------------------------------------------------------
// Central application state. One object so it is easy to inspect/debug.
// ---------------------------------------------------------------------------
const state = {
  // --- Media / canvases -----------------------------------------------------
  video: null,             // hidden <video> carrying the camera stream
  procCanvas: null,        // hidden low-res canvas used for pixel processing
  procCtx: null,
  viewCanvas: null,        // visible canvas: mirrored video + overlays
  viewCtx: null,

  // --- Camera ---------------------------------------------------------------
  stream: null,
  facing: 'user',          // 'user' = front/selfie camera, 'environment' = rear
  running: false,

  // --- Processing resolution ------------------------------------------------
  // Width the camera frame is downscaled to before analysis. Height follows the
  // video aspect ratio. Smaller = faster, larger = less jitter.
  procW: 320,
  procH: 240,

  // --- Detection parameters (user-tunable) ---------------------------------
  darkThreshold: 90,       // luminance 0..255; pixels darker than this = skier
  ballDiameterM: 0.22,     // real ball diameter in metres (size-5 football ~0.22)
  floorPct: 100,           // ignore skier pixels BELOW this % of frame height
                           // (lets the user crop out skis / foreground snow)

  // --- Ball colour target (HSV), picked by tapping the ball -----------------
  ballColor: null,         // { h, s, v } once picked, else null
  hueTol: 22,              // hue tolerance in degrees (0..180 here, see rgb2hsv)
  satMin: 0.35,            // minimum saturation to count as the (neon) ball
  valMin: 0.30,            // minimum value/brightness

  // --- Reusable buffers (allocated once in resizeBuffers) -------------------
  personLabels: null,      // Int32Array: connected-component labels for skier
  ballLabels: null,        // Int32Array: connected-component labels for ball
  personMask: null,        // Uint8Array: 1 where pixel is dark (candidate skier)
  ballMask: null,          // Uint8Array: 1 where pixel matches ball colour
  bfsStack: null,          // Int32Array: explicit stack for flood fill

  // --- Results --------------------------------------------------------------
  areaRaw: null,           // last raw frontal area [m^2]
  areaSmoothed: null,      // exponentially smoothed area for a calm display
  bestArea: null,          // smallest (best/most aero) area seen this session
  refArea: null,           // user-saved reference pose to compare against

  // --- Audio feedback -------------------------------------------------------
  audioOn: false,
  audioCtx: null,
  oscillator: null,
  gainNode: null,
};

// Smoothing factor for the exponential moving average (0..1). Higher = calmer
// but laggier display. 0.8 is a good compromise for a glanceable readout.
const EMA_ALPHA = 0.8;

// =============================================================================
// Colour helpers
// =============================================================================

/**
 * Standard Rec. 601 luminance. Cheap and adequate for "dark vs snow".
 */
function luminance(r, g, b) {
  return 0.299 * r + 0.587 * g + 0.114 * b;
}

/**
 * Convert RGB (0..255) to HSV with hue in 0..360, s and v in 0..1.
 * We match the ball by hue, so HSV is far more lighting-robust than raw RGB.
 */
function rgb2hsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const d = max - min;
  let h = 0;
  if (d !== 0) {
    if (max === r)      h = ((g - b) / d) % 6;
    else if (max === g) h = (b - r) / d + 2;
    else                h = (r - g) / d + 4;
    h *= 60;
    if (h < 0) h += 360;
  }
  const s = max === 0 ? 0 : d / max;
  const v = max;
  return { h, s, v };
}

/**
 * Smallest absolute difference between two hues on the 0..360 circle.
 */
function hueDistance(h1, h2) {
  const d = Math.abs(h1 - h2) % 360;
  return d > 180 ? 360 - d : d;
}

// =============================================================================
// Connected components: find the LARGEST blob in a binary mask
// =============================================================================

/**
 * Label connected components (4-connectivity) of `mask` and return the largest.
 * Uses an explicit stack flood fill to stay within JS recursion limits and to
 * run fast at video rates. Buffers are reused across frames to avoid GC churn.
 *
 * @param {Uint8Array} mask    - 1 = foreground pixel, 0 = background
 * @param {number} w
 * @param {number} h
 * @param {Int32Array} labels  - scratch buffer (w*h), will be overwritten
 * @param {Int32Array} stack   - scratch buffer (>= w*h) for the flood fill
 * @returns {{count:number,minX:number,minY:number,maxX:number,maxY:number,
 *            bestLabel:number}|null}  stats of the largest blob, or null if none
 */
function largestBlob(mask, w, h, labels, stack) {
  labels.fill(0);                 // 0 means "not yet labelled"
  let nextLabel = 0;
  let bestLabel = -1;
  let bestCount = 0;
  let bMinX = 0, bMinY = 0, bMaxX = 0, bMaxY = 0;

  const n = w * h;
  for (let start = 0; start < n; start++) {
    // Skip background or already-labelled pixels.
    if (mask[start] === 0 || labels[start] !== 0) continue;

    nextLabel++;
    let sp = 0;                    // stack pointer
    stack[sp++] = start;
    labels[start] = nextLabel;

    let count = 0;
    let minX = w, minY = h, maxX = 0, maxY = 0;

    // Iterative flood fill of this component.
    while (sp > 0) {
      const idx = stack[--sp];
      const x = idx % w;
      const y = (idx - x) / w;

      count++;
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;

      // 4-connected neighbours; push if foreground and unlabelled.
      if (x > 0) {
        const nIdx = idx - 1;
        if (mask[nIdx] === 1 && labels[nIdx] === 0) { labels[nIdx] = nextLabel; stack[sp++] = nIdx; }
      }
      if (x < w - 1) {
        const nIdx = idx + 1;
        if (mask[nIdx] === 1 && labels[nIdx] === 0) { labels[nIdx] = nextLabel; stack[sp++] = nIdx; }
      }
      if (y > 0) {
        const nIdx = idx - w;
        if (mask[nIdx] === 1 && labels[nIdx] === 0) { labels[nIdx] = nextLabel; stack[sp++] = nIdx; }
      }
      if (y < h - 1) {
        const nIdx = idx + w;
        if (mask[nIdx] === 1 && labels[nIdx] === 0) { labels[nIdx] = nextLabel; stack[sp++] = nIdx; }
      }
    }

    if (count > bestCount) {
      bestCount = count;
      bestLabel = nextLabel;
      bMinX = minX; bMinY = minY; bMaxX = maxX; bMaxY = maxY;
    }
  }

  if (bestLabel === -1) return null;
  return { count: bestCount, minX: bMinX, minY: bMinY, maxX: bMaxX, maxY: bMaxY, bestLabel };
}

// =============================================================================
// Camera setup
// =============================================================================

async function startCamera() {
  // Stop any previous stream first (e.g. when switching front/rear camera).
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }

  try {
    // Request the chosen camera. We ask for a moderate resolution; the browser
    // picks the closest supported mode.
    state.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: state.facing,
        width:  { ideal: 1280 },
        height: { ideal: 720 },
      },
    });
  } catch (err) {
    setStatus('Kamera nicht verfügbar – Zugriff erlauben und Seite neu laden.');
    console.error(err);
    return;
  }

  state.video.srcObject = state.stream;
  await state.video.play();

  // Derive processing height from the actual video aspect ratio so circles stay
  // circular and areas are not distorted.
  const aspect = state.video.videoHeight / state.video.videoWidth || 0.75;
  state.procH = Math.round(state.procW * aspect);
  resizeBuffers();
  sizeViewCanvas();

  state.running = true;
  setStatus('Bereit. Ball antippen, um die Farbe zu wählen.');
  requestAnimationFrame(loop);
}

/**
 * (Re)allocate all per-pixel buffers for the current processing resolution.
 * Called once per camera start / resolution change, never per frame.
 */
function resizeBuffers() {
  const n = state.procW * state.procH;
  state.personMask   = new Uint8Array(n);
  state.ballMask     = new Uint8Array(n);
  state.personLabels = new Int32Array(n);
  state.ballLabels   = new Int32Array(n);
  state.bfsStack     = new Int32Array(n);

  state.procCanvas.width  = state.procW;
  state.procCanvas.height = state.procH;
}

/**
 * Match the visible canvas to its CSS box so overlays line up with what the
 * user sees and tap coordinates map cleanly.
 */
function sizeViewCanvas() {
  const rect = state.viewCanvas.getBoundingClientRect();
  // Internal resolution = displayed CSS size (kept simple; crisp enough here).
  state.viewCanvas.width  = Math.max(1, Math.round(rect.width));
  state.viewCanvas.height = Math.max(1, Math.round(rect.width * state.procH / state.procW));
}

// =============================================================================
// Main loop: process one frame and render
// =============================================================================

function loop() {
  if (!state.running) return;

  processFrame();
  render();

  requestAnimationFrame(loop);
}

/**
 * Pull the current frame, build the two masks, find both blobs and compute area.
 */
function processFrame() {
  const { procW, procH, procCtx, video } = state;

  // Draw the frame MIRRORED into the processing canvas. Mirroring is purely for
  // a natural "mirror" feel; area is invariant under mirroring, so it does not
  // affect the measurement. Processing and display use the SAME mirrored frame,
  // which keeps tap-to-pick coordinates consistent.
  procCtx.save();
  procCtx.translate(procW, 0);
  procCtx.scale(-1, 1);
  procCtx.drawImage(video, 0, 0, procW, procH);
  procCtx.restore();

  const img = procCtx.getImageData(0, 0, procW, procH);
  const data = img.data;

  const { personMask, ballMask, ballColor, darkThreshold } = state;
  const floorY = Math.round(procH * state.floorPct / 100);

  // ---- Build per-pixel masks in a single pass ----------------------------
  for (let y = 0; y < procH; y++) {
    for (let x = 0; x < procW; x++) {
      const p = y * procW + x;
      const o = p * 4;
      const r = data[o], g = data[o + 1], b = data[o + 2];

      // Ball mask: only meaningful once a colour has been picked.
      let isBall = 0;
      if (ballColor) {
        const { h, s, v } = rgb2hsv(r, g, b);
        if (
          s >= state.satMin &&
          v >= state.valMin &&
          hueDistance(h, ballColor.h) <= state.hueTol
        ) {
          isBall = 1;
        }
      }
      ballMask[p] = isBall;

      // Person mask: dark pixel, above the floor line, and NOT a ball pixel.
      // Excluding ball pixels prevents the reference from leaking into the
      // silhouette if their regions ever touch.
      const dark = luminance(r, g, b) < darkThreshold ? 1 : 0;
      personMask[p] = (dark && !isBall && y < floorY) ? 1 : 0;
    }
  }

  // ---- Largest blobs -----------------------------------------------------
  const person = largestBlob(personMask, procW, procH, state.personLabels, state.bfsStack);
  const ball = ballColor
    ? largestBlob(ballMask, procW, procH, state.ballLabels, state.bfsStack)
    : null;

  state._person = person;   // stash for the renderer
  state._ball = ball;

  // ---- Area computation --------------------------------------------------
  // We need both a skier blob and a ball blob to have a metric scale.
  if (person && ball && ball.count > 0) {
    // Ball diameter in pixels: average the bounding-box width and height. This
    // is robust to a central specular highlight (which can punch a hole in the
    // colour-matched pixels but does not change the bounding box).
    const ballW = ball.maxX - ball.minX + 1;
    const ballH = ball.maxY - ball.minY + 1;
    const dPx = (ballW + ballH) / 2;

    if (dPx > 4) {                              // ignore implausibly tiny blobs
      const s = state.ballDiameterM / dPx;      // metres per pixel
      const area = person.count * s * s;        // m^2

      state.areaRaw = area;
      // Exponential moving average for a calm, readable number.
      state.areaSmoothed = state.areaSmoothed == null
        ? area
        : EMA_ALPHA * state.areaSmoothed + (1 - EMA_ALPHA) * area;

      // Track the best (smallest) pose of the session.
      if (state.bestArea == null || state.areaSmoothed < state.bestArea) {
        state.bestArea = state.areaSmoothed;
      }
      updateAudio();
    }
  } else {
    state.areaRaw = null;
  }
}

// =============================================================================
// Rendering: mirrored video + overlays + big readout
// =============================================================================

function render() {
  const { viewCtx, viewCanvas, video, procW, procH } = state;
  const vw = viewCanvas.width;
  const vh = viewCanvas.height;

  // Background: the mirrored live frame.
  viewCtx.save();
  viewCtx.translate(vw, 0);
  viewCtx.scale(-1, 1);
  viewCtx.drawImage(video, 0, 0, vw, vh);
  viewCtx.restore();

  // Scale factors from processing space to view space (no flip needed: both
  // are mirrored identically).
  const sx = vw / procW;
  const sy = vh / procH;

  // --- Tint the detected skier silhouette --------------------------------
  if (state._person) {
    const { personLabels } = state;
    const best = state._person.bestLabel;
    // Build a small overlay image at processing resolution, then scale it up.
    const overlay = state.procCtx.createImageData(procW, procH);
    const od = overlay.data;
    for (let p = 0; p < personLabels.length; p++) {
      if (personLabels[p] === best) {
        const o = p * 4;
        od[o] = 56; od[o + 1] = 189; od[o + 2] = 248; od[o + 3] = 90; // cyan-ish
      }
    }
    // Push the overlay through the hidden proc canvas, then draw scaled.
    state.procCtx.putImageData(overlay, 0, 0);
    viewCtx.imageSmoothingEnabled = false;
    viewCtx.drawImage(state.procCanvas, 0, 0, procW, procH, 0, 0, vw, vh);
  }

  // --- Outline the ball ---------------------------------------------------
  if (state._ball) {
    const b = state._ball;
    const cx = ((b.minX + b.maxX) / 2) * sx;
    const cy = ((b.minY + b.maxY) / 2) * sy;
    const rad = ((b.maxX - b.minX + b.maxY - b.minY) / 4) * ((sx + sy) / 2);
    viewCtx.beginPath();
    viewCtx.arc(cx, cy, rad, 0, Math.PI * 2);
    viewCtx.strokeStyle = '#f59e0b';
    viewCtx.lineWidth = 3;
    viewCtx.stroke();
  }

  // --- Floor line (skier pixels below it are ignored) --------------------
  if (state.floorPct < 100) {
    const y = vh * state.floorPct / 100;
    viewCtx.beginPath();
    viewCtx.moveTo(0, y);
    viewCtx.lineTo(vw, y);
    viewCtx.strokeStyle = 'rgba(255,255,255,0.55)';
    viewCtx.setLineDash([8, 6]);
    viewCtx.lineWidth = 2;
    viewCtx.stroke();
    viewCtx.setLineDash([]);
  }

  // The big numeric readout lives in the DOM (crisp text, easy styling),
  // not on the canvas. Update it here.
  updateReadout();
}

// =============================================================================
// Readout (DOM) + colour coding
// =============================================================================

/**
 * Map the current area to a feedback colour relative to the reference (or, if
 * no reference is saved, relative to the session best). Green = as good or
 * better, red = clearly worse.
 */
function feedbackColor(current) {
  const baseline = state.refArea != null ? state.refArea : state.bestArea;
  if (baseline == null || current == null) return '#e5e7eb'; // neutral light grey

  // Relative deviation; clamp to a +/-20% window for the gradient.
  const dev = (current - baseline) / baseline;
  const t = Math.max(0, Math.min(1, (dev + 0.0) / 0.20)); // 0 at/below baseline, 1 at +20%
  // Interpolate green -> amber -> red.
  if (t < 0.5) {
    return lerpColor('#22c55e', '#f59e0b', t / 0.5);
  }
  return lerpColor('#f59e0b', '#ef4444', (t - 0.5) / 0.5);
}

function lerpColor(a, b, t) {
  const pa = [parseInt(a.slice(1, 3), 16), parseInt(a.slice(3, 5), 16), parseInt(a.slice(5, 7), 16)];
  const pb = [parseInt(b.slice(1, 3), 16), parseInt(b.slice(3, 5), 16), parseInt(b.slice(5, 7), 16)];
  const r = Math.round(pa[0] + (pb[0] - pa[0]) * t);
  const g = Math.round(pa[1] + (pb[1] - pa[1]) * t);
  const bl = Math.round(pa[2] + (pb[2] - pa[2]) * t);
  return `rgb(${r},${g},${bl})`;
}

function updateReadout() {
  const areaEl = document.getElementById('area-value');
  const unitEl = document.getElementById('area-unit');
  const devEl  = document.getElementById('deviation');
  const bestEl = document.getElementById('best-value');

  if (state.areaSmoothed == null) {
    areaEl.textContent = '–';
    areaEl.style.color = '#e5e7eb';
    unitEl.style.opacity = '0.4';
    devEl.textContent = state.ballColor ? 'Suche Ball & Fahrer …' : 'Ball antippen zum Kalibrieren';
    bestEl.textContent = '–';
    return;
  }

  unitEl.style.opacity = '1';
  const a = state.areaSmoothed;
  areaEl.textContent = a.toFixed(3);
  areaEl.style.color = feedbackColor(a);

  // Deviation versus the saved reference, if any.
  if (state.refArea != null) {
    const dev = (a - state.refArea) / state.refArea * 100;
    const sign = dev > 0 ? '+' : '';
    devEl.textContent = `${sign}${dev.toFixed(1)} % zur Referenz`;
  } else {
    devEl.textContent = 'Keine Referenz gesetzt';
  }

  bestEl.textContent = state.bestArea != null ? `${state.bestArea.toFixed(3)} m²` : '–';
}

// =============================================================================
// Audio feedback (optional): pitch rises as the pose gets more aero
// =============================================================================

function ensureAudio() {
  if (state.audioCtx) return;
  state.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  state.oscillator = state.audioCtx.createOscillator();
  state.gainNode = state.audioCtx.createGain();
  state.oscillator.type = 'sine';
  state.gainNode.gain.value = 0.0;          // start silent; we ramp on demand
  state.oscillator.connect(state.gainNode).connect(state.audioCtx.destination);
  state.oscillator.start();
}

function updateAudio() {
  if (!state.audioOn || !state.audioCtx) return;
  const baseline = state.refArea != null ? state.refArea : state.bestArea;
  if (baseline == null || state.areaSmoothed == null) return;

  // ratio >= ~1 when worse than baseline, < 1 when better. Map smaller area to
  // a higher pitch so "better tuck" = "higher tone".
  const ratio = state.areaSmoothed / baseline;
  const freq = Math.max(200, Math.min(900, 600 / ratio));
  state.oscillator.frequency.setTargetAtTime(freq, state.audioCtx.currentTime, 0.05);
  state.gainNode.gain.setTargetAtTime(0.08, state.audioCtx.currentTime, 0.05);
}

function setAudio(on) {
  state.audioOn = on;
  if (on) {
    ensureAudio();
    if (state.audioCtx.state === 'suspended') state.audioCtx.resume();
  } else if (state.gainNode) {
    state.gainNode.gain.setTargetAtTime(0.0, state.audioCtx.currentTime, 0.05);
  }
}

// =============================================================================
// Tap-to-pick the ball colour
// =============================================================================

function onCanvasTap(e) {
  // Map the tap from CSS pixels to processing-canvas pixels.
  const rect = state.viewCanvas.getBoundingClientRect();
  const clientX = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const clientY = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;

  // CSS -> view-canvas internal pixels -> processing pixels.
  const xCanvas = clientX * (state.viewCanvas.width / rect.width);
  const yCanvas = clientY * (state.viewCanvas.height / rect.height);
  const px = Math.round(xCanvas * state.procW / state.viewCanvas.width);
  const py = Math.round(yCanvas * state.procH / state.viewCanvas.height);

  // Sample a small 3x3 neighbourhood and average for a stable colour pick.
  const img = state.procCtx.getImageData(
    Math.max(0, px - 1), Math.max(0, py - 1), 3, 3
  );
  let r = 0, g = 0, b = 0, n = 0;
  for (let i = 0; i < img.data.length; i += 4) {
    r += img.data[i]; g += img.data[i + 1]; b += img.data[i + 2]; n++;
  }
  r /= n; g /= n; b /= n;

  state.ballColor = rgb2hsv(r, g, b);
  // Reset session statistics: a fresh calibration starts a fresh comparison.
  state.areaSmoothed = null;
  state.bestArea = null;
  setStatus(`Ball-Farbe gesetzt (H ${Math.round(state.ballColor.h)}°). Hocke einnehmen.`);
}

// =============================================================================
// UI wiring
// =============================================================================

function setStatus(msg) {
  document.getElementById('status').textContent = msg;
}

function bindUI() {
  // Tap on the video to pick the ball colour (mouse + touch).
  state.viewCanvas.addEventListener('click', onCanvasTap);
  state.viewCanvas.addEventListener('touchstart', (e) => { e.preventDefault(); onCanvasTap(e); }, { passive: false });

  // Save current pose as the reference to compare against.
  document.getElementById('btn-reference').addEventListener('click', () => {
    if (state.areaSmoothed != null) {
      state.refArea = state.areaSmoothed;
      setStatus(`Referenz gespeichert: ${state.refArea.toFixed(3)} m².`);
    } else {
      setStatus('Noch keine Messung – zuerst Ball antippen und Hocke einnehmen.');
    }
  });

  // Reset the session best and the reference.
  document.getElementById('btn-reset').addEventListener('click', () => {
    state.bestArea = null;
    state.refArea = null;
    setStatus('Bestwert und Referenz zurückgesetzt.');
  });

  // Re-pick the ball colour: clear the current colour so the next canvas tap
  // picks a fresh target, and reset all derived measurements.
  document.getElementById('btn-ball').addEventListener('click', () => {
    state.ballColor = null;
    state.areaSmoothed = null;
    state.bestArea = null;
    setStatus('Ball antippen, um die Farbe neu zu wählen.');
  });

  // Switch front / rear camera.
  document.getElementById('btn-flip').addEventListener('click', () => {
    state.facing = state.facing === 'user' ? 'environment' : 'user';
    startCamera();
  });

  // Dark-threshold slider.
  const thr = document.getElementById('threshold');
  thr.addEventListener('input', () => {
    state.darkThreshold = Number(thr.value);
    document.getElementById('threshold-val').textContent = thr.value;
  });

  // Floor-line slider (crop out skis / foreground).
  const floor = document.getElementById('floor');
  floor.addEventListener('input', () => {
    state.floorPct = Number(floor.value);
    document.getElementById('floor-val').textContent = `${floor.value} %`;
  });

  // Ball diameter input (metres).
  const dia = document.getElementById('ball-diameter');
  dia.addEventListener('input', () => {
    const v = parseFloat(dia.value);
    if (!Number.isNaN(v) && v > 0) state.ballDiameterM = v;
  });

  // Processing resolution selector.
  const res = document.getElementById('resolution');
  res.addEventListener('change', () => {
    state.procW = Number(res.value);
    // Recompute processing height from the real video aspect ratio.
    const aspect = state.video.videoHeight / state.video.videoWidth || 0.75;
    state.procH = Math.round(state.procW * aspect);
    resizeBuffers();
    sizeViewCanvas();
  });

  // Audio toggle.
  const audio = document.getElementById('audio-toggle');
  audio.addEventListener('change', () => setAudio(audio.checked));

  // Settings panel show/hide.
  const panel = document.getElementById('panel');
  document.getElementById('btn-settings').addEventListener('click', () => {
    panel.classList.toggle('open');
  });

  // Keep canvases sized correctly on rotation / resize.
  window.addEventListener('resize', () => { if (state.running) sizeViewCanvas(); });
}

// =============================================================================
// Boot
// =============================================================================

function init() {
  state.video      = document.getElementById('video');
  state.procCanvas = document.getElementById('proc');
  state.procCtx    = state.procCanvas.getContext('2d', { willReadFrequently: true });
  state.viewCanvas = document.getElementById('view');
  state.viewCtx    = state.viewCanvas.getContext('2d');

  bindUI();

  // Start the camera on first user gesture (autoplay/permission friendly).
  document.getElementById('btn-start').addEventListener('click', async () => {
    document.getElementById('start-overlay').style.display = 'none';
    await startCamera();
  });

  // Register the service worker for offline use / installability.
  if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('sw.js').catch((e) => console.warn('SW:', e));
  }
}

document.addEventListener('DOMContentLoaded', init);
