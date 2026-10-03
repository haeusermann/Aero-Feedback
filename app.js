

/* =============================================================================
 * Frontfläche - live frontal-area estimator for alpine ski racing
 * -----------------------------------------------------------------------------
 * Goal: while a skier holds a tuck in front of the (rear) camera, show
 * the projected frontal area in m^2, live and glanceable from a few metres away.
 *
 * Setup:
 *   - One person holds the phone and films the skier with the REAR camera.
 *   - A reference ball (known diameter) is placed on the ground in the fixed
 *     "ball zone" at the bottom-LEFT of the frame (from the camera's view).
 *   - The skier stands to the RIGHT of the ball.
 *   - Audio feedback lets the skier hear whether their pose is better/worse.
 *
 * Measurement principle
 * ---------------------
 *   1. The skier is a dark silhouette against a bright background -> detect by
 *      a luminance threshold, then keep only the LARGEST connected dark blob
 *      (this rejects shadows, spectators, distant dark objects, speckle).
 *   2. The ball of KNOWN diameter lies in the fixed bottom-LEFT zone. It is
 *      the metric reference: its apparent pixel diameter gives the scale
 *      s = D_real / d_px  [metres per pixel].
 *   3. Frontal area  A = N_person * s^2   [m^2]
 *      where N_person is the dark-blob pixel count.
 *
 * Zone separation (new):
 *   - BALL zone: bottom-left third of the frame (height = 1/3, width = 1/3).
 *     Only ball-coloured pixels inside this zone contribute to the ball blob.
 *   - PERSON zone: everything OUTSIDE the ball zone. Only dark pixels there
 *     contribute to the person blob.
 *   This guarantees ball and person can never be confused, even if both are
 *   dark on a bright background.
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
  viewCanvas: null,        // visible canvas: video + overlays
  viewCtx: null,

  // --- Camera ---------------------------------------------------------------
  stream: null,
  facing: 'environment',   // DEFAULT: 'environment' = rear camera (realistic use)
                           //          'user'        = front/selfie camera
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

  // --- Ball zone (fixed: bottom-LEFT third of the frame) --------------------
  // Expressed as fractions of the frame; a 1/3 x 1/3 rectangle in the lower-
  // left corner. Change here if you later want it configurable.
  ballZone: { xFrac: 0.0, yFrac: 2 / 3, wFrac: 1 / 3, hFrac: 1 / 3 },

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
// Ball-zone helpers
// =============================================================================

/**
 * Return the ball zone in processing-pixel coordinates for the current frame.
 * x0,y0 inclusive; x1,y1 exclusive.
 */
function getBallZonePx() {
  const { procW, procH, ballZone } = state;
  const x0 = Math.round(ballZone.xFrac * procW);
  const y0 = Math.round(ballZone.yFrac * procH);
  const x1 = Math.round((ballZone.xFrac + ballZone.wFrac) * procW);
  const y1 = Math.round((ballZone.yFrac + ballZone.hFrac) * procH);
  return { x0, y0, x1, y1 };
}

/** True if the processing-pixel (x, y) lies inside the ball zone. */
function inBallZone(x, y, zone) {
  return x >= zone.x0 && x < zone.x1 && y >= zone.y0 && y < zone.y1;
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


  // Prevent the screen from sleeping while the camera is active.
  if ('wakeLock' in navigator) {
    try {
      state.wakeLock = await navigator.wakeLock.request('screen');
    } catch (e) {
      console.warn('Wake Lock nicht verfügbar:', e);
    }
  }

  state.running = true;
  setStatus('Bereit. Ball in die Zone unten links legen und antippen.');
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
 *
 * NOTE: With the rear camera we draw the frame UNMIRRORED (natural orientation).
 * The front camera is still supported via the "flip" button; when active, the
 * frame would appear naturally mirrored on screen. For simplicity and because
 * area is invariant under mirroring, processing and display use the same
 * (unmirrored) frame here. Tap-to-pick works consistently either way.
 */
function processFrame() {
  const { procW, procH, procCtx, video } = state;

  // Draw the frame UNMIRRORED into the processing canvas (rear-cam use case).
  procCtx.drawImage(video, 0, 0, procW, procH);

  const img = procCtx.getImageData(0, 0, procW, procH);
  const data = img.data;

  const { personMask, ballMask, ballColor, darkThreshold } = state;
  const floorY = Math.round(procH * state.floorPct / 100);
  const zone = getBallZonePx();

  // ---- Build per-pixel masks in a single pass ----------------------------
  // Zone separation:
  //   - Inside the ball zone: ONLY ball-colour pixels count (toward ball mask);
  //     person pixels inside the zone are ignored (set to 0).
  //   - Outside the ball zone: ONLY dark pixels count (toward person mask);
  //     ball mask is 0 outside the zone.
  for (let y = 0; y < procH; y++) {
    for (let x = 0; x < procW; x++) {
      const p = y * procW + x;
      const o = p * 4;
      const r = data[o], g = data[o + 1], b = data[o + 2];
      const insideZone = inBallZone(x, y, zone);

      // Ball mask: only meaningful once a colour has been picked AND inside zone.
      let isBall = 0;
      if (ballColor && insideZone) {
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

      // Person mask: dark pixel, above the floor line, OUTSIDE the ball zone.
      // The zone-exclusion guarantees ball and person stay separated regardless
      // of colour: inside the zone, person pixels never contribute.
      const dark = luminance(r, g, b) < darkThreshold ? 1 : 0;
      personMask[p] = (dark && !insideZone && y < floorY) ? 1 : 0;
    }
  }

  // ---- Largest blobs -----------------------------------------------------
  const person = largestBlob(personMask, procW, procH, state.personLabels, state.bfsStack);
  const ball = ballColor
    ? largestBlob(ballMask, procW, procH, state.ballLabels, state.bfsStack)
    : null;

  state._person = person;   // stash for the renderer
  state._ball = ball;
  state._zone = zone;

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
// Rendering: video + overlays + big readout
// =============================================================================

function render() {
  const { viewCtx, viewCanvas, video, procW, procH } = state;
  const vw = viewCanvas.width;
  const vh = viewCanvas.height;

  // Background: the live frame (unmirrored, rear-cam natural orientation).
  viewCtx.drawImage(video, 0, 0, vw, vh);

  // Scale factors from processing space to view space.
  const sx = vw / procW;
  const sy = vh / procH;

  // --- Ball zone rectangle (always visible, so the user knows where
  //     to place the ball) ------------------------------------------------
  if (state._zone) {
    const z = state._zone;
    const zx = z.x0 * sx;
    const zy = z.y0 * sy;
    const zw = (z.x1 - z.x0) * sx;
    const zh = (z.y1 - z.y0) * sy;

    // Soft fill so the zone reads as a reserved area, not a cutout.
    viewCtx.fillStyle = 'rgba(56, 189, 248, 0.10)';   // light cyan tint
    viewCtx.fillRect(zx, zy, zw, zh);

    // Fine dashed border; cyan if no ball yet, amber once calibrated.
    viewCtx.strokeStyle = state.ballColor ? '#f59e0b' : 'rgba(56, 189, 248, 0.9)';
    viewCtx.lineWidth = 2;
    viewCtx.setLineDash([6, 5]);
    viewCtx.strokeRect(zx + 1, zy + 1, zw - 2, zh - 2);
    viewCtx.setLineDash([]);

    // Small label at the top of the zone.
    viewCtx.font = '12px system-ui, -apple-system, sans-serif';
    viewCtx.fillStyle = 'rgba(230, 240, 255, 0.9)';
    viewCtx.textBaseline = 'top';
    viewCtx.fillText('BALL HIER', zx + 6, zy + 4);
  }

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
    devEl.textContent = state.ballColor ? 'Suche Ball & Fahrer …' : 'Ball in der Zone antippen';
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

  // Reject taps outside the ball zone: force the user to tap on the ball
  // where it belongs, which also prevents accidental colour picks on dark
  // parts of the skier or the background.
  const zone = getBallZonePx();
  if (!inBallZone(px, py, zone)) {
    setStatus('Bitte in der markierten Zone unten links auf den Ball tippen.');
    return;
  }

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
    setStatus('Ball in der Zone antippen, um die Farbe neu zu wählen.');
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

  // Ball-Durchmesser Hilfe-Modal öffnen / schliessen.
  const ballHelpModal = document.getElementById('ball-help-modal');
  document.getElementById('btn-ball-help').addEventListener('click', (e) => {
    e.preventDefault();          // verhindert, dass das Label das Input fokussiert
    ballHelpModal.hidden = false;
  });
  document.getElementById('btn-ball-help-close').addEventListener('click', () => {
    ballHelpModal.hidden = true;
  });
  // Tap auf den Hintergrund (ausserhalb der Modal-Card) schliesst ebenfalls.
  ballHelpModal.addEventListener('click', (e) => {
    if (e.target === ballHelpModal) ballHelpModal.hidden = true;
  });

  // Audio toggle.
  const audio = document.getElementById('audio-toggle');
  audio.addEventListener('change', () => setAudio(audio.checked));

  // Settings panel show/hide.
  const panel = document.getElementById('panel');
  document.getElementById('btn-settings').addEventListener('click', () => {
    panel.classList.toggle('open');
  });

  // Drag-to-close (und Tap-to-close) auf dem Grip-Balken oben am Panel.
  const grip = panel.querySelector('.panel-grip');
  if (grip) {
    let startY = null;
    let lastY = null;
    let dragged = false;

    const onStart = (e) => {
      const y = e.touches ? e.touches[0].clientY : e.clientY;
      startY = y;
      lastY = y;
      dragged = false;
      panel.style.transition = 'none';
    };

    const onMove = (e) => {
      if (startY === null) return;
      const y = e.touches ? e.touches[0].clientY : e.clientY;
      lastY = y;
      const delta = y - startY;
      if (Math.abs(delta) > 5) dragged = true;
      if (delta > 0) {
        panel.style.transform = `translateY(${delta}px)`;
        if (e.cancelable) e.preventDefault();
      }
    };

    const onEnd = () => {
      if (startY === null) return;
      const delta = (lastY ?? startY) - startY;
      const panelHeight = panel.getBoundingClientRect().height;
      panel.style.transition = '';
      panel.style.transform = '';
      if (!dragged || delta > panelHeight * 0.25) {
        panel.classList.remove('open');
      }
      startY = null;
      lastY = null;
      dragged = false;
    };

    grip.addEventListener('touchstart', onStart, { passive: false });
    grip.addEventListener('touchmove',  onMove,  { passive: false });
    grip.addEventListener('touchend',   onEnd);
    grip.addEventListener('touchcancel', onEnd);
    grip.addEventListener('mousedown', onStart);
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup',   onEnd);
  }

  // Fullscreen toggle.
  const fsBtn = document.getElementById('btn-fullscreen');
  if (fsBtn) {
    fsBtn.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen().catch((e) =>
          console.warn('Fullscreen abgelehnt:', e)
        );
      } else {
        document.exitFullscreen().catch(() => {});
      }
    });
    document.addEventListener('fullscreenchange', () => {
      fsBtn.textContent = document.fullscreenElement ? '✕' : '⛶';
    });
  }

  // Keep canvases sized correctly on rotation / resize.
  window.addEventListener('resize', () => { if (state.running) sizeViewCanvas(); });
}

// =============================================================================
// Boot
// =============================================================================

function init() {
  // Show the splash image for 2 s, then fade it out.
  const splash = document.getElementById('splash');
  if (splash) {
    setTimeout(() => {
      splash.classList.add('hidden');
      splash.addEventListener('transitionend', () => splash.remove(), { once: true });
    }, 2000);
  }

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

// Wake Lock und Kamera automatisch wiederherstellen, wenn die App aus dem
// Hintergrund zurückkommt.
document.addEventListener('visibilitychange', async () => {
  if (document.visibilityState !== 'visible') return;

  if (state.running) {
    const track = state.stream && state.stream.getVideoTracks()[0];
    const trackDead = !track || track.readyState === 'ended';
    const videoStalled = state.video && state.video.paused;
    if (trackDead || videoStalled) {
      await startCamera();
      return;
    }
  }

  if (state.wakeLock !== null && 'wakeLock' in navigator) {
    try {
      state.wakeLock = await navigator.wakeLock.request('screen');
    } catch (e) {
      console.warn('Wake Lock Wiederherstellung fehlgeschlagen:', e);
    }
  }
});

document.addEventListener('DOMContentLoaded', init);

