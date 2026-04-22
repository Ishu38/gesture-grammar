/**
 * LandmarkSmoother.js — AGGME Phase 2: Exponential Moving Average (EMA) Filter
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Applies an Exponential Moving Average to all 21 × 3 = 63 landmark coordinate
 * channels (x, y, z per landmark) to suppress high-frequency jitter from
 * dyskinetic movement, resting tremor, and camera noise.
 *
 * The smoothed coordinates replace raw MediaPipe output for all downstream
 * consumers: gesture detection, error vectors, cognitive load estimation,
 * spatial grammar mapping, and intentionality detection.
 *
 * AGGME Pipeline Position:
 *   MediaPipe → RestingBoundaryCalibrator → [LandmarkSmoother] → IntentionalityDetector
 *
 * Mathematical formulation:
 *   For each landmark i (0–20) and axis a ∈ {x, y, z}:
 *     P_smooth(t) = α · P_raw(t) + (1 - α) · P_smooth(t-1)
 *
 *   Where α ∈ (0, 1] is the smoothing factor:
 *     α → 1: less smoothing, faster response, more jitter passes through
 *     α → 0: more smoothing, slower response, tremor suppressed but lag introduced
 *
 * Alpha calibration from RestingBoundaryCalibrator:
 *   The resting jitter (mean 3D stddev from Phase 1) determines the optimal α:
 *     - Low resting jitter (stable hand): α = 0.5–0.7 (minimal smoothing needed)
 *     - Medium resting jitter:            α = 0.3–0.5 (moderate smoothing)
 *     - High resting jitter (severe tremor): α = 0.1–0.3 (aggressive smoothing)
 *
 *   Formula:
 *     α = clamp(ALPHA_MAX - (jitter / JITTER_MAX) * (ALPHA_MAX - ALPHA_MIN), ALPHA_MIN, ALPHA_MAX)
 *
 *   This ensures users with severe tremor get aggressive smoothing automatically,
 *   while users with steady hands get near-raw responsiveness.
 *
 * Properties of the EMA:
 *   - O(1) memory per channel (stores only the previous smoothed value)
 *   - No buffer needed (unlike moving average)
 *   - Introduces phase lag proportional to (1-α) — acceptable for gesture
 *     recognition where frames arrive at 30fps and gestures are held for
 *     1–2 seconds
 *   - Exponential decay of older values: weight at lag k = α(1-α)^k
 */

// =============================================================================
// CONSTANTS
// =============================================================================

// Alpha range boundaries.
// These define the smoothing behavior envelope across all tremor severities.
const ALPHA_MIN = 0.10;  // Maximum smoothing (severe tremor)
const ALPHA_MAX = 0.70;  // Minimum smoothing (stable hand)

// Default alpha when no calibration data is available.
const DEFAULT_ALPHA = 0.45;

// Jitter range for alpha mapping.
// Resting jitter below JITTER_LOW → α = ALPHA_MAX (barely smooth)
// Resting jitter above JITTER_HIGH → α = ALPHA_MIN (heavily smooth)
const JITTER_LOW  = 0.002;  // Steady hand
const JITTER_HIGH = 0.020;  // Severe tremor

// Number of landmarks in MediaPipe hand model
const NUM_LANDMARKS = 21;

// =============================================================================
// LANDMARK SMOOTHER
// =============================================================================

export class LandmarkSmoother {
  /**
   * @param {object} config
   * @param {number} [config.alpha]         — fixed smoothing factor (overrides auto-calibration)
   * @param {number} [config.restingJitter] — from RestingBoundaryCalibrator, used to auto-compute alpha
   */
  constructor(config = {}) {
    // Per-landmark smoothed state: array of 21 {x, y, z}
    this._smoothed = null;  // null until first frame arrives

    // Smoothing factor
    if (config.alpha !== undefined) {
      this._alpha = Math.max(ALPHA_MIN, Math.min(ALPHA_MAX, config.alpha));
      this._alphaSource = 'manual';
    } else if (config.restingJitter !== undefined && config.restingJitter > 0) {
      this._alpha = this._computeAlphaFromJitter(config.restingJitter);
      this._alphaSource = 'calibrated';
    } else {
      this._alpha = DEFAULT_ALPHA;
      this._alphaSource = 'default';
    }

    // Frame counter (for diagnostics)
    this._frameCount = 0;
  }

  // ===========================================================================
  // PUBLIC — Core smoothing operation
  // ===========================================================================

  /**
   * Smooth a frame of raw landmark coordinates.
   *
   * Returns a new array of 21 landmarks with smoothed x, y, z values.
   * The returned array is safe to pass directly to detectGestureRaw(),
   * ErrorVectorEngine, CognitiveLoadAdapter, and SpatialGrammarMapper.
   *
   * @param {Array} rawLandmarks — 21 MediaPipe hand landmarks [{x, y, z}, ...]
   * @returns {Array} 21 smoothed landmarks [{x, y, z}, ...]
   */
  smooth(rawLandmarks) {
    if (!rawLandmarks || rawLandmarks.length < NUM_LANDMARKS) {
      return rawLandmarks; // Pass through null/invalid
    }

    this._frameCount++;

    // First frame: initialize smoothed state to raw values (no history yet)
    if (this._smoothed === null) {
      this._smoothed = rawLandmarks.map(lm => ({
        x: lm.x,
        y: lm.y,
        z: lm.z || 0,
      }));
      return this._cloneSmoothed();
    }

    // Apply EMA: P_smooth(t) = α·P_raw(t) + (1-α)·P_smooth(t-1)
    const alpha = this._alpha;
    const oneMinusAlpha = 1 - alpha;

    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const raw = rawLandmarks[i];
      const prev = this._smoothed[i];

      prev.x = alpha * raw.x + oneMinusAlpha * prev.x;
      prev.y = alpha * raw.y + oneMinusAlpha * prev.y;
      prev.z = alpha * (raw.z || 0) + oneMinusAlpha * prev.z;
    }

    return this._cloneSmoothed();
  }

  // ===========================================================================
  // PUBLIC — Alpha management
  // ===========================================================================

  /**
   * Get the current smoothing factor.
   * @returns {number} alpha ∈ [ALPHA_MIN, ALPHA_MAX]
   */
  getAlpha() {
    return this._alpha;
  }

  /**
   * Get how alpha was determined.
   * @returns {'manual' | 'calibrated' | 'default'}
   */
  getAlphaSource() {
    return this._alphaSource;
  }

  /**
   * Set a new alpha manually.
   * @param {number} alpha — ∈ (0, 1]
   */
  setAlpha(alpha) {
    this._alpha = Math.max(ALPHA_MIN, Math.min(ALPHA_MAX, alpha));
    this._alphaSource = 'manual';
  }

  /**
   * Recalibrate alpha from a new resting jitter value.
   * Called after RestingBoundaryCalibrator completes.
   *
   * @param {number} restingJitter — mean 3D stddev from calibration
   */
  calibrateFromJitter(restingJitter) {
    if (restingJitter > 0) {
      this._alpha = this._computeAlphaFromJitter(restingJitter);
      this._alphaSource = 'calibrated';
    }
  }

  /**
   * Reset the smoother state. Next frame will reinitialize.
   */
  reset() {
    this._smoothed = null;
    this._frameCount = 0;
  }

  /**
   * Get the frame count since last reset.
   * @returns {number}
   */
  getFrameCount() {
    return this._frameCount;
  }

  /**
   * Get the last smoothed landmarks (without processing a new frame).
   * @returns {Array|null}
   */
  getLastSmoothed() {
    return this._smoothed ? this._cloneSmoothed() : null;
  }

  /**
   * Get a serializable descriptor for logging/export.
   * @returns {object}
   */
  toDescriptor() {
    return {
      alpha: this._alpha,
      alpha_source: this._alphaSource,
      frame_count: this._frameCount,
      alpha_min: ALPHA_MIN,
      alpha_max: ALPHA_MAX,
    };
  }

  // ===========================================================================
  // PRIVATE — Alpha computation
  // ===========================================================================

  /**
   * Compute smoothing factor from resting jitter.
   *
   * Higher jitter → lower alpha → more smoothing.
   * Linear mapping from [JITTER_LOW, JITTER_HIGH] → [ALPHA_MAX, ALPHA_MIN].
   *
   * @param {number} jitter — resting jitter scalar
   * @returns {number} alpha ∈ [ALPHA_MIN, ALPHA_MAX]
   */
  _computeAlphaFromJitter(jitter) {
    // Normalize jitter to [0, 1] within the expected range
    const normalized = Math.max(0, Math.min(1,
      (jitter - JITTER_LOW) / (JITTER_HIGH - JITTER_LOW)
    ));

    // Invert: high jitter → low alpha
    const alpha = ALPHA_MAX - normalized * (ALPHA_MAX - ALPHA_MIN);

    return Math.round(alpha * 1000) / 1000; // 3 decimal places
  }

  // ===========================================================================
  // PRIVATE — Clone helper
  // ===========================================================================

  _cloneSmoothed() {
    return this._smoothed.map(lm => ({
      x: lm.x,
      y: lm.y,
      z: lm.z,
    }));
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/**
 * Get a human-readable description of the smoothing intensity.
 * @param {number} alpha
 * @returns {string}
 */
export function smoothingIntensityLabel(alpha) {
  if (alpha >= 0.6) return 'Light';
  if (alpha >= 0.4) return 'Moderate';
  if (alpha >= 0.2) return 'Heavy';
  return 'Maximum';
}

/**
 * Get a CSS color for the smoothing intensity.
 * @param {number} alpha
 * @returns {string}
 */
export function smoothingIntensityColor(alpha) {
  if (alpha >= 0.6) return '#4ade80';   // green — light smoothing
  if (alpha >= 0.4) return '#facc15';   // yellow — moderate
  if (alpha >= 0.2) return '#f97316';   // orange — heavy
  return '#f87171';                      // red — maximum smoothing
}

export default LandmarkSmoother;
