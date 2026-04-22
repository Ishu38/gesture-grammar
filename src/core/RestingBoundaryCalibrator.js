/**
 * RestingBoundaryCalibrator.js — AGGME Phase 1: Resting Boundary Estimation
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Captures per-landmark 3D coordinate variance during a calibration window
 * (default 3 seconds at 30fps = 90 frames) while the user holds their hand
 * still. The resulting variance envelope defines the "Resting Boundary" —
 * any movement within this boundary is classified as involuntary tremor,
 * not an intentional gesture.
 *
 * This is critical for motor-impaired users (cerebral palsy, Parkinson's,
 * essential tremor) who cannot hold perfectly still. Without calibration,
 * resting tremor triggers false gesture detections.
 *
 * AGGME Pipeline Position:
 *   MediaPipe → [RestingBoundaryCalibrator] → LandmarkSmoother → IntentionalityDetector
 *
 * Calibration procedure:
 *   1. User places hand in camera view and holds still
 *   2. System captures CALIBRATION_FRAMES frames of landmark data
 *   3. Per-landmark mean and variance computed for x, y, z
 *   4. Resting envelope = per-landmark mean ± k * stddev (k = envelope multiplier)
 *   5. After calibration, each incoming frame is classified:
 *      - All landmarks within envelope → RESTING (tremor)
 *      - Any landmark outside envelope → ACTIVE (intentional movement)
 *
 * Mathematical formulation:
 *   For each landmark i (0–20) and axis a ∈ {x, y, z}:
 *     μ_ia = (1/N) Σ_{t=1..N} L_ia(t)
 *     σ²_ia = (1/N) Σ_{t=1..N} (L_ia(t) - μ_ia)²
 *     σ_ia = sqrt(σ²_ia)
 *
 *   Resting boundary for landmark i, axis a:
 *     [μ_ia - k·σ_ia, μ_ia + k·σ_ia]
 *
 *   Classification at time t:
 *     if ∀i,a: |L_ia(t) - μ_ia| ≤ k·σ_ia → RESTING
 *     else → ACTIVE
 *
 *   Displacement metric (scalar):
 *     D(t) = (1/21) Σ_{i=0..20} sqrt( Σ_{a∈{x,y,z}} ((L_ia(t) - μ_ia) / σ_ia)² )
 *
 *   This is the mean z-scored Euclidean distance across all landmarks.
 *   D(t) < displacement_threshold → RESTING; D(t) ≥ threshold → ACTIVE
 *
 * The displacement_threshold defaults to 1.5 (= 1.5 standard deviations)
 * and is exposed as a configurable parameter.
 */

// =============================================================================
// CONSTANTS
// =============================================================================

// Number of frames to capture during calibration.
// 90 frames ≈ 3 seconds at 30fps. Long enough to capture tremor cycle
// (most pathological tremors are 4–12 Hz; 90 frames captures 3+ full cycles).
const DEFAULT_CALIBRATION_FRAMES = 90;

// Minimum calibration frames (safety floor — cannot calibrate with less).
const MIN_CALIBRATION_FRAMES = 30;

// Envelope multiplier (k).
// 1.5σ captures ~87% of resting variance for a normal distribution.
// Wider than 1σ (68%) to avoid false ACTIVE classifications from normal tremor.
const DEFAULT_ENVELOPE_K = 1.5;

// Displacement threshold: mean z-scored distance above which movement is ACTIVE.
// 1.5 means the average landmark is 1.5 standard deviations from its resting mean.
const DEFAULT_DISPLACEMENT_THRESHOLD = 1.5;

// Floor variance: if computed variance is below this, use this floor.
// Prevents division-by-zero and handles unrealistically stable calibrations.
// 0.0001 in normalized [0,1] image space ≈ sub-pixel jitter.
const VARIANCE_FLOOR = 0.0001;

// =============================================================================
// CALIBRATION STATES
// =============================================================================

const STATE = {
  IDLE:        'IDLE',        // Not started
  CALIBRATING: 'CALIBRATING', // Collecting calibration frames
  READY:       'READY',       // Calibration complete, classifying
  FAILED:      'FAILED',      // Calibration failed (hand lost during capture)
};

// =============================================================================
// RESTING BOUNDARY CALIBRATOR
// =============================================================================

export class RestingBoundaryCalibrator {
  /**
   * @param {object} config
   * @param {number} [config.calibrationFrames] — frames to capture (default 90)
   * @param {number} [config.envelopeK]         — σ multiplier for boundary (default 1.5)
   * @param {number} [config.displacementThreshold] — ACTIVE threshold (default 1.5)
   */
  constructor(config = {}) {
    this._calibrationFrames = Math.max(
      MIN_CALIBRATION_FRAMES,
      config.calibrationFrames || DEFAULT_CALIBRATION_FRAMES
    );
    this._envelopeK = config.envelopeK || DEFAULT_ENVELOPE_K;
    this._displacementThreshold = config.displacementThreshold || DEFAULT_DISPLACEMENT_THRESHOLD;

    // Calibration state
    this._state = STATE.IDLE;
    this._captureBuffer = [];  // Array of 21-landmark frames during calibration

    // Per-landmark resting statistics (populated after calibration)
    // Array of 21 entries, each: { mean: {x,y,z}, stddev: {x,y,z} }
    this._restingProfile = null;

    // Total resting jitter scalar (mean of per-landmark 3D stddev)
    this._restingJitter = 0;

    // How many frames were actually captured
    this._capturedFrameCount = 0;

    // How many consecutive frames had no hand (for failure detection)
    this._noHandFrames = 0;
    this._maxNoHandFrames = 15; // ~0.5s without hand = calibration failure
  }

  // ===========================================================================
  // PUBLIC — Calibration lifecycle
  // ===========================================================================

  /**
   * Begin the calibration process.
   * Call this when the user presses "Calibrate" and has their hand visible.
   */
  startCalibration() {
    this._state = STATE.CALIBRATING;
    this._captureBuffer = [];
    this._noHandFrames = 0;
    this._capturedFrameCount = 0;
    this._restingProfile = null;
    this._restingJitter = 0;
  }

  /**
   * Feed a frame of landmark data during calibration, or classify after calibration.
   *
   * @param {Array|null} landmarks — 21 MediaPipe landmarks, or null if no hand
   * @returns {CalibrationResult}
   */
  processFrame(landmarks) {
    switch (this._state) {
      case STATE.CALIBRATING:
        return this._handleCalibrationFrame(landmarks);
      case STATE.READY:
        return this._classifyFrame(landmarks);
      case STATE.FAILED:
        return { state: STATE.FAILED, classification: null, displacement: 0, progress: 0 };
      case STATE.IDLE:
      default:
        return { state: STATE.IDLE, classification: null, displacement: 0, progress: 0 };
    }
  }

  /**
   * Get the current state of the calibrator.
   * @returns {string} 'IDLE' | 'CALIBRATING' | 'READY' | 'FAILED'
   */
  getState() {
    return this._state;
  }

  /**
   * Get calibration progress (0.0 to 1.0).
   * @returns {number}
   */
  getProgress() {
    if (this._state !== STATE.CALIBRATING) {
      return this._state === STATE.READY ? 1.0 : 0;
    }
    return Math.min(1, this._capturedFrameCount / this._calibrationFrames);
  }

  /**
   * Get the computed resting profile after calibration.
   * @returns {object|null} restingProfile or null if not calibrated
   */
  getRestingProfile() {
    return this._restingProfile;
  }

  /**
   * Get the scalar resting jitter (mean 3D stddev across all landmarks).
   * Useful for calibrating the EMA smoothing alpha.
   * @returns {number}
   */
  getRestingJitter() {
    return this._restingJitter;
  }

  /**
   * Check whether calibration is complete and the system is ready.
   * @returns {boolean}
   */
  isReady() {
    return this._state === STATE.READY;
  }

  /**
   * Reset the calibrator to IDLE state.
   */
  reset() {
    this._state = STATE.IDLE;
    this._captureBuffer = [];
    this._restingProfile = null;
    this._restingJitter = 0;
    this._capturedFrameCount = 0;
    this._noHandFrames = 0;
  }

  /**
   * Get a serializable summary for SessionDataLogger export.
   * @returns {object}
   */
  toDescriptor() {
    return {
      state: this._state,
      calibration_frames: this._calibrationFrames,
      captured_frames: this._capturedFrameCount,
      envelope_k: this._envelopeK,
      displacement_threshold: this._displacementThreshold,
      resting_jitter: this._restingJitter,
      resting_profile: this._restingProfile,
    };
  }

  // ===========================================================================
  // PRIVATE — Calibration frame handling
  // ===========================================================================

  _handleCalibrationFrame(landmarks) {
    // No hand detected during calibration
    if (!landmarks || landmarks.length < 21) {
      this._noHandFrames++;
      if (this._noHandFrames >= this._maxNoHandFrames) {
        this._state = STATE.FAILED;
        return { state: STATE.FAILED, classification: null, displacement: 0, progress: this.getProgress() };
      }
      return { state: STATE.CALIBRATING, classification: null, displacement: 0, progress: this.getProgress() };
    }

    // Hand detected — reset no-hand counter and capture frame
    this._noHandFrames = 0;
    this._captureBuffer.push(this._cloneLandmarks(landmarks));
    this._capturedFrameCount++;

    // Check if calibration is complete
    if (this._capturedFrameCount >= this._calibrationFrames) {
      this._computeRestingProfile();
      this._state = STATE.READY;
      return { state: STATE.READY, classification: 'RESTING', displacement: 0, progress: 1.0 };
    }

    return { state: STATE.CALIBRATING, classification: null, displacement: 0, progress: this.getProgress() };
  }

  // ===========================================================================
  // PRIVATE — Resting profile computation
  // ===========================================================================

  _computeRestingProfile() {
    const N = this._captureBuffer.length;
    const numLandmarks = 21;

    this._restingProfile = new Array(numLandmarks);
    let totalJitter = 0;

    for (let i = 0; i < numLandmarks; i++) {
      // Collect all values for landmark i across all captured frames
      const xs = new Float64Array(N);
      const ys = new Float64Array(N);
      const zs = new Float64Array(N);

      for (let t = 0; t < N; t++) {
        xs[t] = this._captureBuffer[t][i].x;
        ys[t] = this._captureBuffer[t][i].y;
        zs[t] = this._captureBuffer[t][i].z || 0;
      }

      // Compute mean and variance
      const meanX = this._mean(xs);
      const meanY = this._mean(ys);
      const meanZ = this._mean(zs);

      const varX = Math.max(VARIANCE_FLOOR, this._variance(xs, meanX));
      const varY = Math.max(VARIANCE_FLOOR, this._variance(ys, meanY));
      const varZ = Math.max(VARIANCE_FLOOR, this._variance(zs, meanZ));

      const stddevX = Math.sqrt(varX);
      const stddevY = Math.sqrt(varY);
      const stddevZ = Math.sqrt(varZ);

      this._restingProfile[i] = {
        mean:   { x: meanX,   y: meanY,   z: meanZ   },
        stddev: { x: stddevX, y: stddevY, z: stddevZ },
        var:    { x: varX,    y: varY,    z: varZ    },
      };

      // Accumulate total jitter (3D stddev per landmark)
      totalJitter += Math.sqrt(varX + varY + varZ);
    }

    this._restingJitter = totalJitter / numLandmarks;

    // Free capture buffer — no longer needed
    this._captureBuffer = [];
  }

  // ===========================================================================
  // PRIVATE — Frame classification (post-calibration)
  // ===========================================================================

  /**
   * Classify an incoming frame as RESTING or ACTIVE based on the resting profile.
   *
   * Computes the mean z-scored Euclidean displacement across all 21 landmarks.
   * If displacement ≥ threshold → ACTIVE (intentional gesture likely)
   * If displacement < threshold → RESTING (tremor / no intent)
   */
  _classifyFrame(landmarks) {
    if (!landmarks || landmarks.length < 21 || !this._restingProfile) {
      return { state: STATE.READY, classification: null, displacement: 0, progress: 1.0 };
    }

    let totalDisplacement = 0;
    const numLandmarks = 21;

    for (let i = 0; i < numLandmarks; i++) {
      const profile = this._restingProfile[i];
      const lm = landmarks[i];

      // Z-scored displacement per axis
      const zx = (lm.x - profile.mean.x) / profile.stddev.x;
      const zy = (lm.y - profile.mean.y) / profile.stddev.y;
      const zz = ((lm.z || 0) - profile.mean.z) / profile.stddev.z;

      // 3D z-scored Euclidean distance for this landmark
      totalDisplacement += Math.sqrt(zx * zx + zy * zy + zz * zz);
    }

    const meanDisplacement = totalDisplacement / numLandmarks;
    const classification = meanDisplacement >= this._displacementThreshold ? 'ACTIVE' : 'RESTING';

    return {
      state: STATE.READY,
      classification,
      displacement: Math.round(meanDisplacement * 1000) / 1000,
      progress: 1.0,
    };
  }

  // ===========================================================================
  // PRIVATE — Math utilities
  // ===========================================================================

  _mean(arr) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) sum += arr[i];
    return sum / arr.length;
  }

  _variance(arr, mean) {
    let sum = 0;
    for (let i = 0; i < arr.length; i++) {
      const d = arr[i] - mean;
      sum += d * d;
    }
    return sum / arr.length;
  }

  _cloneLandmarks(landmarks) {
    return landmarks.map(lm => ({
      x: lm.x,
      y: lm.y,
      z: lm.z || 0,
    }));
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/**
 * Get a display label for the calibration state.
 * @param {string} state
 * @returns {string}
 */
export function calibrationStateLabel(state) {
  switch (state) {
    case STATE.IDLE:        return 'Not calibrated';
    case STATE.CALIBRATING: return 'Calibrating...';
    case STATE.READY:       return 'Calibrated';
    case STATE.FAILED:      return 'Calibration failed';
    default:                return 'Unknown';
  }
}

/**
 * Get a CSS color for the calibration state.
 * @param {string} state
 * @returns {string}
 */
export function calibrationStateColor(state) {
  switch (state) {
    case STATE.IDLE:        return '#64748b';
    case STATE.CALIBRATING: return '#facc15';
    case STATE.READY:       return '#4ade80';
    case STATE.FAILED:      return '#f87171';
    default:                return '#64748b';
  }
}

export { STATE as CALIBRATION_STATE };
export default RestingBoundaryCalibrator;
