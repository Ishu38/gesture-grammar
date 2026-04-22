/**
 * IntentionalityDetector.js — AGGME Phase 3: Intentional Gesture Detection
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Monitors the smoothed Euclidean displacement of all 21 hand landmarks from
 * the stored resting baseline (computed in Phase 1). When displacement exceeds
 * a dynamically computed threshold, the system transitions from RESTING to
 * GESTURE_ACTIVE — signaling that the user has begun an intentional movement.
 *
 * When displacement returns below the threshold, the system transitions back
 * to RESTING, signaling gesture cessation.
 *
 * This replaces the implicit "gesture present = non-null detectGestureRaw()"
 * assumption with a principled displacement-based onset detector that correctly
 * handles motor-impaired users whose resting tremor would otherwise trigger
 * continuous false detections.
 *
 * AGGME Pipeline Position:
 *   MediaPipe → RestingBoundaryCalibrator → LandmarkSmoother → [IntentionalityDetector] → SpatialGrammarMapper / detectGestureRaw
 *
 * State machine:
 *   RESTING ──[displacement ≥ threshold for N frames]──→ GESTURE_ACTIVE
 *   GESTURE_ACTIVE ──[displacement < threshold for M frames]──→ RESTING
 *
 *   Hysteresis (N, M) prevents flickering at the boundary.
 *
 * Displacement metric:
 *   D(t) = (1/21) Σ_{i=0..20} sqrt( ((x_i(t) - μ_x_i) / σ_x_i)² +
 *                                     ((y_i(t) - μ_y_i) / σ_y_i)² +
 *                                     ((z_i(t) - μ_z_i) / σ_z_i)² )
 *
 *   Where μ and σ are from the RestingBoundaryCalibrator profile.
 *   D(t) is the mean z-scored Euclidean distance across all landmarks.
 *
 * Threshold:
 *   Default = 1.5 × mean resting standard deviation (in z-score space).
 *   This means: "intentional" = average landmark has moved 1.5σ from rest.
 *
 * Outputs per frame:
 *   - intent: 'RESTING' | 'GESTURE_ACTIVE'
 *   - displacement: scalar D(t) for debug display
 *   - onset: boolean (true on the RESTING→ACTIVE transition frame)
 *   - offset: boolean (true on the ACTIVE→RESTING transition frame)
 *   - duration_ms: how long the current gesture has been active (0 if RESTING)
 *   - trajectory: accumulated wrist displacement vector since onset
 */

// =============================================================================
// CONSTANTS
// =============================================================================

// Hysteresis frame counts to prevent flickering at the threshold boundary.
// Onset: require N consecutive frames above threshold before committing to ACTIVE.
const ONSET_HYSTERESIS_FRAMES = 3;
// Offset: require M consecutive frames below threshold before returning to RESTING.
// Longer offset hysteresis to prevent interrupting a gesture during brief pauses.
const OFFSET_HYSTERESIS_FRAMES = 8;

// Default threshold multiplier applied to the resting profile.
// 1.5 = movement must exceed 1.5× the resting standard deviation to be "intentional".
const DEFAULT_THRESHOLD_MULTIPLIER = 1.5;

// Number of landmarks
const NUM_LANDMARKS = 21;

// Wrist landmark index
const WRIST_INDEX = 0;

// =============================================================================
// INTENT STATES
// =============================================================================

const INTENT = {
  RESTING:        'RESTING',
  GESTURE_ACTIVE: 'GESTURE_ACTIVE',
};

// =============================================================================
// INTENTIONALITY DETECTOR
// =============================================================================

export class IntentionalityDetector {
  /**
   * @param {object} config
   * @param {Array}  config.restingProfile  — from RestingBoundaryCalibrator
   * @param {number} [config.thresholdMultiplier] — σ multiplier (default 1.5)
   */
  constructor(config = {}) {
    this._restingProfile = config.restingProfile || null;
    this._thresholdMultiplier = config.thresholdMultiplier || DEFAULT_THRESHOLD_MULTIPLIER;

    // State machine
    this._intent = INTENT.RESTING;
    this._pendingIntent = INTENT.RESTING;
    this._hysteresisCount = 0;

    // Onset tracking
    this._onsetTimestamp = null;  // ms since epoch when GESTURE_ACTIVE began
    this._onsetWristPos = null;  // wrist position at onset {x, y, z}

    // Per-frame displacement (cached for debug display)
    this._displacement = 0;

    // Trajectory accumulation: total wrist displacement vector since onset
    this._previousWristPos = null;
    this._trajectory = { dx: 0, dy: 0, dz: 0, distance: 0 };

    // Gesture duration counter
    this._activeDurationMs = 0;
  }

  // ===========================================================================
  // PUBLIC — Core detection
  // ===========================================================================

  /**
   * Process one frame of smoothed landmarks.
   *
   * @param {Array} landmarks — 21 smoothed landmarks from LandmarkSmoother
   * @returns {IntentResult}
   */
  detect(landmarks) {
    if (!landmarks || landmarks.length < NUM_LANDMARKS || !this._restingProfile) {
      return this._noResult();
    }

    // Compute displacement
    this._displacement = this._computeDisplacement(landmarks);

    // Determine raw intent from displacement
    const rawIntent = this._displacement >= this._thresholdMultiplier
      ? INTENT.GESTURE_ACTIVE
      : INTENT.RESTING;

    // Apply hysteresis
    let onset = false;
    let offset = false;
    const previousIntent = this._intent;

    if (rawIntent !== this._intent) {
      if (rawIntent === this._pendingIntent) {
        this._hysteresisCount++;
        const requiredFrames = rawIntent === INTENT.GESTURE_ACTIVE
          ? ONSET_HYSTERESIS_FRAMES
          : OFFSET_HYSTERESIS_FRAMES;

        if (this._hysteresisCount >= requiredFrames) {
          this._intent = rawIntent;
          this._hysteresisCount = 0;

          if (this._intent === INTENT.GESTURE_ACTIVE) {
            onset = true;
            this._onsetTimestamp = Date.now();
            this._onsetWristPos = {
              x: landmarks[WRIST_INDEX].x,
              y: landmarks[WRIST_INDEX].y,
              z: landmarks[WRIST_INDEX].z || 0,
            };
            this._trajectory = { dx: 0, dy: 0, dz: 0, distance: 0 };
          } else {
            offset = true;
            this._activeDurationMs = this._onsetTimestamp
              ? Date.now() - this._onsetTimestamp
              : 0;
            this._onsetTimestamp = null;
            this._onsetWristPos = null;
          }
        }
      } else {
        this._pendingIntent = rawIntent;
        this._hysteresisCount = 1;
      }
    } else {
      this._pendingIntent = rawIntent;
      this._hysteresisCount = 0;
    }

    // Update trajectory during active gesture
    if (this._intent === INTENT.GESTURE_ACTIVE && this._onsetWristPos) {
      const wrist = landmarks[WRIST_INDEX];
      this._trajectory.dx = wrist.x - this._onsetWristPos.x;
      this._trajectory.dy = wrist.y - this._onsetWristPos.y;
      this._trajectory.dz = (wrist.z || 0) - this._onsetWristPos.z;
      this._trajectory.distance = Math.sqrt(
        this._trajectory.dx ** 2 +
        this._trajectory.dy ** 2 +
        this._trajectory.dz ** 2
      );
    }

    const durationMs = this._intent === INTENT.GESTURE_ACTIVE && this._onsetTimestamp
      ? Date.now() - this._onsetTimestamp
      : 0;

    return {
      intent: this._intent,
      displacement: this._displacement,
      onset,
      offset,
      duration_ms: durationMs,
      trajectory: { ...this._trajectory },
    };
  }

  // ===========================================================================
  // PUBLIC — Configuration
  // ===========================================================================

  /**
   * Set or update the resting profile (from RestingBoundaryCalibrator).
   * Must be called before detect() can classify frames.
   *
   * @param {Array} restingProfile — array of 21 {mean:{x,y,z}, stddev:{x,y,z}}
   */
  setRestingProfile(restingProfile) {
    this._restingProfile = restingProfile;
    this.reset();
  }

  /**
   * Set the threshold multiplier.
   * @param {number} multiplier
   */
  setThresholdMultiplier(multiplier) {
    this._thresholdMultiplier = Math.max(0.5, Math.min(5.0, multiplier));
  }

  /**
   * Get the current intent state.
   * @returns {'RESTING' | 'GESTURE_ACTIVE'}
   */
  getIntent() {
    return this._intent;
  }

  /**
   * Get the last computed displacement.
   * @returns {number}
   */
  getDisplacement() {
    return this._displacement;
  }

  /**
   * Check if a gesture is currently active.
   * @returns {boolean}
   */
  isGestureActive() {
    return this._intent === INTENT.GESTURE_ACTIVE;
  }

  /**
   * Reset the detector to RESTING state.
   */
  reset() {
    this._intent = INTENT.RESTING;
    this._pendingIntent = INTENT.RESTING;
    this._hysteresisCount = 0;
    this._displacement = 0;
    this._onsetTimestamp = null;
    this._onsetWristPos = null;
    this._trajectory = { dx: 0, dy: 0, dz: 0, distance: 0 };
    this._activeDurationMs = 0;
    this._previousWristPos = null;
  }

  /**
   * Get a serializable descriptor for logging/export.
   * @returns {object}
   */
  toDescriptor() {
    return {
      intent: this._intent,
      displacement: this._displacement,
      threshold_multiplier: this._thresholdMultiplier,
      onset_hysteresis: ONSET_HYSTERESIS_FRAMES,
      offset_hysteresis: OFFSET_HYSTERESIS_FRAMES,
    };
  }

  // ===========================================================================
  // PRIVATE — Displacement computation
  // ===========================================================================

  /**
   * Compute mean z-scored Euclidean displacement across all 21 landmarks.
   *
   * For each landmark i:
   *   d_i = sqrt( ((x_i - μ_x_i)/σ_x_i)² + ((y_i - μ_y_i)/σ_y_i)² + ((z_i - μ_z_i)/σ_z_i)² )
   *
   * Return mean(d_i) across all landmarks.
   */
  _computeDisplacement(landmarks) {
    let total = 0;

    for (let i = 0; i < NUM_LANDMARKS; i++) {
      const lm = landmarks[i];
      const profile = this._restingProfile[i];

      const STDDEV_FLOOR = 1e-6;
      const zx = (lm.x - profile.mean.x) / Math.max(profile.stddev.x, STDDEV_FLOOR);
      const zy = (lm.y - profile.mean.y) / Math.max(profile.stddev.y, STDDEV_FLOOR);
      const zz = ((lm.z || 0) - profile.mean.z) / Math.max(profile.stddev.z, STDDEV_FLOOR);

      total += Math.sqrt(zx * zx + zy * zy + zz * zz);
    }

    return total / NUM_LANDMARKS;
  }

  _noResult() {
    return {
      intent: this._intent,
      displacement: 0,
      onset: false,
      offset: false,
      duration_ms: 0,
      trajectory: { dx: 0, dy: 0, dz: 0, distance: 0 },
    };
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/**
 * Get a CSS color for the current intent state.
 * @param {string} intent
 * @returns {string}
 */
export function intentColor(intent) {
  return intent === INTENT.GESTURE_ACTIVE ? '#4ade80' : '#64748b';
}

/**
 * Get a display label for the intent state.
 * @param {string} intent
 * @returns {string}
 */
export function intentLabel(intent) {
  return intent === INTENT.GESTURE_ACTIVE ? 'ACTIVE' : 'RESTING';
}

export { INTENT };
export default IntentionalityDetector;
