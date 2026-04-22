/**
 * CognitiveLoadAdapter.js — Real-Time Cognitive Load Estimation from Motor Jitter
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Estimates cognitive load by measuring positional jitter (instability) across
 * all 21 MediaPipe hand landmarks over a rolling frame window. High jitter
 * indicates motor instability associated with elevated working memory demand.
 *
 * The adapter then adjusts the gesture confidence lock threshold in response:
 * higher load → more frames required → prevents accidental locks when the
 * learner is cognitively overwhelmed.
 *
 * Theoretical grounding:
 *   Birsh & Carreker (2018), Chapter 8: "The Role of Executive Function in
 *   Literacy Instruction" (Gordon-Pershey). Executive function encompasses:
 *     - Working memory: holding the current sentence structure in mind while
 *       forming the next gesture.
 *     - Inhibitory control: suppressing incorrect gesture forms.
 *     - Cognitive flexibility: switching between gesture categories (NP→VP→OBJ).
 *   Research establishes that motor tremor and positional instability increase
 *   under high cognitive load (dual-task interference, Baddeley, 2000).
 *
 *   Birsh & Carreker (2018), Chapter 11: "Multi-Modal Handwriting Instruction"
 *   (Wolf & Berninger). Kinesthetic-motor precision reflects automaticity;
 *   instability signals incomplete motor memory consolidation.
 *
 * Mechanism:
 *   1. Rolling window of N frames of 21-landmark MediaPipe hand data
 *   2. Per-landmark 3D jitter = sqrt(var_x + var_y + var_z) over the window
 *   3. Mean jitter across all 21 landmarks = cognitive load index
 *   4. Thresholds map the index to LOW | MEDIUM | HIGH load levels
 *   5. Each level maps to a confidence frame multiplier applied to the
 *      useSentenceBuilder lock threshold
 *
 * Patent claim:
 *   "A system that dynamically adjusts gesture recognition sensitivity based
 *   on real-time motor jitter analysis as a proxy for cognitive load, thereby
 *   operationalizing executive function theory (Baddeley, 2000; Gordon-Pershey
 *   in Birsh & Carreker, 2018) in a gesture-based language instruction interface
 *   running locally on a mobile device."
 */

// =============================================================================
// CONSTANTS
// =============================================================================

// Rolling window size (frames). 12 frames ≈ 400ms at 30fps — responsive but
// stable enough to filter single-frame noise.
const DEFAULT_WINDOW_SIZE = 12;

// Jitter thresholds (mean 3D std dev across 21 normalized landmarks):
// MediaPipe coordinates are in [0, 1] image space.
// Calibrated against typical hand stability in a seated user with smartphone camera.
const JITTER_LOW_THRESHOLD    = 0.004;  // Below this: stable, low load
const JITTER_MEDIUM_THRESHOLD = 0.012;  // Below this: some instability, medium load
// Above MEDIUM: high instability = high cognitive load

// Confidence frame multipliers per load level.
// Applied to the base threshold from AccessibilityProfile (default: 45 frames).
const MULTIPLIERS = {
  LOW:    1.0,   // No change — baseline
  MEDIUM: 1.25,  // 25% more frames (e.g., 45 → ~56)
  HIGH:   1.6,   // 60% more frames (e.g., 45 → 72) — significant scaffolding
};

// Hard cap to prevent the threshold from becoming unreachably high.
const MAX_THRESHOLD_FRAMES = 120; // 4 seconds at 30fps

// Hysteresis: require the level to persist for this many consecutive frames
// before committing to a level change. Prevents flickering.
const HYSTERESIS_FRAMES = 6;

// =============================================================================
// COGNITIVE LOAD ADAPTER
// =============================================================================

/**
 * Estimates cognitive load from MediaPipe hand landmark jitter and returns
 * an adjusted confidence lock threshold for useSentenceBuilder.
 *
 * Usage (called every camera frame in SandboxMode.processLandmarks):
 *   const result = cognitiveLoadAdapterRef.current.update(handLandmarks);
 *   if (result.levelChanged) {
 *     setCognitiveLoad(result.level);
 *     setConfidenceThreshold(result.recommendedFrames);
 *   }
 */
export class CognitiveLoadAdapter {
  /**
   * @param {object} config
   * @param {number} config.windowSize — rolling window size in frames
   */
  constructor(config = {}) {
    this._windowSize = config.windowSize || DEFAULT_WINDOW_SIZE;
    this._buffer = [];       // Rolling window: array of 21-landmark arrays
    this._level = 'LOW';     // Current confirmed load level
    this._pendingLevel = 'LOW';
    this._hysteresisCount = 0;
    this._jitter = 0;        // Last computed jitter value
    this._baseFrames = 45;   // Will be updated from AccessibilityProfile
  }

  /**
   * Set the base confidence frame count (from AccessibilityProfile).
   * Called once when the profile is known, and on profile changes.
   *
   * @param {number} frames
   */
  setBaseFrames(frames) {
    this._baseFrames = frames || 45;
  }

  /**
   * Process a new frame of landmark data.
   * Called every camera frame in processLandmarks.
   *
   * @param {Array} landmarks — array of 21 {x, y, z} MediaPipe hand landmarks
   * @returns {{ level, jitter, levelChanged, recommendedFrames }}
   */
  update(landmarks) {
    if (!landmarks || landmarks.length < 21) {
      return this._noChangeResult();
    }

    // Add to rolling buffer
    this._buffer.push(landmarks);
    if (this._buffer.length > this._windowSize) {
      this._buffer.shift();
    }

    // Need minimum frames for meaningful computation
    if (this._buffer.length < 3) {
      return this._noChangeResult();
    }

    // Compute jitter
    this._jitter = this._computeJitter();

    // Map to raw level
    const rawLevel = this._jitterToLevel(this._jitter);

    // Apply hysteresis: only change level after rawLevel persists N frames
    let levelChanged = false;
    if (rawLevel !== this._level) {
      if (rawLevel === this._pendingLevel) {
        this._hysteresisCount++;
        if (this._hysteresisCount >= HYSTERESIS_FRAMES) {
          this._level = rawLevel;
          this._hysteresisCount = 0;
          levelChanged = true;
        }
      } else {
        this._pendingLevel = rawLevel;
        this._hysteresisCount = 1;
      }
    } else {
      // Raw level matches confirmed level — reset pending
      this._pendingLevel = rawLevel;
      this._hysteresisCount = 0;
    }

    return {
      level: this._level,
      jitter: this._jitter,
      levelChanged,
      recommendedFrames: this.getRecommendedFrames(),
    };
  }

  /**
   * Get the current confirmed cognitive load level.
   * @returns {'LOW' | 'MEDIUM' | 'HIGH'}
   */
  getLevel() {
    return this._level;
  }

  /**
   * Get the last computed mean jitter value.
   * @returns {number}
   */
  getJitter() {
    return this._jitter;
  }

  /**
   * Compute the recommended confidence frame count based on current level
   * and the base frame count set via setBaseFrames().
   *
   * @param {number} [baseFrames] — override base frames if provided
   * @returns {number}
   */
  getRecommendedFrames(baseFrames) {
    const base = baseFrames || this._baseFrames;
    const multiplier = MULTIPLIERS[this._level] || 1.0;
    return Math.min(MAX_THRESHOLD_FRAMES, Math.round(base * multiplier));
  }

  /**
   * Get the opacity multiplier for the ErrorOverlay.
   * High load → reduce overlay opacity to prevent visual overload.
   *
   * @returns {number} 0.0–1.0
   */
  getOverlayOpacity() {
    switch (this._level) {
      case 'HIGH':   return 0.35;
      case 'MEDIUM': return 0.70;
      case 'LOW':
      default:       return 1.0;
    }
  }

  /**
   * Reset the buffer and return to LOW state.
   * Call when camera stops or sentence is cleared.
   */
  reset() {
    this._buffer = [];
    this._level = 'LOW';
    this._pendingLevel = 'LOW';
    this._hysteresisCount = 0;
    this._jitter = 0;
  }

  // ===========================================================================
  // PRIVATE — Jitter computation
  // ===========================================================================

  /**
   * Compute mean 3D positional jitter across all 21 landmarks over the window.
   *
   * For each landmark i (0–20):
   *   jitter_i = sqrt( var(x_i) + var(y_i) + var(z_i) )
   *
   * Return value = mean of jitter_i across all 21 landmarks.
   *
   * This is the root-mean-square position deviation in 3D space, averaged
   * across all hand joints — a scalar measure of total hand instability.
   */
  _computeJitter() {
    const numLandmarks = 21;
    let totalJitter = 0;

    for (let i = 0; i < numLandmarks; i++) {
      const xs = this._buffer.map(frame => frame[i].x);
      const ys = this._buffer.map(frame => frame[i].y);
      const zs = this._buffer.map(frame => frame[i].z);

      const varX = this._variance(xs);
      const varY = this._variance(ys);
      const varZ = this._variance(zs);

      totalJitter += Math.sqrt(varX + varY + varZ);
    }

    return totalJitter / numLandmarks;
  }

  _variance(arr) {
    const n = arr.length;
    if (n < 2) return 0;
    const mean = arr.reduce((a, b) => a + b, 0) / n;
    return arr.reduce((sum, val) => sum + (val - mean) ** 2, 0) / n;
  }

  _jitterToLevel(jitter) {
    if (jitter < JITTER_LOW_THRESHOLD)    return 'LOW';
    if (jitter < JITTER_MEDIUM_THRESHOLD) return 'MEDIUM';
    return 'HIGH';
  }

  _noChangeResult() {
    return {
      level: this._level,
      jitter: this._jitter,
      levelChanged: false,
      recommendedFrames: this.getRecommendedFrames(),
    };
  }
}

// =============================================================================
// DISPLAY HELPERS (used in SandboxMode UI)
// =============================================================================

/**
 * Get the display color for a cognitive load level.
 * @param {'LOW'|'MEDIUM'|'HIGH'} level
 * @returns {string} CSS hex color
 */
export function loadLevelColor(level) {
  switch (level) {
    case 'HIGH':   return '#f87171';  // red
    case 'MEDIUM': return '#facc15';  // yellow
    case 'LOW':    return '#4ade80';  // green
    default:       return '#64748b';
  }
}

/**
 * Get a human-readable description of the load level for the UI.
 * @param {'LOW'|'MEDIUM'|'HIGH'} level
 * @returns {string}
 */
export function loadLevelDescription(level) {
  switch (level) {
    case 'HIGH':   return 'High — threshold extended';
    case 'MEDIUM': return 'Medium — slight extension';
    case 'LOW':    return 'Low — baseline';
    default:       return 'Unknown';
  }
}

export default CognitiveLoadAdapter;
