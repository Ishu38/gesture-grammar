/**
 * EyeGazeTracker.js
 * Third modality for UMCE trimodal Bayesian fusion.
 *
 * Uses MediaPipe FaceLandmarker to extract gaze direction, blink state,
 * and fixation patterns from the user's face, then computes P(G|S) —
 * the probability of the observed gaze evidence given each communicative state.
 *
 * Architecture mirrors UASAM: standalone class with initialize(), processFrame(),
 * getLastResult(), getGazeLikelihoods(), isActive(), getFeatures(), destroy().
 *
 * Fusion equation (trimodal):
 *   P(S | A, V, G) ∝ P(A|S)^w_A · P(V|S)^w_V · P(G|S)^w_G · P(S)
 *
 * Pipeline:
 *   Camera frame → FaceLandmarker → Iris/Eye landmarks → Feature Extraction
 *                                                              ↓
 *                                                     Gaze State Classification
 *                                                              ↓
 *                                                     P(G|S) Likelihoods → UMCE
 */

import { FaceLandmarker } from '@mediapipe/tasks-vision';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Probability floor for gaze likelihoods. */
const GAZE_PROB_FLOOR = 0.001;

/** EMA alpha for feature smoothing across frames. */
const FEATURE_SMOOTH_ALPHA = 0.35;

/** EAR threshold — below this, eyes are considered closed. */
const EAR_CLOSED_THRESHOLD = 0.15;

/** Saccade velocity threshold — above this, gaze is in rapid movement. */
const SACCADE_VELOCITY_THRESHOLD = 0.08;

/** Fixation stability threshold — gaze position change below this is "stable". */
const FIXATION_STABILITY_THRESHOLD = 0.03;

/** Gaze direction thresholds. */
const GAZE_LEFT_THRESHOLD = 0.35;
const GAZE_RIGHT_THRESHOLD = 0.65;
const GAZE_UP_THRESHOLD = 0.35;
const GAZE_DOWN_THRESHOLD = 0.65;

// =============================================================================
// GAZE STATES
// =============================================================================

const GAZE_STATES = {
  CENTER:     'CENTER',
  LEFT:       'LEFT',
  RIGHT:      'RIGHT',
  UP:         'UP',
  DOWN:       'DOWN',
  SACCADE:    'SACCADE',
  EYES_CLOSED:'EYES_CLOSED',
};

const GAZE_STATE_ORDER = [
  GAZE_STATES.CENTER,
  GAZE_STATES.LEFT,
  GAZE_STATES.RIGHT,
  GAZE_STATES.UP,
  GAZE_STATES.DOWN,
  GAZE_STATES.SACCADE,
  GAZE_STATES.EYES_CLOSED,
];

// =============================================================================
// FACE LANDMARK INDICES
// =============================================================================

/** Iris landmarks (MediaPipe FaceLandmarker 478-point model). */
const IRIS = {
  LEFT_CENTER:  468,
  LEFT_RING:    [469, 470, 471, 472],
  RIGHT_CENTER: 473,
  RIGHT_RING:   [474, 475, 476, 477],
};

/** Eye corner landmarks. */
const EYE_CORNERS = {
  LEFT_OUTER:  33,
  LEFT_INNER:  133,
  RIGHT_INNER: 362,
  RIGHT_OUTER: 263,
};

/** Eye top/bottom landmarks for EAR computation. */
const EYE_VERTICAL = {
  LEFT_TOP:    159,
  LEFT_BOTTOM: 145,
  RIGHT_TOP:   386,
  RIGHT_BOTTOM:374,
};

// =============================================================================
// EMISSION MATRIX: P(gaze_state | communicative_state)
// =============================================================================

/**
 * For each gesture/communicative state S, defines the probability of
 * observing each gaze state. Encodes where a child typically looks
 * when producing each communicative gesture.
 *
 * Rows: communicative states (gesture IDs).
 * Columns: gaze states (CENTER, LEFT, RIGHT, UP, DOWN, SACCADE, EYES_CLOSED).
 * Each row sums to 1.0.
 *
 * Rationale:
 *   - SUBJECT_I → looking down at self / slightly averted
 *   - SUBJECT_YOU → looking at addressee (center)
 *   - SUBJECT_HE → looking right (third person referent)
 *   - SUBJECT_SHE → looking left (third person referent, opposite)
 *   - SUBJECT_WE → center (inclusive gaze)
 *   - SUBJECT_THEY → saccade/scanning (multiple referents)
 *   - Action verbs (GRAB, EAT, DRINK) → looking down at hands/objects
 *   - GO → looking away / scanning direction
 *   - STOP → center (assertive)
 *   - WANT → center/up (desire expression)
 *   - Objects (APPLE, BALL, etc.) → looking down at referent
 *   - BOOK → looking down (reading)
 *   - HOUSE → looking up (large structure)
 */
const GAZE_EMISSION_MATRIX = {
  //                       CENTER  LEFT   RIGHT  UP     DOWN   SACCADE  CLOSED
  // ─── SUBJECTS ──────────────────────────────────────────────────────────
  SUBJECT_I:              [0.15,  0.10,  0.10,  0.05,  0.40,  0.10,    0.10],
  SUBJECT_YOU:            [0.55,  0.05,  0.05,  0.05,  0.10,  0.10,    0.10],
  SUBJECT_HE:             [0.10,  0.05,  0.45,  0.05,  0.10,  0.15,    0.10],
  SUBJECT_SHE:            [0.10,  0.45,  0.05,  0.05,  0.10,  0.15,    0.10],
  SUBJECT_WE:             [0.40,  0.10,  0.10,  0.05,  0.10,  0.15,    0.10],
  SUBJECT_THEY:           [0.10,  0.15,  0.15,  0.05,  0.10,  0.35,    0.10],

  // ─── VERBS ─────────────────────────────────────────────────────────────
  GRAB:                   [0.15,  0.05,  0.05,  0.05,  0.45,  0.15,    0.10],
  GO:                     [0.10,  0.10,  0.10,  0.10,  0.10,  0.40,    0.10],
  EAT:                    [0.15,  0.05,  0.05,  0.05,  0.45,  0.15,    0.10],
  WANT:                   [0.35,  0.05,  0.05,  0.20,  0.10,  0.15,    0.10],
  STOP:                   [0.45,  0.05,  0.05,  0.10,  0.10,  0.15,    0.10],
  DRINK:                  [0.15,  0.05,  0.05,  0.05,  0.45,  0.15,    0.10],

  // ─── OBJECTS ───────────────────────────────────────────────────────────
  APPLE:                  [0.15,  0.05,  0.05,  0.05,  0.45,  0.15,    0.10],
  BALL:                   [0.15,  0.10,  0.10,  0.05,  0.35,  0.15,    0.10],
  WATER:                  [0.15,  0.05,  0.05,  0.05,  0.45,  0.15,    0.10],
  FOOD:                   [0.15,  0.05,  0.05,  0.05,  0.45,  0.15,    0.10],
  BOOK:                   [0.10,  0.05,  0.05,  0.05,  0.55,  0.10,    0.10],
  HOUSE:                  [0.15,  0.05,  0.05,  0.35,  0.10,  0.20,    0.10],
};

// =============================================================================
// EyeGazeTracker CLASS
// =============================================================================

export class EyeGazeTracker {
  constructor() {
    /** @type {FaceLandmarker|null} */
    this._faceLandmarker = null;

    this._isActive = false;
    this._isInitializing = false;
    this._frameCount = 0;

    // Per-frame gaze features (smoothed)
    this._features = {
      gazeX: 0.5,
      gazeY: 0.5,
      earLeft: 0.3,
      earRight: 0.3,
      ear: 0.3,
      velocity: 0,
      fixationDuration: 0,
    };

    // Previous raw features for EMA smoothing
    this._prevFeatures = null;

    // Previous gaze position for velocity computation
    this._prevGazeX = 0.5;
    this._prevGazeY = 0.5;

    // Fixation counter (consecutive stable frames)
    this._fixationFrames = 0;

    // Gaze state probabilities (soft classification)
    this._gazeStateProbabilities = {};
    for (const gs of GAZE_STATE_ORDER) {
      this._gazeStateProbabilities[gs] = gs === GAZE_STATES.CENTER ? 1.0 : 0.0;
    }

    // Hard classification
    this._gazeState = GAZE_STATES.CENTER;

    // Gaze likelihoods: P(G | S) for each communicative state
    this._gazeLikelihoods = {};

    // Last frame result (cached for skipped frames)
    this._lastResult = null;

    // All communicative state IDs from emission matrix
    this._stateIds = Object.keys(GAZE_EMISSION_MATRIX);
  }

  // ===========================================================================
  // PUBLIC — Initialization
  // ===========================================================================

  /**
   * Initialize FaceLandmarker using a shared WASM fileset.
   * Call after FilesetResolver.forVisionTasks() has resolved.
   *
   * @param {object} vision — WASM fileset from FilesetResolver.forVisionTasks()
   * @returns {Promise<boolean>} Whether initialization succeeded
   */
  async initialize(vision) {
    if (this._isActive || this._isInitializing) return this._isActive;
    this._isInitializing = true;

    try {
      this._faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: '/face_landmarker.task',
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numFaces: 1,
        outputFaceBlendshapes: false,
        outputFacialTransformationMatrixes: false,
      });

      this._isActive = true;
      this._isInitializing = false;
      console.log('[EyeGazeTracker] FaceLandmarker initialized successfully');
      return true;
    } catch (err) {
      this._isActive = false;
      this._isInitializing = false;
      console.warn('[EyeGazeTracker] FaceLandmarker init failed:', err.message);
      return false;
    }
  }

  // ===========================================================================
  // PUBLIC — Per-Frame Processing
  // ===========================================================================

  /**
   * Process one video frame. Call at ~10fps (every 3rd camera frame).
   *
   * Extracts gaze features → classifies gaze state → computes P(G|S).
   *
   * @param {HTMLVideoElement} video — the video element
   * @param {number} timestamp — performance.now() timestamp
   * @returns {EyeGazeResult} Gaze analysis result including P(G|S) map
   */
  processFrame(video, timestamp) {
    if (!this._isActive || !this._faceLandmarker) {
      return this._inactiveResult();
    }

    this._frameCount++;

    // =====================================================================
    // Step 1: Run FaceLandmarker
    // =====================================================================
    let faceResult;
    try {
      faceResult = this._faceLandmarker.detectForVideo(video, timestamp);
    } catch {
      return this._lastResult || this._inactiveResult();
    }

    if (!faceResult || !faceResult.faceLandmarks || faceResult.faceLandmarks.length === 0) {
      // No face detected — return cached or inactive
      return this._lastResult || this._inactiveResult();
    }

    const landmarks = faceResult.faceLandmarks[0]; // First face

    // Check we have iris landmarks (need 478+ landmarks)
    if (landmarks.length < 478) {
      return this._lastResult || this._inactiveResult();
    }

    // =====================================================================
    // Step 2: Extract raw gaze features from iris/eye landmarks
    // =====================================================================
    const rawFeatures = this._extractFeatures(landmarks);

    // =====================================================================
    // Step 3: EMA smooth features across frames
    // =====================================================================
    this._smoothFeatures(rawFeatures);

    // =====================================================================
    // Step 4: Classify gaze state (soft probabilities)
    // =====================================================================
    this._classifyGazeState();

    // =====================================================================
    // Step 5: Compute P(G | S) for each communicative state
    // =====================================================================
    this._computeGazeLikelihoods();

    // =====================================================================
    // Step 6: Assemble result
    // =====================================================================
    this._lastResult = {
      features: { ...this._features },
      gaze_state: this._gazeState,
      gaze_state_probabilities: { ...this._gazeStateProbabilities },
      gaze_likelihoods: { ...this._gazeLikelihoods },
      face_detected: true,
      frame: this._frameCount,
    };

    return this._lastResult;
  }

  /**
   * Get the gaze likelihood map P(G|S) from the last processed frame.
   * This is what gets passed into UMCE for Bayesian fusion.
   *
   * @returns {Object} Map<state_id, probability>
   */
  getGazeLikelihoods() {
    return { ...this._gazeLikelihoods };
  }

  /**
   * Get last frame result (cached for skipped frames).
   * @returns {EyeGazeResult|null}
   */
  getLastResult() {
    return this._lastResult;
  }

  /**
   * Get current features for debug display.
   * @returns {object}
   */
  getFeatures() {
    return { ...this._features };
  }

  /**
   * Whether the FaceLandmarker is initialized and active.
   * @returns {boolean}
   */
  isActive() {
    return this._isActive;
  }

  /**
   * Release FaceLandmarker resources.
   */
  destroy() {
    if (this._faceLandmarker) {
      try {
        this._faceLandmarker.close();
      } catch { /* ignore */ }
      this._faceLandmarker = null;
    }
    this._isActive = false;
    this._lastResult = null;
    this._frameCount = 0;
    console.log('[EyeGazeTracker] Destroyed');
  }

  // ===========================================================================
  // PRIVATE — Feature Extraction
  // ===========================================================================

  /**
   * Extract raw gaze features from 478-point face landmarks.
   *
   * Features:
   *   - gazeX: horizontal iris position ratio (0=left, 1=right)
   *   - gazeY: vertical iris position ratio (0=up, 1=down)
   *   - earLeft/earRight: Eye Aspect Ratio per eye
   *   - ear: average EAR
   *   - velocity: frame-to-frame gaze displacement
   *   - fixationDuration: consecutive stable frames
   *
   * @param {Array} lm — 478 face landmarks [{x, y, z}, ...]
   * @returns {object} Raw (unsmoothed) features
   */
  _extractFeatures(lm) {
    // ── Iris positions ──
    const leftIris  = lm[IRIS.LEFT_CENTER];
    const rightIris = lm[IRIS.RIGHT_CENTER];

    // ── Eye corners ──
    const leftOuter  = lm[EYE_CORNERS.LEFT_OUTER];
    const leftInner  = lm[EYE_CORNERS.LEFT_INNER];
    const rightInner = lm[EYE_CORNERS.RIGHT_INNER];
    const rightOuter = lm[EYE_CORNERS.RIGHT_OUTER];

    // ── Eye vertical ──
    const leftTop    = lm[EYE_VERTICAL.LEFT_TOP];
    const leftBottom = lm[EYE_VERTICAL.LEFT_BOTTOM];
    const rightTop   = lm[EYE_VERTICAL.RIGHT_TOP];
    const rightBottom= lm[EYE_VERTICAL.RIGHT_BOTTOM];

    // ── Horizontal gaze ratio (per eye, then averaged) ──
    const leftEyeWidth  = dist2D(leftOuter, leftInner);
    const rightEyeWidth = dist2D(rightInner, rightOuter);

    const leftGazeX  = leftEyeWidth  > 0.001
      ? (leftIris.x  - leftOuter.x)  / leftEyeWidth  : 0.5;
    const rightGazeX = rightEyeWidth > 0.001
      ? (rightIris.x - rightInner.x) / rightEyeWidth : 0.5;

    const gazeX = clamp((leftGazeX + rightGazeX) / 2, 0, 1);

    // ── Vertical gaze ratio ──
    const leftEyeHeight  = dist2D(leftTop, leftBottom);
    const rightEyeHeight = dist2D(rightTop, rightBottom);

    const leftGazeY  = leftEyeHeight  > 0.001
      ? (leftIris.y  - leftTop.y)  / leftEyeHeight  : 0.5;
    const rightGazeY = rightEyeHeight > 0.001
      ? (rightIris.y - rightTop.y) / rightEyeHeight : 0.5;

    const gazeY = clamp((leftGazeY + rightGazeY) / 2, 0, 1);

    // ── Eye Aspect Ratio (EAR) — blink/closed detection ──
    const earLeft  = leftEyeWidth  > 0.001 ? leftEyeHeight  / leftEyeWidth  : 0.3;
    const earRight = rightEyeWidth > 0.001 ? rightEyeHeight / rightEyeWidth : 0.3;
    const ear = (earLeft + earRight) / 2;

    // ── Velocity (frame-to-frame gaze displacement) ──
    const dx = gazeX - this._prevGazeX;
    const dy = gazeY - this._prevGazeY;
    const velocity = Math.sqrt(dx * dx + dy * dy);

    this._prevGazeX = gazeX;
    this._prevGazeY = gazeY;

    // ── Fixation duration ──
    if (velocity < FIXATION_STABILITY_THRESHOLD) {
      this._fixationFrames++;
    } else {
      this._fixationFrames = 0;
    }

    return {
      gazeX,
      gazeY,
      earLeft,
      earRight,
      ear,
      velocity,
      fixationDuration: this._fixationFrames,
    };
  }

  // ===========================================================================
  // PRIVATE — Feature Smoothing
  // ===========================================================================

  /**
   * Exponential Moving Average smoothing across frames.
   * Mirrors UASAM._smoothFeatures().
   *
   * @param {object} raw — raw features from _extractFeatures()
   */
  _smoothFeatures(raw) {
    if (!this._prevFeatures) {
      this._features = { ...raw };
      this._prevFeatures = { ...raw };
      return;
    }

    const a = FEATURE_SMOOTH_ALPHA;
    this._features.gazeX     = a * raw.gazeX     + (1 - a) * this._prevFeatures.gazeX;
    this._features.gazeY     = a * raw.gazeY     + (1 - a) * this._prevFeatures.gazeY;
    this._features.earLeft   = a * raw.earLeft   + (1 - a) * this._prevFeatures.earLeft;
    this._features.earRight  = a * raw.earRight  + (1 - a) * this._prevFeatures.earRight;
    this._features.ear       = a * raw.ear       + (1 - a) * this._prevFeatures.ear;
    this._features.velocity  = a * raw.velocity  + (1 - a) * this._prevFeatures.velocity;
    this._features.fixationDuration = raw.fixationDuration; // counter, not smoothed

    this._prevFeatures = { ...this._features };
  }

  // ===========================================================================
  // PRIVATE — Gaze State Classification
  // ===========================================================================

  /**
   * Soft classification of gaze state from smoothed features.
   * Produces probability distribution over GAZE_STATE_ORDER.
   * Mirrors UASAM._classifyVocalizationState().
   */
  _classifyGazeState() {
    const { gazeX, gazeY, ear, velocity } = this._features;

    // Raw scores (unnormalized)
    const scores = {};

    // Eyes closed check
    if (ear < EAR_CLOSED_THRESHOLD) {
      scores[GAZE_STATES.EYES_CLOSED] = 3.0;
      scores[GAZE_STATES.CENTER]  = 0.1;
      scores[GAZE_STATES.LEFT]    = 0.1;
      scores[GAZE_STATES.RIGHT]   = 0.1;
      scores[GAZE_STATES.UP]      = 0.1;
      scores[GAZE_STATES.DOWN]    = 0.1;
      scores[GAZE_STATES.SACCADE] = 0.1;
    } else if (velocity > SACCADE_VELOCITY_THRESHOLD) {
      // Saccade — rapid eye movement
      scores[GAZE_STATES.SACCADE]     = 2.5;
      scores[GAZE_STATES.EYES_CLOSED] = 0.05;
      scores[GAZE_STATES.CENTER]  = 0.2;
      scores[GAZE_STATES.LEFT]    = 0.2;
      scores[GAZE_STATES.RIGHT]   = 0.2;
      scores[GAZE_STATES.UP]      = 0.2;
      scores[GAZE_STATES.DOWN]    = 0.2;
    } else {
      // Directional classification via soft boundaries
      scores[GAZE_STATES.EYES_CLOSED] = 0.05;
      scores[GAZE_STATES.SACCADE]     = sigmoid((velocity - SACCADE_VELOCITY_THRESHOLD) * 30);

      // Horizontal
      const leftScore   = sigmoid((GAZE_LEFT_THRESHOLD  - gazeX) * 15);
      const rightScore  = sigmoid((gazeX - GAZE_RIGHT_THRESHOLD) * 15);
      const centerXScore = 1.0 - leftScore - rightScore;

      // Vertical
      const upScore     = sigmoid((GAZE_UP_THRESHOLD   - gazeY) * 15);
      const downScore   = sigmoid((gazeY - GAZE_DOWN_THRESHOLD) * 15);
      const centerYScore = 1.0 - upScore - downScore;

      scores[GAZE_STATES.LEFT]   = Math.max(0, leftScore  * centerYScore);
      scores[GAZE_STATES.RIGHT]  = Math.max(0, rightScore * centerYScore);
      scores[GAZE_STATES.UP]     = Math.max(0, upScore    * centerXScore);
      scores[GAZE_STATES.DOWN]   = Math.max(0, downScore  * centerXScore);
      scores[GAZE_STATES.CENTER] = Math.max(0, centerXScore * centerYScore);
    }

    // Normalize to probability distribution
    let total = 0;
    for (const gs of GAZE_STATE_ORDER) {
      total += scores[gs] || 0;
    }
    if (total > 0) {
      for (const gs of GAZE_STATE_ORDER) {
        this._gazeStateProbabilities[gs] = (scores[gs] || 0) / total;
      }
    }

    // Hard classification (argmax)
    let maxProb = 0;
    for (const gs of GAZE_STATE_ORDER) {
      if (this._gazeStateProbabilities[gs] > maxProb) {
        maxProb = this._gazeStateProbabilities[gs];
        this._gazeState = gs;
      }
    }
  }

  // ===========================================================================
  // PRIVATE — Gaze Likelihood Computation
  // ===========================================================================

  /**
   * Compute P(G | S) for each communicative state via total probability.
   *
   *   P(G|S) = Σ_g P(gazeState=g | features) × P(g | S)
   *
   * where P(gazeState=g | features) comes from _classifyGazeState()
   * and P(g | S) comes from the GAZE_EMISSION_MATRIX.
   *
   * Mirrors UASAM._computeAcousticLikelihoods() (lines 826-845).
   */
  _computeGazeLikelihoods() {
    for (const stateId of this._stateIds) {
      const emissions = GAZE_EMISSION_MATRIX[stateId];
      if (!emissions) {
        this._gazeLikelihoods[stateId] = GAZE_PROB_FLOOR;
        continue;
      }

      // P(G|S) = Σ_g P(g|features) · P(g|S)
      let likelihood = 0;
      for (let g = 0; g < GAZE_STATE_ORDER.length; g++) {
        const gazeState = GAZE_STATE_ORDER[g];
        const pGazeGivenFeatures = this._gazeStateProbabilities[gazeState] || 0;
        const pGazeGivenState = emissions[g];
        likelihood += pGazeGivenFeatures * pGazeGivenState;
      }

      this._gazeLikelihoods[stateId] = Math.max(GAZE_PROB_FLOOR, likelihood);
    }
  }

  // ===========================================================================
  // PRIVATE — Utility
  // ===========================================================================

  /** Result when FaceLandmarker is inactive. Returns uniform likelihoods. */
  _inactiveResult() {
    const uniform = 1 / this._stateIds.length;
    const likelihoods = {};
    for (const stateId of this._stateIds) {
      likelihoods[stateId] = uniform;
    }
    return {
      features: { ...this._features },
      gaze_state: GAZE_STATES.CENTER,
      gaze_state_probabilities: { [GAZE_STATES.CENTER]: 1.0 },
      gaze_likelihoods: likelihoods,
      face_detected: false,
      frame: this._frameCount,
    };
  }
}

// =============================================================================
// MODULE-LEVEL HELPERS
// =============================================================================

/** Sigmoid for soft classification. */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/** Euclidean distance in 2D (x, y). */
function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Clamp value to [min, max]. */
function clamp(v, min, max) {
  return v < min ? min : v > max ? max : v;
}
