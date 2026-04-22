/**
 * generators.js — Synthetic Data Generators for AGGME & UMCE Testing
 *
 * Generates deterministic, reproducible landmark sequences and acoustic
 * likelihood arrays for rigorous pipeline testing.
 *
 * All generators use seeded pseudo-random noise so results are reproducible
 * across test runs.
 */

// =============================================================================
// SEEDED PRNG (Mulberry32 — deterministic across runs)
// =============================================================================

function mulberry32(seed) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller transform: uniform → Gaussian */
function gaussianPair(rng) {
  const u1 = rng();
  const u2 = rng();
  const mag = Math.sqrt(-2 * Math.log(u1 + 1e-10));
  return [mag * Math.cos(2 * Math.PI * u2), mag * Math.sin(2 * Math.PI * u2)];
}

function gaussian(rng) {
  return gaussianPair(rng)[0];
}

// =============================================================================
// BASE HAND POSES (21 landmarks each)
// =============================================================================

/**
 * Neutral resting hand — relaxed open hand at screen center.
 * This is the "zero" position from which tremor is added.
 */
export function neutralHand() {
  return [
    { x: 0.50, y: 0.60, z: 0.00 },  // 0: wrist
    { x: 0.48, y: 0.55, z: -0.01 }, // 1: thumb_cmc
    { x: 0.45, y: 0.50, z: -0.02 }, // 2: thumb_mcp
    { x: 0.43, y: 0.46, z: -0.02 }, // 3: thumb_ip
    { x: 0.41, y: 0.43, z: -0.02 }, // 4: thumb_tip
    { x: 0.47, y: 0.45, z: -0.01 }, // 5: index_mcp
    { x: 0.46, y: 0.38, z: -0.01 }, // 6: index_pip
    { x: 0.46, y: 0.33, z: -0.01 }, // 7: index_dip
    { x: 0.46, y: 0.29, z: -0.01 }, // 8: index_tip
    { x: 0.50, y: 0.44, z: -0.01 }, // 9: middle_mcp
    { x: 0.50, y: 0.36, z: -0.01 }, // 10: middle_pip
    { x: 0.50, y: 0.31, z: -0.01 }, // 11: middle_dip
    { x: 0.50, y: 0.27, z: -0.01 }, // 12: middle_tip
    { x: 0.53, y: 0.45, z: -0.01 }, // 13: ring_mcp
    { x: 0.53, y: 0.38, z: -0.01 }, // 14: ring_pip
    { x: 0.53, y: 0.33, z: -0.01 }, // 15: ring_dip
    { x: 0.53, y: 0.30, z: -0.01 }, // 16: ring_tip
    { x: 0.56, y: 0.47, z: -0.01 }, // 17: pinky_mcp
    { x: 0.56, y: 0.42, z: -0.01 }, // 18: pinky_pip
    { x: 0.56, y: 0.38, z: -0.01 }, // 19: pinky_dip
    { x: 0.56, y: 0.35, z: -0.01 }, // 20: pinky_tip
  ];
}

/**
 * GRAB gesture — all fingertips bunched close to thumb tip (claw shape).
 * avgThumbDist < 0.06 triggers detectGestureRaw → 'GRAB'.
 */
export function grabHand() {
  return [
    { x: 0.50, y: 0.55, z: 0.00 },  // wrist
    { x: 0.48, y: 0.50, z: -0.01 },
    { x: 0.47, y: 0.46, z: -0.02 },
    { x: 0.46, y: 0.43, z: -0.02 },
    { x: 0.46, y: 0.41, z: -0.02 }, // thumb_tip
    { x: 0.48, y: 0.46, z: -0.01 },
    { x: 0.47, y: 0.43, z: -0.01 },
    { x: 0.47, y: 0.42, z: -0.01 },
    { x: 0.47, y: 0.41, z: -0.01 }, // index_tip (near thumb)
    { x: 0.49, y: 0.45, z: -0.01 },
    { x: 0.49, y: 0.43, z: -0.01 },
    { x: 0.48, y: 0.42, z: -0.01 },
    { x: 0.48, y: 0.41, z: -0.01 }, // middle_tip (near thumb)
    { x: 0.50, y: 0.46, z: -0.01 },
    { x: 0.50, y: 0.44, z: -0.01 },
    { x: 0.49, y: 0.42, z: -0.01 },
    { x: 0.48, y: 0.41, z: -0.01 }, // ring_tip (near thumb)
    { x: 0.51, y: 0.47, z: -0.01 },
    { x: 0.51, y: 0.45, z: -0.01 },
    { x: 0.50, y: 0.43, z: -0.01 },
    { x: 0.49, y: 0.42, z: -0.01 }, // pinky_tip (near thumb)
  ];
}

/**
 * STOP gesture — all fingers extended, hand vertical (open palm).
 */
export function stopHand() {
  return [
    { x: 0.50, y: 0.70, z: 0.00 },  // wrist (low = vertical hand)
    { x: 0.43, y: 0.60, z: -0.01 },
    { x: 0.40, y: 0.55, z: -0.02 },
    { x: 0.38, y: 0.50, z: -0.02 },
    { x: 0.36, y: 0.47, z: -0.02 }, // thumb_tip (extended)
    { x: 0.46, y: 0.52, z: -0.01 },
    { x: 0.45, y: 0.44, z: -0.01 },
    { x: 0.45, y: 0.39, z: -0.01 },
    { x: 0.45, y: 0.35, z: -0.01 }, // index_tip (extended up)
    { x: 0.50, y: 0.51, z: -0.01 },
    { x: 0.50, y: 0.42, z: -0.01 },
    { x: 0.50, y: 0.37, z: -0.01 },
    { x: 0.50, y: 0.33, z: -0.01 }, // middle_tip (extended up)
    { x: 0.54, y: 0.52, z: -0.01 },
    { x: 0.54, y: 0.44, z: -0.01 },
    { x: 0.54, y: 0.39, z: -0.01 },
    { x: 0.54, y: 0.35, z: -0.01 }, // ring_tip (extended up)
    { x: 0.58, y: 0.54, z: -0.01 },
    { x: 0.58, y: 0.47, z: -0.01 },
    { x: 0.58, y: 0.42, z: -0.01 },
    { x: 0.58, y: 0.38, z: -0.01 }, // pinky_tip (extended up)
  ];
}

/**
 * SUBJECT_I gesture — thumb pointing inward, four fingers curled.
 */
export function subjectIHand() {
  return [
    { x: 0.50, y: 0.55, z: 0.00 },  // wrist
    { x: 0.52, y: 0.50, z: -0.01 },
    { x: 0.55, y: 0.47, z: -0.02 },
    { x: 0.57, y: 0.44, z: -0.02 },
    { x: 0.59, y: 0.42, z: -0.02 }, // thumb_tip (x > wrist.x + 0.05 = inward)
    { x: 0.48, y: 0.48, z: -0.01 },
    { x: 0.47, y: 0.45, z: -0.01 },
    { x: 0.47, y: 0.47, z: -0.01 },
    { x: 0.47, y: 0.50, z: -0.01 }, // index_tip (curled: tip.y > pip.y)
    { x: 0.50, y: 0.47, z: -0.01 },
    { x: 0.50, y: 0.44, z: -0.01 },
    { x: 0.50, y: 0.46, z: -0.01 },
    { x: 0.50, y: 0.49, z: -0.01 }, // middle_tip (curled)
    { x: 0.52, y: 0.48, z: -0.01 },
    { x: 0.52, y: 0.45, z: -0.01 },
    { x: 0.52, y: 0.47, z: -0.01 },
    { x: 0.52, y: 0.50, z: -0.01 }, // ring_tip (curled)
    { x: 0.54, y: 0.50, z: -0.01 },
    { x: 0.54, y: 0.47, z: -0.01 },
    { x: 0.54, y: 0.49, z: -0.01 },
    { x: 0.54, y: 0.52, z: -0.01 }, // pinky_tip (curled)
  ];
}

// =============================================================================
// LANDMARK SEQUENCE GENERATORS
// =============================================================================

/**
 * Generate a resting tremor sequence — base hand + Gaussian noise.
 *
 * @param {number} numFrames — number of frames to generate
 * @param {number} sigma — noise standard deviation (controls tremor intensity)
 * @param {Function} [basePose] — base hand pose function (default: neutralHand)
 * @param {number} [seed] — PRNG seed for reproducibility
 * @returns {Array} Array of frames, each frame = 21 landmarks
 */
export function generateTremorSequence(numFrames, sigma, basePose = neutralHand, seed = 42) {
  const rng = mulberry32(seed);
  const base = basePose();
  const frames = [];

  for (let f = 0; f < numFrames; f++) {
    const frame = base.map(lm => ({
      x: lm.x + gaussian(rng) * sigma,
      y: lm.y + gaussian(rng) * sigma,
      z: lm.z + gaussian(rng) * sigma,
    }));
    frames.push(frame);
  }

  return frames;
}

/**
 * Generate an intentional gesture trajectory — smooth interpolation
 * from one pose to another, with optional noise overlay.
 *
 * Uses cosine interpolation for natural acceleration/deceleration.
 *
 * @param {number} numFrames — total frames for the transition
 * @param {Function} startPose — starting hand pose function
 * @param {Function} endPose — ending hand pose function
 * @param {number} [sigma] — noise to overlay (default: 0.002)
 * @param {number} [seed] — PRNG seed
 * @returns {Array} Array of frames
 */
export function generateGestureTrajectory(numFrames, startPose, endPose, sigma = 0.002, seed = 123) {
  const rng = mulberry32(seed);
  const start = startPose();
  const end = endPose();
  const frames = [];

  for (let f = 0; f < numFrames; f++) {
    // Cosine interpolation: smooth ease-in-out (guard against numFrames=1)
    const divisor = Math.max(1, numFrames - 1);
    const t = 0.5 * (1 - Math.cos(Math.PI * f / divisor));
    const frame = start.map((sLm, i) => {
      const eLm = end[i];
      return {
        x: sLm.x + (eLm.x - sLm.x) * t + gaussian(rng) * sigma,
        y: sLm.y + (eLm.y - sLm.y) * t + gaussian(rng) * sigma,
        z: sLm.z + (eLm.z - sLm.z) * t + gaussian(rng) * sigma,
      };
    });
    frames.push(frame);
  }

  return frames;
}

/**
 * Generate a complete test sequence:
 *   [resting tremor] → [onset transition] → [sustained gesture] → [offset transition] → [resting tremor]
 *
 * @param {object} config
 * @param {number} config.restingFrames — frames of pure tremor before & after gesture
 * @param {number} config.transitionFrames — frames for onset/offset transitions
 * @param {number} config.sustainedFrames — frames holding the target gesture
 * @param {number} config.tremorSigma — noise σ for resting tremor
 * @param {number} config.gestureSigma — noise σ during gesture (typically lower)
 * @param {Function} config.gesturePose — target gesture pose function
 * @param {number} [config.seed] — PRNG seed
 * @returns {{ frames: Array, labels: Array<string>, phases: object }}
 */
export function generateFullTestSequence(config) {
  const {
    restingFrames = 90,
    transitionFrames = 15,
    sustainedFrames = 60,
    tremorSigma = 0.008,
    gestureSigma = 0.003,
    gesturePose = grabHand,
    seed = 42,
  } = config;

  const allFrames = [];
  const labels = [];
  let seedOffset = seed;

  // Phase 1: Resting tremor (pre-gesture)
  const restingPre = generateTremorSequence(restingFrames, tremorSigma, neutralHand, seedOffset++);
  allFrames.push(...restingPre);
  for (let i = 0; i < restingFrames; i++) labels.push('RESTING');

  // Phase 2: Onset transition (neutral → gesture)
  const onset = generateGestureTrajectory(transitionFrames, neutralHand, gesturePose, gestureSigma, seedOffset++);
  allFrames.push(...onset);
  for (let i = 0; i < transitionFrames; i++) labels.push('TRANSITION_ONSET');

  // Phase 3: Sustained gesture
  const sustained = generateTremorSequence(sustainedFrames, gestureSigma, gesturePose, seedOffset++);
  allFrames.push(...sustained);
  for (let i = 0; i < sustainedFrames; i++) labels.push('GESTURE_ACTIVE');

  // Phase 4: Offset transition (gesture → neutral)
  const offset = generateGestureTrajectory(transitionFrames, gesturePose, neutralHand, gestureSigma, seedOffset++);
  allFrames.push(...offset);
  for (let i = 0; i < transitionFrames; i++) labels.push('TRANSITION_OFFSET');

  // Phase 5: Resting tremor (post-gesture)
  const restingPost = generateTremorSequence(restingFrames, tremorSigma, neutralHand, seedOffset++);
  allFrames.push(...restingPost);
  for (let i = 0; i < restingFrames; i++) labels.push('RESTING');

  return {
    frames: allFrames,
    labels,
    totalFrames: allFrames.length,
    phases: {
      resting_pre:   { start: 0, end: restingFrames },
      onset:         { start: restingFrames, end: restingFrames + transitionFrames },
      sustained:     { start: restingFrames + transitionFrames, end: restingFrames + transitionFrames + sustainedFrames },
      offset:        { start: restingFrames + transitionFrames + sustainedFrames, end: restingFrames + 2 * transitionFrames + sustainedFrames },
      resting_post:  { start: restingFrames + 2 * transitionFrames + sustainedFrames, end: 2 * restingFrames + 2 * transitionFrames + sustainedFrames },
    },
  };
}

/**
 * Generate a hand at a specific screen position (for spatial zone testing).
 *
 * @param {number} screenX — normalized x position [0, 1] (after mirror flip)
 * @param {number} screenY — normalized y position [0, 1]
 * @param {Function} [basePose] — base hand pose
 * @returns {Array} 21 landmarks shifted to the target position
 */
export function generateHandAtPosition(screenX, screenY, basePose = neutralHand) {
  const base = basePose();
  const wrist = base[0];
  // Mirror compensation: screenX = 1.0 - rawX → rawX = 1.0 - screenX
  const rawX = 1.0 - screenX;
  const dx = rawX - wrist.x;
  const dy = screenY - wrist.y;
  return base.map(lm => ({
    x: lm.x + dx,
    y: lm.y + dy,
    z: lm.z,
  }));
}

// =============================================================================
// ACOUSTIC LIKELIHOOD GENERATORS (for UMCE testing)
// =============================================================================

/**
 * Generate a mock P(A|S) array where one gesture has high likelihood.
 *
 * @param {string} targetGesture — gesture ID to peak at
 * @param {number} peakProb — probability mass for the target
 * @param {Array} allGestureIds — all gesture IDs in the state space
 * @returns {Object} Map<gesture_id, probability>
 */
export function generateAcousticPeak(targetGesture, peakProb, allGestureIds) {
  const likelihoods = {};
  const remaining = 1 - peakProb;
  const uniformRest = remaining / (allGestureIds.length - 1);

  for (const gId of allGestureIds) {
    likelihoods[gId] = gId === targetGesture ? peakProb : uniformRest;
  }

  return likelihoods;
}

/**
 * Generate a uniform (non-informative) P(A|S) array.
 *
 * @param {Array} allGestureIds
 * @returns {Object} Map<gesture_id, probability>
 */
export function generateUniformAcoustic(allGestureIds) {
  const likelihoods = {};
  const uniform = 1 / allGestureIds.length;
  for (const gId of allGestureIds) {
    likelihoods[gId] = uniform;
  }
  return likelihoods;
}

/**
 * Generate a mock spatial result for a specific zone.
 *
 * @param {string} zone — 'SUBJECT_ZONE' | 'VERB_ZONE' | 'OBJECT_ZONE'
 * @param {number} confidence — zone confidence [0, 1]
 * @returns {object} SpatialGrammarMapper-compatible result
 */
export function generateSpatialResult(zone, confidence = 0.8) {
  const labels = {
    SUBJECT_ZONE: 'Subject (NP)',
    VERB_ZONE: 'Verb (VP)',
    OBJECT_ZONE: 'Object (NP)',
  };
  const tense = 'PRESENT';
  return {
    syntactic_zone: zone,
    syntactic_role: zone.replace('_ZONE', ''),
    syntactic_label: labels[zone] || zone,
    tense_zone: tense,
    tense_label: 'Present',
    zone_confidence: confidence,
    movement_intensity: 0.001,
    duration_in_zone_ms: 500,
    zone_changed: false,
    tense_changed: false,
  };
}

/**
 * Generate a mock intent result.
 *
 * @param {string} intent — 'RESTING' | 'GESTURE_ACTIVE'
 * @param {number} displacement — z-scored displacement value
 * @returns {object} IntentionalityDetector-compatible result
 */
export function generateIntentResult(intent, displacement = 2.0) {
  return {
    intent,
    displacement,
    onset: false,
    offset: false,
    duration_ms: 0,
    trajectory: { dx: 0, dy: 0, dz: 0, distance: 0 },
  };
}

// =============================================================================
// VARIANCE MEASUREMENT UTILITY
// =============================================================================

/**
 * Compute the mean 3D variance across all landmarks in a frame sequence.
 *
 * @param {Array} frames — array of 21-landmark frames
 * @returns {number} mean variance
 */
export function computeSequenceVariance(frames) {
  if (frames.length < 2) return 0;

  const numLandmarks = frames[0].length;
  let totalVariance = 0;

  for (let lm = 0; lm < numLandmarks; lm++) {
    // Compute variance for each axis
    for (const axis of ['x', 'y', 'z']) {
      const values = frames.map(f => f[lm][axis]);
      const mean = values.reduce((s, v) => s + v, 0) / values.length;
      const variance = values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length;
      totalVariance += variance;
    }
  }

  return totalVariance / (numLandmarks * 3); // mean per-axis per-landmark
}
