/**
 * SGRM.js — Syntactic Gesture Reference Model
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * The reference database for the closed-loop error correction system.
 * For each gesture, stores ideal constraint parameters with tolerance bands.
 *
 * Exports:
 *   - GESTURE_REFERENCE_DB — full reference model for all gestures
 *   - computeErrorVector(currentLandmarks, gestureId) — per-constraint directional errors
 *   - findClosestGesture(landmarks) — nearest valid gesture + distance
 *   - isWithinTolerance(landmarks, gestureId) — Boolean
 */

import { euclideanDistance3D, angleBetweenPoints3D, normalizeToWrist } from '../utils/vectorGeometry';

// =============================================================================
// LANDMARK INDEX MAP (for readability)
// =============================================================================

const LM = {
  WRIST: 0,
  THUMB_CMC: 1, THUMB_MCP: 2, THUMB_IP: 3, THUMB_TIP: 4,
  INDEX_MCP: 5, INDEX_PIP: 6, INDEX_DIP: 7, INDEX_TIP: 8,
  MIDDLE_MCP: 9, MIDDLE_PIP: 10, MIDDLE_DIP: 11, MIDDLE_TIP: 12,
  RING_MCP: 13, RING_PIP: 14, RING_DIP: 15, RING_TIP: 16,
  PINKY_MCP: 17, PINKY_PIP: 18, PINKY_DIP: 19, PINKY_TIP: 20,
};

// =============================================================================
// CONSTRAINT EVALUATORS
// =============================================================================

/**
 * Evaluate a displacement constraint (distance between two landmarks along an axis).
 */
function evaluateDisplacement(normalized, constraint) {
  const lmA = normalized[constraint.landmark_a];
  const lmB = normalized[constraint.landmark_b];
  const axis = constraint.axis || 'x';
  const actual = lmA[axis] - lmB[axis];
  const tolerance = constraint.tolerance > 0 ? constraint.tolerance : 1;
  return {
    actual,
    ideal: constraint.ideal,
    deviation: actual - constraint.ideal,
    normalized_error: Math.abs(actual - constraint.ideal) / tolerance,
    direction: actual < constraint.ideal ? +1 : -1,
  };
}

/**
 * Evaluate an angle constraint (joint angle at a vertex).
 */
function evaluateAngle(normalized, constraint) {
  const a = normalized[constraint.joints[0]];
  const b = normalized[constraint.joints[1]];
  const c = normalized[constraint.joints[2]];
  const actual = angleBetweenPoints3D(a, b, c);
  const tolerance = constraint.tolerance > 0 ? constraint.tolerance : 1;
  return {
    actual,
    ideal: constraint.ideal_angle,
    deviation: actual - constraint.ideal_angle,
    normalized_error: Math.abs(actual - constraint.ideal_angle) / tolerance,
    direction: actual < constraint.ideal_angle ? +1 : -1,
  };
}

/**
 * Evaluate a distance constraint (Euclidean distance between two landmarks).
 */
function evaluateDistance(normalized, constraint) {
  const lmA = normalized[constraint.landmark_a];
  const lmB = normalized[constraint.landmark_b];
  const actual = euclideanDistance3D(lmA, lmB);
  return {
    actual,
    ideal: constraint.ideal,
    deviation: actual - constraint.ideal,
    normalized_error: Math.abs(actual - constraint.ideal) / (constraint.tolerance > 0 ? constraint.tolerance : 1),
    direction: actual < constraint.ideal ? +1 : -1,
  };
}

// =============================================================================
// GESTURE REFERENCE DATABASE
// =============================================================================

/**
 * The complete Syntactic Gesture Reference Model.
 * Each entry defines the ideal configuration and tolerance bands for a gesture.
 */
export const GESTURE_REFERENCE_DB = {
  SUBJECT_I: {
    gesture_id: 'SUBJECT_I',
    display: 'I',
    description: 'Thumb pointing inward (at self), four fingers curled — first person',
    ideal_constraints: [
      {
        id: 'thumb_inward',
        type: 'displacement',
        landmark_a: LM.THUMB_TIP,
        landmark_b: LM.WRIST,
        axis: 'x',
        ideal: 0.12,
        tolerance: 0.05,
        weight: 0.4,
        correction: 'Point thumb more toward yourself',
      },
      {
        id: 'index_curled',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl index finger tighter',
      },
      {
        id: 'middle_curled',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl middle finger tighter',
      },
      {
        id: 'ring_curled',
        type: 'angle',
        joints: [LM.RING_MCP, LM.RING_PIP, LM.RING_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl ring finger tighter',
      },
      {
        id: 'pinky_curled',
        type: 'angle',
        joints: [LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl pinky finger tighter',
      },
    ],
    grammar_binding: {
      grammar_id: 'SUBJECT_I',
      category: 'NP',
      features: { person: 1, number: 'singular' },
    },
  },

  SUBJECT_YOU: {
    gesture_id: 'SUBJECT_YOU',
    display: 'You',
    description: 'Index finger extended pointing forward, other fingers curled — second person',
    ideal_constraints: [
      {
        id: 'index_extended',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.3,
        correction: 'Straighten your index finger',
      },
      {
        id: 'index_forward',
        type: 'displacement',
        landmark_a: LM.INDEX_TIP,
        landmark_b: LM.WRIST,
        axis: 'z',
        ideal: -0.08,
        tolerance: 0.04,
        weight: 0.2,
        correction: 'Point index finger more toward the camera',
      },
      {
        id: 'middle_curled',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.17,
        correction: 'Curl middle finger tighter',
      },
      {
        id: 'ring_curled',
        type: 'angle',
        joints: [LM.RING_MCP, LM.RING_PIP, LM.RING_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.17,
        correction: 'Curl ring finger tighter',
      },
      {
        id: 'pinky_curled',
        type: 'angle',
        joints: [LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.16,
        correction: 'Curl pinky finger tighter',
      },
    ],
    grammar_binding: {
      grammar_id: 'SUBJECT_YOU',
      category: 'NP',
      features: { person: 2, number: 'singular' },
    },
  },

  SUBJECT_HE: {
    gesture_id: 'SUBJECT_HE',
    display: 'He/She',
    description: 'Thumb pointing outward (hitchhiker), four fingers curled — third person',
    ideal_constraints: [
      {
        id: 'thumb_outward',
        type: 'displacement',
        landmark_a: LM.THUMB_TIP,
        landmark_b: LM.WRIST,
        axis: 'x',
        ideal: -0.12,
        tolerance: 0.04,
        weight: 0.4,
        correction: 'Move thumb further left (outward)',
      },
      {
        id: 'index_curled',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl index finger tighter',
      },
      {
        id: 'middle_curled',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl middle finger tighter',
      },
      {
        id: 'ring_curled',
        type: 'angle',
        joints: [LM.RING_MCP, LM.RING_PIP, LM.RING_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl ring finger tighter',
      },
      {
        id: 'pinky_curled',
        type: 'angle',
        joints: [LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl pinky finger tighter',
      },
    ],
    grammar_binding: {
      grammar_id: 'SUBJECT_HE',
      category: 'NP',
      features: { person: 3, number: 'singular' },
    },
  },

  GRAB: {
    gesture_id: 'GRAB',
    display: 'Grab',
    description: 'All fingertips brought close together (claw/pinch) — transitive verb',
    ideal_constraints: [
      {
        id: 'thumb_index_close',
        type: 'distance',
        landmark_a: LM.THUMB_TIP,
        landmark_b: LM.INDEX_TIP,
        ideal: 0.03,
        tolerance: 0.03,
        weight: 0.25,
        correction: 'Bring thumb and index finger closer together',
      },
      {
        id: 'thumb_middle_close',
        type: 'distance',
        landmark_a: LM.THUMB_TIP,
        landmark_b: LM.MIDDLE_TIP,
        ideal: 0.03,
        tolerance: 0.03,
        weight: 0.25,
        correction: 'Bring thumb and middle finger closer together',
      },
      {
        id: 'thumb_ring_close',
        type: 'distance',
        landmark_a: LM.THUMB_TIP,
        landmark_b: LM.RING_TIP,
        ideal: 0.04,
        tolerance: 0.03,
        weight: 0.25,
        correction: 'Bring thumb and ring finger closer together',
      },
      {
        id: 'thumb_pinky_close',
        type: 'distance',
        landmark_a: LM.THUMB_TIP,
        landmark_b: LM.PINKY_TIP,
        ideal: 0.05,
        tolerance: 0.03,
        weight: 0.25,
        correction: 'Bring thumb and pinky closer together',
      },
    ],
    grammar_binding: {
      grammar_id: 'GRAB',
      category: 'VP',
      features: { transitivity: 'transitive', requires_object: true },
    },
  },

  STOP: {
    gesture_id: 'STOP',
    display: 'Stop',
    description: 'Open palm, all fingers extended, hand vertical — intransitive verb',
    ideal_constraints: [
      {
        id: 'index_extended',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your index finger',
      },
      {
        id: 'middle_extended',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your middle finger',
      },
      {
        id: 'ring_extended',
        type: 'angle',
        joints: [LM.RING_MCP, LM.RING_PIP, LM.RING_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your ring finger',
      },
      {
        id: 'pinky_extended',
        type: 'angle',
        joints: [LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your pinky finger',
      },
      {
        id: 'hand_vertical',
        type: 'displacement',
        landmark_a: LM.WRIST,
        landmark_b: LM.MIDDLE_TIP,
        axis: 'y',
        ideal: 0.25,
        tolerance: 0.1,
        weight: 0.2,
        correction: 'Raise your hand more vertically',
      },
    ],
    grammar_binding: {
      grammar_id: 'STOP',
      category: 'VP',
      features: { transitivity: 'intransitive', requires_object: false },
    },
  },

  APPLE: {
    gesture_id: 'APPLE',
    display: 'Apple',
    description: 'Cupped hand — all fingers slightly curved',
    ideal_constraints: [
      {
        id: 'index_curved',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 120,
        tolerance: 30,
        weight: 0.25,
        correction: 'Curve your index finger more (like holding a ball)',
      },
      {
        id: 'middle_curved',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 120,
        tolerance: 30,
        weight: 0.25,
        correction: 'Curve your middle finger more',
      },
      {
        id: 'ring_curved',
        type: 'angle',
        joints: [LM.RING_MCP, LM.RING_PIP, LM.RING_DIP],
        ideal_angle: 120,
        tolerance: 30,
        weight: 0.25,
        correction: 'Curve your ring finger more',
      },
      {
        id: 'pinky_curved',
        type: 'angle',
        joints: [LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP],
        ideal_angle: 120,
        tolerance: 30,
        weight: 0.25,
        correction: 'Curve your pinky finger more',
      },
    ],
    grammar_binding: {
      grammar_id: 'APPLE',
      category: 'OBJ',
      features: { noun_type: 'concrete', countable: true },
    },
  },

  BOOK: {
    gesture_id: 'BOOK',
    display: 'Book',
    description: 'Flat palm facing up — all fingers extended horizontally',
    ideal_constraints: [
      {
        id: 'index_extended',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.25,
        correction: 'Straighten your index finger',
      },
      {
        id: 'middle_extended',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.25,
        correction: 'Straighten your middle finger',
      },
      {
        id: 'hand_horizontal',
        type: 'displacement',
        landmark_a: LM.WRIST,
        landmark_b: LM.MIDDLE_TIP,
        axis: 'y',
        ideal: 0.0,
        tolerance: 0.1,
        weight: 0.25,
        correction: 'Hold your hand more horizontally',
      },
      {
        id: 'palm_up',
        type: 'displacement',
        landmark_a: LM.MIDDLE_MCP,
        landmark_b: LM.WRIST,
        axis: 'z',
        ideal: 0.02,
        tolerance: 0.03,
        weight: 0.25,
        correction: 'Turn your palm to face upward',
      },
    ],
    grammar_binding: {
      grammar_id: 'BOOK',
      category: 'OBJ',
      features: { noun_type: 'concrete', countable: true },
    },
  },

  HOUSE: {
    gesture_id: 'HOUSE',
    display: 'House',
    description: 'Index and middle fingertips touching (roof shape)',
    ideal_constraints: [
      {
        id: 'index_extended',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your index finger',
      },
      {
        id: 'middle_extended',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your middle finger',
      },
      {
        id: 'fingertips_touching',
        type: 'distance',
        landmark_a: LM.INDEX_TIP,
        landmark_b: LM.MIDDLE_TIP,
        ideal: 0.01,
        tolerance: 0.03,
        weight: 0.3,
        correction: 'Bring index and middle fingertips closer together',
      },
      {
        id: 'ring_curled',
        type: 'angle',
        joints: [LM.RING_MCP, LM.RING_PIP, LM.RING_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl your ring finger',
      },
      {
        id: 'pinky_curled',
        type: 'angle',
        joints: [LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.15,
        correction: 'Curl your pinky finger',
      },
    ],
    grammar_binding: {
      grammar_id: 'HOUSE',
      category: 'OBJ',
      features: { noun_type: 'concrete', countable: true },
    },
  },

  WATER: {
    gesture_id: 'WATER',
    display: 'Water',
    description: 'W-shape — index, middle, ring extended and spread',
    ideal_constraints: [
      {
        id: 'index_extended',
        type: 'angle',
        joints: [LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your index finger',
      },
      {
        id: 'middle_extended',
        type: 'angle',
        joints: [LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your middle finger',
      },
      {
        id: 'ring_extended',
        type: 'angle',
        joints: [LM.RING_MCP, LM.RING_PIP, LM.RING_DIP],
        ideal_angle: 170,
        tolerance: 20,
        weight: 0.2,
        correction: 'Straighten your ring finger',
      },
      {
        id: 'pinky_curled',
        type: 'angle',
        joints: [LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP],
        ideal_angle: 60,
        tolerance: 25,
        weight: 0.2,
        correction: 'Curl your pinky finger',
      },
      {
        id: 'fingers_spread',
        type: 'distance',
        landmark_a: LM.INDEX_TIP,
        landmark_b: LM.MIDDLE_TIP,
        ideal: 0.08,
        tolerance: 0.04,
        weight: 0.2,
        correction: 'Spread your fingers apart more',
      },
    ],
    grammar_binding: {
      grammar_id: 'WATER',
      category: 'OBJ',
      features: { noun_type: 'concrete', countable: false },
    },
  },
};

// =============================================================================
// ERROR VECTOR COMPUTATION
// =============================================================================

/**
 * Compute the error vector between current hand landmarks and a target gesture.
 * Returns per-constraint directional errors with correction instructions.
 *
 * @param {Array<{x:number, y:number, z?:number}>} currentLandmarks — 21 landmarks
 * @param {string} gestureId — target gesture ID from GESTURE_REFERENCE_DB
 * @returns {object|null} error vector with per-constraint details
 */
export function computeErrorVector(currentLandmarks, gestureId) {
  const ref = GESTURE_REFERENCE_DB[gestureId];
  if (!ref) return null;
  if (!currentLandmarks || currentLandmarks.length < 21) return null;

  const normalized = normalizeToWrist(currentLandmarks);
  const errors = [];
  let totalWeightedError = 0;
  let totalWeight = 0;

  for (const constraint of ref.ideal_constraints) {
    let result;

    switch (constraint.type) {
      case 'displacement':
        result = evaluateDisplacement(normalized, constraint);
        break;
      case 'angle':
        result = evaluateAngle(normalized, constraint);
        break;
      case 'distance':
        result = evaluateDistance(normalized, constraint);
        break;
      default:
        continue;
    }

    const constraintError = {
      constraint_id: constraint.id,
      type: constraint.type,
      actual_value: result.actual,
      ideal_value: result.ideal,
      deviation: result.deviation,
      normalized_error: result.normalized_error,
      direction: result.direction,
      weight: constraint.weight,
      correction_instruction: constraint.correction,
      within_tolerance: result.normalized_error <= 1.0,
    };

    errors.push(constraintError);
    totalWeightedError += result.normalized_error * constraint.weight;
    totalWeight += constraint.weight;
  }

  const aggregateError = totalWeight > 0 ? totalWeightedError / totalWeight : 0;

  return {
    gesture_target: gestureId,
    per_constraint_errors: errors,
    aggregate_error: aggregateError,
    is_within_tolerance: aggregateError <= 1.0,
    constraints_met: errors.filter(e => e.within_tolerance).length,
    constraints_total: errors.length,
  };
}

// =============================================================================
// FIND CLOSEST GESTURE
// =============================================================================

/**
 * Find the gesture in the reference database that is closest to the current landmarks.
 *
 * @param {Array<{x:number, y:number, z?:number}>} landmarks — 21 landmarks
 * @returns {object} { gesture_id, distance, is_within_tolerance, error_vector }
 */
export function findClosestGesture(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  let closest = null;
  let minError = Infinity;

  for (const gestureId of Object.keys(GESTURE_REFERENCE_DB)) {
    const errorVector = computeErrorVector(landmarks, gestureId);
    if (!errorVector) continue;

    if (errorVector.aggregate_error < minError) {
      minError = errorVector.aggregate_error;
      closest = {
        gesture_id: gestureId,
        distance: errorVector.aggregate_error,
        is_within_tolerance: errorVector.is_within_tolerance,
        error_vector: errorVector,
      };
    }
  }

  return closest;
}

// =============================================================================
// TOLERANCE CHECK
// =============================================================================

/**
 * Check if the current landmarks are within the tolerance band of a target gesture.
 *
 * @param {Array<{x:number, y:number, z?:number}>} landmarks — 21 landmarks
 * @param {string} gestureId — target gesture ID
 * @param {number} toleranceMultiplier — multiplier for tolerance bands (default: 1.0)
 * @returns {boolean}
 */
export function isWithinTolerance(landmarks, gestureId, toleranceMultiplier = 1.0) {
  const errorVector = computeErrorVector(landmarks, gestureId);
  if (!errorVector) return false;

  // With multiplier, all normalized errors must be within multiplied tolerance
  return errorVector.per_constraint_errors.every(
    e => e.normalized_error <= toleranceMultiplier
  );
}
