/**
 * MLAF — Client-Side Random Forest Gesture Classifier
 *
 * O(1) gesture classification regardless of vocabulary size.
 * Loads the exported gesture_classifier_v1.json and runs inference
 * entirely in the browser — no server roundtrip.
 *
 * Replaces the sequential predicate-testing loop in SyntacticGesture.js
 * with a single RF traversal: O(tree_depth × n_trees), constant w.r.t.
 * number of gesture classes.
 */

import { normalizeToWrist, euclideanDistance3D, angleBetweenPoints3D } from '../utils/vectorGeometry.js';

// ---------------------------------------------------------------------------
// Gesture ID mappings (must match training/config.py exactly)
// ---------------------------------------------------------------------------

const GESTURE_IDS = [
  'subject_i', 'subject_you', 'subject_he', 'subject_she',
  'subject_we', 'subject_they',
  'verb_want', 'verb_eat', 'verb_see', 'verb_grab',
  'verb_drink', 'verb_go', 'verb_stop',
  'object_food', 'object_water', 'object_book',
  'object_apple', 'object_ball', 'object_house',
];

// Frontend IDs must match the grammar_id values used by LEXICON, parser, and
// SyntacticGesture.ALL_GESTURES: subjects are SUBJECT_X, verbs/objects are unprefixed.
const IDX_TO_FRONTEND_ID = {
  0:  'SUBJECT_I',   1:  'SUBJECT_YOU',  2:  'SUBJECT_HE',
  3:  'SUBJECT_SHE', 4:  'SUBJECT_WE',   5:  'SUBJECT_THEY',
  6:  'WANT',        7:  'EAT',          8:  'SEE',
  9:  'GRAB',        10: 'DRINK',        11: 'GO',
  12: 'STOP',
  13: 'FOOD',        14: 'WATER',        15: 'BOOK',
  16: 'APPLE',       17: 'BALL',         18: 'HOUSE',
};

const GESTURE_LABELS = {
  SUBJECT_I: 'I', SUBJECT_YOU: 'You', SUBJECT_HE: 'He',
  SUBJECT_SHE: 'She', SUBJECT_WE: 'We', SUBJECT_THEY: 'They',
  WANT: 'want', EAT: 'eat', SEE: 'see',
  GRAB: 'grab', DRINK: 'drink', GO: 'go', STOP: 'stop',
  FOOD: 'food', WATER: 'water', BOOK: 'book',
  APPLE: 'apple', BALL: 'ball', HOUSE: 'house',
};

const GESTURE_CATEGORIES = {
  SUBJECT_I: 'NP', SUBJECT_YOU: 'NP', SUBJECT_HE: 'NP',
  SUBJECT_SHE: 'NP', SUBJECT_WE: 'NP', SUBJECT_THEY: 'NP',
  WANT: 'VP', EAT: 'VP', SEE: 'VP',
  GRAB: 'VP', DRINK: 'VP', GO: 'VP', STOP: 'VP',
  FOOD: 'NP', WATER: 'NP', BOOK: 'NP',
  APPLE: 'NP', BALL: 'NP', HOUSE: 'NP',
};

// Fingertip and MCP indices (MediaPipe hand model)
const TIPS = [4, 8, 12, 16, 20];
const MCPS = [2, 5, 9, 13, 17];
const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'];

// ---------------------------------------------------------------------------
// Feature engineering (must match preprocess.py exactly)
// ---------------------------------------------------------------------------

/**
 * Normalize landmarks: translate wrist to origin, scale max distance to 1.
 * @param {Array} landmarks - 21 {x,y,z} objects
 * @returns {Array} Normalized landmarks
 */
function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const centered = landmarks.map(lm => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: (lm.z || 0) - (wrist.z || 0),
  }));

  let maxDist = 0;
  for (let i = 0; i < 21; i++) {
    const d = Math.sqrt(centered[i].x ** 2 + centered[i].y ** 2 + centered[i].z ** 2);
    if (d > maxDist) maxDist = d;
  }

  if (maxDist < 1e-8) return centered;
  return centered.map(lm => ({
    x: lm.x / maxDist,
    y: lm.y / maxDist,
    z: lm.z / maxDist,
  }));
}

/**
 * Extract full feature vector (63 raw + 23 engineered = 86 features).
 * Matches the Python preprocess.py pipeline exactly.
 * @param {Array} landmarks - 21 {x,y,z} objects (raw from MediaPipe)
 * @returns {Float64Array} Feature vector
 */
export function extractFeatures(landmarks) {
  const norm = normalizeLandmarks(landmarks);
  const features = [];

  // 1. Raw 63 landmark coordinates
  for (let i = 0; i < 21; i++) {
    features.push(norm[i].x, norm[i].y, norm[i].z);
  }

  // 2. Inter-finger distances (10 features) — all pairs of fingertips
  for (let i = 0; i < TIPS.length; i++) {
    for (let j = i + 1; j < TIPS.length; j++) {
      features.push(euclideanDistance3D(norm[TIPS[i]], norm[TIPS[j]]));
    }
  }

  // 3. Finger curl angles (5 features) — tip-MCP-wrist angle
  const wrist = norm[0];
  for (let f = 0; f < 5; f++) {
    const tip = norm[TIPS[f]];
    const mcp = norm[MCPS[f]];
    const v1x = tip.x - mcp.x, v1y = tip.y - mcp.y, v1z = tip.z - mcp.z;
    const v2x = wrist.x - mcp.x, v2y = wrist.y - mcp.y, v2z = wrist.z - mcp.z;
    const dot = v1x * v2x + v1y * v2y + v1z * v2z;
    const mag1 = Math.sqrt(v1x * v1x + v1y * v1y + v1z * v1z);
    const mag2 = Math.sqrt(v2x * v2x + v2y * v2y + v2z * v2z);
    const cosAngle = Math.max(-1, Math.min(1, dot / (mag1 * mag2 + 1e-8)));
    features.push(Math.acos(cosAngle));
  }

  // 4. Thumb-to-finger distance ratios (4 features)
  const thumbTip = norm[TIPS[0]];
  const palmSpan = euclideanDistance3D(norm[TIPS[1]], norm[TIPS[4]]) + 1e-8;
  for (let f = 1; f < 5; f++) {
    features.push(euclideanDistance3D(thumbTip, norm[TIPS[f]]) / palmSpan);
  }

  // 5. Hand spread (1 feature)
  let maxPairDist = 0;
  for (let i = 0; i < 21; i++) {
    for (let j = i + 1; j < 21; j++) {
      const d = euclideanDistance3D(norm[i], norm[j]);
      if (d > maxPairDist) maxPairDist = d;
    }
  }
  features.push(maxPairDist);

  // 6. Center of mass offset (3 features)
  let comX = 0, comY = 0, comZ = 0;
  for (let i = 0; i < 21; i++) {
    comX += norm[i].x; comY += norm[i].y; comZ += norm[i].z;
  }
  features.push(comX / 21, comY / 21, comZ / 21);

  return new Float64Array(features);
}

// ---------------------------------------------------------------------------
// Random Forest inference engine
// ---------------------------------------------------------------------------

/**
 * Traverse a single decision tree.
 * @param {Object} tree - { feature, threshold, children_left, children_right, value }
 * @param {Float64Array} features - Input feature vector
 * @returns {Float64Array} Class vote counts at the leaf node
 */
function traverseTree(tree, features) {
  let node = 0;
  while (tree.children_left[node] !== -1) { // -1 = leaf
    if (features[tree.feature[node]] <= tree.threshold[node]) {
      node = tree.children_left[node];
    } else {
      node = tree.children_right[node];
    }
  }
  return tree.value[node];
}

/**
 * GestureClassifierRF — Client-side Random Forest gesture classifier.
 *
 * After loading, classify() runs in O(tree_depth × n_trees) time
 * regardless of vocabulary size. No network calls.
 */
export class GestureClassifierRF {
  constructor() {
    this.trees = null;
    this.meta = null;
    this.loaded = false;
    this.numClasses = GESTURE_IDS.length;
  }

  /**
   * Load model from JSON URL or object.
   * Auto-syncs gesture ID mappings from model metadata when available,
   * falling back to hardcoded mappings if metadata lacks class info.
   * @param {string|Object} source - URL to gesture_classifier_v1.json or parsed object
   */
  async load(source) {
    let data;
    if (typeof source === 'string') {
      const resp = await fetch(source);
      if (!resp.ok) throw new Error(`Failed to load classifier: ${resp.status}`);
      data = await resp.json();
    } else {
      data = source;
    }

    this.meta = data._meta;
    this.trees = data.trees;
    this.numClasses = this.meta.num_classes;

    // Auto-sync gesture IDs from model metadata if available.
    // Models may use either 'class_names' or 'gesture_ids' as the key.
    const classNames = this.meta.class_names || this.meta.gesture_ids;
    if (classNames && Array.isArray(classNames)) {
      this._syncGestureIds(classNames);
    }

    this.loaded = true;
  }

  /**
   * Sync frontend gesture ID mappings from model's class_names metadata.
   * Converts training IDs (e.g. 'subject_i' → 'SUBJECT_I', 'verb_grab' → 'GRAB')
   * to match the unprefixed convention used by LEXICON and parser.
   */
  _syncGestureIds(classNames) {
    // Build instance-local mapping instead of mutating shared module object
    this._idxToFrontendId = {};
    for (let i = 0; i < classNames.length; i++) {
      const upper = classNames[i].toUpperCase();
      // Subjects keep their prefix (SUBJECT_I); verbs/objects strip it
      if (upper.startsWith('SUBJECT_')) {
        this._idxToFrontendId[i] = upper;
      } else if (upper.startsWith('VERB_')) {
        this._idxToFrontendId[i] = upper.replace('VERB_', '');
      } else if (upper.startsWith('OBJECT_')) {
        this._idxToFrontendId[i] = upper.replace('OBJECT_', '');
      } else {
        this._idxToFrontendId[i] = upper;
      }
    }
    this.numClasses = classNames.length;
  }

  /**
   * Classify landmarks into a gesture. O(1) w.r.t. vocabulary size.
   * @param {Array} landmarks - 21 {x,y,z} hand landmarks from MediaPipe
   * @param {number} [confidenceThreshold=0.4] - Minimum confidence to return a result
   * @returns {Object|null} { id, label, category, confidence, probabilities }
   */
  classify(landmarks, confidenceThreshold = 0.4) {
    if (!this.loaded || !landmarks || landmarks.length < 21) return null;

    // Extract features (matches Python pipeline)
    const features = extractFeatures(landmarks);

    // Aggregate votes across all trees
    const votes = new Float64Array(this.numClasses);
    for (let t = 0; t < this.trees.length; t++) {
      const leafVotes = traverseTree(this.trees[t], features);
      for (let c = 0; c < this.numClasses; c++) {
        votes[c] += leafVotes[c] || 0;
      }
    }

    // Normalize to probabilities
    let total = 0;
    for (let c = 0; c < this.numClasses; c++) total += votes[c];
    if (total === 0) return null;

    const probabilities = new Float64Array(this.numClasses);
    let bestIdx = 0;
    let bestProb = 0;
    for (let c = 0; c < this.numClasses; c++) {
      probabilities[c] = votes[c] / total;
      if (probabilities[c] > bestProb) {
        bestProb = probabilities[c];
        bestIdx = c;
      }
    }

    if (bestProb < confidenceThreshold) return null;

    const frontendId = (this._idxToFrontendId || IDX_TO_FRONTEND_ID)[bestIdx];
    return {
      id: frontendId,
      label: GESTURE_LABELS[frontendId],
      category: GESTURE_CATEGORIES[frontendId],
      confidence: bestProb,
      classIndex: bestIdx,
      gestureId: (bestIdx < GESTURE_IDS.length) ? GESTURE_IDS[bestIdx] : frontendId,
      probabilities,
    };
  }
}
