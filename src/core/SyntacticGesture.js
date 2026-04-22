/**
 * MLAF — SyntacticGesture
 *
 * Formal mathematical definition of a Syntactic Gesture:
 *   G = (C, τ, Φ, κ, ψ)
 *
 * Where:
 *   C  — Configuration vector (63 floats from 21 landmarks × 3 axes)
 *   τ  — Temporal class: STATIC | DYNAMIC
 *   Φ  — Constraint predicate (Boolean function over C)
 *   κ  — Grammar binding (token with morphosyntactic features)
 *   ψ  — Spatial modifier (tense derived from wrist Y/Z position)
 */

import { normalizeToWrist, flattenToConfigVector } from '../utils/vectorGeometry.js';
import { analyzeFingerStates } from '../utils/fingerAnalysis.js';
import { detectSubject, detectVerb, detectObject, detectSpatialModifier } from '../utils/gestureDetection.js';
import { GestureClassifierRF } from './GestureClassifierRF.js';

export const TEMPORAL_CLASS = Object.freeze({
  STATIC: 'STATIC',
  DYNAMIC: 'DYNAMIC',
});

/**
 * A SyntacticGesture is the formal unit of the MLAF framework.
 * It encapsulates both the physical hand configuration and its linguistic binding.
 */
export class SyntacticGesture {
  /**
   * @param {Float64Array} config - C: 63-dimensional configuration vector
   * @param {string} temporalClass - τ: STATIC or DYNAMIC
   * @param {Function} predicate - Φ: Boolean constraint function over landmarks
   * @param {Object} grammarBinding - κ: { grammar_id, category, features }
   * @param {string} spatialModifier - ψ: 'past' | 'present' | 'future'
   */
  constructor(config, temporalClass, predicate, grammarBinding, spatialModifier) {
    this.config = config;             // C ∈ ℝ^63
    this.temporalClass = temporalClass; // τ ∈ {STATIC, DYNAMIC}
    this.predicate = predicate;       // Φ: C → {true, false}
    this.grammarBinding = grammarBinding; // κ
    this.spatialModifier = spatialModifier; // ψ
    this.confidence = 0;
    this.errorVector = null;
    this.timestamp = performance.now();
  }

  /**
   * Test if given landmarks satisfy this gesture's constraint predicate.
   * @param {Array} landmarks - 21 hand landmarks
   * @returns {boolean}
   */
  test(landmarks) {
    return this.predicate(landmarks);
  }

  /**
   * Serialize this gesture into a prompt token for PromptTokenInterface.
   * @returns {object} Token with value, category, features, confidence, error_vector, temporal, tense
   */
  toPromptToken() {
    return {
      value: this.grammarBinding?.grammar_id || null,
      category: this.grammarBinding?.category || null,
      features: this.grammarBinding?.features || {},
      confidence: this.confidence,
      error_vector: this.errorVector,
      temporal: this.temporalClass,
      tense: this.spatialModifier,
    };
  }
}

/**
 * Registry of all defined SyntacticGestures in the system.
 */
export class SyntacticGestureSpace {
  constructor() {
    this.gestures = new Map();
  }

  register(id, gesture) {
    this.gestures.set(id, gesture);
  }

  get(id) {
    return this.gestures.get(id);
  }

  all() {
    return Array.from(this.gestures.values());
  }

  ids() {
    return Array.from(this.gestures.keys());
  }
}

/**
 * Full gesture vocabulary — all 19 gestures with morphosyntactic features.
 *
 * IDs here MUST match the grammar_id values returned by gestureDetection.js
 * and the keys used in GrammarEngine.js LEXICON / GestureGrammar.js parser:
 *   Subjects: SUBJECT_I, SUBJECT_YOU, etc.  (prefixed — matches detection)
 *   Verbs:    GRAB, EAT, SEE, etc.          (unprefixed — matches LEXICON)
 *   Objects:  APPLE, BALL, WATER, etc.       (unprefixed — matches LEXICON)
 */
const ALL_GESTURES = [
  // Subjects (D-heads)
  { id: 'SUBJECT_I',    label: 'I',    category: 'NP', features: { person: 1, number: 'singular' } },
  { id: 'SUBJECT_YOU',  label: 'You',  category: 'NP', features: { person: 2, number: 'singular' } },
  { id: 'SUBJECT_HE',   label: 'He',   category: 'NP', features: { person: 3, number: 'singular' } },
  { id: 'SUBJECT_SHE',  label: 'She',  category: 'NP', features: { person: 3, number: 'singular' } },
  { id: 'SUBJECT_WE',   label: 'We',   category: 'NP', features: { person: 1, number: 'plural' } },
  { id: 'SUBJECT_THEY', label: 'They', category: 'NP', features: { person: 3, number: 'plural' } },
  // Verbs (V-heads) — IDs match LEXICON keys (no VERB_ prefix)
  { id: 'WANT',  label: 'want',  category: 'VP', features: { tense: 'present', transitive: true } },
  { id: 'EAT',   label: 'eat',   category: 'VP', features: { tense: 'present', transitive: true } },
  { id: 'SEE',   label: 'see',   category: 'VP', features: { tense: 'present', transitive: true } },
  { id: 'GRAB',  label: 'grab',  category: 'VP', features: { tense: 'present', transitive: true } },
  { id: 'DRINK', label: 'drink', category: 'VP', features: { tense: 'present', transitive: true } },
  { id: 'GO',    label: 'go',    category: 'VP', features: { tense: 'present', transitive: false } },
  { id: 'STOP',  label: 'stop',  category: 'VP', features: { tense: 'present', transitive: false } },
  // Objects (N-heads) — IDs match LEXICON keys (no OBJECT_ prefix)
  { id: 'FOOD',  label: 'food',  category: 'NP', features: { type: 'noun', countable: true } },
  { id: 'WATER', label: 'water', category: 'NP', features: { type: 'noun', countable: false } },
  { id: 'BOOK',  label: 'book',  category: 'NP', features: { type: 'noun', countable: true } },
  { id: 'APPLE', label: 'apple', category: 'NP', features: { type: 'noun', countable: true } },
  { id: 'BALL',  label: 'ball',  category: 'NP', features: { type: 'noun', countable: true } },
  { id: 'HOUSE', label: 'house', category: 'NP', features: { type: 'noun', countable: true } },
];

/** Lookup table for gesture metadata by frontend ID. */
const GESTURE_META = Object.fromEntries(ALL_GESTURES.map(g => [g.id, g]));

/**
 * Build the default SyntacticGestureSpace with all 19 defined gestures.
 * Each gesture retains a heuristic predicate for the original 11 gestures
 * (used as fallback when the RF classifier is not loaded).
 * @returns {SyntacticGestureSpace}
 */
export function buildDefaultGestureSpace() {
  const space = new SyntacticGestureSpace();

  // Heuristic predicates for ALL 19 gestures (fallback when RF classifier unavailable).
  // Each predicate delegates to the corresponding detect*() function and checks
  // the grammar_id field (not .id — detect functions return grammar_id).
  const heuristicPredicates = {
    // Subjects — grammar_id matches ALL_GESTURES id (SUBJECT_X format)
    'SUBJECT_I':    (lm) => { const r = detectSubject(lm); return r !== null && r.grammar_id === 'SUBJECT_I'; },
    'SUBJECT_YOU':  (lm) => { const r = detectSubject(lm); return r !== null && r.grammar_id === 'SUBJECT_YOU'; },
    'SUBJECT_HE':   (lm) => { const r = detectSubject(lm); return r !== null && r.grammar_id === 'SUBJECT_HE'; },
    'SUBJECT_SHE':  (lm) => { const r = detectSubject(lm); return r !== null && r.grammar_id === 'SUBJECT_SHE'; },
    'SUBJECT_WE':   (lm) => { const r = detectSubject(lm); return r !== null && r.grammar_id === 'SUBJECT_WE'; },
    'SUBJECT_THEY': (lm) => { const r = detectSubject(lm); return r !== null && r.grammar_id === 'SUBJECT_THEY'; },
    // Verbs — grammar_id from detectVerb() is unprefixed (GRAB, not VERB_GRAB)
    'WANT':         (lm) => { const r = detectVerb(lm); return r !== null && r.grammar_id === 'WANT'; },
    'EAT':          (lm) => { const r = detectVerb(lm); return r !== null && r.grammar_id === 'EAT'; },
    'SEE':          (lm) => { const r = detectVerb(lm); return r !== null && r.grammar_id === 'SEE'; },
    'GRAB':         (lm) => { const r = detectVerb(lm); return r !== null && r.grammar_id === 'GRAB'; },
    'DRINK':        (lm) => { const r = detectVerb(lm); return r !== null && r.grammar_id === 'DRINK'; },
    'GO':           (lm) => { const r = detectVerb(lm); return r !== null && r.grammar_id === 'GO'; },
    'STOP':         (lm) => { const r = detectVerb(lm); return r !== null && r.grammar_id === 'STOP'; },
    // Objects — grammar_id from detectObject() is unprefixed (APPLE, not OBJECT_APPLE)
    'FOOD':         (lm) => { const r = detectObject(lm); return r !== null && r.grammar_id === 'FOOD'; },
    'WATER':        (lm) => { const r = detectObject(lm); return r !== null && r.grammar_id === 'WATER'; },
    'BOOK':         (lm) => { const r = detectObject(lm); return r !== null && r.grammar_id === 'BOOK'; },
    'APPLE':        (lm) => { const r = detectObject(lm); return r !== null && r.grammar_id === 'APPLE'; },
    'BALL':         (lm) => { const r = detectObject(lm); return r !== null && r.grammar_id === 'BALL'; },
    'HOUSE':        (lm) => { const r = detectObject(lm); return r !== null && r.grammar_id === 'HOUSE'; },
  };

  for (const g of ALL_GESTURES) {
    const predicate = heuristicPredicates[g.id] || (() => false);
    space.register(g.id, new SyntacticGesture(
      null,
      TEMPORAL_CLASS.STATIC,
      predicate,
      { grammar_id: g.id, category: g.category, features: g.features },
      'present'
    ));
  }

  return space;
}

// ---------------------------------------------------------------------------
// Singleton RF classifier (loaded once, used for all recognition calls)
// ---------------------------------------------------------------------------

let _rfClassifier = null;

/** Model load status — exposed for UI feedback */
let _modelLoadStatus = { loaded: false, error: null, attempted: false };

/**
 * Load the RF classifier model. Call once at app startup.
 * @param {string} [modelUrl] - URL to gesture_classifier_v1.json
 * @returns {Promise<boolean>} true if loaded successfully
 */
export async function loadGestureClassifier(modelUrl) {
  if (_rfClassifier?.loaded) return true;
  // Allow retry after a failed attempt (user may fix network/path issues)
  _modelLoadStatus.attempted = true;

  try {
    _rfClassifier = new GestureClassifierRF();
    const url = modelUrl || '/models/gesture_classifier_v1.json';
    await _rfClassifier.load(url);
    _modelLoadStatus = { loaded: true, error: null, attempted: true };
    console.log(`[MLAF] RF gesture classifier loaded (${_rfClassifier.trees.length} trees)`);
    return true;
  } catch (err) {
    const errorMsg = `ML model unavailable: ${err.message}. Using heuristic fallback (lower accuracy).`;
    _modelLoadStatus = { loaded: false, error: errorMsg, attempted: true };
    console.warn('[MLAF] RF classifier not available, using heuristic fallback:', err.message);
    _rfClassifier = null;
    return false;
  }
}

/**
 * Get the model load status for UI feedback.
 * @returns {{ loaded: boolean, error: string|null, attempted: boolean }}
 */
export function getModelLoadStatus() {
  return _modelLoadStatus;
}

/**
 * Get the loaded classifier instance (or null).
 * @returns {GestureClassifierRF|null}
 */
export function getGestureClassifier() {
  return _rfClassifier?.loaded ? _rfClassifier : null;
}

/**
 * Recognize a SyntacticGesture from raw landmarks.
 * Maps ℝ^63 → SyntacticGesture ∪ {∅}
 *
 * Strategy:
 *   1. If RF classifier is loaded → O(1) inference (independent of vocab size)
 *   2. Else → fallback to sequential heuristic predicates (O(n) over gestures)
 *
 * @param {Array} landmarks - 21 hand landmarks from MediaPipe
 * @param {SyntacticGestureSpace} space - The gesture space to search
 * @returns {SyntacticGesture|null} The recognized gesture or null
 */
export function recognize(landmarks, space) {
  if (!landmarks || landmarks.length < 21) return null;

  const normalized = normalizeToWrist(landmarks);
  const config = flattenToConfigVector(normalized);
  const spatialMod = detectSpatialModifier(landmarks);

  // --- Path 1: RF classifier (O(1) w.r.t. vocabulary size) ---
  const rf = getGestureClassifier();
  if (rf) {
    const result = rf.classify(landmarks);
    if (result) {
      const gestureEntry = space.get(result.id) || GESTURE_META[result.id];
      if (gestureEntry) {
        const binding = gestureEntry.grammarBinding || {
          grammar_id: result.id,
          category: result.category,
          features: {},
        };
        const recognized = new SyntacticGesture(
          config,
          TEMPORAL_CLASS.STATIC,
          () => true,
          binding,
          spatialMod,
        );
        recognized.confidence = result.confidence;
        recognized.classifierResult = result; // attach full result for debugging
        return recognized;
      }
    }
    // RF returned null (below confidence threshold) — no fallback, return null
    return null;
  }

  // --- Path 2: Heuristic fallback (O(n) over gestures) ---
  for (const [id, gesture] of space.gestures) {
    if (gesture.test(landmarks)) {
      const recognized = new SyntacticGesture(
        config,
        gesture.temporalClass,
        gesture.predicate,
        gesture.grammarBinding,
        spatialMod
      );
      recognized.confidence = 1.0;
      return recognized;
    }
  }

  return null; // ∅ — no gesture recognized
}
