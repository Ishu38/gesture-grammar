/**
 * UMCE.js — Unified Multimodal Classification Engine
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BAYESIAN BIMODAL LATE FUSION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The MERN frontend processes audio and video independently, generating
 * two separate probability arrays. The UMCE then mathematically fuses
 * these independent probabilities using a Bayesian framework to calculate
 * the final posterior probability of a communicative state S, given
 * acoustic data A and visual gesture data V:
 *
 *     P(S | A, V) ∝ P(A | S) · P(V | S) · P(S)
 *
 * Where:
 *   P(S)      is the prior — syntactic expectation, temporal context,
 *             mastery history (e.g., how likely the child is to produce
 *             a SUBJECT next given that the sentence is empty).
 *   P(A | S)  is the acoustic likelihood from UASAM — probability of
 *             the observed spectral features given state S.
 *   P(V | S)  is the visual likelihood from AGGME — probability of
 *             the observed hand configuration given state S.
 *
 * This is true late fusion: each modality produces its own independent
 * probability array over the full state space, and the UMCE multiplies
 * them under Bayes' rule. The two channels never see each other's raw
 * data — only their probability outputs are combined.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * VISUAL CHANNEL: P(V | S)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The visual likelihood is itself a fusion of 4 visual sub-modalities
 * combined via weighted log-linear pooling:
 *
 *   log P(V | S) = w_shape · log L_shape(S)
 *                + w_spatial · log L_spatial(S)
 *                + w_intent · log L_intent
 *                + w_temporal · log L_temporal(S)
 *
 *   V1: Shape Classifier (w=0.45) — sigmoid-activated geometric constraints
 *   V2: Spatial Zone (w=0.25) — SVO zone congruence from SpatialGrammarMapper
 *   V3: Intentionality (w=0.15) — displacement gate from IntentionalityDetector
 *   V4: Temporal (w=0.15) — frame-consistency from sliding window
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ACOUSTIC CHANNEL: P(A | S)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Produced by UASAM (Unified Acoustic State Analysis Module):
 *   Microphone → AudioContext → AnalyserNode → Feature Extraction
 *   → Vocalization Classification → Emission Matrix → P(A | S)
 *
 * When microphone is inactive, P(A|S) = uniform (non-informative),
 * and the Bayesian product degrades gracefully to visual-only mode.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PRIOR: P(S)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   - Syntactic context: SVO ordering → SUBJECT boosted when sentence empty,
 *     VERB boosted after subject, OBJECT boosted after verb.
 *   - Mastery history: gestures the user has mastered get a slight prior
 *     boost (they are more likely to be intentionally produced).
 *   - Temporal regularity: frequency of each state in recent session history.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * FUSION EQUATION (element-wise for each state S_i)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   unnormalized_i = P(A | S_i) · P(V | S_i) · P(S_i)
 *   P(S_i | A, V) = unnormalized_i / Σ_j unnormalized_j
 *
 * In log-space (numerically stable):
 *   log_score_i = log P(A|S_i) + log P(V|S_i) + log P(S_i)
 *   P(S_i|A,V) = softmax(log_score / T)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * OUTPUT
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   {
 *     top1:       { gesture_id, probability, category },
 *     top3:       [ { gesture_id, probability, category }, ... ],
 *     posterior:   Map<gesture_id, probability>,
 *     entropy:     number (bits),
 *     margin:      number (top1 - top2),
 *     modality_contributions: {
 *       visual:   { top1, P_V, sub_modalities: {shape, spatial, intent, temporal} },
 *       acoustic: { top1, P_A, is_active, vocalization_state },
 *       prior:    { top1, expected_category },
 *     },
 *     fusion_mode: 'TRIMODAL' | 'BIMODAL_VA' | 'BIMODAL_VG' | 'VISUAL_ONLY',
 *     decision_quality: 'HIGH' | 'MEDIUM' | 'LOW' | 'REJECT',
 *   }
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PIPELINE POSITION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   ┌─ VISUAL CHANNEL (AGGME) ──────────────────────────────┐
 *   │ MediaPipe → Calibrator → Smoother                     │
 *   │          → IntentionalityDetector ───┐                │
 *   │          → SpatialGrammarMapper ──┐  │                │
 *   │          → Shape Constraints ──┐  │  │                │
 *   │          → Temporal Buffer ──┐ │  │  │                │
 *   │                              ↓ ↓  ↓  ↓                │
 *   │                        P(V | S) = Σ w_m·log L_m       │
 *   └───────────────────────────────┬───────────────────────┘
 *                                   │
 *   ┌─ ACOUSTIC CHANNEL (UASAM) ───┤───────────────────────┐
 *   │ Microphone → AudioContext     │                       │
 *   │          → AnalyserNode       │                       │
 *   │          → Features (F1–F8)   │                       │
 *   │          → VocState Classifier│                       │
 *   │          → Emission Matrix    │                       │
 *   │                  ↓            │                       │
 *   │              P(A | S)         │                       │
 *   └──────────────────┬────────────┘                       │
 *                      │            │                       │
 *                      ↓            ↓                       │
 *              ┌───────────────────────────┐                │
 *              │    UMCE Bayesian Fusion   │                │
 *              │ P(S|A,V) ∝ P(A|S)·P(V|S)·P(S) │          │
 *              └───────────┬───────────────┘                │
 *                          ↓                                │
 *                   Fused Classification                    │
 *                          ↓                                │
 *                   ConfidenceLock / Sentence                │
 */

import { LEXICON } from '../utils/GrammarEngine';
import { detectSubject, detectVerb, detectObject } from '../utils/gestureDetection';

// =============================================================================
// CONSTANTS
// =============================================================================

/** Visual sub-modality weights — must sum to 1.0.
 *  These control the internal fusion within the visual channel P(V|S). */
const DEFAULT_VISUAL_WEIGHTS = {
  shape:    0.45,
  spatial:  0.25,
  intent:   0.15,
  temporal: 0.15,
};

/** When CNN classifier is active, redistribute visual weights to include it.
 *  CNN acts as a 5th visual sub-modality with high weight since it's
 *  trained on supervised data. */
const CNN_ACTIVE_VISUAL_WEIGHTS = {
  shape:    0.25,
  spatial:  0.20,
  intent:   0.10,
  temporal: 0.10,
  cnn:      0.35,
};

/** Trimodal channel weights for the top-level Bayesian product.
 *  When a modality is inactive, its weight is redistributed.
 *  These scale the log-likelihoods before the product:
 *    log P(S|A,V,G) = w_V·log P(V|S) + w_A·log P(A|S) + w_G·log P(G|S) + log P(S) */
const DEFAULT_CHANNEL_WEIGHTS = {
  visual:   0.55,
  acoustic: 0.25,
  gaze:     0.20,
};

/** Temporal consistency window (frames) */
const TEMPORAL_WINDOW = 12;

/** Minimum probability to report (below this → 0) */
const PROBABILITY_FLOOR = 0.001;

/** Softmax temperature — controls sharpness of the distribution.
 *  Lower T → sharper (more decisive), Higher T → flatter (more uncertain).
 *  T = 1.0 is standard softmax. */
const SOFTMAX_TEMPERATURE = 1.0;

/** Decision quality thresholds */
const QUALITY_THRESHOLDS = {
  HIGH:   { minProbability: 0.60, minMargin: 0.25 },
  MEDIUM: { minProbability: 0.35, minMargin: 0.10 },
  LOW:    { minProbability: 0.15, minMargin: 0.03 },
  // Below LOW → REJECT
};

/** Intent gate: how displacement maps to the intent modulation factor.
 *  Below GATE_LOW, the intent gate → 0 (full suppression).
 *  Above GATE_HIGH, the intent gate → 1 (full pass-through). */
const INTENT_GATE_LOW  = 0.5;   // Below this displacement: suppressed
const INTENT_GATE_HIGH = 2.0;   // Above this displacement: fully active

/** Small epsilon to prevent log(0) */
const LOG_EPSILON = 1e-10;

/** Syntactic prior boost factor for grammatically valid next gestures */
const SYNTACTIC_PRIOR_BOOST = 2.5;

// =============================================================================
// GESTURE CATALOG — All recognizable gesture IDs with geometric constraints
// =============================================================================

/**
 * For each gesture in the LEXICON, define the geometric constraints that
 * the shape classifier evaluates. Each constraint is a function that
 * returns a score in [0, 1] indicating how well the landmark configuration
 * matches the gesture's ideal shape.
 *
 * This replaces the binary pass/fail logic of detectGestureRaw() with
 * continuous soft scores suitable for probabilistic fusion.
 */
const GESTURE_SHAPE_SPECS = {
  // =========================================================================
  // SUBJECTS — Directional pronoun gestures
  // =========================================================================
  SUBJECT_I: {
    category: 'SUBJECT',
    constraints: (lm) => {
      const wrist = lm[0], thumbTip = lm[4], thumbIP = lm[3];
      // Thumb pointing inward
      const thumbInward = sigmoid((thumbTip.x - wrist.x - 0.05) * 20);
      // Thumb extended
      const thumbExtended = sigmoid((thumbIP.y - thumbTip.y + 0.1) * 10);
      // Four fingers curled
      const fingersCurled = fourFingerCurlScore(lm);
      return [
        { id: 'thumb_inward',   score: thumbInward,    weight: 0.35 },
        { id: 'thumb_extended', score: thumbExtended,  weight: 0.25 },
        { id: 'fingers_curled', score: fingersCurled,  weight: 0.40 },
      ];
    },
  },

  SUBJECT_YOU: {
    category: 'SUBJECT',
    constraints: (lm) => {
      const wrist = lm[0], indexTip = lm[8];
      // Index extended
      const indexExt = sigmoid((lm[6].y - indexTip.y) * 15);
      // Index pointing forward (z-depth)
      const indexForward = sigmoid(((wrist.z || 0) - (indexTip.z || 0) - 0.02) * 30);
      // Other fingers curled
      const othersCurled = threeFingerCurlScore(lm);
      // Thumb neutral
      const thumbNeutral = sigmoid((wrist.x + 0.1 - lm[4].x) * 15);
      return [
        { id: 'index_extended',  score: indexExt,      weight: 0.30 },
        { id: 'index_forward',   score: indexForward,  weight: 0.25 },
        { id: 'others_curled',   score: othersCurled,  weight: 0.25 },
        { id: 'thumb_neutral',   score: thumbNeutral,  weight: 0.20 },
      ];
    },
  },

  SUBJECT_HE: {
    category: 'SUBJECT',
    constraints: (lm) => {
      const wrist = lm[0], thumbTip = lm[4], thumbIP = lm[3];
      // Thumb pointing outward
      const thumbOutward = sigmoid((wrist.x - thumbTip.x - 0.08) * 15);
      // Thumb extended
      const thumbExtended = sigmoid((thumbIP.y + 0.05 - thumbTip.y) * 10);
      // Four fingers curled
      const fingersCurled = fourFingerCurlScore(lm);
      return [
        { id: 'thumb_outward',  score: thumbOutward,   weight: 0.35 },
        { id: 'thumb_extended', score: thumbExtended,  weight: 0.25 },
        { id: 'fingers_curled', score: fingersCurled,  weight: 0.40 },
      ];
    },
  },

  // SHE uses same shape as HE (differentiated by context, not shape)
  SUBJECT_SHE: {
    category: 'SUBJECT',
    constraints: (lm) => {
      // Same geometric shape as HE — context-dependent disambiguation
      const wrist = lm[0], thumbTip = lm[4], thumbIP = lm[3];
      const thumbOutward = sigmoid((wrist.x - thumbTip.x - 0.08) * 15);
      const thumbExtended = sigmoid((thumbIP.y + 0.05 - thumbTip.y) * 10);
      const fingersCurled = fourFingerCurlScore(lm);
      return [
        { id: 'thumb_outward',  score: thumbOutward,   weight: 0.35 },
        { id: 'thumb_extended', score: thumbExtended,  weight: 0.25 },
        { id: 'fingers_curled', score: fingersCurled,  weight: 0.40 },
      ];
    },
  },

  SUBJECT_WE: {
    category: 'SUBJECT',
    // WE = index + middle extended, close together, ring + pinky curled
    // Distinguishes from I (thumb inward, all four fingers curled)
    constraints: (lm) => {
      const indexTip = lm[8], middleTip = lm[12];
      // Index and middle extended (using 3D angles)
      const indexExt = 1 - fingerCurlScore3D(lm, 5, 6, 7);
      const middleExt = 1 - fingerCurlScore3D(lm, 9, 10, 11);
      // Ring and pinky curled
      const ringCurl = fingerCurlScore3D(lm, 13, 14, 15);
      const pinkyCurl = fingerCurlScore3D(lm, 17, 18, 19);
      // Fingers close together (distinguishes from SEE where fingers are spread)
      const fingersTogether = sigmoid((0.06 - dist2D(indexTip, middleTip)) * 25);
      return [
        { id: 'index_extended',    score: indexExt,        weight: 0.25 },
        { id: 'middle_extended',   score: middleExt,       weight: 0.25 },
        { id: 'ring_curled',       score: ringCurl,        weight: 0.15 },
        { id: 'pinky_curled',      score: pinkyCurl,       weight: 0.15 },
        { id: 'fingers_together',  score: fingersTogether, weight: 0.20 },
      ];
    },
  },

  SUBJECT_THEY: {
    category: 'SUBJECT',
    // THEY = sweeping index point — same shape as YOU but with lateral motion.
    // Shape-only classifier adds a slight wrist-offset bias to differentiate.
    constraints: (lm) => {
      const wrist = lm[0], indexTip = lm[8];
      const indexExt = 1 - fingerCurlScore3D(lm, 5, 6, 7);
      const indexForward = sigmoid(((wrist.z || 0) - (indexTip.z || 0) - 0.02) * 30);
      const othersCurled = threeFingerCurlScore(lm);
      // Wrist further from center (sweeping outward motion bias)
      const wristOffset = sigmoid((Math.abs(wrist.x - 0.5) - 0.15) * 10);
      return [
        { id: 'index_extended',  score: indexExt,      weight: 0.30 },
        { id: 'index_forward',   score: indexForward,  weight: 0.20 },
        { id: 'others_curled',   score: othersCurled,  weight: 0.25 },
        { id: 'wrist_offset',    score: wristOffset,   weight: 0.25 },
      ];
    },
  },

  // =========================================================================
  // VERBS — Action-based gestures
  // =========================================================================
  GRAB: {
    category: 'VERB',
    constraints: (lm) => {
      // Claw shape: all fingertips close together
      const avgThumbDist = averageThumbToFingerDist(lm);
      const clawScore = sigmoid((0.06 - avgThumbDist) * 40);
      return [
        { id: 'claw_shape', score: clawScore, weight: 1.0 },
      ];
    },
  },

  DRINK: {
    category: 'VERB',
    constraints: (lm) => {
      const thumbTip = lm[4], indexTip = lm[8];
      const thumbIndexDist = dist2D(thumbTip, indexTip);
      const thumbIndexYDiff = Math.abs(thumbTip.y - indexTip.y);
      // C-shape: thumb-index apart at similar height
      const cShape = sigmoid((thumbIndexDist - 0.08) * 20) *
                     sigmoid((0.2 - thumbIndexDist) * 20);
      const sameHeight = sigmoid((0.08 - thumbIndexYDiff) * 20);
      // Others curled
      const othersCurled = threeFingerCurlScore(lm);
      return [
        { id: 'c_shape',      score: cShape,       weight: 0.40 },
        { id: 'same_height',  score: sameHeight,   weight: 0.25 },
        { id: 'others_curled', score: othersCurled, weight: 0.35 },
      ];
    },
  },

  STOP: {
    category: 'VERB',
    constraints: (lm) => {
      const wrist = lm[0], middleTip = lm[12];
      // All fingers extended
      const allExtended = allFingersExtendedScore(lm);
      // Hand vertical
      const handVert = sigmoid((wrist.y - middleTip.y - 0.1) * 10);
      return [
        { id: 'all_extended',  score: allExtended, weight: 0.55 },
        { id: 'hand_vertical', score: handVert,    weight: 0.45 },
      ];
    },
  },

  EAT: {
    category: 'VERB',
    constraints: (lm) => {
      // Pinch fingers together near mouth (similar to grab but more delicate)
      const avgThumbDist = averageThumbToFingerDist(lm);
      const pinchScore = sigmoid((0.08 - avgThumbDist) * 30);
      // Hand higher up (near face area, y < 0.4)
      const nearFace = sigmoid((0.4 - lm[0].y) * 10);
      return [
        { id: 'pinch_shape', score: pinchScore, weight: 0.60 },
        { id: 'near_face',   score: nearFace,   weight: 0.40 },
      ];
    },
  },

  WANT: {
    category: 'VERB',
    constraints: (lm) => {
      // Open hand pulling toward self — fingers slightly curved, hand vertical
      // Matches gestureDetection.js: all fingers in slight-curve range + hand vertical
      const fingersCurled = fourFingerCurlScore(lm);
      const allExtended = allFingersExtendedScore(lm);
      // Slight curve: not fully extended, not fully curled
      const slightCurve = (1 - allExtended) * (1 - fingersCurled);
      // Hand vertical (wrist below fingertips) — matches gestureDetection.js isHandVertical
      const wrist = lm[0], middleTip = lm[12];
      const handVertical = sigmoid((wrist.y - middleTip.y - 0.05) * 10);
      return [
        { id: 'slight_curve',   score: slightCurve,  weight: 0.50 },
        { id: 'hand_vertical',  score: handVertical,  weight: 0.50 },
      ];
    },
  },

  SEE: {
    category: 'VERB',
    constraints: (lm) => {
      // V-shape: index + middle extended and spread, ring + pinky curled
      const indexTip = lm[8], middleTip = lm[12];
      const indexExt = sigmoid((lm[6].y - indexTip.y) * 12);
      const middleExt = sigmoid((lm[10].y - middleTip.y) * 12);
      const ringCurl = sigmoid((lm[16].y - lm[14].y) * 12);
      const pinkyCurl = sigmoid((lm[20].y - lm[18].y) * 12);
      // Fingers spread apart (distinguishes from WE where fingers are together)
      const fingerSpread = sigmoid((dist2D(indexTip, middleTip) - 0.06) * 25);
      return [
        { id: 'index_extended',  score: indexExt,     weight: 0.25 },
        { id: 'middle_extended', score: middleExt,    weight: 0.25 },
        { id: 'ring_curled',     score: ringCurl,     weight: 0.15 },
        { id: 'pinky_curled',    score: pinkyCurl,    weight: 0.15 },
        { id: 'fingers_spread',  score: fingerSpread, weight: 0.20 },
      ];
    },
  },

  GO: {
    category: 'VERB',
    constraints: (lm) => {
      // Index pointing forward strongly (like YOU but with motion context)
      const wrist = lm[0], indexTip = lm[8];
      const indexExt = sigmoid((lm[6].y - indexTip.y) * 15);
      const indexForward = sigmoid(((wrist.z || 0) - (indexTip.z || 0) - 0.02) * 30);
      const othersCurled = threeFingerCurlScore(lm);
      return [
        { id: 'index_extended', score: indexExt,     weight: 0.35 },
        { id: 'index_forward',  score: indexForward, weight: 0.35 },
        { id: 'others_curled',  score: othersCurled, weight: 0.30 },
      ];
    },
  },

  // =========================================================================
  // OBJECTS — Shape-mimicry gestures
  // =========================================================================
  APPLE: {
    category: 'OBJECT',
    constraints: (lm) => {
      // Cupped hand: all fingers slightly curved
      const indexCurved = slightCurveScore(lm, 8, 7, 6);
      const middleCurved = slightCurveScore(lm, 12, 11, 10);
      const ringCurved = slightCurveScore(lm, 16, 15, 14);
      const pinkyCurved = slightCurveScore(lm, 20, 19, 18);
      const thumbCupped = sigmoid((lm[4].y - lm[3].y + 0.05) * 15);
      return [
        { id: 'index_curved',  score: indexCurved,  weight: 0.20 },
        { id: 'middle_curved', score: middleCurved, weight: 0.20 },
        { id: 'ring_curved',   score: ringCurved,   weight: 0.20 },
        { id: 'pinky_curved',  score: pinkyCurved,  weight: 0.20 },
        { id: 'thumb_cupped',  score: thumbCupped,  weight: 0.20 },
      ];
    },
  },

  BOOK: {
    category: 'OBJECT',
    constraints: (lm) => {
      // Flat palm up
      const allExtended = allFingersExtendedScore(lm);
      const wrist = lm[0], middleMCP = lm[9], middleTip = lm[12];
      const palmUp = sigmoid(((middleMCP.z || 0) - (wrist.z || 0) + 0.02) * 20);
      const fingersFlat = sigmoid((0.15 - Math.abs(middleTip.y - middleMCP.y)) * 10);
      const notVertical = sigmoid((0.1 - (wrist.y - middleTip.y)) * 8);
      return [
        { id: 'all_extended',  score: allExtended,  weight: 0.30 },
        { id: 'palm_up',       score: palmUp,       weight: 0.30 },
        { id: 'fingers_flat',  score: fingersFlat,  weight: 0.20 },
        { id: 'not_vertical',  score: notVertical,  weight: 0.20 },
      ];
    },
  },

  HOUSE: {
    category: 'OBJECT',
    constraints: (lm) => {
      const indexTip = lm[8], middleTip = lm[12];
      // Index and middle extended
      const indexExt = sigmoid((lm[6].y - indexTip.y) * 15);
      const middleExt = sigmoid((lm[10].y - middleTip.y) * 15);
      // Ring and pinky curled
      const ringCurled = sigmoid((lm[16].y - lm[14].y) * 12);
      const pinkyCurled = sigmoid((lm[20].y - lm[18].y) * 12);
      // Fingertips touching
      const tipDist = dist2D(indexTip, middleTip);
      const tipsTouching = sigmoid((0.04 - tipDist) * 50);
      return [
        { id: 'index_extended',   score: indexExt,      weight: 0.20 },
        { id: 'middle_extended',  score: middleExt,     weight: 0.20 },
        { id: 'ring_curled',      score: ringCurled,    weight: 0.15 },
        { id: 'pinky_curled',     score: pinkyCurled,   weight: 0.15 },
        { id: 'tips_touching',    score: tipsTouching,  weight: 0.30 },
      ];
    },
  },

  WATER: {
    category: 'OBJECT',
    constraints: (lm) => {
      const indexTip = lm[8], middleTip = lm[12], ringTip = lm[16];
      // Index, middle, ring extended
      const indexExt = sigmoid((lm[6].y - indexTip.y) * 15);
      const middleExt = sigmoid((lm[10].y - middleTip.y) * 15);
      const ringExt = sigmoid((lm[14].y - ringTip.y) * 15);
      // Pinky curled
      const pinkyCurled = sigmoid((lm[20].y - lm[18].y) * 12);
      // Fingers spread
      const indexMiddleSpread = sigmoid((dist2D(indexTip, middleTip) - 0.04) * 25);
      const middleRingSpread = sigmoid((dist2D(middleTip, ringTip) - 0.04) * 25);
      return [
        { id: 'index_extended',       score: indexExt,          weight: 0.15 },
        { id: 'middle_extended',      score: middleExt,         weight: 0.15 },
        { id: 'ring_extended',        score: ringExt,           weight: 0.15 },
        { id: 'pinky_curled',         score: pinkyCurled,       weight: 0.15 },
        { id: 'index_middle_spread',  score: indexMiddleSpread, weight: 0.20 },
        { id: 'middle_ring_spread',   score: middleRingSpread,  weight: 0.20 },
      ];
    },
  },

  FOOD: {
    category: 'OBJECT',
    constraints: (lm) => {
      // Open cupped hand — similar to APPLE but looser
      const indexCurved = slightCurveScore(lm, 8, 7, 6);
      const middleCurved = slightCurveScore(lm, 12, 11, 10);
      const avgThumbDist = averageThumbToFingerDist(lm);
      const openCup = sigmoid((avgThumbDist - 0.06) * 15) * sigmoid((0.15 - avgThumbDist) * 15);
      return [
        { id: 'index_curved',  score: indexCurved,  weight: 0.30 },
        { id: 'middle_curved', score: middleCurved, weight: 0.30 },
        { id: 'open_cup',      score: openCup,      weight: 0.40 },
      ];
    },
  },

  BALL: {
    category: 'OBJECT',
    constraints: (lm) => {
      // Rounded cupped hand — all fingers curved, wider than APPLE
      const indexCurved = slightCurveScore(lm, 8, 7, 6);
      const middleCurved = slightCurveScore(lm, 12, 11, 10);
      const ringCurved = slightCurveScore(lm, 16, 15, 14);
      const pinkyCurved = slightCurveScore(lm, 20, 19, 18);
      // Wider spread than apple
      const spread = dist2D(lm[8], lm[20]);
      const wideSpread = sigmoid((spread - 0.08) * 15);
      return [
        { id: 'index_curved',  score: indexCurved,  weight: 0.20 },
        { id: 'middle_curved', score: middleCurved, weight: 0.20 },
        { id: 'ring_curved',   score: ringCurved,   weight: 0.20 },
        { id: 'pinky_curved',  score: pinkyCurved,  weight: 0.20 },
        { id: 'wide_spread',   score: wideSpread,   weight: 0.20 },
      ];
    },
  },
};

// S-form verbs share the same shape as their base form
const S_FORM_PAIRS = {
  GRABS: 'GRAB', GOES: 'GO', EATS: 'EAT', SEES: 'SEE',
  WANTS: 'WANT', STOPS: 'STOP', DRINKS: 'DRINK',
};

// =============================================================================
// GEOMETRIC HELPER FUNCTIONS (Continuous soft scores)
// =============================================================================

/** Sigmoid activation — converts distance to [0, 1] score */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/** 2D Euclidean distance */
function dist2D(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/** Average thumb-to-finger distance */
function averageThumbToFingerDist(lm) {
  const t = lm[4]; // thumb tip
  return (dist2D(t, lm[8]) + dist2D(t, lm[12]) + dist2D(t, lm[16]) + dist2D(t, lm[20])) / 4;
}

/** 3D angle at joint b formed by points a→b→c (degrees). */
function jointAngle3D(a, b, c) {
  const v1 = { x: a.x - b.x, y: a.y - b.y, z: (a.z || 0) - (b.z || 0) };
  const v2 = { x: c.x - b.x, y: c.y - b.y, z: (c.z || 0) - (b.z || 0) };
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.sqrt(v1.x ** 2 + v1.y ** 2 + v1.z ** 2);
  const m2 = Math.sqrt(v2.x ** 2 + v2.y ** 2 + v2.z ** 2);
  if (m1 < 1e-8 || m2 < 1e-8) return 180;
  return Math.acos(Math.max(-1, Math.min(1, dot / (m1 * m2)))) * (180 / Math.PI);
}

/** Single finger curl score using 3D PIP joint angle.
 *  Low angle = curled (score→1), high angle = extended (score→0). */
function fingerCurlScore3D(lm, mcpIdx, pipIdx, dipIdx) {
  const angle = jointAngle3D(lm[mcpIdx], lm[pipIdx], lm[dipIdx]);
  // Below 130° → curled, sigmoid centered at 130
  return sigmoid((130 - angle) * 0.15);
}

/** Four-finger curl score using 3D joint angles (rotation-invariant).
 *  Returns average [0, 1] where 1 = all curled. */
function fourFingerCurlScore(lm) {
  return (
    fingerCurlScore3D(lm, 5, 6, 7) +   // index: MCP→PIP→DIP
    fingerCurlScore3D(lm, 9, 10, 11) +  // middle
    fingerCurlScore3D(lm, 13, 14, 15) + // ring
    fingerCurlScore3D(lm, 17, 18, 19)   // pinky
  ) / 4;
}

/** Three-finger curl score: middle, ring, pinky (not index). */
function threeFingerCurlScore(lm) {
  return (
    fingerCurlScore3D(lm, 9, 10, 11) +
    fingerCurlScore3D(lm, 13, 14, 15) +
    fingerCurlScore3D(lm, 17, 18, 19)
  ) / 3;
}

/** All-fingers-extended score using 3D joint angles. */
function allFingersExtendedScore(lm) {
  // Thumb uses IP joint (2→3→4), fingers use PIP joint
  const thumbAngle = jointAngle3D(lm[2], lm[3], lm[4]);
  const thumbExt = sigmoid((thumbAngle - 155) * 0.15);
  const indexExt = 1 - fingerCurlScore3D(lm, 5, 6, 7);
  const middleExt = 1 - fingerCurlScore3D(lm, 9, 10, 11);
  const ringExt = 1 - fingerCurlScore3D(lm, 13, 14, 15);
  const pinkyExt = 1 - fingerCurlScore3D(lm, 17, 18, 19);
  return (thumbExt + indexExt + middleExt + ringExt + pinkyExt) / 5;
}

/** Slight-curve score for a single finger (cupped hand detection).
 *  Uses 3D angle: between curled and extended range (~100-155°). */
function slightCurveScore(lm, tipIdx, dipIdx, pipIdx) {
  // Find the MCP for this finger chain (pipIdx - 1 in the finger chain)
  const mcpIdx = pipIdx - 1;
  const angle = jointAngle3D(lm[mcpIdx], lm[pipIdx], lm[dipIdx]);
  // Peak around 130°: score high when angle is between 100-155
  const notTooExtended = sigmoid((155 - angle) * 0.15);
  const notTooCurled = sigmoid((angle - 100) * 0.15);
  return notTooExtended * notTooCurled;
}

// =============================================================================
// SPATIAL ZONE → CATEGORY MAPPING
// =============================================================================

const ZONE_CATEGORY_MAP = {
  SUBJECT_ZONE: 'SUBJECT',
  VERB_ZONE:    'VERB',
  OBJECT_ZONE:  'OBJECT',
};

// =============================================================================
// UNIFIED MULTIMODAL CLASSIFICATION ENGINE
// =============================================================================

export class UMCE {
  /**
   * @param {object} config
   * @param {object} [config.visualWeights]    — visual sub-modality weights {shape, spatial, intent, temporal}
   * @param {object} [config.channelWeights]   — bimodal channel weights {visual, acoustic}
   * @param {number} [config.temperature]      — softmax temperature
   * @param {number} [config.temporalWindow]   — frame history window size
   */
  constructor(config = {}) {
    // Visual sub-modality weights (for internal P(V|S) fusion)
    this._visualWeights = { ...DEFAULT_VISUAL_WEIGHTS, ...(config.visualWeights || {}) };
    this._normalizeWeightObj(this._visualWeights);

    // Bimodal channel weights (for top-level Bayesian product)
    this._channelWeights = { ...DEFAULT_CHANNEL_WEIGHTS, ...(config.channelWeights || {}) };

    // Softmax temperature
    this._temperature = config.temperature || SOFTMAX_TEMPERATURE;

    // Temporal consistency buffer (circular buffer of per-frame top-1 IDs)
    this._temporalWindow = config.temporalWindow || TEMPORAL_WINDOW;
    this._temporalBuffer = [];

    // All gesture IDs we classify over (base forms only).
    // S-form verbs (GRABS, EATS, etc.) share identical hand shapes with their
    // base forms — agreement is resolved downstream by useSentenceBuilder via
    // getCorrectVerbForm(). Use getSFormId() if S-form resolution is needed.
    this._gestureIds = Object.keys(GESTURE_SHAPE_SPECS);

    // Sentence context for syntactic prior
    this._sentenceContext = [];

    // Mastery data for prior computation
    this._masteryData = null;

    // Frame counter
    this._frameCount = 0;

    // Fusion mode tracking
    this._fusionMode = 'VISUAL_ONLY';

    // Last fusion result (cached for debug display)
    this._lastResult = null;

    // Cached per-frame intermediate arrays (for debug/logging)
    this._lastVisualLikelihoods = {};
    this._lastAcousticLikelihoods = {};
    this._lastGazeLikelihoods = {};
    this._lastPriors = {};
  }

  // ===========================================================================
  // PUBLIC — Core fusion
  // ===========================================================================

  /**
   * Bayesian trimodal late fusion.
   *
   * Computes: P(S | A, V, G) ∝ P(A|S)^w_A · P(V|S)^w_V · P(G|S)^w_G · P(S)
   *
   * The visual channel P(V|S) is itself a weighted log-linear fusion of
   * 4 sub-modalities (shape, spatial, intent, temporal).
   * The acoustic channel P(A|S) comes from UASAM.
   * The gaze channel P(G|S) comes from EyeGazeTracker.
   * P(S) is the prior (syntactic + mastery + temporal context).
   *
   * When a modality is inactive, its P(·|S) = uniform → no effect on fusion.
   *
   * @param {object} inputs
   * @param {Array}  inputs.landmarks           — 21 smoothed hand landmarks
   * @param {object} inputs.spatialResult       — from SpatialGrammarMapper.map()
   * @param {object} inputs.intentResult        — from IntentionalityDetector.detect()
   * @param {object} inputs.cognitiveLoad       — { level, jitter }
   * @param {object} [inputs.acousticLikelihoods] — P(A|S) map from UASAM.getAcousticLikelihoods()
   * @param {boolean} [inputs.acousticActive]   — whether UASAM microphone is active
   * @param {string} [inputs.vocalizationState] — current vocalization state from UASAM
   * @param {object} [inputs.gazeLikelihoods]   — P(G|S) map from EyeGazeTracker.getGazeLikelihoods()
   * @param {boolean} [inputs.gazeActive]       — whether EyeGazeTracker is active
   * @param {string} [inputs.gazeState]         — current gaze state from EyeGazeTracker
   * @param {object} [inputs.cnnResult]          — from GestureClassifierCNN.classify() { id, confidence, probabilities }
   * @param {string} inputs.rawGesture          — from detectGestureRaw() (backward compat)
   * @returns {UMCEResult}
   */
  fuse(inputs) {
    const {
      landmarks, spatialResult, intentResult, cognitiveLoad,
      acousticLikelihoods, acousticActive, vocalizationState,
      gazeLikelihoods, gazeActive, gazeState,
      cnnResult,
      rawGesture,
    } = inputs;

    if (!landmarks || landmarks.length < 21) {
      return this._noResult();
    }

    this._frameCount++;

    // =====================================================================
    // VISUAL CHANNEL: P(V | S) — fusion of 4 visual sub-modalities
    // =====================================================================

    // V1: Shape Classifier — soft likelihood per gesture
    const shapeLikelihoods = this._computeShapeLikelihoods(landmarks);

    // V2: Spatial Zone Congruence — category-based likelihood
    const spatialLikelihoods = this._computeSpatialLikelihoods(spatialResult);

    // V3: Intentionality Gate — scalar modulation
    const intentGate = this._computeIntentGate(intentResult);

    // V4: Temporal Consistency — frame history stability
    const temporalLikelihoods = this._computeTemporalLikelihoods();

    // V5: CNN Classifier — supervised neural network likelihoods (optional)
    // CNN probabilities are indexed by class (matching GESTURE_IDS order in config.py):
    //   0=SUBJECT_I, 1=SUBJECT_YOU, ..., 6=WANT, 7=EAT, ..., 18=HOUSE
    // We map these to UMCE gesture IDs via the CNN_IDX_TO_FRONTEND_ID lookup.
    const isCnnActive = cnnResult && cnnResult.probabilities && cnnResult.probabilities.length > 0;
    const cnnLikelihoods = {};
    if (isCnnActive) {
      const CNN_IDX_MAP = [
        'SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_HE', 'SUBJECT_SHE',
        'SUBJECT_WE', 'SUBJECT_THEY',
        'WANT', 'EAT', 'SEE', 'GRAB', 'DRINK', 'GO', 'STOP',
        'FOOD', 'WATER', 'BOOK', 'APPLE', 'BALL', 'HOUSE',
      ];
      const uniform = 1 / this._gestureIds.length;
      // Initialize all gestures to uniform
      for (const gId of this._gestureIds) {
        cnnLikelihoods[gId] = uniform;
      }
      // Map CNN probabilities by gesture ID, not by UMCE index
      for (let i = 0; i < Math.min(cnnResult.probabilities.length, CNN_IDX_MAP.length); i++) {
        const gId = CNN_IDX_MAP[i];
        if (cnnLikelihoods.hasOwnProperty(gId)) {
          cnnLikelihoods[gId] = Math.max(cnnResult.probabilities[i], LOG_EPSILON);
        }
      }
    }

    // Consolidate visual sub-modalities into P(V | S) via weighted log-linear:
    //   log P(V|S_i) = w_shape · log L_shape(S_i) + w_spatial · log L_spatial(S_i)
    //                + w_temporal · log L_temporal(S_i) + w_intent · log(intent_gate)
    //                + [w_cnn · log L_cnn(S_i)]  (when CNN is active)
    const visualLogLikelihoods = {};
    const vw = isCnnActive ? CNN_ACTIVE_VISUAL_WEIGHTS : this._visualWeights;

    for (const gId of this._gestureIds) {
      const logShape    = Math.log(Math.max(shapeLikelihoods[gId]    || 0, LOG_EPSILON));
      const logSpatial  = Math.log(Math.max(spatialLikelihoods[gId]  || 0, LOG_EPSILON));
      const logTemporal = Math.log(Math.max(temporalLikelihoods[gId] || 0, LOG_EPSILON));
      const logIntent   = Math.log(Math.max(intentGate, LOG_EPSILON));

      let logSum = (
        vw.shape    * logShape +
        vw.spatial  * logSpatial +
        vw.temporal * logTemporal +
        vw.intent   * logIntent
      );

      if (isCnnActive && vw.cnn) {
        const logCnn = Math.log(Math.max(cnnLikelihoods[gId] || 0, LOG_EPSILON));
        logSum += vw.cnn * logCnn;
      }

      visualLogLikelihoods[gId] = logSum;
    }

    // Normalize visual log-likelihoods into a proper probability via softmax
    const visualProbabilities = this._softmax(visualLogLikelihoods);
    this._lastVisualLikelihoods = visualProbabilities;

    // =====================================================================
    // ACOUSTIC CHANNEL: P(A | S) — from UASAM
    // =====================================================================
    const isAcousticActive = acousticActive && acousticLikelihoods &&
      Object.keys(acousticLikelihoods).length > 0;

    const isGazeActive = gazeActive && gazeLikelihoods &&
      Object.keys(gazeLikelihoods).length > 0;

    // Determine fusion mode based on active modalities
    if (isAcousticActive && isGazeActive) {
      this._fusionMode = 'TRIMODAL';
    } else if (isAcousticActive) {
      this._fusionMode = 'BIMODAL_VA';
    } else if (isGazeActive) {
      this._fusionMode = 'BIMODAL_VG';
    } else {
      this._fusionMode = 'VISUAL_ONLY';
    }

    // If UASAM is inactive, P(A|S) = uniform → log P(A|S) = const → no effect
    const acousticProbs = {};
    const uniform = 1 / this._gestureIds.length;
    for (const gId of this._gestureIds) {
      acousticProbs[gId] = isAcousticActive
        ? Math.max(acousticLikelihoods[gId] || 0, LOG_EPSILON)
        : uniform;
    }
    this._lastAcousticLikelihoods = acousticProbs;

    // If EyeGazeTracker is inactive, P(G|S) = uniform → no effect
    const gazeProbs = {};
    for (const gId of this._gestureIds) {
      gazeProbs[gId] = isGazeActive
        ? Math.max(gazeLikelihoods[gId] || 0, LOG_EPSILON)
        : uniform;
    }
    this._lastGazeLikelihoods = gazeProbs;

    // =====================================================================
    // PRIOR: P(S) — syntactic context + mastery + temporal regularity
    // =====================================================================
    const priors = this._computePriors();
    this._lastPriors = priors;

    // =====================================================================
    // BAYESIAN LATE FUSION:
    //   log P(S_i | A, V, G) = w_V·log P(V|S_i) + w_A·log P(A|S_i)
    //                        + w_G·log P(G|S_i) + log P(S_i)
    //
    // Then softmax normalization to get proper posterior.
    //
    // This implements: P(S|A,V,G) ∝ P(A|S)^w_A · P(V|S)^w_V · P(G|S)^w_G · P(S)
    // which is the weighted product-of-experts form of Bayes' rule.
    // =====================================================================
    const bayesianLogScores = {};
    const cw = this._channelWeights;

    for (const gId of this._gestureIds) {
      const logVisual   = Math.log(Math.max(visualProbabilities[gId] || 0, LOG_EPSILON));
      const logAcoustic = Math.log(Math.max(acousticProbs[gId]      || 0, LOG_EPSILON));
      const logGaze     = Math.log(Math.max(gazeProbs[gId]          || 0, LOG_EPSILON));
      const logPrior    = Math.log(Math.max(priors[gId]             || 0, LOG_EPSILON));

      bayesianLogScores[gId] = (
        cw.visual   * logVisual +
        cw.acoustic * logAcoustic +
        cw.gaze     * logGaze +
        logPrior
      );
    }

    // =====================================================================
    // SOFTMAX NORMALIZATION → Posterior: P(S | A, V, G)
    // =====================================================================
    const posterior = this._softmax(bayesianLogScores);

    // =====================================================================
    // EXTRACT TOP-K and DECISION METRICS
    // =====================================================================
    const sorted = Object.entries(posterior)
      .sort((a, b) => b[1] - a[1]);

    const top1 = sorted[0] ? { gesture_id: sorted[0][0], probability: sorted[0][1] } : null;
    const top2 = sorted[1] ? { gesture_id: sorted[1][0], probability: sorted[1][1] } : null;
    const top3 = sorted.slice(0, 3).map(([gId, prob]) => ({
      gesture_id: gId,
      probability: prob,
      category: GESTURE_SHAPE_SPECS[gId]?.category || LEXICON[gId]?.type || 'UNKNOWN',
    }));

    const margin = top1 && top2 ? top1.probability - top2.probability : 0;
    const entropy = this._shannonEntropy(posterior);

    // Update temporal buffer with this frame's top-1
    if (top1 && top1.probability > PROBABILITY_FLOOR) {
      this._temporalBuffer.push(top1.gesture_id);
      if (this._temporalBuffer.length > this._temporalWindow) {
        this._temporalBuffer.shift();
      }
    }

    // Decision quality assessment
    const decisionQuality = this._assessDecisionQuality(top1?.probability || 0, margin);

    // =====================================================================
    // MODALITY CONTRIBUTION BREAKDOWN (for debug + research logging)
    // =====================================================================
    const modalityContributions = {
      visual: {
        top1: this._argmax(visualProbabilities),
        weight: cw.visual,
        P_V: { ...visualProbabilities },
        sub_modalities: {
          shape: {
            top1: this._argmax(shapeLikelihoods),
            weight: vw.shape,
          },
          spatial: {
            top1: this._argmax(spatialLikelihoods),
            weight: vw.spatial,
            congruent_category: spatialResult?.syntactic_zone
              ? ZONE_CATEGORY_MAP[spatialResult.syntactic_zone] : null,
            zone_confidence: spatialResult?.zone_confidence || 0,
          },
          intent: {
            displacement: intentResult?.displacement || 0,
            gate_value: intentGate,
            weight: vw.intent,
          },
          temporal: {
            top1: this._argmax(temporalLikelihoods),
            weight: vw.temporal,
            window_size: this._temporalBuffer.length,
            consistency: this._temporalConsistency(),
          },
          cnn: isCnnActive ? {
            top1: cnnResult.id,
            confidence: cnnResult.confidence,
            weight: vw.cnn || 0,
            is_active: true,
          } : { is_active: false },
        },
      },
      acoustic: {
        top1: isAcousticActive ? this._argmax(acousticProbs) : null,
        weight: cw.acoustic,
        P_A: isAcousticActive ? { ...acousticProbs } : null,
        is_active: isAcousticActive,
        vocalization_state: vocalizationState || null,
      },
      gaze: {
        top1: isGazeActive ? this._argmax(gazeProbs) : null,
        weight: cw.gaze,
        P_G: isGazeActive ? { ...gazeProbs } : null,
        is_active: isGazeActive,
        gaze_state: gazeState || null,
      },
      prior: {
        top1: this._argmax(priors),
        expected_category: this._getExpectedCategory(),
        P_S: { ...priors },
      },
    };

    this._lastResult = {
      top1: top1 ? {
        gesture_id: top1.gesture_id,
        probability: round4(top1.probability),
        category: GESTURE_SHAPE_SPECS[top1.gesture_id]?.category ||
                  LEXICON[top1.gesture_id]?.type || 'UNKNOWN',
      } : null,
      top3,
      posterior,
      entropy: round4(entropy),
      margin: round4(margin),
      modality_contributions: modalityContributions,
      fusion_mode: this._fusionMode,
      decision_quality: decisionQuality,
      intent_gate: round4(intentGate),
      frame: this._frameCount,
      raw_gesture_agrees: rawGesture ? (top1?.gesture_id === rawGesture) : null,
    };

    return this._lastResult;
  }

  // ===========================================================================
  // PUBLIC — Context updates
  // ===========================================================================

  /**
   * Update the sentence context for syntactic prior computation.
   * Called whenever the sentence changes in useSentenceBuilder.
   *
   * @param {Array} sentence — current sentence array [{type, grammar_id, ...}, ...]
   */
  setSentenceContext(sentence) {
    this._sentenceContext = sentence || [];
  }

  /**
   * Get the effective gesture ID to pass to the ConfidenceLock / processGestureInput.
   * Returns the top-1 gesture if decision quality is not REJECT, otherwise null.
   *
   * @param {UMCEResult} result — from fuse()
   * @returns {string|null} gesture_id or null
   */
  getClassification(result) {
    if (!result || !result.top1 || result.decision_quality === 'REJECT') {
      return null;
    }
    return result.top1.gesture_id;
  }

  /**
   * Resolve a base verb ID to its S-form if needed for 3rd person singular.
   * E.g., getSFormId('GRAB') → 'GRABS', getSFormId('GO') → 'GOES'.
   * Returns the original ID if no S-form exists (non-verbs, already S-form).
   * @param {string} gestureId — base form gesture ID
   * @returns {string} S-form gesture ID or original
   */
  getSFormId(gestureId) {
    const sFormEntry = Object.entries(S_FORM_PAIRS).find(([, base]) => base === gestureId);
    return sFormEntry ? sFormEntry[0] : gestureId;
  }

  /**
   * Set visual sub-modality weights dynamically.
   * @param {object} weights — partial or full weight object {shape, spatial, intent, temporal}
   */
  setVisualWeights(weights) {
    Object.assign(this._visualWeights, weights);
    this._normalizeWeightObj(this._visualWeights);
  }

  /**
   * Set bimodal channel weights dynamically.
   * @param {object} weights — {visual, acoustic}
   */
  setChannelWeights(weights) {
    Object.assign(this._channelWeights, weights);
  }

  /**
   * Set mastery data for prior computation.
   * @param {object} masteryReport — from GestureMasteryGate.getMasteryReport()
   */
  setMasteryData(masteryReport) {
    this._masteryData = masteryReport;
  }

  /**
   * Set softmax temperature.
   * @param {number} temperature
   */
  setTemperature(temperature) {
    this._temperature = Math.max(0.1, Math.min(5.0, temperature));
  }

  /**
   * Get the last fusion result.
   * @returns {UMCEResult|null}
   */
  getLastResult() {
    return this._lastResult;
  }

  /**
   * Reset all state.
   */
  reset() {
    this._temporalBuffer = [];
    this._sentenceContext = [];
    this._masteryData = null;
    this._frameCount = 0;
    this._fusionMode = 'VISUAL_ONLY';
    this._lastResult = null;
    this._lastVisualLikelihoods = {};
    this._lastAcousticLikelihoods = {};
    this._lastPriors = {};
  }

  /**
   * Get serializable descriptor for logging/export.
   * @returns {object}
   */
  toDescriptor() {
    return {
      visual_weights: { ...this._visualWeights },
      channel_weights: { ...this._channelWeights },
      temperature: this._temperature,
      temporal_window: this._temporalWindow,
      gesture_count: this._gestureIds.length,
      frame_count: this._frameCount,
      fusion_mode: this._fusionMode,
      last_top1: this._lastResult?.top1?.gesture_id || null,
      last_entropy: this._lastResult?.entropy || null,
      last_quality: this._lastResult?.decision_quality || null,
    };
  }

  /**
   * Get the current fusion mode.
   * @returns {'BIMODAL' | 'VISUAL_ONLY'}
   */
  getFusionMode() {
    return this._fusionMode;
  }

  /**
   * Get the visual probability array P(V|S) from the last frame.
   * @returns {Object} Map<gesture_id, probability>
   */
  getVisualLikelihoods() {
    return { ...this._lastVisualLikelihoods };
  }

  /**
   * Get the acoustic probability array P(A|S) from the last frame.
   * @returns {Object} Map<gesture_id, probability>
   */
  getAcousticLikelihoods() {
    return { ...this._lastAcousticLikelihoods };
  }

  /**
   * Get the prior array P(S) from the last frame.
   * @returns {Object} Map<gesture_id, probability>
   */
  getPriors() {
    return { ...this._lastPriors };
  }

  // ===========================================================================
  // PRIVATE — Modality likelihood computations
  // ===========================================================================

  /**
   * M1: Shape Classifier — compute soft likelihood for each gesture.
   *
   * For each gesture g, evaluate its geometric constraint set against the
   * current landmarks. The likelihood is the weighted sum of individual
   * constraint satisfaction scores.
   *
   * @param {Array} landmarks — 21 smoothed landmarks
   * @returns {Object} Map<gesture_id, likelihood ∈ [0, 1]>
   */
  _computeShapeLikelihoods(landmarks) {
    const likelihoods = {};

    for (const gId of this._gestureIds) {
      const spec = GESTURE_SHAPE_SPECS[gId];
      if (!spec) {
        likelihoods[gId] = PROBABILITY_FLOOR;
        continue;
      }

      try {
        const constraints = spec.constraints(landmarks);
        let totalScore = 0;
        let totalWeight = 0;

        for (const c of constraints) {
          totalScore += c.score * c.weight;
          totalWeight += c.weight;
        }

        likelihoods[gId] = totalWeight > 0
          ? Math.max(PROBABILITY_FLOOR, totalScore / totalWeight)
          : PROBABILITY_FLOOR;
      } catch {
        likelihoods[gId] = PROBABILITY_FLOOR;
      }
    }

    return likelihoods;
  }

  /**
   * M2: Spatial Zone Congruence — likelihood based on wrist position
   * matching the expected zone for each gesture's category.
   *
   * If wrist is in SUBJECT_ZONE, all SUBJECT gestures get boosted likelihood.
   * The boost is proportional to zone_confidence (how centered in the zone).
   *
   * @param {object} spatialResult — from SpatialGrammarMapper
   * @returns {Object} Map<gesture_id, likelihood ∈ [0, 1]>
   */
  _computeSpatialLikelihoods(spatialResult) {
    const likelihoods = {};
    const uniform = 1 / this._gestureIds.length;

    if (!spatialResult || !spatialResult.syntactic_zone) {
      // No spatial data → uniform (non-informative)
      for (const gId of this._gestureIds) {
        likelihoods[gId] = uniform;
      }
      return likelihoods;
    }

    const currentZone = spatialResult.syntactic_zone;
    const congruentCategory = ZONE_CATEGORY_MAP[currentZone];
    const confidence = spatialResult.zone_confidence || 0.5;

    for (const gId of this._gestureIds) {
      const category = GESTURE_SHAPE_SPECS[gId]?.category || LEXICON[gId]?.type;

      if (category === congruentCategory) {
        // Congruent: boost proportional to zone confidence
        // At confidence=1.0 (zone center), boost = 3x uniform
        // At confidence=0.0 (zone edge), boost = 1.5x uniform
        likelihoods[gId] = uniform * (1.5 + 1.5 * confidence);
      } else {
        // Incongruent: suppressed proportional to zone confidence
        // At confidence=1.0, suppression = 0.3x uniform
        // At confidence=0.0, suppression = 0.7x uniform
        likelihoods[gId] = uniform * (0.7 - 0.4 * confidence);
      }

      likelihoods[gId] = Math.max(PROBABILITY_FLOOR, likelihoods[gId]);
    }

    return likelihoods;
  }

  /**
   * M3: Intentionality Gate — scalar that modulates all likelihoods.
   *
   * Maps displacement from IntentionalityDetector to [0, 1]:
   *   - displacement < GATE_LOW  → 0 (full suppression, resting)
   *   - displacement > GATE_HIGH → 1 (full activation, intentional)
   *   - Between → linear interpolation
   *
   * @param {object} intentResult — from IntentionalityDetector
   * @returns {number} gate value ∈ [0, 1]
   */
  _computeIntentGate(intentResult) {
    if (!intentResult || intentResult.intent === 'RESTING') {
      return 0.05; // Near-zero but not zero (avoids log(0))
    }

    const d = intentResult.displacement || 0;

    if (d < INTENT_GATE_LOW) {
      return 0.05 + 0.45 * (d / INTENT_GATE_LOW); // 0.05 → 0.5
    }
    if (d > INTENT_GATE_HIGH) {
      return 1.0;
    }

    // Linear interpolation between LOW and HIGH
    return 0.5 + 0.5 * ((d - INTENT_GATE_LOW) / (INTENT_GATE_HIGH - INTENT_GATE_LOW));
  }

  /**
   * M4: Temporal Consistency — how often each gesture appeared in recent frames.
   *
   * Counts occurrences of each gesture in the temporal buffer and normalizes
   * to a likelihood. Gestures that appeared consistently get higher scores.
   *
   * @returns {Object} Map<gesture_id, likelihood ∈ [0, 1]>
   */
  _computeTemporalLikelihoods() {
    const likelihoods = {};
    const uniform = 1 / this._gestureIds.length;

    if (this._temporalBuffer.length === 0) {
      for (const gId of this._gestureIds) {
        likelihoods[gId] = uniform;
      }
      return likelihoods;
    }

    // Count occurrences with exponential recency weighting
    const counts = {};
    const n = this._temporalBuffer.length;
    let totalWeight = 0;

    for (let i = 0; i < n; i++) {
      const gId = this._temporalBuffer[i];
      // Exponential recency: more recent frames get higher weight
      const recencyWeight = Math.exp((i - n + 1) * 0.3); // Most recent = 1.0
      counts[gId] = (counts[gId] || 0) + recencyWeight;
      totalWeight += recencyWeight;
    }

    // Convert to likelihoods with Laplace smoothing
    const smoothingAlpha = 0.1; // Smoothing constant
    for (const gId of this._gestureIds) {
      const count = counts[gId] || 0;
      likelihoods[gId] = (count + smoothingAlpha) / (totalWeight + smoothingAlpha * this._gestureIds.length);
    }

    return likelihoods;
  }

  /**
   * Compute prior distribution P(S) incorporating:
   *   1. Syntactic context — SVO ordering expectation
   *   2. Mastery history — mastered gestures slightly more likely
   *   3. Uniform base — prevents any gesture from having zero prior
   *
   * If sentence already has a SUBJECT, verbs get boosted prior (more likely next).
   * If sentence has SUBJECT + VERB, objects get boosted.
   * Mastered gestures receive a 1.5x boost (the child can produce them reliably).
   *
   * @returns {Object} Map<gesture_id, prior ∈ (0, 1)>
   */
  _computePriors() {
    const priors = {};
    const uniform = 1 / this._gestureIds.length;

    // 1. Syntactic context prior
    const expectedCategory = this._getExpectedCategory();

    for (const gId of this._gestureIds) {
      const category = GESTURE_SHAPE_SPECS[gId]?.category || LEXICON[gId]?.type;
      priors[gId] = (category === expectedCategory)
        ? uniform * SYNTACTIC_PRIOR_BOOST
        : uniform;
    }

    // 2. Mastery prior boost — mastered gestures are more likely intended
    if (this._masteryData && this._masteryData.gestures) {
      const MASTERY_BOOST = 1.5;
      for (const gId of this._gestureIds) {
        const gestureData = this._masteryData.gestures?.[gId];
        if (gestureData && gestureData.mastered) {
          priors[gId] *= MASTERY_BOOST;
        }
      }
    }

    // Normalize priors to sum to 1
    const total = Object.values(priors).reduce((s, v) => s + v, 0);
    for (const gId of this._gestureIds) {
      priors[gId] /= total;
    }

    return priors;
  }

  /**
   * Determine the syntactically expected category given current sentence.
   * @returns {string} 'SUBJECT' | 'VERB' | 'OBJECT'
   */
  _getExpectedCategory() {
    const hasSubject = this._sentenceContext.some(w => w.type === 'SUBJECT');
    const hasVerb = this._sentenceContext.some(w => w.type === 'VERB');

    if (!hasSubject) return 'SUBJECT';
    if (!hasVerb) return 'VERB';
    return 'OBJECT';
  }

  // ===========================================================================
  // PRIVATE — Softmax and entropy
  // ===========================================================================

  /**
   * Softmax normalization over log-scores.
   *
   * P(g) = exp(score_g / T) / Σ_g' exp(score_g' / T)
   *
   * Uses the log-sum-exp trick for numerical stability:
   *   log Σ exp(x_i) = max(x) + log Σ exp(x_i - max(x))
   *
   * @param {Object} logScores — Map<gesture_id, log-score>
   * @returns {Object} Map<gesture_id, probability>
   */
  _softmax(logScores) {
    const T = this._temperature;
    const scaled = {};
    let maxScore = -Infinity;

    // Scale by temperature and find max for numerical stability
    for (const gId of this._gestureIds) {
      scaled[gId] = (logScores[gId] || -100) / T;
      if (scaled[gId] > maxScore) maxScore = scaled[gId];
    }

    // Compute exp(score - max) and sum
    let sumExp = 0;
    const expValues = {};
    for (const gId of this._gestureIds) {
      expValues[gId] = Math.exp(scaled[gId] - maxScore);
      sumExp += expValues[gId];
    }

    // Normalize
    const posterior = {};
    for (const gId of this._gestureIds) {
      posterior[gId] = sumExp > 0
        ? Math.max(PROBABILITY_FLOOR, expValues[gId] / sumExp)
        : 1 / this._gestureIds.length;
    }

    return posterior;
  }

  /**
   * Shannon entropy of the posterior distribution (in bits).
   * H = -Σ P(g) · log₂(P(g))
   *
   * Lower entropy = more certain classification.
   * For |G| = 18 gestures, max entropy = log₂(18) ≈ 4.17 bits (uniform).
   *
   * @param {Object} posterior — Map<gesture_id, probability>
   * @returns {number} entropy in bits
   */
  _shannonEntropy(posterior) {
    let H = 0;
    for (const gId of this._gestureIds) {
      const p = posterior[gId] || 0;
      if (p > PROBABILITY_FLOOR) {
        H -= p * Math.log2(p);
      }
    }
    return H;
  }

  // ===========================================================================
  // PRIVATE — Decision quality
  // ===========================================================================

  /**
   * Assess decision quality based on top-1 probability and margin.
   *
   * @param {number} topProb — top-1 probability
   * @param {number} margin  — top1 - top2 probability gap
   * @returns {'HIGH' | 'MEDIUM' | 'LOW' | 'REJECT'}
   */
  _assessDecisionQuality(topProb, margin) {
    if (topProb >= QUALITY_THRESHOLDS.HIGH.minProbability &&
        margin >= QUALITY_THRESHOLDS.HIGH.minMargin) {
      return 'HIGH';
    }
    if (topProb >= QUALITY_THRESHOLDS.MEDIUM.minProbability &&
        margin >= QUALITY_THRESHOLDS.MEDIUM.minMargin) {
      return 'MEDIUM';
    }
    if (topProb >= QUALITY_THRESHOLDS.LOW.minProbability &&
        margin >= QUALITY_THRESHOLDS.LOW.minMargin) {
      return 'LOW';
    }
    return 'REJECT';
  }

  // ===========================================================================
  // PRIVATE — Utility
  // ===========================================================================

  /** Find the gesture ID with the highest value in a map. */
  _argmax(map) {
    let maxId = null;
    let maxVal = -Infinity;
    for (const [gId, val] of Object.entries(map)) {
      if (val > maxVal) {
        maxVal = val;
        maxId = gId;
      }
    }
    return maxId;
  }

  /** Compute temporal consistency: fraction of buffer matching top-1. */
  _temporalConsistency() {
    if (this._temporalBuffer.length === 0) return 0;
    const counts = {};
    for (const gId of this._temporalBuffer) {
      counts[gId] = (counts[gId] || 0) + 1;
    }
    const maxCount = Math.max(...Object.values(counts));
    return maxCount / this._temporalBuffer.length;
  }

  /** Normalize a weight object to sum to 1.0. */
  _normalizeWeightObj(obj) {
    const total = Object.values(obj).reduce((s, v) => s + v, 0);
    if (total > 0 && Math.abs(total - 1.0) > 0.001) {
      for (const key of Object.keys(obj)) {
        obj[key] /= total;
      }
    }
  }

  /** Return empty result when landmarks are missing. */
  _noResult() {
    return {
      top1: null,
      top3: [],
      posterior: {},
      entropy: 0,
      margin: 0,
      modality_contributions: null,
      fusion_mode: this._fusionMode,
      decision_quality: 'REJECT',
      intent_gate: 0,
      frame: this._frameCount,
      raw_gesture_agrees: null,
    };
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/** Round to 4 decimal places. */
function round4(n) {
  return Math.round(n * 10000) / 10000;
}

/**
 * Get CSS color for decision quality.
 * @param {string} quality
 * @returns {string}
 */
export function decisionQualityColor(quality) {
  switch (quality) {
    case 'HIGH':   return '#4ade80';  // green
    case 'MEDIUM': return '#facc15';  // yellow
    case 'LOW':    return '#f97316';  // orange
    case 'REJECT': return '#f87171';  // red
    default:       return '#64748b';
  }
}

/**
 * Get display label for decision quality.
 * @param {string} quality
 * @returns {string}
 */
export function decisionQualityLabel(quality) {
  switch (quality) {
    case 'HIGH':   return 'High Confidence';
    case 'MEDIUM': return 'Moderate';
    case 'LOW':    return 'Low Confidence';
    case 'REJECT': return 'Rejected';
    default:       return '--';
  }
}

/**
 * Format posterior probability as percentage string.
 * @param {number} prob
 * @returns {string}
 */
export function formatProbability(prob) {
  return `${(prob * 100).toFixed(1)}%`;
}

/**
 * Get color for a probability value (red → yellow → green gradient).
 * @param {number} prob — [0, 1]
 * @returns {string}
 */
export function probabilityColor(prob) {
  if (prob >= 0.6) return '#4ade80';
  if (prob >= 0.35) return '#facc15';
  if (prob >= 0.15) return '#f97316';
  return '#f87171';
}

export {
  DEFAULT_VISUAL_WEIGHTS as UMCE_VISUAL_WEIGHTS,
  DEFAULT_CHANNEL_WEIGHTS as UMCE_CHANNEL_WEIGHTS,
  QUALITY_THRESHOLDS,
};
export default UMCE;
