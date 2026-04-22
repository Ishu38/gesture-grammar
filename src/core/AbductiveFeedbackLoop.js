/**
 * AbductiveFeedbackLoop.js — Closed-Loop Neural↔Symbolic Feedback Engine
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BEYOND VANILLA NEURO-SYMBOLIC: BIDIRECTIONAL ABDUCTIVE FEEDBACK
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Vanilla neuro-symbolic AI:
 *   Neural → Symbolic (one-way: perception feeds into reasoning)
 *
 * MLAF Advanced:
 *   Neural ⇄ Symbolic (bidirectional: reasoning ADAPTS perception)
 *
 * When the symbolic layer (GraphRAG Layer 3 — Abductive Diagnosis) detects
 * a learner error, it traces the cause through the knowledge graph and
 * feeds corrective signals BACK into the neural layer:
 *
 *   1. CONFIDENCE ADAPTATION — Raise RF classifier thresholds for confused
 *      gesture pairs (e.g., YOU↔DRINK: threshold 0.4 → 0.65)
 *
 *   2. BAYESIAN PRIOR UPDATING — Shift P(S) priors in UMCE based on error
 *      history. Gestures the learner struggles with get lower prior → need
 *      stronger evidence to lock.
 *
 *   3. CURRICULUM GATING — When abduction points to a curriculum stage the
 *      learner hasn't mastered, gate gestures from later stages.
 *
 *   4. INTERFERENCE COUNTER-WEIGHTING — When L1 (ISL) transfer is detected,
 *      boost weight of correct SVO ordering signals and increase lock time
 *      to force deliberate sequencing.
 *
 *   5. TEMPORAL DECAY — Adaptations decay over time as the learner improves,
 *      preventing permanent over-correction.
 *
 * This creates a homeostatic loop: errors tighten constraints, mastery
 * relaxes them — mirroring how the brain's error-monitoring system
 * (anterior cingulate cortex) modulates motor output.
 *
 * All computation is in-browser. No data leaves the client.
 */

// =============================================================================
// CONSTANTS
// =============================================================================

/** Minimum confidence boost for confused pairs */
const MIN_CONFUSION_BOOST = 0.05;

/** Maximum confidence threshold (never go above this) */
const MAX_CONFIDENCE_THRESHOLD = 0.85;

/** Base confidence threshold (the RF classifier default) */
const BASE_CONFIDENCE_THRESHOLD = 0.40;

/** Decay rate per successful recognition (exponential decay toward baseline) */
const ADAPTATION_DECAY_RATE = 0.92;

/** How many error events before an adaptation kicks in */
const ERROR_TRIGGER_COUNT = 2;

/** Lock frame multiplier when L1 interference is active */
const INTERFERENCE_LOCK_MULTIPLIER = 1.4;

/** Maximum number of events in the error history ring buffer */
const MAX_HISTORY_SIZE = 200;

/** Time window (ms) for computing error frequency */
const ERROR_FREQUENCY_WINDOW_MS = 60000; // 1 minute

// =============================================================================
// FATIGUE DETECTION CONSTANTS
// =============================================================================

/** Rolling window (ms) for fatigue detection — 5 minutes */
const FATIGUE_WINDOW_MS = 300000;

/** Minimum events in the fatigue window before fatigue can be assessed */
const FATIGUE_MIN_EVENTS = 10;

/** Error rate threshold: above this in the recent window = fatigue detected */
const FATIGUE_ERROR_RATE_THRESHOLD = 0.45;

/** Confidence relaxation factor when fatigued (multiplied against threshold) */
const FATIGUE_CONFIDENCE_RELAXATION = 0.7;

/** Tolerance expansion factor when fatigued (multiplied against toleranceMultiplier) */
const FATIGUE_TOLERANCE_EXPANSION = 1.5;

/** How many consecutive correct events to exit fatigue state */
const FATIGUE_EXIT_STREAK = 8;

// =============================================================================
// ERROR EVENT TYPES
// =============================================================================

export const ERROR_EVENTS = {
  WRONG_VERB_FORM:     'WRONG_VERB_FORM',
  WRONG_WORD_ORDER:    'WRONG_WORD_ORDER',
  MISSING_OBJECT:      'MISSING_OBJECT',
  EXTRA_OBJECT:        'EXTRA_OBJECT',
  GESTURE_CONFUSION:   'GESTURE_CONFUSION',
  TYPE_MISMATCH:       'TYPE_MISMATCH',
  AGREEMENT_VIOLATION: 'AGREEMENT_VIOLATION',
};

// =============================================================================
// ABDUCTIVE FEEDBACK LOOP
// =============================================================================

export class AbductiveFeedbackLoop {
  /**
   * @param {import('./GraphRAG').GraphRAG} graphRAG — the 4-layer reasoning engine
   * @param {object} [options]
   * @param {number} [options.decayRate] — how fast adaptations decay (0-1)
   * @param {number} [options.errorTriggerCount] — errors before adaptation fires
   */
  constructor(graphRAG, options = {}) {
    this.graphRAG = graphRAG;
    this.decayRate = options.decayRate || ADAPTATION_DECAY_RATE;
    this.errorTriggerCount = options.errorTriggerCount || ERROR_TRIGGER_COUNT;

    // ── State ──
    /** @type {Array<ErrorEvent>} Ring buffer of recent error events */
    this._errorHistory = [];

    /** @type {Map<string, number>} Gesture-specific confidence threshold overrides */
    this._confidenceAdaptations = new Map();

    /** @type {Map<string, number>} Gesture pair confusion counts: "A|B" → count */
    this._confusionCounts = new Map();

    /** @type {Map<string, number>} Bayesian prior adjustments per gesture */
    this._priorAdjustments = new Map();

    /** @type {number} Lock frame multiplier (increased when interference detected) */
    this._lockMultiplier = 1.0;

    /** @type {Set<string>} Currently gated gestures (from curriculum gating) */
    this._gatedGestures = new Set();

    /** @type {number} Total error count (lifetime) */
    this._totalErrors = 0;

    /** @type {number} Total correct recognitions (lifetime) */
    this._totalCorrect = 0;

    /** @type {number} Consecutive correct count (for decay) */
    this._consecutiveCorrect = 0;

    // ── Fatigue Detection State ──
    /** @type {boolean} Whether fatigue detection is enabled */
    this._fatigueDetectionEnabled = options.fatigueDetection || false;

    /** @type {boolean} Whether fatigue is currently active */
    this._fatigueActive = false;

    /** @type {Array<{correct: boolean, timestamp: number}>} Rolling event log for fatigue assessment */
    this._fatigueEventLog = [];

    /** @type {number} Session start timestamp */
    this._sessionStartTime = Date.now();

    /** @type {number} Fatigue confidence relaxation currently applied */
    this._fatigueRelaxation = 1.0;

    /** @type {number} Fatigue tolerance expansion currently applied */
    this._fatigueToleranceExpansion = 1.0;
  }

  // ===========================================================================
  // PUBLIC — Event Recording
  // ===========================================================================

  /**
   * Record a successful gesture recognition.
   * Triggers temporal decay of all adaptations.
   *
   * @param {string} gestureId — the correctly recognized gesture
   */
  recordSuccess(gestureId) {
    this._totalCorrect++;
    this._consecutiveCorrect++;

    // Fatigue tracking
    if (this._fatigueDetectionEnabled) {
      this._fatigueEventLog.push({ correct: true, timestamp: Date.now() });
      this._assessFatigue();
    }

    // Temporal decay: every correct recognition relaxes adaptations
    this._applyDecay();

    // If 5 consecutive correct, accelerate decay
    if (this._consecutiveCorrect >= 5) {
      this._applyDecay(); // double decay for streaks
    }
  }

  /**
   * Record an error event and trigger abductive feedback.
   * This is the main entry point — call this when:
   *   - Parser rejects a sentence
   *   - Agreement violation detected
   *   - Wrong word order (ISL interference)
   *   - Gesture confusion (RF outputs wrong class)
   *
   * @param {string} errorType — one of ERROR_EVENTS
   * @param {object} context — error context
   * @param {string} [context.intendedGesture] — what the user meant to do
   * @param {string} [context.recognizedGesture] — what the system recognized
   * @param {string} [context.subjectId] — the subject in the sentence
   * @param {string} [context.verbId] — the verb in the sentence
   * @param {string[]} [context.sentenceTokens] — current sentence token sequence
   * @param {Array} [context.sentenceWords] — full word objects
   * @returns {FeedbackResult} — the adaptations applied
   */
  recordError(errorType, context = {}) {
    this._totalErrors++;
    this._consecutiveCorrect = 0;

    // Fatigue tracking
    if (this._fatigueDetectionEnabled) {
      this._fatigueEventLog.push({ correct: false, timestamp: Date.now() });
      this._assessFatigue();
    }

    // Store event in ring buffer
    const event = {
      type: errorType,
      context,
      timestamp: Date.now(),
    };
    this._errorHistory.push(event);
    if (this._errorHistory.length > MAX_HISTORY_SIZE) {
      this._errorHistory.shift();
    }

    // ── Abductive Diagnosis via Graph RAG ──
    const diagnosis = this.graphRAG
      ? this.graphRAG.diagnoseError(errorType, context)
      : { remediation: [], interferencePatterns: [], explanation: null };

    // ── Apply feedback based on error type ──
    const adaptations = [];

    // 1. Confidence Adaptation (for gesture confusion)
    if (errorType === ERROR_EVENTS.GESTURE_CONFUSION && context.intendedGesture && context.recognizedGesture) {
      const adaptation = this._adaptConfidence(context.intendedGesture, context.recognizedGesture);
      if (adaptation) adaptations.push(adaptation);
    }

    // 2. Bayesian Prior Update (for agreement violations)
    if (errorType === ERROR_EVENTS.WRONG_VERB_FORM || errorType === ERROR_EVENTS.AGREEMENT_VIOLATION) {
      const priorAdapt = this._adaptPrior(context.verbId, diagnosis);
      if (priorAdapt) adaptations.push(priorAdapt);
    }

    // 3. Curriculum Gating (based on remediation path)
    if (diagnosis.remediation && diagnosis.remediation.length > 0) {
      const gating = this._applyCurriculumGating(diagnosis.remediation);
      if (gating) adaptations.push(gating);
    }

    // 4. Interference Counter-Weighting
    if (errorType === ERROR_EVENTS.WRONG_WORD_ORDER && diagnosis.interferencePatterns && diagnosis.interferencePatterns.length > 0) {
      const counterWeight = this._applyInterferenceCounterWeight(diagnosis.interferencePatterns);
      if (counterWeight) adaptations.push(counterWeight);
    }

    // 5. Type mismatch → boost lock time for the misplaced gesture
    if (errorType === ERROR_EVENTS.TYPE_MISMATCH && context.recognizedGesture) {
      const lockAdapt = this._adaptLockTime(context.recognizedGesture);
      if (lockAdapt) adaptations.push(lockAdapt);
    }

    return {
      errorType,
      diagnosis: diagnosis.diagnosis,
      adaptations,
      abductiveChain: diagnosis.diagnosis?.abductiveChain || null,
      interferenceDetected: diagnosis.interferencePatterns.length > 0,
      remediationStages: diagnosis.remediation.map(r => r.stage),
      totalErrors: this._totalErrors,
      totalCorrect: this._totalCorrect,
      errorRate: this._computeErrorRate(),
    };
  }

  // ===========================================================================
  // PUBLIC — Query Current Adaptations
  // ===========================================================================

  /**
   * Get the adapted confidence threshold for a specific gesture.
   * The RF classifier should call this before accepting a classification.
   *
   * @param {string} gestureId
   * @returns {number} — adapted threshold (higher = stricter)
   */
  getAdaptedThreshold(gestureId) {
    const adaptation = this._confidenceAdaptations.get(gestureId);
    if (!adaptation) return BASE_CONFIDENCE_THRESHOLD;
    return Math.min(BASE_CONFIDENCE_THRESHOLD + adaptation, MAX_CONFIDENCE_THRESHOLD);
  }

  /**
   * Get the Bayesian prior adjustment for a gesture.
   * UMCE should multiply P(S) by this factor.
   *
   * @param {string} gestureId
   * @returns {number} — prior multiplier (< 1.0 means reduce prior)
   */
  getPriorAdjustment(gestureId) {
    return this._priorAdjustments.get(gestureId) || 1.0;
  }

  /**
   * Get the current lock frame multiplier.
   * useSentenceBuilder should multiply its lock threshold by this.
   *
   * @returns {number}
   */
  getLockMultiplier() {
    return this._lockMultiplier;
  }

  /**
   * Check if a gesture is currently gated by curriculum.
   *
   * @param {string} gestureId
   * @returns {boolean}
   */
  isGated(gestureId) {
    return this._gatedGestures.has(gestureId);
  }

  /**
   * Get all current adaptations as a structured report.
   * @returns {AdaptationReport}
   */
  getAdaptationReport() {
    const confusionPairs = [];
    for (const [pair, count] of this._confusionCounts) {
      const [a, b] = pair.split('|');
      confusionPairs.push({
        gestureA: a,
        gestureB: b,
        confusionCount: count,
        thresholdA: this.getAdaptedThreshold(a),
        thresholdB: this.getAdaptedThreshold(b),
      });
    }

    return {
      totalErrors: this._totalErrors,
      totalCorrect: this._totalCorrect,
      errorRate: this._computeErrorRate(),
      recentErrorRate: this._computeRecentErrorRate(),
      consecutiveCorrect: this._consecutiveCorrect,
      lockMultiplier: this._lockMultiplier,
      gatedGestures: [...this._gatedGestures],
      confusionPairs,
      confidenceAdaptations: Object.fromEntries(this._confidenceAdaptations),
      priorAdjustments: Object.fromEntries(this._priorAdjustments),
      historySize: this._errorHistory.length,
      fatigue: this.getFatigueStatus(),
    };
  }

  /**
   * Get the error frequency for a specific error type within the time window.
   * @param {string} errorType
   * @returns {number} — errors per minute
   */
  getErrorFrequency(errorType) {
    const now = Date.now();
    const recent = this._errorHistory.filter(
      e => e.type === errorType && (now - e.timestamp) < ERROR_FREQUENCY_WINDOW_MS
    );
    return recent.length; // count per minute window
  }

  // ===========================================================================
  // PUBLIC — Fatigue Detection (CP)
  // ===========================================================================

  /**
   * Enable or disable fatigue detection.
   * When enabled, the system monitors declining gesture precision over a
   * rolling 5-minute window. If error rate exceeds the fatigue threshold,
   * confidence thresholds are RELAXED (not tightened) and tolerance bands
   * are EXPANDED — the opposite of normal error-driven adaptation.
   *
   * @param {boolean} enabled
   */
  setFatigueDetection(enabled) {
    this._fatigueDetectionEnabled = enabled;
    if (!enabled) {
      this._fatigueActive = false;
      this._fatigueRelaxation = 1.0;
      this._fatigueToleranceExpansion = 1.0;
    }
  }

  /**
   * Check if fatigue is currently detected.
   * @returns {boolean}
   */
  isFatigueActive() {
    return this._fatigueActive;
  }

  /**
   * Get the fatigue-adjusted confidence threshold for a gesture.
   * When fatigued, this returns a LOWER threshold than getAdaptedThreshold().
   *
   * @param {string} gestureId
   * @returns {number}
   */
  getFatigueAdjustedThreshold(gestureId) {
    const baseAdapted = this.getAdaptedThreshold(gestureId);
    if (!this._fatigueActive) return baseAdapted;
    // Relax: multiply the threshold by the relaxation factor (< 1.0)
    return Math.max(0.15, baseAdapted * this._fatigueRelaxation);
  }

  /**
   * Get the fatigue-adjusted tolerance multiplier.
   * When fatigued, returns a HIGHER multiplier (wider tolerance bands).
   *
   * @param {number} baseTolerance — from AccessibilityProfile.getToleranceBands()
   * @returns {number}
   */
  getFatigueAdjustedTolerance(baseTolerance) {
    if (!this._fatigueActive) return baseTolerance;
    return baseTolerance * this._fatigueToleranceExpansion;
  }

  /**
   * Get fatigue diagnostic info for debug/session report.
   * @returns {object}
   */
  getFatigueStatus() {
    const now = Date.now();
    const recentEvents = this._fatigueEventLog.filter(
      e => (now - e.timestamp) < FATIGUE_WINDOW_MS
    );
    const errors = recentEvents.filter(e => !e.correct).length;
    const total = recentEvents.length;

    return {
      enabled: this._fatigueDetectionEnabled,
      active: this._fatigueActive,
      recentErrorRate: total > 0 ? errors / total : 0,
      recentTotal: total,
      recentErrors: errors,
      sessionDurationMs: now - this._sessionStartTime,
      confidenceRelaxation: this._fatigueRelaxation,
      toleranceExpansion: this._fatigueToleranceExpansion,
    };
  }

  /**
   * Reset all adaptations to baseline.
   */
  reset() {
    this._errorHistory = [];
    this._confidenceAdaptations.clear();
    this._confusionCounts.clear();
    this._priorAdjustments.clear();
    this._lockMultiplier = 1.0;
    this._gatedGestures.clear();
    this._totalErrors = 0;
    this._totalCorrect = 0;
    this._consecutiveCorrect = 0;
    this._fatigueActive = false;
    this._fatigueEventLog = [];
    this._fatigueRelaxation = 1.0;
    this._fatigueToleranceExpansion = 1.0;
    this._sessionStartTime = Date.now();
  }

  // ===========================================================================
  // PRIVATE — Feedback Mechanisms
  // ===========================================================================

  /**
   * Mechanism 1: Confidence Adaptation
   * When two gestures are confused, raise threshold for both.
   * Uses the Graph RAG's confusion pair data to determine how much.
   */
  _adaptConfidence(intended, recognized) {
    const pairKey = [intended, recognized].sort().join('|');
    const count = (this._confusionCounts.get(pairKey) || 0) + 1;
    this._confusionCounts.set(pairKey, count);

    if (count < this.errorTriggerCount) return null;

    // Compute boost based on confusion frequency and graph-derived ambiguity
    const graphPairs = this.graphRAG?.getConfusionPairs(intended) || [];
    const isKnownPair = graphPairs.some(p => p.confusedWith === recognized);

    // Known confusion pairs get a larger boost (the graph predicted this)
    const boost = isKnownPair
      ? MIN_CONFUSION_BOOST * (1 + count * 0.1)
      : MIN_CONFUSION_BOOST * (1 + count * 0.05);

    // Apply to both gestures in the pair
    this._confidenceAdaptations.set(intended,
      Math.max(this._confidenceAdaptations.get(intended) || 0, boost));
    this._confidenceAdaptations.set(recognized,
      Math.max(this._confidenceAdaptations.get(recognized) || 0, boost));

    return {
      type: 'CONFIDENCE_ADAPTATION',
      pair: [intended, recognized],
      boost,
      newThresholdA: this.getAdaptedThreshold(intended),
      newThresholdB: this.getAdaptedThreshold(recognized),
      isKnownConfusionPair: isKnownPair,
      reason: `Confused ${intended}↔${recognized} ${count} times → raised threshold`,
    };
  }

  /**
   * Mechanism 2: Bayesian Prior Updating
   * Lower the prior P(S) for gestures the learner consistently mis-produces.
   * This makes UMCE require stronger sensory evidence before accepting.
   */
  _adaptPrior(gestureId, diagnosis) {
    if (!gestureId) return null;

    // Count errors for this gesture's agreement rule
    const errorCount = this._errorHistory.filter(
      e => (e.type === ERROR_EVENTS.WRONG_VERB_FORM || e.type === ERROR_EVENTS.AGREEMENT_VIOLATION) &&
           (e.context.verbId === gestureId)
    ).length;

    if (errorCount < this.errorTriggerCount) return null;

    // Prior suppression: more errors → lower prior (floor at 0.3)
    const suppression = Math.max(0.3, 1.0 - (errorCount * 0.08));
    this._priorAdjustments.set(gestureId, suppression);

    // Also suppress the s-form pair (if agreement is the issue)
    const profile = this.graphRAG?.getTokenProfile(gestureId);
    if (profile?.sFormPair) {
      this._priorAdjustments.set(profile.sFormPair, suppression);
    }
    if (profile?.baseFormPair) {
      this._priorAdjustments.set(profile.baseFormPair, suppression);
    }

    return {
      type: 'PRIOR_ADJUSTMENT',
      gestureId,
      suppression,
      reason: `${errorCount} agreement errors → P(S) for ${gestureId} reduced to ${suppression.toFixed(2)}`,
      remediationStage: diagnosis.remediation?.[0]?.stage || null,
    };
  }

  /**
   * Mechanism 3: Curriculum Gating
   * If the abductive chain points to a curriculum stage the learner hasn't
   * mastered, gate gestures from that and later stages.
   */
  _applyCurriculumGating(remediation) {
    // Get the earliest remediation stage
    const stages = remediation.map(r => r.stage).sort((a, b) => a - b);
    const targetStage = stages[0];

    if (!targetStage || targetStage <= 2) return null; // Don't gate basic stages

    // Gate gestures that belong to stages after the remediation target
    const llmContext = this.graphRAG?.buildLLMContext([]);
    const curriculum = llmContext?.graph_rag_context?.curriculum;
    if (!curriculum?.stages) return null;
    const gated = [];

    for (const stage of curriculum.stages) {
      if (stage.stage > targetStage) {
        for (const gesture of stage.gestures) {
          this._gatedGestures.add(gesture);
          gated.push(gesture);
        }
      }
    }

    if (gated.length === 0) return null;

    return {
      type: 'CURRICULUM_GATING',
      targetStage,
      gatedGestures: gated,
      reason: `Abductive diagnosis → remediate at Stage ${targetStage} → gating ${gated.length} later gestures`,
    };
  }

  /**
   * Mechanism 4: Interference Counter-Weighting
   * When L1 (ISL) word order transfer is detected, increase lock time
   * to force deliberate, sequential ordering.
   */
  _applyInterferenceCounterWeight(interferencePatterns) {
    const sovDetected = interferencePatterns.some(p => p.id === 'SOV_ORDER');
    const topicFronting = interferencePatterns.some(p => p.id === 'TOPIC_FRONTING');

    // Increase lock multiplier — more interference = more deliberate sequencing
    const boost = (sovDetected ? 0.2 : 0) + (topicFronting ? 0.15 : 0);
    this._lockMultiplier = Math.min(
      this._lockMultiplier + boost,
      INTERFERENCE_LOCK_MULTIPLIER
    );

    return {
      type: 'INTERFERENCE_COUNTER_WEIGHT',
      patterns: interferencePatterns.map(p => p.id),
      lockMultiplier: this._lockMultiplier,
      reason: `L1 transfer detected (${interferencePatterns.map(p => p.id).join(', ')}) → lock multiplier increased to ${this._lockMultiplier.toFixed(2)}`,
    };
  }

  /**
   * Mechanism 5: Lock Time Adaptation for Type Mismatches
   * If a gesture keeps being placed in the wrong slot, increase its lock time.
   */
  _adaptLockTime(gestureId) {
    const typeMismatches = this._errorHistory.filter(
      e => e.type === ERROR_EVENTS.TYPE_MISMATCH && e.context.recognizedGesture === gestureId
    ).length;

    if (typeMismatches < this.errorTriggerCount) return null;

    // Increase confidence threshold for this gesture (makes it harder to lock)
    const boost = MIN_CONFUSION_BOOST * typeMismatches;
    this._confidenceAdaptations.set(gestureId,
      Math.max(this._confidenceAdaptations.get(gestureId) || 0, boost));

    return {
      type: 'LOCK_TIME_ADAPTATION',
      gestureId,
      newThreshold: this.getAdaptedThreshold(gestureId),
      reason: `${typeMismatches} type mismatches for ${gestureId} → threshold raised`,
    };
  }

  // ===========================================================================
  // PRIVATE — Fatigue Detection
  // ===========================================================================

  /**
   * Assess fatigue from the rolling event log.
   *
   * Fatigue signature in CP:
   *   - Error rate in the recent window exceeds FATIGUE_ERROR_RATE_THRESHOLD
   *   - Session has been running for at least 2 minutes (avoid false detection
   *     during initial calibration)
   *   - Declining accuracy trend: the second half of the window has a higher
   *     error rate than the first half (performance is getting worse, not
   *     just consistently bad)
   *
   * When fatigue is detected:
   *   - Confidence thresholds are RELAXED (multiplied by 0.7)
   *   - Tolerance bands are EXPANDED (multiplied by 1.5)
   *   - Normal error-driven tightening is suppressed
   *
   * When fatigue resolves (consecutive correct streak):
   *   - Relaxation and expansion gradually return to baseline
   */
  _assessFatigue() {
    if (!this._fatigueDetectionEnabled) return;

    const now = Date.now();

    // Prune old events beyond the fatigue window
    this._fatigueEventLog = this._fatigueEventLog.filter(
      e => (now - e.timestamp) < FATIGUE_WINDOW_MS
    );

    const events = this._fatigueEventLog;
    const sessionDuration = now - this._sessionStartTime;

    // Need minimum events and session duration
    if (events.length < FATIGUE_MIN_EVENTS || sessionDuration < 120000) {
      return;
    }

    // Compute error rate in the window
    const errors = events.filter(e => !e.correct).length;
    const errorRate = errors / events.length;

    // Check for declining accuracy trend (second half worse than first)
    const midpoint = Math.floor(events.length / 2);
    const firstHalf = events.slice(0, midpoint);
    const secondHalf = events.slice(midpoint);
    const firstHalfErrorRate = firstHalf.filter(e => !e.correct).length / Math.max(firstHalf.length, 1);
    const secondHalfErrorRate = secondHalf.filter(e => !e.correct).length / Math.max(secondHalf.length, 1);
    const isDeclining = secondHalfErrorRate > firstHalfErrorRate;

    if (errorRate >= FATIGUE_ERROR_RATE_THRESHOLD && isDeclining) {
      // Fatigue detected — activate relaxation
      if (!this._fatigueActive) {
        this._fatigueActive = true;
      }
      // Scale relaxation by how far above threshold we are
      const severity = Math.min(1.0, (errorRate - FATIGUE_ERROR_RATE_THRESHOLD) / 0.3);
      this._fatigueRelaxation = FATIGUE_CONFIDENCE_RELAXATION - (severity * 0.15);
      this._fatigueToleranceExpansion = FATIGUE_TOLERANCE_EXPANSION + (severity * 0.3);
    } else if (this._fatigueActive && this._consecutiveCorrect >= FATIGUE_EXIT_STREAK) {
      // Fatigue resolving — gradual recovery
      this._fatigueRelaxation = Math.min(1.0, this._fatigueRelaxation + 0.05);
      this._fatigueToleranceExpansion = Math.max(1.0, this._fatigueToleranceExpansion - 0.08);

      if (this._fatigueRelaxation >= 0.98 && this._fatigueToleranceExpansion <= 1.02) {
        this._fatigueActive = false;
        this._fatigueRelaxation = 1.0;
        this._fatigueToleranceExpansion = 1.0;
      }
    }
  }

  // ===========================================================================
  // PRIVATE — Temporal Decay
  // ===========================================================================

  /**
   * Apply exponential decay to all adaptations.
   * Called on every successful recognition.
   * Adaptations approach baseline over time.
   */
  _applyDecay() {
    // Decay confidence adaptations
    for (const [gesture, boost] of this._confidenceAdaptations) {
      const decayed = boost * this.decayRate;
      if (decayed < 0.01) {
        this._confidenceAdaptations.delete(gesture);
      } else {
        this._confidenceAdaptations.set(gesture, decayed);
      }
    }

    // Decay prior adjustments (toward 1.0) using additive interpolation.
    // Additive decay (vs multiplicative for confidence/lock) ensures priors
    // converge to neutral (1.0) at a fixed rate regardless of current value,
    // preventing prior biases from persisting indefinitely.
    for (const [gesture, prior] of this._priorAdjustments) {
      const newPrior = prior + (1.0 - prior) * (1 - this.decayRate);
      if (Math.abs(newPrior - 1.0) < 0.01) {
        this._priorAdjustments.delete(gesture);
      } else {
        this._priorAdjustments.set(gesture, newPrior);
      }
    }

    // Decay lock multiplier (toward 1.0)
    this._lockMultiplier = 1.0 + (this._lockMultiplier - 1.0) * this.decayRate;
    if (Math.abs(this._lockMultiplier - 1.0) < 0.01) {
      this._lockMultiplier = 1.0;
    }

    // Ungate gestures after sustained correct performance
    if (this._consecutiveCorrect >= 10 && this._gatedGestures.size > 0) {
      this._gatedGestures.clear();
    }
  }

  // ===========================================================================
  // PRIVATE — Analytics
  // ===========================================================================

  _computeErrorRate() {
    const total = this._totalErrors + this._totalCorrect;
    return total === 0 ? 0 : this._totalErrors / total;
  }

  _computeRecentErrorRate() {
    const now = Date.now();
    const recentErrors = this._errorHistory.filter(
      e => (now - e.timestamp) < ERROR_FREQUENCY_WINDOW_MS
    ).length;
    // Approximate recent total (errors + estimated correct)
    const estimatedRecentTotal = Math.max(recentErrors, 1) / Math.max(this._computeErrorRate(), 0.01);
    return recentErrors / Math.max(estimatedRecentTotal, 1);
  }
}

export default AbductiveFeedbackLoop;
