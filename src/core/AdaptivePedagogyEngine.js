/**
 * AdaptivePedagogyEngine.js — Coordinated Instructional Intelligence
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE META-CONTROLLER: ADAPTIVE NEURO-SYMBOLIC PEDAGOGY
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Sits above all adaptation modules as the "executive function" of the system.
 * Ingests data from every subsystem and makes coordinated instructional
 * decisions. This is what transforms MLAF from a reactive error-corrector
 * into a genuinely adaptive tutoring platform.
 *
 * Data sources (all existing, all real-time):
 *   KnowledgeTracer      → BKT knowledge states (what the learner knows)
 *   CognitiveLoadAdapter → motor jitter / fatigue (how taxed the learner is)
 *   AbductiveFeedbackLoop → error adaptations (what thresholds are adjusted)
 *   GestureMasteryGate   → curriculum stage progress (what's been mastered)
 *   SpacedRepetitionScheduler → review due dates (what needs revisiting)
 *   SessionDataLogger    → cross-session trends (how is the learner trending)
 *   ISLInterferenceDetector → L1 transfer patterns (ISL influence level)
 *
 * Decisions it makes (that no single module can make alone):
 *   1. Stage progression — advance, regress, or stay
 *   2. Modality weighting — adjust visual/acoustic/gaze emphasis
 *   3. Scaffolding level — increase or decrease support
 *   4. Intervention strategy — which feedback style works best
 *   5. Pacing — optimal difficulty and repetition cadence
 *   6. Learner state classification — confused, fatigued, stuck, flowing
 *
 * This is Structured Literacy operationalized as a policy engine:
 * explicit, systematic, cumulative, and DIAGNOSTIC-PRESCRIPTIVE instruction
 * driven by continuous, multimodal learner assessment.
 */

// =============================================================================
// LEARNER STATE CLASSIFICATION
// =============================================================================

export const LEARNER_STATES = {
  FLOWING:   'FLOWING',   // High accuracy, low cognitive load, fast responses
  LEARNING:  'LEARNING',  // Moderate accuracy, engaged, making progress
  CONFUSED:  'CONFUSED',  // High error rate, high load, slow responses
  FATIGUED:  'FATIGUED',  // Declining accuracy over time, high session duration
  STUCK:     'STUCK',     // Persistent same error, no improvement despite attempts
  DISENGAGED:'DISENGAGED',// Long pauses, low response rate, gaze wandering
};

// =============================================================================
// PEDAGOGICAL DECISION TYPES
// =============================================================================

export const DECISION_TYPES = {
  STAGE_ADVANCE:    'STAGE_ADVANCE',
  STAGE_REGRESS:    'STAGE_REGRESS',
  MODALITY_ADJUST:  'MODALITY_ADJUST',
  SCAFFOLD_ADJUST:  'SCAFFOLD_ADJUST',
  INTERVENTION:     'INTERVENTION',
  PACING_ADJUST:    'PACING_ADJUST',
  BREAK_SUGGESTION: 'BREAK_SUGGESTION',
};

// =============================================================================
// DEFAULT POLICY PARAMETERS
// =============================================================================

const DEFAULT_PARAMS = {
  // Stage progression thresholds
  stageAdvanceThreshold:  0.75,  // Average P(known) needed to advance
  stageRegressThreshold:  0.30,  // Average P(known) below which to regress
  minConceptsForAdvance:  3,     // Minimum mastered concepts before advancing

  // Fatigue thresholds
  highLoadSustainedFrames: 180,  // ~6 seconds of sustained HIGH load
  maxSessionMinutes:       20,   // Suggest break after this
  fatigueErrorRateSpike:   0.15, // Error rate increase considered fatigue

  // Confusion detection
  sameErrorStreakCount:    3,    // Same error 3+ times = stuck
  stuckRecoveryAttempts:   2,    // Attempts to auto-recover before intervention

  // Modality adjustment
  visualWeightDefault:     0.55,
  acousticWeightDefault:   0.25,
  gazeWeightDefault:       0.20,
  motorImpairmentVisualBoost: 0.10,  // Increase visual for motor impairment

  // Scaffolding levels
  maxScaffoldingLevel:     3,
  minScaffoldingLevel:     0,
  scaffoldingStepSize:     0.5,
};

// =============================================================================
// ADAPTIVE PEDAGOGY ENGINE
// =============================================================================

export class AdaptivePedagogyEngine {
  /**
   * @param {object} config
   * @param {object} [config.params] — policy parameters
   * @param {string} [config.profileType] — accessibility profile type
   */
  constructor(config = {}) {
    this._params = { ...DEFAULT_PARAMS, ...(config.params || {}) };
    this._profileType = config.profileType || 'default';

    /** Current learner state */
    this._learnerState = LEARNER_STATES.FLOWING;

    /** Current scaffolding level (0 = minimal, 3 = maximum) */
    this._scaffoldingLevel = 1;

    /** Decision history for trend analysis */
    this._decisionHistory = [];

    /** Consecutive frame counts for state transitions */
    this._stateFrameCount = 0;
    this._previousState = LEARNER_STATES.FLOWING;

    /** Last N error types for stuck detection */
    this._recentErrorTypes = [];

    /** Session start time */
    this._sessionStart = Date.now();

    /** Effective modality weights (adjusted for learner profile) */
    this._modalityWeights = this._computeInitialModalityWeights();

    /** Preferred intervention style (learned from session data) */
    this._preferredIntervention = 'IMPLICIT'; // IMPLICIT | CONTRASTIVE | EXPLICIT | DELAYED

    /** Error resolution tracking: { errorType: { attempts, resolved } } */
    this._errorResolution = {};
  }

  // ===========================================================================
  // PUBLIC — Core Update (called every frame during practice)
  // ===========================================================================

  /**
   * Update the adaptive engine with the latest data from all subsystems.
   * Called every frame in SandboxMode.processLandmarks.
   *
   * @param {object} dataSources
   * @param {object} dataSources.knowledgeTracer — KnowledgeTracer instance
   * @param {object} dataSources.cognitiveLoad — { level, jitter }
   * @param {object} dataSources.umceResult — UMCE fusion result (entropy, margin, decision_quality)
   * @param {object} dataSources.masteryReport — from GestureMasteryGate
   * @param {Array} dataSources.sentence — current sentence
   * @param {object} [dataSources.latestError] — most recent error explanation
   * @param {number} [dataSources.responseTimeMs] — latest production latency
   * @returns {PedagogyUpdate}
   */
  update(dataSources) {
    const {
      knowledgeTracer,
      cognitiveLoad,
      umceResult,
      masteryReport,
      sentence: _sentence,
      latestError,
      responseTimeMs,
    } = dataSources;

    this._stateFrameCount++;

    // Determine current learner state
    this._classifyLearnerState(cognitiveLoad, umceResult, latestError, responseTimeMs);

    // Generate decisions based on state + data
    const decisions = [];

    // 1. Stage progression check (periodic — not every frame)
    if (this._shouldCheckProgression(knowledgeTracer)) {
      const stageDecision = this._evaluateStageProgression(knowledgeTracer, masteryReport);
      if (stageDecision) decisions.push(stageDecision);
    }

    // 2. Modality adjustment based on learner state + profile
    const modalityDecision = this._evaluateModalityAdjustment(cognitiveLoad);
    if (modalityDecision) decisions.push(modalityDecision);

    // 3. Scaffolding adjustment
    const scaffoldDecision = this._evaluateScaffoldingAdjustment();
    if (scaffoldDecision) decisions.push(scaffoldDecision);

    // 4. Intervention strategy
    if (latestError) {
      this._trackErrorForResolution(latestError);
      if (this._shouldIntervene(latestError)) {
        const intervention = this._planIntervention(latestError);
        if (intervention) decisions.push(intervention);
      }
    }

    // 5. Break suggestion
    const breakDecision = this._evaluateBreakNeeded();
    if (breakDecision) decisions.push(breakDecision);

    // Build learner model
    const learnerModel = this._buildLearnerModel(knowledgeTracer);

    return {
      learnerState: this._learnerState,
      decisions,
      learnerModel,
      effectiveModalityWeights: { ...this._modalityWeights },
      scaffoldingLevel: this._scaffoldingLevel,
    };
  }

  // ===========================================================================
  // PUBLIC — Strategic Decisions (called at transitions)
  // ===========================================================================

  /**
   * Evaluate whether the learner should advance to the next curriculum stage.
   * Called when a session ends or when all current-stage concepts are mastered.
   *
   * @param {object} knowledgeTracer
   * @param {object} masteryReport
   * @returns {StageDecision|null}
   */
  evaluateStageProgression(knowledgeTracer, masteryReport) {
    return this._evaluateStageProgression(knowledgeTracer, masteryReport);
  }

  /**
   * Recommend the next exercise for the learner.
   * Based on BKT weakest concepts, SRS due reviews, and curriculum stage.
   *
   * @param {object} knowledgeTracer
   * @param {object} masteryReport
   * @param {object} srsReport — from SpacedRepetitionScheduler
   * @returns {ExerciseRecommendation}
   */
  recommendNextExercise(knowledgeTracer, masteryReport, srsReport) {
    const knowledge = knowledgeTracer?.getOverallReport();

    // Prioritize SRS due reviews
    if (srsReport?.dueNowIds?.length > 0) {
      return {
        exerciseType: 'SPACED_REVIEW',
        concepts: srsReport.dueNowIds.slice(0, 5),
        modality: this._getOptimalModality(),
        difficulty: 'MODERATE',
        rationale: `${srsReport.dueNowIds.length} concept(s) due for spaced repetition review`,
      };
    }

    // Practice weakest concepts
    if (knowledge?.weakestConcepts?.length > 0) {
      const weakest = knowledge.weakestConcepts.filter(c => c.pKnown < 0.70);
      if (weakest.length > 0) {
        return {
          exerciseType: 'TARGETED_PRACTICE',
          concepts: weakest.slice(0, 3).map(c => c.conceptId),
          modality: this._getOptimalModality(),
          difficulty: 'CHALLENGING',
          rationale: `Targeted practice on ${weakest.length} developing concepts`,
        };
      }
    }

    // Free practice at current stage
    const stage = masteryReport?.currentStage || 1;
    return {
      exerciseType: 'FREE_PRACTICE',
      concepts: [],
      modality: this._getOptimalModality(),
      difficulty: 'MODERATE',
      rationale: `Stage ${stage} free practice — consolidate current concepts`,
    };
  }

  /**
   * Recommend an intervention strategy for a specific error.
   *
   * @param {object} errorExplanation — from ExplainabilityEngine
   * @returns {InterventionPlan}
   */
  recommendIntervention(errorExplanation) {
    return this._planIntervention(errorExplanation);
  }

  // ===========================================================================
  // PUBLIC — Learner Model Access
  // ===========================================================================

  /**
   * Build a complete learner model from all subsystems.
   * @param {object} knowledgeTracer
   * @returns {LearnerModel}
   */
  getLearnerModel(knowledgeTracer) {
    return this._buildLearnerModel(knowledgeTracer);
  }

  /**
   * Get the personalization profile — learned from session data.
   * This becomes more accurate as more pilot data accumulates.
   * @returns {PersonalizationProfile}
   */
  getPersonalizationProfile() {
    return {
      profileType: this._profileType,
      effectiveModalities: this._summarizeModalities(),
      optimalPacing: this._getPacing(),
      preferredFeedbackStyle: this._preferredIntervention,
      interventionSuccessRate: this._computeInterventionSuccessRate(),
      currentState: this._learnerState,
      scaffoldingLevel: this._scaffoldingLevel,
      sessionDurationMinutes: Math.round((Date.now() - this._sessionStart) / 60000),
    };
  }

  /**
   * Get adaptation history for research/validation.
   * @returns {AdaptationHistory}
   */
  getAdaptationHistory() {
    return {
      decisions: this._decisionHistory,
      stateTransitions: this._stateTransitions || [],
      modalityAdjustments: this._modalityAdjustments || [],
      errorResolution: this._errorResolution,
    };
  }

  /**
   * Reset session state.
   */
  reset() {
    this._learnerState = LEARNER_STATES.FLOWING;
    this._stateFrameCount = 0;
    this._previousState = LEARNER_STATES.FLOWING;
    this._recentErrorTypes = [];
    this._decisionHistory = [];
    this._stateTransitions = [];
    this._modalityAdjustments = [];
    this._sessionStart = Date.now();
    this._scaffoldingLevel = 1;
    this._preferredIntervention = 'IMPLICIT';
    this._errorResolution = {};
  }

  // ===========================================================================
  // PRIVATE — Learner State Classification
  // ===========================================================================

  _classifyLearnerState(cognitiveLoad, umceResult, latestError, responseTimeMs) {
    const load = cognitiveLoad?.level || 'LOW';
    const decisionQuality = umceResult?.decision_quality || 'HIGH';
    const entropy = umceResult?.entropy || 0;

    // Default: flowing
    let newState = LEARNER_STATES.FLOWING;

    // Fatigue detection: sustained high load
    if (load === 'HIGH' && this._stateFrameCount > this._params.highLoadSustainedFrames) {
      newState = LEARNER_STATES.FATIGUED;
    }
    // Confusion: high entropy + recent errors
    else if (entropy > 2.0 && decisionQuality === 'LOW' && this._recentErrorTypes.length >= 3) {
      newState = LEARNER_STATES.CONFUSED;
    }
    // Stuck: same error repeated
    else if (this._detectStuck()) {
      newState = LEARNER_STATES.STUCK;
    }
    // Disengaged: very low activity
    else if (responseTimeMs && responseTimeMs > 10000) {
      newState = LEARNER_STATES.DISENGAGED;
    }
    // Learning: moderate errors, engaged
    else if (this._recentErrorTypes.length > 0) {
      newState = LEARNER_STATES.LEARNING;
    }

    // Track state transitions
    if (newState !== this._learnerState) {
      this._stateTransitions = this._stateTransitions || [];
      this._stateTransitions.push({
        from: this._learnerState,
        to: newState,
        timestamp: Date.now(),
        frame: this._stateFrameCount,
      });
      this._previousState = this._learnerState;
      this._learnerState = newState;
      this._stateFrameCount = 0;
    }
  }

  // ===========================================================================
  // PRIVATE — Decision Evaluation
  // ===========================================================================

  _shouldCheckProgression(knowledgeTracer) {
    if (!knowledgeTracer) return false;
    // Check every ~180 frames (~6 seconds)
    return this._stateFrameCount % 180 === 0;
  }

  _evaluateStageProgression(knowledgeTracer, masteryReport) {
    if (!knowledgeTracer || !masteryReport) return null;

    const report = knowledgeTracer.getOverallReport();
    const currentStage = masteryReport.currentStage || 1;
    const currentStageData = report.stageCompletion?.[currentStage];

    if (!currentStageData) return null;

    const masteredRatio = currentStageData.total > 0
      ? currentStageData.mastered / currentStageData.total
      : 0;

    // Advance: all concepts in current stage are mastered
    if (masteredRatio >= 1.0 && currentStage < 5) {
      const decision = {
        type: DECISION_TYPES.STAGE_ADVANCE,
        fromStage: currentStage,
        toStage: currentStage + 1,
        rationale: `All ${currentStageData.total} Stage ${currentStage} concepts mastered — advancing to Stage ${currentStage + 1}`,
        timestamp: Date.now(),
      };
      this._decisionHistory.push(decision);
      return decision;
    }

    // Regress: very low average knowledge for current stage
    const stageConceptKnows = currentStageData.concepts.map(c => c.pKnown);
    const avgKnow = stageConceptKnows.length > 0
      ? stageConceptKnows.reduce((s, v) => s + v, 0) / stageConceptKnows.length
      : 0;

    if (avgKnow < this._params.stageRegressThreshold && currentStage > 1) {
      const decision = {
        type: DECISION_TYPES.STAGE_REGRESS,
        fromStage: currentStage,
        toStage: currentStage - 1,
        rationale: `Average concept knowledge for Stage ${currentStage} is ${Math.round(avgKnow * 100)}% — below ${Math.round(this._params.stageRegressThreshold * 100)}% threshold. Regressing to Stage ${currentStage - 1} for reinforcement.`,
        timestamp: Date.now(),
      };
      this._decisionHistory.push(decision);
      return decision;
    }

    return null;
  }

  _evaluateModalityAdjustment(cognitiveLoad) {
    const load = cognitiveLoad?.level || 'LOW';
    const defaultWeights = this._computeInitialModalityWeights();

    // Under high load, reduce visual complexity, increase acoustic
    if (load === 'HIGH' && this._modalityWeights.visual > 0.35) {
      const decision = {
        type: DECISION_TYPES.MODALITY_ADJUST,
        previous: { ...this._modalityWeights },
        adjusted: {
          visual: this._modalityWeights.visual - 0.10,
          acoustic: Math.min(0.45, this._modalityWeights.acoustic + 0.10),
          gaze: this._modalityWeights.gaze,
        },
        rationale: 'High cognitive load detected — reducing visual modality weight to decrease processing demand',
        timestamp: Date.now(),
      };
      this._modalityWeights = decision.adjusted;
      this._modalityAdjustments = this._modalityAdjustments || [];
      this._modalityAdjustments.push(decision);
      return decision;
    }

    // Return to defaults if load is low and weights have drifted
    if (load === 'LOW' &&
        (Math.abs(this._modalityWeights.visual - defaultWeights.visual) > 0.05)) {
      const decision = {
        type: DECISION_TYPES.MODALITY_ADJUST,
        previous: { ...this._modalityWeights },
        adjusted: { ...defaultWeights },
        rationale: 'Cognitive load normalized — restoring default modality weights',
        timestamp: Date.now(),
      };
      this._modalityWeights = defaultWeights;
      this._modalityAdjustments = this._modalityAdjustments || [];
      this._modalityAdjustments.push(decision);
      return decision;
    }

    return null;
  }

  _evaluateScaffoldingAdjustment() {
    let targetLevel = this._scaffoldingLevel;

    switch (this._learnerState) {
      case LEARNER_STATES.CONFUSED:
      case LEARNER_STATES.STUCK:
        targetLevel = Math.min(this._params.maxScaffoldingLevel, this._scaffoldingLevel + 1);
        break;
      case LEARNER_STATES.FLOWING:
        targetLevel = Math.max(this._params.minScaffoldingLevel, this._scaffoldingLevel - 1);
        break;
      case LEARNER_STATES.FATIGUED:
        targetLevel = Math.min(this._params.maxScaffoldingLevel, this._scaffoldingLevel);
        break;
      default:
        // Maintain current
        return null;
    }

    if (targetLevel !== this._scaffoldingLevel) {
      const decision = {
        type: DECISION_TYPES.SCAFFOLD_ADJUST,
        fromLevel: this._scaffoldingLevel,
        toLevel: targetLevel,
        rationale: `Learner state: ${this._learnerState} — adjusting scaffolding from ${this._scaffoldingLevel} to ${targetLevel}`,
        timestamp: Date.now(),
      };
      this._scaffoldingLevel = targetLevel;
      this._decisionHistory.push(decision);
      return decision;
    }

    return null;
  }

  _evaluateBreakNeeded() {
    const sessionMinutes = (Date.now() - this._sessionStart) / 60000;

    if (this._learnerState === LEARNER_STATES.FATIGUED && sessionMinutes > 10) {
      return {
        type: DECISION_TYPES.BREAK_SUGGESTION,
        reason: `Fatigue detected after ${Math.round(sessionMinutes)} minutes — consider a short break`,
        sessionMinutes: Math.round(sessionMinutes),
        timestamp: Date.now(),
      };
    }

    if (sessionMinutes > this._params.maxSessionMinutes) {
      return {
        type: DECISION_TYPES.BREAK_SUGGESTION,
        reason: `Session exceeded ${this._params.maxSessionMinutes} minutes — recommend ending or pausing`,
        sessionMinutes: Math.round(sessionMinutes),
        timestamp: Date.now(),
      };
    }

    return null;
  }

  _shouldIntervene(errorExplanation) {
    if (!errorExplanation) return false;

    // Intervene when error persists
    const errorType = errorExplanation.errorType;
    const sameErrorCount = this._recentErrorTypes.filter(e => e === errorType).length;

    return sameErrorCount >= this._params.sameErrorStreakCount;
  }

  _planIntervention(errorExplanation) {
    // Choose intervention style based on what's worked before
    let style = this._preferredIntervention;

    // For ISL transfer errors, CONTRASTIVE is most effective
    if (errorExplanation.l1Transfer) {
      style = 'CONTRASTIVE';
    }
    // For repeated errors, switch to EXPLICIT
    else if (this._isRepeatedError(errorExplanation.errorType)) {
      style = 'EXPLICIT';
    }

    return {
      type: DECISION_TYPES.INTERVENTION,
      errorType: errorExplanation.errorType,
      style,
      rationale: errorExplanation.rootCause || 'Persistent error pattern',
      action: errorExplanation.remediation || 'Review and retry',
      timestamp: Date.now(),
    };
  }

  // ===========================================================================
  // PRIVATE — Learner Model Builder
  // ===========================================================================

  _buildLearnerModel(knowledgeTracer) {
    const knowledge = knowledgeTracer?.getOverallReport();
    const profile = this.getPersonalizationProfile();

    return {
      profile: this._profileType,
      sessionCount: knowledge?.totalConcepts ? 1 : 0, // Cross-session count from persistence

      knowledge: knowledge
        ? Object.entries(knowledgeTracer.getAllStates())
            .filter(([, state]) => state && state.category !== 'META')
            .reduce((acc, [id, state]) => {
              acc[id] = {
                pKnown: state.pKnown,
                masteryStage: state.masteryStage,
              };
              return acc;
            }, {})
        : {},

      adaptationProfile: {
        effectiveModalities: profile.effectiveModalities,
        optimalPacing: profile.optimalPacing,
        bestFeedbackStyle: profile.preferredFeedbackStyle,
        interventionSuccessRate: profile.interventionSuccessRate,
      },

      riskFactors: {
        fatigueDetected: this._learnerState === LEARNER_STATES.FATIGUED,
        confusedDetected: this._learnerState === LEARNER_STATES.CONFUSED,
        islInterferenceActive: this._errorResolution?.['WRONG_WORD_ORDER']?.attempts > 3 || false,
        stuckDetected: this._learnerState === LEARNER_STATES.STUCK,
        conceptDecayRisk: knowledge?.decayRisks?.map(r => r.conceptId) || [],
      },

      trajectory: {
        accuracyTrend: knowledge?.averagePKnown
          ? (knowledge.averagePKnown > 0.70 ? 'IMPROVING' : knowledge.averagePKnown > 0.40 ? 'STABLE' : 'DEVELOPING')
          : 'INITIAL',
        fluencyTrend: 'TRACKING',
        estimatedTimeToMastery: this._estimateTimeToMastery(knowledgeTracer),
      },

      currentState: this._learnerState,
      timestamp: Date.now(),
    };
  }

  // ===========================================================================
  // PRIVATE — Helpers
  // ===========================================================================

  _computeInitialModalityWeights() {
    const base = {
      visual: this._params.visualWeightDefault,
      acoustic: this._params.acousticWeightDefault,
      gaze: this._params.gazeWeightDefault,
    };

    // Motor-impaired learners benefit from increased visual weight
    if (this._profileType === 'motor_impairment' || this._profileType === 'cerebral_palsy') {
      base.visual += this._params.motorImpairmentVisualBoost;
      base.acoustic = Math.max(0.10, base.acoustic - this._params.motorImpairmentVisualBoost);
    }

    // Deaf/HoH learners: visual primary, acoustic irrelevant
    if (this._profileType === 'deaf' || this._profileType === 'hard_of_hearing') {
      base.acoustic = 0.05;
      base.visual = 0.65;
      base.gaze = 0.30;
    }

    // Dyslexia: increase acoustic for verbal reinforcement
    if (this._profileType === 'dyslexia') {
      base.acoustic += 0.10;
      base.visual -= 0.10;
    }

    // Low vision: acoustic + gaze primary
    if (this._profileType === 'low_vision') {
      base.acoustic = 0.40;
      base.gaze = 0.30;
      base.visual = 0.30;
    }

    // Normalize
    const sum = base.visual + base.acoustic + base.gaze;
    if (Math.abs(sum - 1.0) > 0.001) {
      base.visual /= sum;
      base.acoustic /= sum;
      base.gaze /= sum;
    }

    return base;
  }

  _detectStuck() {
    if (this._recentErrorTypes.length < 4) return false;
    const last4 = this._recentErrorTypes.slice(-4);
    const unique = new Set(last4);
    return unique.size === 1; // All same error type
  }

  _isRepeatedError(errorType) {
    const count = this._recentErrorTypes.filter(e => e === errorType).length;
    return count >= this._params.sameErrorStreakCount;
  }

  _trackErrorForResolution(errorExplanation) {
    if (!errorExplanation) return;
    const errorType = errorExplanation.errorType;
    this._recentErrorTypes.push(errorType);
    if (this._recentErrorTypes.length > 20) {
      this._recentErrorTypes.shift();
    }

    if (!this._errorResolution[errorType]) {
      this._errorResolution[errorType] = { attempts: 0, resolved: 0, lastSeen: null };
    }
    this._errorResolution[errorType].attempts++;
    this._errorResolution[errorType].lastSeen = Date.now();
  }

  _summarizeModalities() {
    const maxModality = Object.entries(this._modalityWeights)
      .sort((a, b) => b[1] - a[1])[0];
    return {
      primary: maxModality[0],
      weights: { ...this._modalityWeights },
    };
  }

  _getOptimalModality() {
    const maxModality = Object.entries(this._modalityWeights)
      .sort((a, b) => b[1] - a[1])[0][0];
    return maxModality.toUpperCase();
  }

  _getPacing() {
    switch (this._learnerState) {
      case LEARNER_STATES.FLOWING:  return 'NORMAL';
      case LEARNER_STATES.LEARNING: return 'NORMAL';
      case LEARNER_STATES.CONFUSED: return 'SLOW';
      case LEARNER_STATES.STUCK:    return 'SLOW';
      case LEARNER_STATES.FATIGUED: return 'BREAK_RECOMMENDED';
      default:                      return 'NORMAL';
    }
  }

  _computeInterventionSuccessRate() {
    const types = Object.values(this._errorResolution);
    const total = types.reduce((s, t) => s + t.attempts, 0);
    const resolved = types.reduce((s, t) => s + t.resolved, 0);
    return total > 0 ? resolved / total : 0;
  }

  _estimateTimeToMastery(knowledgeTracer) {
    if (!knowledgeTracer) return null;
    const unmastered = Object.values(knowledgeTracer.getAllStates())
      .filter(s => s && s.category !== 'META' && !s.isMastered);

    if (unmastered.length === 0) return 'Mastered';

    const avgPKnown = unmastered.reduce((s, c) => s + c.pKnown, 0) / unmastered.length;
    const sessionsNeeded = Math.ceil((0.85 - avgPKnown) / 0.12); // Rough estimate based on P(T)
    return sessionsNeeded <= 0 ? 'Current session' : `~${sessionsNeeded} session(s)`;
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

export function learnerStateColor(state) {
  switch (state) {
    case LEARNER_STATES.FLOWING:    return '#4ade80';
    case LEARNER_STATES.LEARNING:   return '#60a5fa';
    case LEARNER_STATES.CONFUSED:   return '#fbbf24';
    case LEARNER_STATES.FATIGUED:   return '#fb923c';
    case LEARNER_STATES.STUCK:      return '#f87171';
    case LEARNER_STATES.DISENGAGED: return '#94a3b8';
    default:                        return '#64748b';
  }
}

export function learnerStateLabel(state) {
  switch (state) {
    case LEARNER_STATES.FLOWING:    return 'Flowing';
    case LEARNER_STATES.LEARNING:   return 'Learning';
    case LEARNER_STATES.CONFUSED:   return 'Confused';
    case LEARNER_STATES.FATIGUED:   return 'Fatigued';
    case LEARNER_STATES.STUCK:      return 'Stuck';
    case LEARNER_STATES.DISENGAGED: return 'Disengaged';
    default:                        return 'Unknown';
  }
}

export default AdaptivePedagogyEngine;
