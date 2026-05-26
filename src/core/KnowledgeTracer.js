/**
 * KnowledgeTracer.js — Bayesian Knowledge Tracing (BKT) for Gesture Grammar
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BAYESIAN KNOWLEDGE TRACING (Corbett & Anderson, 1995)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Estimates latent knowledge state P(known) for each grammar concept via
 * Bayesian update on every practice opportunity. Goes beyond simple
 * production-count mastery (GestureMasteryGate) by modeling:
 *
 *   - P(L)    probability the learner knows the concept
 *   - P(T)    probability of learning per opportunity
 *   - P(G)    probability of correct guess when unknown
 *   - P(S)    probability of slip (incorrect despite knowing)
 *   - P(L0)   initial knowledge prior
 *
 * BKT update equations:
 *   Correct:   P(L|correct)   = P(L)*(1-S) / [P(L)*(1-S) + (1-P(L))*G]
 *   Incorrect: P(L|incorrect) = P(L)*S     / [P(L)*S + (1-P(L))*(1-G)]
 *   Learning:  P(L_next)      = P(L|obs) + (1-P(L|obs))*T
 *
 * Response time integration:
 *   Fast correct response (< 500ms) → boosts P(L|correct) (automaticity)
 *   Slow correct response (> 3000ms) → reduces P(L|correct) (deliberation)
 *   Fast incorrect → higher slip probability
 *
 * Persists per-concept state across sessions via localStorage.
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'mlaf_bkt_v1';

/** Default BKT parameter set (calibrated for initial learning) */
const DEFAULT_PARAMS = {
  pKnown0: 0.20,  // Initial knowledge prior
  pLearn:  0.12,  // Probability of learning per opportunity
  pGuess:  0.14,  // Probability of correct guess
  pSlip:   0.08,  // Probability of slip (error despite knowing)
};

/** Automaticity timing thresholds (ms) */
const FAST_RESPONSE_MS = 500;
const SLOW_RESPONSE_MS = 3000;

/** Automaticity boost factor: how much a fast response increases P(L|correct) */
const AUTOMATICITY_BOOST = 0.12;

/** Automaticity penalty: how much a slow response reduces P(L|correct) */
const AUTOMATICITY_PENALTY = 0.06;

/** Decay rate per day for retention forecasting */
const DAILY_DECAY_RATE = 0.03;

/** Probability threshold for mastery */
const MASTERY_THRESHOLD = 0.85;

/** Minimum probability floor (never go to absolute zero) */
const P_FLOOR = 0.01;

/** Maximum probability ceiling (never go to absolute one) */
const P_CEILING = 0.999;

/**
 * Complete concept map for BKT tracking.
 * 
 * 18 gesture concepts (6 subjects + 7 verbs + 6 objects + S-form variants)
 * 4 meta-concepts (grammar rules that span multiple gestures)
 */
const CONCEPT_MAP = {
  // ── Subject gestures ──
  SUBJECT_I:    { category: 'SUBJECT', display: 'I',   stage: 1 },
  SUBJECT_YOU:  { category: 'SUBJECT', display: 'You', stage: 5 },
  SUBJECT_HE:   { category: 'SUBJECT', display: 'He',  stage: 2 },
  SUBJECT_SHE:  { category: 'SUBJECT', display: 'She', stage: 2 },
  SUBJECT_WE:   { category: 'SUBJECT', display: 'We',  stage: 5 },
  SUBJECT_THEY: { category: 'SUBJECT', display: 'They', stage: 5 },

  // ── Base verb gestures ──
  GRAB:  { category: 'VERB', display: 'grab',  sForm: 'GRABS',  transitive: true,  stage: 2 },
  EAT:   { category: 'VERB', display: 'eat',   sForm: 'EATS',   transitive: true,  stage: 2 },
  WANT:  { category: 'VERB', display: 'want',  sForm: 'WANTS',  transitive: true,  stage: 5 },
  DRINK: { category: 'VERB', display: 'drink', sForm: 'DRINKS', transitive: true,  stage: 5 },
  SEE:   { category: 'VERB', display: 'see',   sForm: 'SEES',   transitive: true,  stage: 5 },
  GO:    { category: 'VERB', display: 'go',    sForm: 'GOES',   transitive: false, stage: 5 },
  STOP:  { category: 'VERB', display: 'stop',  sForm: 'STOPS',  transitive: false, stage: 1 },

  // ── S-form verb gestures (tracked separately from base) ──
  GRABS:  { category: 'VERB', display: 'grabs',  baseForm: 'GRAB',  stage: 4 },
  EATS:   { category: 'VERB', display: 'eats',   baseForm: 'EAT',   stage: 4 },
  WANTS:  { category: 'VERB', display: 'wants',  baseForm: 'WANT',  stage: 4 },
  DRINKS: { category: 'VERB', display: 'drinks', baseForm: 'DRINK', stage: 4 },
  SEES:   { category: 'VERB', display: 'sees',   baseForm: 'SEE',   stage: 4 },
  GOES:   { category: 'VERB', display: 'goes',   baseForm: 'GO',    stage: 4 },
  STOPS:  { category: 'VERB', display: 'stops',  baseForm: 'STOP',  stage: 4 },

  // ── Object gestures ──
  APPLE: { category: 'OBJECT', display: 'apple', stage: 3 },
  BALL:  { category: 'OBJECT', display: 'ball',  stage: 3 },
  WATER: { category: 'OBJECT', display: 'water', stage: 5 },
  FOOD:  { category: 'OBJECT', display: 'food',  stage: 5 },
  BOOK:  { category: 'OBJECT', display: 'book',  stage: 5 },
  HOUSE: { category: 'OBJECT', display: 'house', stage: 5 },
};

/**
 * Meta-concepts: higher-order grammar rules that span multiple gestures.
 * Updated whenever any relevant gesture opportunity occurs or a sentence
 * is validated.
 */
const META_CONCEPTS = {
  SV_AGREEMENT: {
    display: 'Subject-Verb Agreement',
    description: 'Matching verb form to subject person/number',
    stage: 4,
  },
  SVO_ORDER: {
    display: 'English SVO Word Order',
    description: 'Subject → Verb → Object sequencing',
    stage: 3,
  },
  TRANSITIVITY: {
    display: 'Verb Transitivity',
    description: 'Whether a verb requires or forbids an object',
    stage: 3,
  },
  S_FORM_MARKING: {
    display: 'Third-Person S-Form Marking',
    description: 'Adding -s/-es for He/She subjects in present tense',
    stage: 4,
  },
};

/** Mastery stage labels */
const MASTERY_STAGES = {
  ACQUIRING:      { min: 0.00, label: 'Acquiring',       color: '#f87171' },
  DEVELOPING:     { min: 0.40, label: 'Developing',      color: '#fbbf24' },
  CONSOLIDATING:  { min: 0.70, label: 'Consolidating',   color: '#60a5fa' },
  MASTERED:       { min: 0.85, label: 'Mastered',        color: '#4ade80' },
  AUTOMATIC:      { min: 0.95, label: 'Automatic',       color: '#22d3ee' },
};

// =============================================================================
// BAYESIAN KNOWLEDGE TRACER
// =============================================================================

export class KnowledgeTracer {
  /**
   * @param {object} config
   * @param {object} [config.params] — BKT parameters {pKnown0, pLearn, pGuess, pSlip}
   * @param {string} [config.storageKey] — localStorage key override
   */
  constructor(config = {}) {
    this._params = { ...DEFAULT_PARAMS, ...(config.params || {}) };
    this._storageKey = config.storageKey || STORAGE_KEY;

    /** Per-concept state: Map<conceptId, ConceptState> */
    this._states = {};

    /** Per-concept opportunity history for computation */
    this._history = {};

    /** Transfer latency tracking: "fromConcept|toconcept" → {sum, count} */
    this._transferData = {};

    /** Session start time */
    this._sessionStart = Date.now();

    // Initialize all concepts
    for (const conceptId of Object.keys(CONCEPT_MAP)) {
      this._initConcept(conceptId);
    }
    for (const conceptId of Object.keys(META_CONCEPTS)) {
      this._initConcept(conceptId);
    }

    // Load persisted state
    this._load();
  }

  // ===========================================================================
  // PUBLIC — Record Practice Opportunity
  // ===========================================================================

  /**
   * Record a practice opportunity for a gesture concept.
   * Called on every gesture lock (successful word addition).
   *
   * @param {string} conceptId — e.g., 'GRAB', 'SUBJECT_I', 'APPLE'
   * @param {boolean} correct — whether the production was correct (always true for locks)
   * @param {object} [context]
   * @param {number} [context.responseTimeMs] — production latency
   * @param {string} [context.tense] — tense zone
   * @returns {ConceptReport}
   */
  recordOpportunity(conceptId, correct, context = {}) {
    const state = this._getOrInit(conceptId);
    if (!state) return null;

    const { pKnown, pGuess, pSlip, pLearn } = state;
    const oldPKnown = pKnown;

    let pKnownGivenObs;
    let effectivePGuess = pGuess;
    let effectivePSlip = pSlip;

    if (correct) {
      // Bayes: P(L | correct) = P(L)*(1-S) / [P(L)*(1-S) + (1-P(L))*G]
      const numerator = pKnown * (1 - effectivePSlip);
      const denominator = numerator + (1 - pKnown) * effectivePGuess;
      pKnownGivenObs = denominator > 0 ? numerator / denominator : pKnown;

      // Automaticity bonus: fast correct responses increase certainty
      const rt = context.responseTimeMs || 0;
      if (rt > 0 && rt < FAST_RESPONSE_MS && pKnownGivenObs > P_FLOOR) {
        pKnownGivenObs = Math.min(
          P_CEILING,
          pKnownGivenObs + AUTOMATICITY_BOOST * (1 - pKnownGivenObs)
        );
      } else if (rt > SLOW_RESPONSE_MS && pKnownGivenObs > P_FLOOR) {
        pKnownGivenObs = Math.max(
          P_FLOOR,
          pKnownGivenObs - AUTOMATICITY_PENALTY * pKnownGivenObs
        );
      }
    } else {
      // Bayes: P(L | incorrect) = P(L)*S / [P(L)*S + (1-P(L))*(1-G)]
      const numerator = pKnown * effectivePSlip;
      const denominator = numerator + (1 - pKnown) * (1 - effectivePGuess);
      pKnownGivenObs = denominator > 0 ? numerator / denominator : Math.max(P_FLOOR, pKnown * 0.8);
    }

    // Learning: P(L_next) = P(L|obs) + (1 - P(L|obs)) * T
    const pKnownNext = pKnownGivenObs + (1 - pKnownGivenObs) * pLearn;

    // Clamp
    state.pKnown = Math.max(P_FLOOR, Math.min(P_CEILING, pKnownNext));
    state.totalAttempts++;
    if (correct) state.correctAttempts++;
    state.lastSeen = Date.now();
    state.lastResponseTime = context.responseTimeMs || null;

    // Track history for transfer learning detection
    this._trackTransfer(conceptId, context);

    // Persist
    this._save();

    return {
      conceptId,
      pKnown: state.pKnown,
      pKnownPrevious: oldPKnown,
      delta: state.pKnown - oldPKnown,
      masteryStage: this._getMasteryStage(state.pKnown),
      totalAttempts: state.totalAttempts,
      correctAttempts: state.correctAttempts,
    };
  }

  /**
   * Record a sentence-level error (agreement, word order, transitivity).
   * Updates the relevant meta-concept.
   *
   * @param {'SV_AGREEMENT'|'SVO_ORDER'|'TRANSITIVITY'|'S_FORM_MARKING'} metaConceptId
   * @param {boolean} correct
   * @returns {ConceptReport|null}
   */
  recordMetaOpportunity(metaConceptId, correct) {
    return this.recordOpportunity(metaConceptId, correct, {});
  }

  /**
   * Record a subject-verb agreement attempt.
   * Updates both the specific S-form concept, the base verb concept,
   * and the SV_AGREEMENT meta-concept.
   *
   * @param {string} subjectId — e.g., 'SUBJECT_HE'
   * @param {string} usedVerbId — e.g., 'GRAB' (what the learner used)
   * @param {string} correctVerbId — e.g., 'GRABS' (what they should have used)
   * @param {boolean} wasCorrect — whether the correct form was used
   * @returns {{ subject: ConceptReport, verb: ConceptReport, agreement: ConceptReport }}
   */
  recordAgreementAttempt(subjectId, usedVerbId, correctVerbId, wasCorrect) {
    // Track the specific verb gesture that was used
    const verbReport = this.recordOpportunity(usedVerbId, wasCorrect);

    // Track the S-form that was needed (records as correct if used correctly)
    const sFormReport = correctVerbId !== usedVerbId
      ? this.recordOpportunity(correctVerbId, wasCorrect)
      : verbReport;

    // Track the meta-concept
    const agreementReport = this.recordMetaOpportunity('SV_AGREEMENT', wasCorrect);

    // Track S_FORM_MARKING meta-concept
    const sFormMarking = wasCorrect || correctVerbId !== usedVerbId
      ? this.recordMetaOpportunity('S_FORM_MARKING', wasCorrect)
      : null;

    return { subjectId, verb: verbReport, sForm: sFormReport, agreement: agreementReport, sFormMarking };
  }

  // ===========================================================================
  // PUBLIC — Knowledge State Queries
  // ===========================================================================

  /**
   * Get the full knowledge state for a concept.
   * @param {string} conceptId
   * @returns {ConceptState|null}
   */
  getKnowledgeState(conceptId) {
    const state = this._states[conceptId];
    if (!state) return null;

    const conceptDef = CONCEPT_MAP[conceptId] || META_CONCEPTS[conceptId];
    return {
      conceptId,
      display: conceptDef?.display || conceptId,
      category: conceptDef?.category || 'META',
      stage: conceptDef?.stage || null,
      pKnown: state.pKnown,
      pGuess: state.pGuess,
      pSlip: state.pSlip,
      pLearn: state.pLearn,
      totalAttempts: state.totalAttempts,
      correctAttempts: state.correctAttempts,
      accuracy: state.totalAttempts > 0
        ? state.correctAttempts / state.totalAttempts
        : 0,
      lastSeen: state.lastSeen,
      masteryStage: this._getMasteryStage(state.pKnown),
      isMastered: this._isMastered(state.pKnown),
    };
  }

  /**
   * Get all concept states.
   * @returns {Map<string, ConceptState>}
   */
  getAllStates() {
    const result = {};
    for (const id of Object.keys(this._states)) {
      result[id] = this.getKnowledgeState(id);
    }
    return result;
  }

  /**
   * Get mastery estimate with predicted attempts to reach mastery.
   * @param {string} conceptId
   * @returns {{ pKnown: number, isMastered: boolean, attemptsNeeded: number|null }}
   */
  getMasteryEstimate(conceptId) {
    const state = this._states[conceptId];
    if (!state) return { pKnown: 0, isMastered: false, attemptsNeeded: null };

    const pKnown = state.pKnown;
    if (this._isMastered(pKnown)) {
      return { pKnown, isMastered: true, attemptsNeeded: 0 };
    }

    // Simulate: how many more correct attempts at current P(T)?
    const pLearn = state.pLearn;
    let simulated = pKnown;
    let attempts = 0;
    const MAX_SIM = 100;
    while (simulated < MASTERY_THRESHOLD && attempts < MAX_SIM) {
      simulated += (1 - simulated) * pLearn;
      attempts++;
    }
    return {
      pKnown,
      isMastered: false,
      attemptsNeeded: attempts < MAX_SIM ? attempts : null,
    };
  }

  // ===========================================================================
  // PUBLIC — Retention & Transfer
  // ===========================================================================

  /**
   * Forecast retention: estimate P(known) after N days without practice.
   * Uses exponential decay model.
   *
   * @param {string} conceptId
   * @param {number} daysFromNow
   * @returns {{ pKnownProjected: number, needsReview: boolean, recommendedReviewDate: number }}
   */
  getRetentionForecast(conceptId, daysFromNow = 1) {
    const state = this._states[conceptId];
    if (!state) return { pKnownProjected: 0, needsReview: true, recommendedReviewDate: 0 };

    const pKnown = state.pKnown;
    const masteryStage = this._getMasteryStage(pKnown);

    // Decay rate depends on mastery stage (well-learned decays slower)
    const stageDecayRates = {
      AUTOMATIC:     0.01,
      MASTERED:      0.02,
      CONSOLIDATING: 0.04,
      DEVELOPING:    0.06,
      ACQUIRING:     0.08,
    };
    const decayRate = stageDecayRates[masteryStage] || DAILY_DECAY_RATE;
    const pProjected = Math.max(P_FLOOR, pKnown * Math.exp(-decayRate * daysFromNow));

    // When will it drop below 0.7? (consolidation boundary)
    const daysToReview = decayRate > 0
      ? Math.log(pKnown / 0.7) / decayRate
      : Infinity;
    const recommendedReviewDate = Date.now() + Math.min(daysToReview * 86400000, 7 * 86400000);

    return {
      pKnownProjected: pProjected,
      needsReview: pProjected < 0.7,
      recommendedReviewDate: Math.round(recommendedReviewDate),
      masteryStage,
      decayRate,
    };
  }

  /**
   * Get transfer latency: time between mastering concept A and mastering concept B.
   * @param {string} fromConcept
   * @param {string} toConcept
   * @returns {{ hours: number|null, evidenceCount: number }}
   */
  getTransferLatency(fromConcept, toConcept) {
    const key = `${fromConcept}|${toConcept}`;
    const data = this._transferData[key];
    if (!data || data.count === 0) return { hours: null, evidenceCount: 0 };

    return {
      hours: Math.round(data.sum / data.count / 3600000 * 10) / 10,
      evidenceCount: data.count,
    };
  }

  // ===========================================================================
  // PUBLIC — Reports
  // ===========================================================================

  /**
   * Get a comprehensive knowledge report for all concepts.
   * @returns {KnowledgeReport}
   */
  getOverallReport() {
    const all = this.getAllStates();
    const entries = Object.values(all).filter(e => e !== null);

    const gestureConcepts = entries.filter(e => e.category !== 'META');
    const metaConcepts = entries.filter(e => e.category === 'META');

    // Average P(known) across gesture concepts
    const avgPKnown = gestureConcepts.length > 0
      ? gestureConcepts.reduce((s, e) => s + e.pKnown, 0) / gestureConcepts.length
      : 0;

    // Sorted by pKnown (weakest first)
    const sorted = [...gestureConcepts].sort((a, b) => a.pKnown - b.pKnown);
    const weakest = sorted.slice(0, 5);
    const strongest = sorted.slice(-5).reverse();

    // Concepts ready for next stage
    const mastered = gestureConcepts.filter(e => e.isMastered);

    // Concepts at risk of decay
    const decayRisks = gestureConcepts
      .map(e => ({ ...e, forecast: this.getRetentionForecast(e.conceptId) }))
      .filter(e => e.forecast.needsReview && e.pKnown > 0)
      .sort((a, b) => a.forecast.pKnownProjected - b.forecast.pKnownProjected)
      .slice(0, 5);

    // Stage completion
    const stageCompletion = {};
    for (const e of gestureConcepts) {
      if (e.stage) {
        if (!stageCompletion[e.stage]) stageCompletion[e.stage] = { total: 0, mastered: 0, concepts: [] };
        stageCompletion[e.stage].total++;
        if (e.isMastered) stageCompletion[e.stage].mastered++;
        stageCompletion[e.stage].concepts.push({
          conceptId: e.conceptId,
          pKnown: e.pKnown,
          masteryStage: e.masteryStage,
        });
      }
    }

    return {
      totalConcepts: gestureConcepts.length,
      totalMetaConcepts: metaConcepts.length,
      averagePKnown: Math.round(avgPKnown * 1000) / 1000,
      masteryCount: mastered.length,
      weakestConcepts: weakest.map(e => ({ conceptId: e.conceptId, pKnown: e.pKnown, stage: e.stage })),
      strongestConcepts: strongest.map(e => ({ conceptId: e.conceptId, pKnown: e.pKnown, stage: e.stage })),
      decayRisks: decayRisks.map(e => ({
        conceptId: e.conceptId,
        pKnown: e.pKnown,
        pKnownProjected: e.forecast?.pKnownProjected || 0,
        needsReview: e.forecast?.needsReview || false,
      })),
      stageCompletion,
      conceptsReadyForNextStage: this._getConceptsReadyForPromotion(gestureConcepts),
      metaConcepts: metaConcepts.map(e => ({
        conceptId: e.conceptId,
        pKnown: e.pKnown,
        isMastered: e.isMastered,
      })),
      timestamp: Date.now(),
    };
  }

  /**
   * Get a concept decay report: which concepts need review soonest?
   * @returns {Array}
   */
  getConceptDecayReport() {
    const all = this.getAllStates();
    const entries = Object.values(all).filter(e => e && e.category !== 'META');

    const now = Date.now();
    return entries
      .map(e => ({
        ...e,
        daysSinceLastPractice: e.lastSeen
          ? Math.round((now - e.lastSeen) / 86400000 * 10) / 10
          : Infinity,
        retention: this.getRetentionForecast(e.conceptId, 1),
      }))
      .sort((a, b) => a.retention.pKnownProjected - b.retention.pKnownProjected);
  }

  /**
   * Get the set of concepts that the learner has mastered but the curriculum
   * hasn't yet promoted. Used by AdaptivePedagogyEngine for stage advancement.
   */
  getConceptsReadyForPromotion() {
    const report = this.getOverallReport();
    return report.conceptsReadyForNextStage;
  }

  /**
   * Get current parameters.
   * @returns {object}
   */
  getParameters() {
    return { ...this._params };
  }

  /**
   * Get a serializable snapshot for session logging.
   * @returns {object}
   */
  getSnapshot() {
    const states = {};
    for (const [id, state] of Object.entries(this._states)) {
      states[id] = {
        pKnown: state.pKnown,
        totalAttempts: state.totalAttempts,
        correctAttempts: state.correctAttempts,
        masteryStage: this._getMasteryStage(state.pKnown),
      };
    }
    return {
      parameters: { ...this._params },
      states,
      timestamp: Date.now(),
    };
  }

  /**
   * Get a descriptor for logging/export.
   * @returns {object}
   */
  toDescriptor() {
    const report = this.getOverallReport();
    return {
      model: 'BKT (Corbett & Anderson, 1995)',
      parameters: this._params,
      totalConcepts: report.totalConcepts,
      averagePKnown: report.averagePKnown,
      masteryCount: report.masteryCount,
      conceptsTracked: Object.keys(this._states).length,
    };
  }

  // ===========================================================================
  // PUBLIC — Lifecycle
  // ===========================================================================

  /**
   * Reset all state to initial priors.
   */
  reset() {
    this._states = {};
    this._history = {};
    this._transferData = {};
    for (const conceptId of Object.keys(CONCEPT_MAP)) {
      this._initConcept(conceptId);
    }
    for (const conceptId of Object.keys(META_CONCEPTS)) {
      this._initConcept(conceptId);
    }
    this._sessionStart = Date.now();
    this._save();
  }

  // ===========================================================================
  // PRIVATE — Concept Initialization
  // ===========================================================================

  _initConcept(conceptId) {
    this._states[conceptId] = {
      pKnown: this._params.pKnown0,
      pGuess: this._params.pGuess,
      pSlip: this._params.pSlip,
      pLearn: this._params.pLearn,
      totalAttempts: 0,
      correctAttempts: 0,
      lastSeen: null,
      lastResponseTime: null,
      firstSeen: Date.now(),
    };
    this._history[conceptId] = [];
  }

  _getOrInit(conceptId) {
    if (!this._states[conceptId]) {
      this._initConcept(conceptId);
    }
    return this._states[conceptId];
  }

  // ===========================================================================
  // PRIVATE — Helpers
  // ===========================================================================

  _getMasteryStage(pKnown) {
    const stages = Object.entries(MASTERY_STAGES)
      .sort((a, b) => b[1].min - a[1].min);
    for (const [key, def] of stages) {
      if (pKnown >= def.min) return key;
    }
    return 'ACQUIRING';
  }

  _isMastered(pKnown) {
    return pKnown >= MASTERY_THRESHOLD;
  }

  _getConceptsReadyForPromotion(gestureConcepts) {
    // Concepts that are mastered but belong to a stage beyond what the
    // curriculum has formally unlocked.
    const masteredSet = new Set(
      gestureConcepts.filter(e => e.isMastered).map(e => e.conceptId)
    );

    // Find the highest stage with at least 1 mastered concept
    let maxMasteredStage = 0;
    for (const e of gestureConcepts) {
      if (e.isMastered && e.stage && e.stage > maxMasteredStage) {
        maxMasteredStage = e.stage;
      }
    }

    return {
      masteredConceptIds: [...masteredSet],
      maxMasteredStage,
      totalMastered: masteredSet.size,
    };
  }

  _trackTransfer(conceptId, context) {
    if (!context._previousConcept) return;

    const prevConcept = context._previousConcept;
    const key = `${prevConcept}|${conceptId}`;
    if (!this._transferData[key]) {
      this._transferData[key] = { sum: 0, count: 0 };
    }

    // Track time between productions of different concepts
    const state = this._states[conceptId];
    const prevState = this._states[prevConcept];
    if (state?.lastSeen && prevState?.lastSeen) {
      const delta = state.lastSeen - prevState.lastSeen;
      if (delta > 0 && delta < 600000) { // Within 10 minutes
        this._transferData[key].sum += delta;
        this._transferData[key].count++;
      }
    }
  }

  // ===========================================================================
  // PRIVATE — Persistence
  // ===========================================================================

  _save() {
    try {
      const data = {};
      for (const [id, state] of Object.entries(this._states)) {
        data[id] = { ...state };
      }
      const payload = {
        states: data,
        transferData: this._transferData,
        savedAt: Date.now(),
      };
      localStorage.setItem(this._storageKey, JSON.stringify(payload));
    } catch { /* ignore */ }
  }

  _load() {
    try {
      const raw = localStorage.getItem(this._storageKey);
      if (!raw) return;
      const payload = JSON.parse(raw);
      if (payload.states) {
        for (const [id, savedState] of Object.entries(payload.states)) {
          if (this._states[id]) {
            Object.assign(this._states[id], savedState);
          }
        }
      }
      if (payload.transferData) {
        this._transferData = payload.transferData;
      }
    } catch { /* ignore — reset to defaults */ }
  }
}

// =============================================================================
// EXPORT CONSTANTS
// =============================================================================

export { CONCEPT_MAP, META_CONCEPTS, MASTERY_STAGES, MASTERY_THRESHOLD, DEFAULT_PARAMS };

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

export function masteryStageColor(stage) {
  return MASTERY_STAGES[stage]?.color || '#64748b';
}

export function masteryStageLabel(stage) {
  return MASTERY_STAGES[stage]?.label || 'Unknown';
}

export function knowledgeProbabilityColor(pKnown) {
  if (pKnown >= 0.95) return '#22d3ee';
  if (pKnown >= 0.85) return '#4ade80';
  if (pKnown >= 0.70) return '#60a5fa';
  if (pKnown >= 0.40) return '#fbbf24';
  return '#f87171';
}

export function formatPKnown(pKnown) {
  if (pKnown >= 0.95) return '★★★';
  if (pKnown >= 0.85) return '★★☆';
  if (pKnown >= 0.70) return '★☆☆';
  return '···';
}

export default KnowledgeTracer;
