/**
 * ExplainabilityEngine.js — Human-Readable Explanation Generator
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Converts structured error diagnosis from the GraphRAG abductive layer
 * into natural-language explanations accessible to educators, therapists,
 * researchers, and licensees. No black-box AI — every explanation is
 * traceable to specific graph nodes and edges.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * EXPLANATION ARCHITECTURE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Each explanation has 5 components:
 *   narrative     — plain-English description of what happened
 *   rootCause     — the underlying linguistic/cognitive reason
 *   remediation   — what the curriculum or system did/will do about it
 *   l1Transfer    — whether L1 (ISL) transfer is involved
 *   systemAction  — what the adaptive system changed in response
 *
 * All five are constructed by traversing the knowledge graph's structured
 * error → cause → remediation edges, filled with concrete data from the
 * session's BKT states and error history.
 */

import { MASTERY_STAGES } from './KnowledgeTracer.js';

// =============================================================================
// ERROR TYPE CONSTANTS
// =============================================================================

export const ERROR_TYPES = {
  TYPE_MISMATCH:       'TYPE_MISMATCH',
  WRONG_VERB_FORM:     'WRONG_VERB_FORM',
  WRONG_WORD_ORDER:    'WRONG_WORD_ORDER',
  MISSING_OBJECT:      'MISSING_OBJECT',
  EXTRA_OBJECT:        'EXTRA_OBJECT',
  GESTURE_CONFUSION:   'GESTURE_CONFUSION',
  AGREEMENT_VIOLATION: 'AGREEMENT_VIOLATION',
  SENTENCE_REJECTED:   'SENTENCE_REJECTED',
};

const SEVERITY = {
  INFO:     'INFO',
  WARNING:  'WARNING',
  CRITICAL: 'CRITICAL',
};

// =============================================================================
// HUMAN-READABLE LABELS FOR CONCEPTS
// =============================================================================

const CONCEPT_LABELS = {
  SUBJECT_I:    'I',
  SUBJECT_YOU:  'You',
  SUBJECT_HE:   'He',
  SUBJECT_SHE:  'She',
  SUBJECT_WE:   'We',
  SUBJECT_THEY: 'They',
  GRAB:  'grab',  GRABS:  'grabs',
  EAT:   'eat',   EATS:   'eats',
  WANT:  'want',  WANTS:  'wants',
  DRINK: 'drink', DRINKS: 'drinks',
  SEE:   'see',   SEES:   'sees',
  GO:    'go',    GOES:   'goes',
  STOP:  'stop',  STOPS:  'stops',
  APPLE: 'apple', BALL: 'ball', WATER: 'water',
  FOOD:  'food',  BOOK: 'book', HOUSE: 'house',
};

const CATEGORY_LABELS = {
  SUBJECT: 'subject',
  VERB:    'verb',
  OBJECT:  'object',
};

function labelFor(conceptId) {
  return CONCEPT_LABELS[conceptId] || conceptId;
}

// =============================================================================
// EXPLAINABILITY ENGINE
// =============================================================================

export class ExplainabilityEngine {
  /**
   * @param {object} graphRAG — the GraphRAG instance for abductive diagnosis
   * @param {object} knowledgeTracer — the KnowledgeTracer instance for concept state
   */
  constructor(graphRAG, knowledgeTracer) {
    this.graphRAG = graphRAG || null;
    this.knowledgeTracer = knowledgeTracer || null;

    /** Accumulated explanations for session narrative */
    this._explanations = [];

    /** Session-level counters for narrative synthesis */
    this._sessionCounters = {
      totalErrors: 0,
      errorsByType: {},
      islTransferCount: 0,
      agreementErrors: 0,
      confusionPairs: {},
    };
  }

  // ===========================================================================
  // PUBLIC — Per-Error Explanation
  // ===========================================================================

  /**
   * Generate a human-readable explanation for an error event.
   *
   * @param {string} errorType — one of ERROR_TYPES
   * @param {object} context — error context from SandboxMode
   * @param {string} [context.recognizedGesture] — what the system saw
   * @param {string} [context.intendedGesture] — what was intended
   * @param {string[]} [context.sentenceTokens] — current sentence tokens
   * @param {Array} [context.sentenceWords] — current sentence word objects
   * @param {string} [context.subjectId] — subject grammar ID
   * @param {string} [context.verbId] — verb grammar ID
   * @param {object} [context.validationResult] — from GrammarEngine
   * @returns {Explanation}
   */
  explainError(errorType, context = {}) {
    // Run abductive diagnosis through GraphRAG
    const diagnosis = this.graphRAG
      ? this.graphRAG.diagnoseError(errorType, context)
      : { causes: [], remediation: [], interferencePatterns: [] };

    // Build explanation based on error type
    let explanation;
    switch (errorType) {
      case ERROR_TYPES.TYPE_MISMATCH:
        explanation = this._explainTypeMismatch(context, diagnosis);
        break;
      case ERROR_TYPES.WRONG_VERB_FORM:
      case ERROR_TYPES.AGREEMENT_VIOLATION:
        explanation = this._explainAgreementError(context, diagnosis);
        break;
      case ERROR_TYPES.WRONG_WORD_ORDER:
        explanation = this._explainWordOrder(context, diagnosis);
        break;
      case ERROR_TYPES.MISSING_OBJECT:
        explanation = this._explainMissingObject(context, diagnosis);
        break;
      case ERROR_TYPES.EXTRA_OBJECT:
        explanation = this._explainExtraObject(context, diagnosis);
        break;
      case ERROR_TYPES.GESTURE_CONFUSION:
        explanation = this._explainGestureConfusion(context, diagnosis);
        break;
      case ERROR_TYPES.SENTENCE_REJECTED:
        explanation = this._explainSentenceRejected(context, diagnosis);
        break;
      default:
        explanation = this._explainGeneric(errorType, context, diagnosis);
    }

    // Enrich with knowledge tracing data if available
    if (this.knowledgeTracer) {
      explanation = this._enrichWithKnowledgeState(explanation, context);
    }

    // Track for session narrative
    this._explanations.push(explanation);
    this._updateSessionCounters(explanation);

    return explanation;
  }

  // ===========================================================================
  // PUBLIC — Session Narrative
  // ===========================================================================

  /**
   * Generate an end-of-session narrative synthesizing all errors, progress,
   * and recommendations into a coherent report for educators/researchers.
   *
   * @param {Array} [sessionEvents] — from SessionDataLogger
   * @param {object} [knowledgeReport] — from KnowledgeTracer.getOverallReport()
   * @param {object} [masteryReport] — from GestureMasteryGate
   * @returns {SessionNarrative}
   */
  generateSessionNarrative(knowledgeReport = null, masteryReport = null) {
    const errors = this._explanations;
    const counters = this._sessionCounters;

    // Headline: high-level summary
    const headline = this._generateHeadline(counters, knowledgeReport);

    // Strengths: what went well
    const strengths = this._identifyStrengths(knowledgeReport, masteryReport);

    // Challenges: what needs work
    const challenges = this._identifyChallenges(counters, knowledgeReport, errors);

    // ISL transfer summary
    const islSummary = counters.islTransferCount > 0
      ? `ISL (Indian Sign Language) transfer detected in ${counters.islTransferCount} sentence${counters.islTransferCount !== 1 ? 's' : ''}. `
        + `This is expected for deaf/HoH learners and the system adaptively adjusts spatial-zone dwell times to reinforce English SVO word order.`
      : null;

    // Agreement summary
    const agreementSummary = counters.agreementErrors > 0
      ? `${counters.agreementErrors} subject-verb agreement error${counters.agreementErrors !== 1 ? 's' : ''} detected. `
        + `Third-person singular S-form marking is an English-specific morphological feature that learners typically acquire in Stage 4.`
      : null;

    // Recommendations
    const recommendations = this._generateRecommendations(counters, knowledgeReport, masteryReport, errors);

    // Progress summary
    const progressSummary = this._generateProgressSummary(knowledgeReport, masteryReport);

    return {
      headline,
      strengths,
      challenges,
      islSummary,
      agreementSummary,
      recommendations,
      progressSummary,
      errorCount: counters.totalErrors,
      errorsByType: counters.errorsByType,
      topExplanations: errors.slice(-5), // Last 5 explanations for detail
      generatedAt: Date.now(),
    };
  }

  /**
   * Get the most recent explanations (for live UI).
   * @param {number} [count=3]
   * @returns {Explanation[]}
   */
  getRecentExplanations(count = 3) {
    return this._explanations.slice(-count);
  }

  /**
   * Get all accumulated explanations for this session.
   * @returns {Explanation[]}
   */
  getAllExplanations() {
    return [...this._explanations];
  }

  /**
   * Get the learner profile summary.
   * @param {Array} [sessionHistory] — from SessionDataLogger
   * @param {object} [knowledgeReport] — from KnowledgeTracer
   * @returns {LearnerProfileSummary}
   */
  getLearnerProfile(sessionHistory = [], knowledgeReport = null) {
    const profile = {
      sessionsCompleted: sessionHistory.length || 0,
      knowledge: knowledgeReport
        ? `${Math.round(knowledgeReport.averagePKnown * 100)}% average concept mastery`
        : 'No knowledge data yet',
    };

    // Primary challenge area
    if (knowledgeReport?.weakestConcepts?.length > 0) {
      const weakest = knowledgeReport.weakestConcepts[0];
      profile.primaryChallenge = {
        concept: labelFor(weakest.conceptId),
        masteryLevel: Math.round(weakest.pKnown * 100),
      };
    }

    // ISL inference level
    if (this._sessionCounters.islTransferCount > 5) {
      profile.islInfluenceLevel = 'Significant — ISL-informed pedagogy recommended';
    } else if (this._sessionCounters.islTransferCount > 0) {
      profile.islInfluenceLevel = 'Occasional — monitor for ISL transfer patterns';
    } else {
      profile.islInfluenceLevel = 'Minimal or not assessed';
    }

    // Learning rate estimate
    if (sessionHistory.length >= 2) {
      profile.learningRate = this._estimateLearningRate(sessionHistory);
    }

    return profile;
  }

  /**
   * Reset accumulated session data.
   */
  reset() {
    this._explanations = [];
    this._sessionCounters = {
      totalErrors: 0,
      errorsByType: {},
      islTransferCount: 0,
      agreementErrors: 0,
      confusionPairs: {},
    };
  }

  // ===========================================================================
  // PRIVATE — Error-Type Specific Explanations
  // ===========================================================================

  _explainTypeMismatch(context, diagnosis) {
    const gesture = context.recognizedGesture || 'Unknown';
    const tokens = context.sentenceTokens || [];
    const sentenceDisplay = tokens.map(t => labelFor(t)).join(' · ') || '(empty)';

    // Determine what category was expected
    const expectedCategory = this._inferExpectedCategory(tokens);
    const gestureCategory = this._inferGestureCategory(gesture);

    const narrative = gestureCategory && expectedCategory
      ? `${labelFor(gesture)} (${gestureCategory.toLowerCase()}) was produced but the sentence expected a ${expectedCategory.toLowerCase()} at position ${tokens.length + 1}. ` +
        `The sentence currently reads: "${sentenceDisplay}".`
      : `${labelFor(gesture)} was produced but is not grammatically valid at this position. ` +
        `Current sentence: "${sentenceDisplay}".`;

    let remediation = '';
    if (diagnosis.remediation?.length > 0) {
      const stage = diagnosis.remediation[0];
      remediation = `Curriculum Stage ${stage.stage || '?'}: ${stage.label || 'Review'} — ${stage.reason || 'practice the expected gesture category.'}`;
    } else if (expectedCategory) {
      remediation = `Try producing a ${expectedCategory.toLowerCase()} gesture instead.`;
    }

    return {
      errorType: ERROR_TYPES.TYPE_MISMATCH,
      narrative,
      rootCause: diagnosis.causes?.[0]?.description ||
        `${CATEGORY_LABELS[gestureCategory] || 'gesture'} placed in wrong syntactic slot`,
      remediation: remediation || 'Review which gestures belong in each sentence position.',
      l1Transfer: null,
      systemAction: diagnosis.interferencePatterns?.length > 0
        ? 'Spatial zone lock times adjusted to guide correct category selection.'
        : 'Confidence thresholds normalized for this gesture.',
      severity: SEVERITY.WARNING,
      context: {
        gesture: labelFor(gesture),
        gestureCategory,
        expectedCategory,
        currentSentence: sentenceDisplay,
      },
    };
  }

  _explainAgreementError(context, diagnosis) {
    const subject = context.subjectId || context.sentenceTokens?.[0] || 'subject';
    const verb = context.verbId || context.recognizedGesture || 'verb';
    const subjectLabel = labelFor(subject);
    const verbLabel = labelFor(verb);

    // Figure out the correct form
    let correctForm = verb;
    if (verb.endsWith('S') && !verb.endsWith('SS')) {
      correctForm = verb.slice(0, -1);
    } else if (!verb.endsWith('S')) {
      correctForm = verb + 'S';
    }

    const person = ['SUBJECT_HE', 'SUBJECT_SHE'].includes(subject) ? 'third-person singular' : 'non-third-person';
    const rule = ['SUBJECT_HE', 'SUBJECT_SHE'].includes(subject)
      ? 'requires the S-form'
      : 'requires the base form';

    const narrative = diagnosis.causes?.[0]?.rule
      ? `${subjectLabel} (${person} subject) ${rule} of the verb. ` +
        `The learner produced "${verbLabel}" but the correct form is "${labelFor(correctForm)}".`
      : `Subject-verb agreement error: "${subjectLabel} ${verbLabel}" — ` +
        `${subjectLabel} ${rule} "${labelFor(correctForm)}".`;

    // Link to knowledge tracing if available
    const sFormConcept = verb.endsWith('S') ? verb : verb + 'S';
    const knowledgeNote = this.knowledgeTracer
      ? this._knowledgeNote(sFormConcept)
      : '';

    const remediation = `Practice S-form marking with ${subjectLabel}: "${subjectLabel} ${labelFor(correctForm)}". ` +
      `This skill is taught in Stage 4 (Subject-Verb Agreement) of the curriculum.` +
      knowledgeNote;

    return {
      errorType: ERROR_TYPES.AGREEMENT_VIOLATION,
      narrative,
      rootCause: diagnosis.causes?.[0]?.rule ||
        'Third-person singular S-form agreement rule not yet internalized',
      remediation,
      l1Transfer: null, // Agreement errors are morphological, not word-order transfer
      systemAction: `Confidence threshold increased for "${verbLabel}" to require stronger evidence. Prior P(S) reduced in Bayesian fusion.`,
      severity: SEVERITY.WARNING,
      context: {
        subject: subjectLabel,
        verbUsed: verbLabel,
        verbCorrect: labelFor(correctForm),
        person,
      },
    };
  }

  _explainWordOrder(context, diagnosis) {
    const tokens = context.sentenceTokens || [];
    const sentenceDisplay = tokens.map(t => labelFor(t)).join(' · ');

    // Detect the specific pattern
    const isSOV = tokens.length >= 3 &&
      this._isType(tokens[0], 'SUBJECT') &&
      this._isType(tokens[1], 'OBJECT');
    const isTopicFronting = tokens.length >= 2 && this._isType(tokens[0], 'OBJECT');

    let narrative, l1Transfer;

    if (isSOV) {
      narrative = `ISL Subject-Object-Verb (SOV) word order detected: "${sentenceDisplay}". ` +
        `English requires Subject-Verb-Object (SVO) order. The learner is transferring their ISL word order structure.`;
      l1Transfer = 'ISL canonical SOV order transferred to English. ISL places the verb at the end; English places it between subject and object.';
    } else if (isTopicFronting) {
      narrative = `ISL topic fronting detected: the object appears before the subject in "${sentenceDisplay}". ` +
        `ISL is a topic-prominent language where the discourse topic comes first. English is subject-prominent.`;
      l1Transfer = 'ISL topic fronting: objects are placed first for topicalization, a common ISL discourse structure.';
    } else {
      narrative = `Incorrect word order in: "${sentenceDisplay}". ` +
        `Expected SVO order: Subject → Verb → Object.`;
    }

    let remediation = '';
    if (diagnosis.remediation?.length > 0) {
      remediation = diagnosis.remediation
        .map(r => r.strategy || r.explanation || 'Reorder to SVO')
        .join('. ');
    } else {
      remediation = 'Start with the subject, then verb, then object. Try: "I grab apple".';
    }

    return {
      errorType: ERROR_TYPES.WRONG_WORD_ORDER,
      narrative,
      rootCause: l1Transfer || 'Non-standard word order',
      remediation,
      l1Transfer: l1Transfer || null,
      systemAction: 'Spatial-zone lock multiplier increased to enforce sequential SVO construction. Verb-zone dwell time extended.',
      severity: SEVERITY.WARNING,
      context: {
        currentSentence: sentenceDisplay,
        pattern: isSOV ? 'SOV' : isTopicFronting ? 'TOPIC_FRONTING' : 'NON_SVO',
      },
    };
  }

  _explainMissingObject(context, diagnosis) {
    const tokens = context.sentenceTokens || [];
    const sentenceDisplay = tokens.map(t => labelFor(t)).join(' · ');
    const verbToken = tokens.find(t => this._isVerb(t));

    let narrative = `Transitive verb used without an object: "${sentenceDisplay}". ` +
      `In English, transitive verbs like "${labelFor(verbToken || 'grab')}" require an explicit object.`;

    let l1Transfer = null;
    if (diagnosis.interferencePatterns?.length > 0) {
      l1Transfer = 'ISL pro-drop: objects are contextually implied from the signing environment and may be omitted. English requires explicit objects.';
      narrative += ' This pattern mirrors ISL where objects can be dropped when contextually obvious.';
    }

    return {
      errorType: ERROR_TYPES.MISSING_OBJECT,
      narrative,
      rootCause: l1Transfer || 'Transitive verb missing its required object',
      remediation: `Add an object noun after "${labelFor(verbToken || 'the verb')}" — for example: "${sentenceDisplay} ${labelFor('APPLE')}".`,
      l1Transfer,
      systemAction: 'Object-zone spatial cue intensified. Lock window extended for object gestures to prompt the learner to complete the sentence.',
      severity: SEVERITY.WARNING,
      context: {
        currentSentence: sentenceDisplay,
        verbToken: labelFor(verbToken),
      },
    };
  }

  _explainExtraObject(context) {
    const tokens = context.sentenceTokens || [];
    const sentenceDisplay = tokens.map(t => labelFor(t)).join(' · ');

    const narrative = `Intransitive verb used with an object: "${sentenceDisplay}". ` +
      `Verbs like "stop" and "go" do not take direct objects in English.`;

    return {
      errorType: ERROR_TYPES.EXTRA_OBJECT,
      narrative,
      rootCause: 'Intransitive verb incorrectly used with an object',
      remediation: 'Remove the object after the intransitive verb, or use a transitive verb instead.',
      l1Transfer: null,
      systemAction: 'Intransitive verb flag raised — object gestures gated when following intransitive verbs.',
      severity: SEVERITY.WARNING,
      context: { currentSentence: sentenceDisplay },
    };
  }

  _explainGestureConfusion(context, diagnosis) {
    const intended = context.intendedGesture || 'intended gesture';
    const recognized = context.recognizedGesture || 'recognized gesture';

    let distinguishingFeature = '';
    if (diagnosis.causes?.length > 0) {
      distinguishingFeature = diagnosis.causes[0].distinguishing_feature || '';
    }

    const narrative = `Gesture confusion: "${labelFor(recognized)}" was recognized but "${labelFor(intended)}" was likely intended. ` +
      (distinguishingFeature
        ? `These gestures are visually similar — ${distinguishingFeature}.`
        : 'These gestures share similar hand configurations.');

    return {
      errorType: ERROR_TYPES.GESTURE_CONFUSION,
      narrative,
      rootCause: `Visual similarity between "${labelFor(intended)}" and "${labelFor(recognized)}" gestures`,
      remediation: `Practice both gestures with deliberate, slow hand positioning. The system will show contrastive visual feedback.`,
      l1Transfer: null,
      systemAction: `Confidence threshold raised for both "${labelFor(intended)}" and "${labelFor(recognized)}" to reduce confusion.`,
      severity: SEVERITY.WARNING,
      context: {
        intended: labelFor(intended),
        recognized: labelFor(recognized),
        distinguishingFeature,
      },
    };
  }

  _explainSentenceRejected(context, diagnosis) {
    const sentenceDisplay = (context.sentenceTokens || []).map(t => labelFor(t)).join(' · ');
    const validation = context.validationResult || {};

    let reason = 'The sentence does not follow standard English grammar rules.';
    if (validation.error) {
      reason = validation.error;
    } else if (diagnosis.causes?.length > 0) {
      reason = diagnosis.causes[0].description || reason;
    }

    return {
      errorType: ERROR_TYPES.SENTENCE_REJECTED,
      narrative: `The sentence "${sentenceDisplay}" was rejected: ${reason}`,
      rootCause: diagnosis.causes?.[0]?.description || 'Grammatical violation',
      remediation: diagnosis.remediation?.length > 0
        ? diagnosis.remediation.map(r => r.strategy || r.explanation).join('. ')
        : 'Review sentence structure and try again.',
      l1Transfer: diagnosis.interferencePatterns?.length > 0
        ? `ISL transfer detected: ${diagnosis.interferencePatterns.map(p => p.title).join(', ')}`
        : null,
      systemAction: 'Sentence rejected. Adaptive thresholds maintained for current curriculum stage.',
      severity: SEVERITY.INFO,
      context: { sentence: sentenceDisplay, reason },
    };
  }

  _explainGeneric(errorType, context, diagnosis) {
    return {
      errorType,
      narrative: `An error occurred: ${errorType}. Context: ${JSON.stringify(context).slice(0, 200)}`,
      rootCause: diagnosis.causes?.[0]?.description || 'Unknown',
      remediation: diagnosis.remediation?.[0]?.strategy || 'Review and retry.',
      l1Transfer: null,
      systemAction: 'Monitoring for recurrence.',
      severity: SEVERITY.INFO,
      context,
    };
  }

  // ===========================================================================
  // PRIVATE — Knowledge State Enrichment
  // ===========================================================================

  _enrichWithKnowledgeState(explanation, context) {
    if (!this.knowledgeTracer) return explanation;

    const relevantConcept = context.recognizedGesture ||
      context.intendedGesture ||
      context.verbId ||
      context.subjectId;

    if (relevantConcept) {
      const state = this.knowledgeTracer.getKnowledgeState(relevantConcept);
      if (state) {
        const stage = MASTERY_STAGES[state.masteryStage];
        explanation.knowledgeState = {
          concept: labelFor(relevantConcept),
          pKnown: state.pKnown,
          masteryStage: state.masteryStage,
          stageLabel: stage?.label || 'Unknown',
        };

        // Add knowledge-level severity adjustment
        if (state.pKnown < 0.30 && explanation.severity === SEVERITY.WARNING) {
          explanation.severity = SEVERITY.CRITICAL;
          explanation.narrative += ` [This concept is in early acquisition — the system will increase scaffolding.]`;
        }
      }
    }

    return explanation;
  }

  _knowledgeNote(conceptId) {
    if (!this.knowledgeTracer) return '';
    const state = this.knowledgeTracer.getKnowledgeState(conceptId);
    if (!state) return '';
    if (state.pKnown < 0.30) {
      return ` The learner is in the early acquisition stage for ${labelFor(conceptId)} (${Math.round(state.pKnown * 100)}% estimated knowledge).`;
    } else if (state.pKnown < 0.70) {
      return ` The learner is developing this concept (${Math.round(state.pKnown * 100)}% estimated knowledge).`;
    }
    return '';
  }

  // ===========================================================================
  // PRIVATE — Session Narrative Synthesis
  // ===========================================================================

  _generateHeadline(counters, knowledgeReport) {
    const errorCount = counters.totalErrors;
    const islCount = counters.islTransferCount;
    const agreementCount = counters.agreementErrors;

    if (errorCount === 0) {
      return 'Excellent session! All gestures were produced correctly with no errors detected.';
    }

    let parts = [];

    if (knowledgeReport?.averagePKnown) {
      const pct = Math.round(knowledgeReport.averagePKnown * 100);
      if (pct >= 85) {
        parts.push(`Strong overall concept mastery at ${pct}%`);
      } else if (pct >= 60) {
        parts.push(`Developing concept mastery at ${pct}%`);
      } else {
        parts.push(`Early-stage concept acquisition at ${pct}%`);
      }
    }

    if (errorCount > 0) {
      parts.push(`${errorCount} grammatical error${errorCount !== 1 ? 's' : ''} detected`);
    }

    if (islCount > 0) {
      parts.push(`${islCount} ISL transfer pattern${islCount !== 1 ? 's' : ''} observed`);
    }

    if (agreementCount > 0) {
      parts.push(`${agreementCount} subject-verb agreement issue${agreementCount !== 1 ? 's' : ''}`);
    }

    return parts.join(' · ') + '.';
  }

  _identifyStrengths(knowledgeReport, masteryReport) {
    const strengths = [];

    if (knowledgeReport?.strongestConcepts?.length > 0) {
      const best = knowledgeReport.strongestConcepts.slice(0, 3);
      best.forEach(c => {
        strengths.push(`Strong command of "${labelFor(c.conceptId)}" (${Math.round(c.pKnown * 100)}% mastery)`);
      });
    }

    if (knowledgeReport?.averagePKnown && knowledgeReport.averagePKnown >= 0.70) {
      strengths.push('Consistent sentence structure with correct SVO word order');
    }

    if (masteryReport?.highestMastered && masteryReport.highestMastered >= 2) {
      strengths.push(`Progressed through Stage ${masteryReport.highestMastered} of the curriculum`);
    }

    if (this._sessionCounters.totalErrors === 0) {
      strengths.push('Error-free session demonstrating solid grammar understanding');
    }

    if (strengths.length === 0) {
      strengths.push('Active engagement with gesture-based grammar practice');
    }

    return strengths;
  }

  _identifyChallenges(counters, knowledgeReport) {
    const challenges = [];

    if (counters.islTransferCount > 0) {
      challenges.push(
        `ISL word order transfer (${counters.islTransferCount} occurrences) — the system is adapting spatial-zone timing to reinforce English SVO`
      );
    }

    if (counters.agreementErrors > 0) {
      challenges.push(
        `Subject-verb agreement (${counters.agreementErrors} errors) — focus on Stage 4 S-form exercises with He/She subjects`
      );
    }

    // Confusion pairs
    const confusionPairs = counters.confusionPairs || {};
    const topConfusions = Object.entries(confusionPairs)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 2);
    for (const [pair, count] of topConfusions) {
      const [a, b] = pair.split('|');
      challenges.push(`Gesture confusion "${labelFor(a)}" ↔ "${labelFor(b)}" (${count}x) — practice both with contrastive feedback`);
    }

    if (knowledgeReport?.weakestConcepts?.length > 0) {
      const weak = knowledgeReport.weakestConcepts.slice(0, 3);
      const weakLabels = weak.map(c => `"${labelFor(c.conceptId)}" (${Math.round(c.pKnown * 100)}%)`).join(', ');
      challenges.push(`Concepts needing practice: ${weakLabels}`);
    }

    if (challenges.length === 0 && counters.totalErrors > 0) {
      challenges.push('Occasional errors — continue consistent practice');
    }

    if (challenges.length === 0) {
      challenges.push('No significant challenges detected this session');
    }

    return challenges;
  }

  _generateRecommendations(counters, knowledgeReport, masteryReport) {
    const recs = [];

    // Stage-gated recommendations
    if (counters.agreementErrors >= 3) {
      recs.push({
        priority: 'HIGH',
        action: 'Focus on Stage 4 curriculum: Subject-Verb Agreement exercises with He/She + S-form verbs',
        rationale: `${counters.agreementErrors} agreement errors indicate this is the primary learning gap`,
      });
    }

    if (counters.islTransferCount >= 3) {
      recs.push({
        priority: 'HIGH',
        action: 'Use explicit contrastive display: show the ISL SOV structure alongside the English SVO target',
        rationale: 'Explicit contrastive analysis is most effective for L1 syntactic transfer (Birsh & Carreker, 2018, Ch.19)',
      });
    }

    // Knowledge-tracing based
    if (knowledgeReport?.decayRisks?.length > 0) {
      const risks = knowledgeReport.decayRisks.slice(0, 3).map(r => `"${labelFor(r.conceptId)}"`).join(', ');
      recs.push({
        priority: 'MEDIUM',
        action: `Schedule spaced repetition review for: ${risks}`,
        rationale: 'These concepts are at risk of decay — review within 24 hours to maintain retention',
      });
    }

    // Stage progression
    if (knowledgeReport?.stageCompletion) {
      const currentStage = masteryReport?.currentStage || 1;
      const stageData = knowledgeReport.stageCompletion[currentStage];
      if (stageData && stageData.mastered === stageData.total) {
        recs.push({
          priority: 'MEDIUM',
          action: `Advance to Stage ${currentStage + 1} — all Stage ${currentStage} concepts are mastered`,
          rationale: `Learner demonstrates mastery of ${stageData.total}/${stageData.total} concepts at current stage`,
        });
      }
    }

    if (recs.length === 0) {
      recs.push({
        priority: 'LOW',
        action: 'Continue consistent practice sessions (3x per week recommended)',
        rationale: 'Maintain momentum and build automaticity',
      });
    }

    return recs;
  }

  _generateProgressSummary(knowledgeReport, masteryReport) {
    if (!knowledgeReport) return null;

    const avgKnow = Math.round(knowledgeReport.averagePKnown * 100);
    const mastered = knowledgeReport.masteryCount || 0;
    const total = knowledgeReport.totalConcepts || 0;
    const stage = masteryReport?.currentStage || 1;

    return {
      averageKnowledge: `${avgKnow}%`,
      conceptsMastered: `${mastered}/${total}`,
      currentStage: stage,
      description: mastered >= total * 0.7
        ? `Approaching full mastery — ${mastered} of ${total} concepts consolidated or better`
        : mastered >= total * 0.4
          ? `Steady progress — ${mastered} of ${total} concepts at developing level or higher`
          : `Early stage — building foundational gesture-grammar mappings`,
    };
  }

  _estimateLearningRate(sessionHistory) {
    if (sessionHistory.length < 2) return null;

    const recentAccuracy = sessionHistory.map(s => s.accuracy_rate || 0);
    const firstHalf = recentAccuracy.slice(0, Math.floor(recentAccuracy.length / 2));
    const secondHalf = recentAccuracy.slice(-Math.floor(recentAccuracy.length / 2));

    const firstAvg = firstHalf.reduce((s, v) => s + v, 0) / Math.max(firstHalf.length, 1);
    const secondAvg = secondHalf.reduce((s, v) => s + v, 0) / Math.max(secondHalf.length, 1);
    const change = secondAvg - firstAvg;

    if (change > 0.1) return 'RAPID';
    if (change > 0.03) return 'STEADY';
    if (change > -0.03) return 'STABLE';
    return 'NEEDS_SUPPORT';
  }

  _updateSessionCounters(explanation) {
    this._sessionCounters.totalErrors++;
    this._sessionCounters.errorsByType[explanation.errorType] =
      (this._sessionCounters.errorsByType[explanation.errorType] || 0) + 1;

    if (explanation.l1Transfer) {
      this._sessionCounters.islTransferCount++;
    }

    if (explanation.errorType === ERROR_TYPES.AGREEMENT_VIOLATION || 
        explanation.errorType === ERROR_TYPES.WRONG_VERB_FORM) {
      this._sessionCounters.agreementErrors++;
    }

    if (explanation.errorType === ERROR_TYPES.GESTURE_CONFUSION && explanation.context) {
      const pair = [explanation.context.intended, explanation.context.recognized]
        .sort().join('|');
      this._sessionCounters.confusionPairs[pair] =
        (this._sessionCounters.confusionPairs[pair] || 0) + 1;
    }
  }

  // ===========================================================================
  // PRIVATE — Category Inference Helpers
  // ===========================================================================

  _inferExpectedCategory(tokens) {
    if (tokens.length === 0) return 'SUBJECT';
    const last = tokens[tokens.length - 1];
    if (this._isType(last, 'SUBJECT')) return 'VERB';
    if (this._isType(last, 'VERB') && this._isTransitive(last)) return 'OBJECT';
    return null; // Sentence may be complete
  }

  _inferGestureCategory(gestureId) {
    if (!gestureId) return null;
    if (gestureId.startsWith('SUBJECT_')) return 'SUBJECT';
    const objects = ['APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE'];
    if (objects.includes(gestureId)) return 'OBJECT';
    return 'VERB';
  }

  _isType(token, type) {
    if (!token) return false;
    if (token.startsWith('SUBJECT_')) return type === 'SUBJECT';
    if (['APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE'].includes(token)) return type === 'OBJECT';
    return type === 'VERB';
  }

  _isVerb(token) {
    if (!token) return false;
    return !token.startsWith('SUBJECT_') &&
      !['APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE'].includes(token);
  }

  _isTransitive(token) {
    if (!token) return false;
    const intransitive = ['GO', 'STOP', 'GOES', 'STOPS'];
    return !intransitive.includes(token);
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

export function severityColor(severity) {
  switch (severity) {
    case SEVERITY.CRITICAL: return '#f87171';
    case SEVERITY.WARNING:  return '#fbbf24';
    case SEVERITY.INFO:     return '#60a5fa';
    default:                return '#94a3b8';
  }
}

export function severityLabel(severity) {
  switch (severity) {
    case SEVERITY.CRITICAL: return 'Critical';
    case SEVERITY.WARNING:  return 'Warning';
    case SEVERITY.INFO:     return 'Info';
    default:                return 'Unknown';
  }
}

export function explanationToLine(explanation) {
  if (!explanation || !explanation.narrative) return '';
  const maxLen = 140;
  return explanation.narrative.length > maxLen
    ? explanation.narrative.slice(0, maxLen).replace(/\s+\S*$/, '') + '…'
    : explanation.narrative;
}

export default ExplainabilityEngine;
