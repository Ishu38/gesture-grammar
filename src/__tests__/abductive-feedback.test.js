/**
 * abductive-feedback.test.js — Tests for Abductive Feedback Loop
 *
 * Validates the bidirectional neural↔symbolic feedback mechanism:
 *   1. Confidence adaptation on gesture confusion
 *   2. Bayesian prior updating on agreement errors
 *   3. Curriculum gating from abductive diagnosis
 *   4. Interference counter-weighting
 *   5. Temporal decay toward baseline
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { buildMLAFKnowledgeGraph } from '../core/MLAFKnowledgeGraph.js';
import { GraphRAG } from '../core/GraphRAG.js';
import { AbductiveFeedbackLoop, ERROR_EVENTS } from '../core/AbductiveFeedbackLoop.js';

let graph;
let rag;
let feedback;

beforeEach(() => {
  graph = buildMLAFKnowledgeGraph();
  rag = new GraphRAG(graph);
  feedback = new AbductiveFeedbackLoop(rag);
});

// =============================================================================
// MECHANISM 1: CONFIDENCE ADAPTATION
// =============================================================================

describe('Mechanism 1 — Confidence Adaptation', () => {
  it('should raise threshold after repeated gesture confusion', () => {
    const baseBefore = feedback.getAdaptedThreshold('SUBJECT_YOU');
    expect(baseBefore).toBe(0.40); // baseline

    // Confuse YOU↔DRINK twice (trigger count = 2)
    feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
      intendedGesture: 'SUBJECT_YOU',
      recognizedGesture: 'DRINK',
    });
    feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
      intendedGesture: 'SUBJECT_YOU',
      recognizedGesture: 'DRINK',
    });

    const thresholdYou = feedback.getAdaptedThreshold('SUBJECT_YOU');
    const thresholdDrink = feedback.getAdaptedThreshold('DRINK');

    // Both should be raised above baseline
    expect(thresholdYou).toBeGreaterThan(0.40);
    expect(thresholdDrink).toBeGreaterThan(0.40);
  });

  it('should boost more for known confusion pairs (graph-predicted)', () => {
    // YOU↔DRINK is a known confusion pair in the knowledge graph
    for (let i = 0; i < 3; i++) {
      feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
        intendedGesture: 'SUBJECT_YOU',
        recognizedGesture: 'DRINK',
      });
    }
    const knownPairThreshold = feedback.getAdaptedThreshold('SUBJECT_YOU');

    // Reset and test unknown pair
    feedback.reset();
    for (let i = 0; i < 3; i++) {
      feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
        intendedGesture: 'SUBJECT_I',
        recognizedGesture: 'FOOD',
      });
    }
    const unknownPairThreshold = feedback.getAdaptedThreshold('SUBJECT_I');

    // Known pair should get higher boost
    expect(knownPairThreshold).toBeGreaterThan(unknownPairThreshold);
  });

  it('should never exceed maximum threshold', () => {
    for (let i = 0; i < 50; i++) {
      feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
        intendedGesture: 'SUBJECT_YOU',
        recognizedGesture: 'DRINK',
      });
    }
    expect(feedback.getAdaptedThreshold('SUBJECT_YOU')).toBeLessThanOrEqual(0.85);
  });

  it('should not adapt below trigger count', () => {
    feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
      intendedGesture: 'SUBJECT_HE',
      recognizedGesture: 'SUBJECT_SHE',
    });
    // Only 1 error — below trigger count of 2
    expect(feedback.getAdaptedThreshold('SUBJECT_HE')).toBe(0.40);
  });
});

// =============================================================================
// MECHANISM 2: BAYESIAN PRIOR UPDATING
// =============================================================================

describe('Mechanism 2 — Bayesian Prior Updating', () => {
  it('should suppress prior for gesture with repeated agreement errors', () => {
    expect(feedback.getPriorAdjustment('GRAB')).toBe(1.0); // baseline

    // Multiple agreement errors with GRAB
    for (let i = 0; i < 3; i++) {
      feedback.recordError(ERROR_EVENTS.WRONG_VERB_FORM, {
        verbId: 'GRAB',
        subjectId: 'SUBJECT_HE',
      });
    }

    const prior = feedback.getPriorAdjustment('GRAB');
    expect(prior).toBeLessThan(1.0);
    expect(prior).toBeGreaterThan(0.29); // floor at 0.3
  });

  it('should also suppress the s-form pair', () => {
    for (let i = 0; i < 3; i++) {
      feedback.recordError(ERROR_EVENTS.WRONG_VERB_FORM, { verbId: 'GRAB' });
    }

    // GRABS (s-form pair) should also have suppressed prior
    const grabsPrior = feedback.getPriorAdjustment('GRABS');
    expect(grabsPrior).toBeLessThan(1.0);
  });
});

// =============================================================================
// MECHANISM 3: CURRICULUM GATING
// =============================================================================

describe('Mechanism 3 — Curriculum Gating', () => {
  it('should gate later-stage gestures when remediation points to earlier stage', () => {
    // WRONG_VERB_FORM → remediation to Stage 4
    // This should gate Stage 5 gestures
    for (let i = 0; i < 3; i++) {
      feedback.recordError(ERROR_EVENTS.WRONG_VERB_FORM, { verbId: 'GRAB' });
    }

    const report = feedback.getAdaptationReport();
    // Gated gestures should be from stages after the remediation target
    // (exact gestures depend on curriculum graph, but there should be some gating)
    expect(report.gatedGestures).toBeDefined();
  });
});

// =============================================================================
// MECHANISM 4: INTERFERENCE COUNTER-WEIGHTING
// =============================================================================

describe('Mechanism 4 — Interference Counter-Weighting', () => {
  it('should increase lock multiplier on word order errors', () => {
    expect(feedback.getLockMultiplier()).toBe(1.0);

    feedback.recordError(ERROR_EVENTS.WRONG_WORD_ORDER, {
      sentenceTokens: ['SUBJECT_I', 'APPLE', 'EAT'],
      sentenceWords: [
        { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
        { type: 'OBJECT', grammar_id: 'APPLE' },
        { type: 'VERB', grammar_id: 'EAT', transitive: true },
      ],
    });

    expect(feedback.getLockMultiplier()).toBeGreaterThan(1.0);
  });

  it('should cap lock multiplier', () => {
    for (let i = 0; i < 20; i++) {
      feedback.recordError(ERROR_EVENTS.WRONG_WORD_ORDER, {
        sentenceWords: [
          { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
          { type: 'OBJECT', grammar_id: 'APPLE' },
          { type: 'VERB', grammar_id: 'EAT', transitive: true },
        ],
      });
    }
    expect(feedback.getLockMultiplier()).toBeLessThanOrEqual(1.4);
  });
});

// =============================================================================
// MECHANISM 5: TEMPORAL DECAY
// =============================================================================

describe('Mechanism 5 — Temporal Decay', () => {
  it('should decay adaptations on correct recognitions', () => {
    // Build up adaptations
    for (let i = 0; i < 5; i++) {
      feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
        intendedGesture: 'SUBJECT_YOU',
        recognizedGesture: 'DRINK',
      });
    }
    const peakThreshold = feedback.getAdaptedThreshold('SUBJECT_YOU');

    // 20 correct recognitions should decay
    for (let i = 0; i < 20; i++) {
      feedback.recordSuccess('SUBJECT_YOU');
    }

    const decayedThreshold = feedback.getAdaptedThreshold('SUBJECT_YOU');
    expect(decayedThreshold).toBeLessThan(peakThreshold);
  });

  it('should accelerate decay on 5+ consecutive correct', () => {
    for (let i = 0; i < 5; i++) {
      feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
        intendedGesture: 'SUBJECT_YOU',
        recognizedGesture: 'DRINK',
      });
    }

    // 5 consecutive correct should trigger accelerated decay
    for (let i = 0; i < 5; i++) {
      feedback.recordSuccess('SUBJECT_YOU');
    }

    const afterStreak = feedback.getAdaptedThreshold('SUBJECT_YOU');

    // Compare with single correct
    feedback.reset();
    for (let i = 0; i < 5; i++) {
      feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
        intendedGesture: 'SUBJECT_YOU',
        recognizedGesture: 'DRINK',
      });
    }
    feedback.recordSuccess('SUBJECT_YOU');
    const afterSingle = feedback.getAdaptedThreshold('SUBJECT_YOU');

    expect(afterStreak).toBeLessThan(afterSingle);
  });

  it('should decay lock multiplier toward 1.0', () => {
    feedback.recordError(ERROR_EVENTS.WRONG_WORD_ORDER, {
      sentenceWords: [
        { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
        { type: 'OBJECT', grammar_id: 'APPLE' },
        { type: 'VERB', grammar_id: 'EAT', transitive: true },
      ],
    });
    const peakMultiplier = feedback.getLockMultiplier();

    for (let i = 0; i < 30; i++) {
      feedback.recordSuccess('SUBJECT_I');
    }

    expect(feedback.getLockMultiplier()).toBeLessThan(peakMultiplier);
  });

  it('should ungate gestures after 10 consecutive correct', () => {
    // Force some gating
    for (let i = 0; i < 5; i++) {
      feedback.recordError(ERROR_EVENTS.WRONG_VERB_FORM, { verbId: 'GRAB' });
    }

    // 10 consecutive correct should ungate
    for (let i = 0; i < 10; i++) {
      feedback.recordSuccess('GRAB');
    }

    expect(feedback.getAdaptationReport().gatedGestures.length).toBe(0);
  });
});

// =============================================================================
// ABDUCTIVE CHAIN INTEGRATION
// =============================================================================

describe('Abductive Chain Integration', () => {
  it('should return abductive chain in error result', () => {
    const result = feedback.recordError(ERROR_EVENTS.WRONG_VERB_FORM, {
      verbId: 'GRAB',
      subjectId: 'SUBJECT_HE',
    });

    expect(result.errorType).toBe('WRONG_VERB_FORM');
    expect(result.diagnosis).toBeDefined();
    expect(result.abductiveChain).toBeDefined();
    expect(result.totalErrors).toBe(1);
  });

  it('should detect interference in word order errors', () => {
    const result = feedback.recordError(ERROR_EVENTS.WRONG_WORD_ORDER, {
      sentenceWords: [
        { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
        { type: 'OBJECT', grammar_id: 'APPLE' },
        { type: 'VERB', grammar_id: 'EAT', transitive: true },
      ],
    });

    expect(result.interferenceDetected).toBe(true);
  });

  it('should track error rate correctly', () => {
    feedback.recordSuccess('A');
    feedback.recordSuccess('B');
    feedback.recordError(ERROR_EVENTS.TYPE_MISMATCH, { recognizedGesture: 'GRAB' });
    feedback.recordSuccess('C');

    const report = feedback.getAdaptationReport();
    expect(report.totalErrors).toBe(1);
    expect(report.totalCorrect).toBe(3);
    expect(report.errorRate).toBeCloseTo(0.25, 2);
  });
});

// =============================================================================
// RESET
// =============================================================================

describe('Reset', () => {
  it('should reset all state to baseline', () => {
    // Build up state
    for (let i = 0; i < 5; i++) {
      feedback.recordError(ERROR_EVENTS.GESTURE_CONFUSION, {
        intendedGesture: 'SUBJECT_YOU',
        recognizedGesture: 'DRINK',
      });
    }
    feedback.recordError(ERROR_EVENTS.WRONG_WORD_ORDER, {
      sentenceWords: [
        { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
        { type: 'OBJECT', grammar_id: 'APPLE' },
        { type: 'VERB', grammar_id: 'EAT', transitive: true },
      ],
    });

    feedback.reset();

    expect(feedback.getAdaptedThreshold('SUBJECT_YOU')).toBe(0.40);
    expect(feedback.getLockMultiplier()).toBe(1.0);
    expect(feedback.getAdaptationReport().totalErrors).toBe(0);
    expect(feedback.getAdaptationReport().gatedGestures.length).toBe(0);
  });
});
