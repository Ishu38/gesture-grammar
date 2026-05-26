/**
 * AdaptivePedagogyEngine Tests
 */
import { describe, it, expect, beforeEach } from 'vitest';
import {
  AdaptivePedagogyEngine,
  LEARNER_STATES,
  DECISION_TYPES,
  learnerStateColor,
  learnerStateLabel,
} from '../core/AdaptivePedagogyEngine';

function createMockKnowledgeTracer() {
  let _states = {};
  const ids = [
    'SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_HE', 'SUBJECT_SHE', 'SUBJECT_WE', 'SUBJECT_THEY',
    'GRAB', 'EAT', 'WANT', 'DRINK', 'SEE', 'GO', 'STOP',
    'GRABS', 'EATS', 'WANTS', 'DRINKS', 'SEES', 'GOES', 'STOPS',
    'APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE',
  ];
  ids.forEach(id => {
    _states[id] = { pKnown: 0.20, masteryStage: 'ACQUIRING', category: 'GESTURE', isMastered: false, stage: 1, totalAttempts: 0, correctAttempts: 0 };
  });

  return {
    getKnowledgeState(id) {
      return _states[id] || { pKnown: 0, masteryStage: 'ACQUIRING' };
    },
    getAllStates() {
      return Object.fromEntries(Object.entries(_states).map(([id, s]) => [id, { ...s, conceptId: id, category: s.category || 'GESTURE' }]));
    },
    getOverallReport() {
      const entries = Object.values(_states);
      const avg = entries.reduce((s, e) => s + e.pKnown, 0) / entries.length;
      return {
        totalConcepts: entries.length,
        totalMetaConcepts: 4,
        averagePKnown: avg,
        masteryCount: entries.filter(e => e.pKnown >= 0.85).length,
        weakestConcepts: entries.filter(e => e.pKnown < 0.70).slice(0, 5).map(e => ({ conceptId: 'GRAB', pKnown: e.pKnown, stage: 1 })),
        strongestConcepts: entries.filter(e => e.pKnown >= 0.70).slice(-5).map(e => ({ conceptId: 'GRAB', pKnown: e.pKnown, stage: 1 })),
        decayRisks: [],
        stageCompletion: {},
        conceptsReadyForNextStage: { masteredConceptIds: [], maxMasteredStage: 1, totalMastered: 0 },
        metaConcepts: [],
      };
    },
    recordOpportunity(id, correct) {
      if (_states[id]) {
        if (correct) {
          _states[id].pKnown = Math.min(0.99, _states[id].pKnown + 0.1);
        } else {
          _states[id].pKnown = Math.max(0.01, _states[id].pKnown - 0.05);
        }
        _states[id].totalAttempts++;
        _states[id].correctAttempts += correct ? 1 : 0;
      }
    },
  };
}

describe('AdaptivePedagogyEngine', () => {
  let engine;
  let mockKT;

  beforeEach(() => {
    mockKT = createMockKnowledgeTracer();
    engine = new AdaptivePedagogyEngine({ profileType: 'default' });
  });

  describe('initialization', () => {
    it('starts in flowing state', () => {
      const model = engine.getPersonalizationProfile();
      expect(model.currentState).toBe(LEARNER_STATES.FLOWING);
      expect(model.scaffoldingLevel).toBe(1);
    });

    it('computes initial modality weights', () => {
      const model = engine.getPersonalizationProfile();
      expect(model.effectiveModalities.primary).toBeDefined();
      const weights = model.effectiveModalities.weights;
      expect(weights.visual + weights.acoustic + weights.gaze).toBeCloseTo(1.0, 1);
    });

    it('adjusts weights for motor_impairment profile', () => {
      const motorEngine = new AdaptivePedagogyEngine({ profileType: 'motor_impairment' });
      const profile = motorEngine.getPersonalizationProfile();
      expect(profile.effectiveModalities.weights.visual).toBeGreaterThan(0.60);
    });

    it('adjusts weights for deaf profile', () => {
      const deafEngine = new AdaptivePedagogyEngine({ profileType: 'deaf' });
      const profile = deafEngine.getPersonalizationProfile();
      expect(profile.effectiveModalities.weights.acoustic).toBeLessThan(0.10);
      expect(profile.effectiveModalities.weights.visual).toBeGreaterThan(0.50);
    });
  });

  describe('update', () => {
    it('produces update without errors on normal data', () => {
      const update = engine.update({
        knowledgeTracer: mockKT,
        cognitiveLoad: { level: 'LOW', jitter: 0.002 },
        masteryReport: { currentStage: 1, highestMastered: 0 },
        sentence: [],
        latestError: null,
        responseTimeMs: 500,
      });

      expect(update).toHaveProperty('learnerState');
      expect(update).toHaveProperty('decisions');
      expect(update).toHaveProperty('learnerModel');
      expect(update).toHaveProperty('effectiveModalityWeights');
    });

    it('detects state transitions', () => {
      // Simulate multiple updates with errors (confused state)
      for (let i = 0; i < 10; i++) {
        engine.update({
          knowledgeTracer: mockKT,
          cognitiveLoad: { level: 'LOW', jitter: 0.002 },
          masteryReport: { currentStage: 1, highestMastered: 0 },
          sentence: [],
          latestError: { errorType: 'TYPE_MISMATCH', narrative: 'error' },
          responseTimeMs: 500,
        });
      }
      // Should be tracking errors
      const profile = engine.getPersonalizationProfile();
      expect(profile.currentState).toBeDefined();
    });
  });

  describe('recommendNextExercise', () => {
    it('recommends free practice when nothing is due', () => {
      const rec = engine.recommendNextExercise(mockKT, { currentStage: 1 }, { dueNowIds: [], dueNow: 0 });
      expect(rec).toHaveProperty('exerciseType');
      expect(rec).toHaveProperty('difficulty');
    });

    it('recommends spaced review when due', () => {
      const rec = engine.recommendNextExercise(
        mockKT,
        { currentStage: 1 },
        { dueNowIds: ['GRAB', 'EAT'], dueNow: 2 }
      );
      expect(rec.exerciseType).toBe('SPACED_REVIEW');
      expect(rec.concepts).toContain('GRAB');
    });
  });

  describe('getLearnerModel', () => {
    it('builds learner model with all sections', () => {
      const model = engine.getLearnerModel(mockKT, { currentStage: 1, highestMastered: 0 });
      expect(model).toHaveProperty('profile');
      expect(model).toHaveProperty('knowledge');
      expect(model).toHaveProperty('adaptationProfile');
      expect(model).toHaveProperty('riskFactors');
      expect(model).toHaveProperty('trajectory');
      expect(model).toHaveProperty('currentState');
    });

    it('returns effective modality weights in model', () => {
      const model = engine.getLearnerModel(mockKT, {});
      expect(model.adaptationProfile.effectiveModalities.primary).toBeDefined();
    });
  });

  describe('reset', () => {
    it('clears session state', () => {
      engine.update({
        knowledgeTracer: mockKT,
        cognitiveLoad: { level: 'LOW', jitter: 0.002 },
        masteryReport: { currentStage: 1, highestMastered: 0 },
        sentence: [],
        latestError: { errorType: 'TYPE_MISMATCH' },
      });
      engine.reset();
      const profile = engine.getPersonalizationProfile();
      expect(profile.currentState).toBe(LEARNER_STATES.FLOWING);
      expect(profile.scaffoldingLevel).toBe(1);
    });
  });
});

describe('Display helpers', () => {
  it('learnerStateColor returns CSS colors', () => {
    expect(learnerStateColor(LEARNER_STATES.FLOWING)).toBe('#4ade80');
    expect(learnerStateColor(LEARNER_STATES.STUCK)).toBe('#f87171');
  });

  it('learnerStateLabel returns readable labels', () => {
    expect(learnerStateLabel(LEARNER_STATES.FLOWING)).toBe('Flowing');
    expect(learnerStateLabel(LEARNER_STATES.FATIGUED)).toBe('Fatigued');
  });
});
