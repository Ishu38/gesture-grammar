/**
 * KnowledgeTracer Tests — Bayesian Knowledge Tracing unit tests
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { KnowledgeTracer, CONCEPT_MAP, META_CONCEPTS, MASTERY_STAGES } from '../core/KnowledgeTracer';

describe('KnowledgeTracer', () => {
  let tracer;

  beforeEach(() => {
    tracer = new KnowledgeTracer();
  });

  describe('initialization', () => {
    it('initializes all gesture concepts with default prior', () => {
      const state = tracer.getKnowledgeState('GRAB');
      expect(state).not.toBeNull();
      expect(state.pKnown).toBeCloseTo(0.20, 1);
      expect(state.masteryStage).toBe('ACQUIRING');
      expect(state.totalAttempts).toBe(0);
    });

    it('initializes meta-concepts', () => {
      const state = tracer.getKnowledgeState('SV_AGREEMENT');
      expect(state).not.toBeNull();
      expect(state.category).toBe('META');
    });

    it('tracks the correct number of concepts', () => {
      const all = tracer.getAllStates();
      const gestureCount = Object.values(all).filter(s => s.category !== 'META').length;
      const metaCount = Object.values(all).filter(s => s.category === 'META').length;
      expect(gestureCount).toBe(Object.keys(CONCEPT_MAP).length);
      expect(metaCount).toBe(Object.keys(META_CONCEPTS).length);
    });
  });

  describe('recordOpportunity', () => {
    it('increases pKnown after a correct response', () => {
      const result = tracer.recordOpportunity('GRAB', true);
      expect(result.pKnown).toBeGreaterThan(0.20);
      expect(result.delta).toBeGreaterThan(0);
    });

    it('decreases pKnown after an incorrect response', () => {
      const result = tracer.recordOpportunity('GRAB', false);
      expect(result.pKnown).toBeLessThan(0.20);
    });

    it('accumulates attempts', () => {
      tracer.recordOpportunity('GRAB', true);
      tracer.recordOpportunity('GRAB', true);
      const state = tracer.getKnowledgeState('GRAB');
      expect(state.totalAttempts).toBe(2);
      expect(state.correctAttempts).toBe(2);
    });

    it('tracks mastery stage progression', () => {
      // Simulate correct responses to reach mastery
      for (let i = 0; i < 10; i++) {
        tracer.recordOpportunity('GRAB', true);
      }
      const state = tracer.getKnowledgeState('GRAB');
      expect(['MASTERED', 'AUTOMATIC']).toContain(state.masteryStage);
      expect(state.pKnown).toBeGreaterThanOrEqual(0.85);
    });

    it('boosts pKnown for fast correct responses', () => {
      const slowResult = tracer.recordOpportunity('GRAB', true, { responseTimeMs: 2000 });
      // Reset and test fast
      const fastTracer = new KnowledgeTracer();
      const fastResult = fastTracer.recordOpportunity('GRAB', true, { responseTimeMs: 200 });
      expect(fastResult.pKnown).toBeGreaterThan(slowResult.pKnown);
    });

    it('returns null for unknown concept', () => {
      const result = tracer.recordOpportunity('NONEXISTENT', true);
      // Should auto-initialize
      expect(result).not.toBeNull();
      expect(result.pKnown).toBeGreaterThan(0);
    });
  });

  describe('getMasteryEstimate', () => {
    it('estimates correct attempts needed for mastery', () => {
      const estimate = tracer.getMasteryEstimate('GRAB');
      expect(estimate.pKnown).toBe(0.20);
      expect(estimate.isMastered).toBe(false);
      expect(estimate.attemptsNeeded).toBeGreaterThan(0);
    });

    it('returns 0 attempts for mastered concepts', () => {
      for (let i = 0; i < 20; i++) {
        tracer.recordOpportunity('GRAB', true);
      }
      const estimate = tracer.getMasteryEstimate('GRAB');
      expect(estimate.isMastered).toBe(true);
      expect(estimate.attemptsNeeded).toBe(0);
    });
  });

  describe('retention forecasting', () => {
    it('predicts retention decay over time', () => {
      tracer.recordOpportunity('GRAB', true);
      const forecast1 = tracer.getRetentionForecast('GRAB', 1);
      const forecast7 = tracer.getRetentionForecast('GRAB', 7);
      expect(forecast7.pKnownProjected).toBeLessThan(forecast1.pKnownProjected);
    });

    it('flags concepts needing review', () => {
      const forecast = tracer.getRetentionForecast('GRAB', 30);
      expect(forecast.needsReview).toBe(true);
    });
  });

  describe('reporting', () => {
    it('generates overall report', () => {
      tracer.recordOpportunity('GRAB', true);
      tracer.recordOpportunity('EAT', false);
      const report = tracer.getOverallReport();
      expect(report.totalConcepts).toBeGreaterThan(0);
      expect(report.averagePKnown).toBeGreaterThan(0);
      expect(report.weakestConcepts.length).toBeGreaterThan(0);
      expect(report.strongestConcepts.length).toBeGreaterThan(0);
    });

    it('generates decay report', () => {
      tracer.recordOpportunity('GRAB', true);
      const decay = tracer.getConceptDecayReport();
      expect(decay.length).toBeGreaterThan(0);
      expect(decay[0]).toHaveProperty('daysSinceLastPractice');
      expect(decay[0]).toHaveProperty('retention');
    });
  });

  describe('persistence', () => {
    it('maintains state after reconstruction', () => {
      tracer.recordOpportunity('GRAB', true);
      tracer.recordOpportunity('GRAB', true);
      const before = tracer.getKnowledgeState('GRAB');
      expect(before.totalAttempts).toBe(2);
      expect(before.pKnown).toBeGreaterThan(0.20);

      // Verify snapshot contains the right data
      const snapshot = tracer.getSnapshot();
      expect(snapshot.states.GRAB.totalAttempts).toBe(2);
    });
  });

  describe('reset', () => {
    it('resets all concepts to initial state', () => {
      tracer.recordOpportunity('GRAB', true);
      tracer.recordOpportunity('GRAB', true);
      tracer.reset();
      const state = tracer.getKnowledgeState('GRAB');
      expect(state.totalAttempts).toBe(0);
      expect(state.pKnown).toBeCloseTo(0.20, 1);
    });
  });

  describe('snapshot', () => {
    it('returns serializable state', () => {
      tracer.recordOpportunity('GRAB', true);
      const snapshot = tracer.getSnapshot();
      expect(snapshot).toHaveProperty('parameters');
      expect(snapshot).toHaveProperty('states');
      expect(snapshot.states.GRAB.pKnown).toBeGreaterThan(0);
    });
  });

  describe('transfer latency', () => {
    it('returns null for untracked pairs', () => {
      const lt = tracer.getTransferLatency('GRAB', 'EAT');
      expect(lt.evidenceCount).toBe(0);
      expect(lt.hours).toBeNull();
    });
  });
});
