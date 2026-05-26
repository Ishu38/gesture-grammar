/**
 * ExplainabilityEngine Tests
 */
import { describe, it, expect, beforeEach } from 'vitest';
import { ExplainabilityEngine, ERROR_TYPES } from '../core/ExplainabilityEngine';

// Minimal mock GraphRAG for testing
function createMockGraphRAG() {
  return {
    diagnoseError(errorType) {
      const results = {
        causes: [],
        remediation: [],
        interferencePatterns: [],
      };

      if (errorType === 'WRONG_VERB_FORM' || errorType === 'AGREEMENT_VIOLATION') {
        results.causes = [{
          rule: 'Third-person singular subjects require S-form verbs',
          description: 'Third-person singular S-form agreement rule not yet internalized',
        }];
        results.remediation = [{
          stage: 4,
          label: 'Stage 4 — Subject-Verb Agreement',
          reason: 'Stage 4 teaches S-V agreement',
        }];
      }

      if (errorType === 'WRONG_WORD_ORDER') {
        results.causes = [{ description: 'ISL SOV word order transfer' }];
        results.interferencePatterns = [{
          id: 'SOV_ORDER',
          title: 'ISL Word Order (SOV)',
          l1_structure: 'Subject → Object → Verb',
          l2_target: 'Subject → Verb → Object',
        }];
        results.remediation = [{
          strategy: 'Reorder to Subject → Verb → Object',
          explanation: 'In English, the verb comes between subject and object',
        }];
      }

      if (errorType === 'MISSING_OBJECT') {
        results.causes = [{ description: 'Transitive verb requires explicit object' }];
        results.interferencePatterns = [{
          id: 'TRANSITIVE_OBJECT_DROP',
          title: 'Missing Object (Pro-drop)',
        }];
        results.remediation = [{
          strategy: 'Add an explicit object noun',
        }];
      }

      return {
        diagnosis: {
          error: { title: errorType },
          description: results.causes[0]?.description || 'Unknown error',
          abductiveChain: `${errorType} → cause`,
        },
        ...results,
      };
    },
  };
}

function createMockKnowledgeTracer() {
  return {
    getKnowledgeState(conceptId) {
      const states = {
        GRAB: { pKnown: 0.85, masteryStage: 'MASTERED' },
        GRABS: { pKnown: 0.25, masteryStage: 'ACQUIRING' },
        SUBJECT_HE: { pKnown: 0.90, masteryStage: 'MASTERED' },
        SUBJECT_I: { pKnown: 0.95, masteryStage: 'AUTOMATIC' },
      };
      return states[conceptId] || { pKnown: 0.20, masteryStage: 'ACQUIRING' };
    },
  };
}

describe('ExplainabilityEngine', () => {
  let engine;
  let mockGraphRAG;
  let mockKT;

  beforeEach(() => {
    mockGraphRAG = createMockGraphRAG();
    mockKT = createMockKnowledgeTracer();
    engine = new ExplainabilityEngine(mockGraphRAG, mockKT);
  });

  describe('explainError', () => {
    it('generates explanation for type mismatch', () => {
      const result = engine.explainError(ERROR_TYPES.TYPE_MISMATCH, {
        recognizedGesture: 'APPLE',
        sentenceTokens: ['SUBJECT_I'],
      });

      expect(result).toHaveProperty('narrative');
      expect(result).toHaveProperty('rootCause');
      expect(result).toHaveProperty('remediation');
      expect(result.errorType).toBe(ERROR_TYPES.TYPE_MISMATCH);
      expect(result.narrative).toContain('apple');
      expect(result.narrative).toContain('I');
    });

    it('generates explanation for agreement violation', () => {
      const result = engine.explainError(ERROR_TYPES.AGREEMENT_VIOLATION, {
        subjectId: 'SUBJECT_HE',
        verbId: 'GRAB',
        recognizedGesture: 'GRAB',
      });

      expect(result).toHaveProperty('narrative');
      expect(result.rootCause).toContain('S-form');
      expect(result.remediation).toContain('Stage 4');
    });

    it('generates explanation for word order error with ISL transfer', () => {
      const result = engine.explainError(ERROR_TYPES.WRONG_WORD_ORDER, {
        sentenceTokens: ['SUBJECT_I', 'APPLE', 'GRAB'],
      });

      expect(result).toHaveProperty('narrative');
      expect(result.l1Transfer).toBeTruthy();
      expect(result.l1Transfer).toContain('ISL');
      expect(result.narrative).toContain('SOV');
    });

    it('generates explanation for missing object', () => {
      const result = engine.explainError(ERROR_TYPES.MISSING_OBJECT, {
        sentenceTokens: ['SUBJECT_I', 'GRAB'],
      });

      expect(result).toHaveProperty('narrative');
      expect(result.narrative).toContain('object');
      expect(result.l1Transfer).toBeTruthy();
    });

    it('includes knowledge state enrichment', () => {
      const result = engine.explainError(ERROR_TYPES.AGREEMENT_VIOLATION, {
        subjectId: 'SUBJECT_HE',
        verbId: 'GRABS',
        recognizedGesture: 'GRABS',
      });

      expect(result).toHaveProperty('knowledgeState');
      if (result.knowledgeState) {
        expect(result.knowledgeState.concept).toBeDefined();
      }
    });

    it('assigns severity correctly', () => {
      const result = engine.explainError(ERROR_TYPES.AGREEMENT_VIOLATION, {
        subjectId: 'SUBJECT_HE',
        verbId: 'GRABS',
      });

      expect(['WARNING', 'CRITICAL', 'INFO']).toContain(result.severity);
    });
  });

  describe('session narrative', () => {
    it('generates narrative after accumulating errors', () => {
      engine.explainError(ERROR_TYPES.TYPE_MISMATCH, {
        recognizedGesture: 'APPLE', sentenceTokens: ['SUBJECT_I'],
      });
      engine.explainError(ERROR_TYPES.AGREEMENT_VIOLATION, {
        subjectId: 'SUBJECT_HE', verbId: 'GRAB',
      });

      const narrative = engine.generateSessionNarrative();
      expect(narrative).toHaveProperty('headline');
      expect(narrative).toHaveProperty('strengths');
      expect(narrative).toHaveProperty('challenges');
      expect(narrative).toHaveProperty('recommendations');
      expect(narrative.errorCount).toBe(2);
    });

    it('generates appropriate headline with no errors', () => {
      const narrative = engine.generateSessionNarrative();
      expect(narrative.headline).toContain('Excellent');
      expect(narrative.errorCount).toBe(0);
    });
  });

  describe('reset', () => {
    it('clears accumulated explanations', () => {
      engine.explainError(ERROR_TYPES.TYPE_MISMATCH, {
        recognizedGesture: 'APPLE', sentenceTokens: ['SUBJECT_I'],
      });
      engine.reset();
      expect(engine.getAllExplanations().length).toBe(0);
    });
  });

  describe('getRecentExplanations', () => {
    it('returns most recent explanations', () => {
      engine.explainError(ERROR_TYPES.TYPE_MISMATCH, {
        recognizedGesture: 'APPLE', sentenceTokens: ['SUBJECT_I'],
      });
      engine.explainError(ERROR_TYPES.AGREEMENT_VIOLATION, {
        subjectId: 'SUBJECT_HE', verbId: 'GRAB',
      });
      engine.explainError(ERROR_TYPES.WRONG_WORD_ORDER, {
        sentenceTokens: ['SUBJECT_I', 'APPLE', 'GRAB'],
      });

      const recent = engine.getRecentExplanations(2);
      expect(recent.length).toBe(2);
      expect(recent[0].errorType).toBe(ERROR_TYPES.AGREEMENT_VIOLATION);
      expect(recent[1].errorType).toBe(ERROR_TYPES.WRONG_WORD_ORDER);
    });
  });

  describe('category inference', () => {
    it('identifies subject gestures', () => {
      const result = engine.explainError(ERROR_TYPES.TYPE_MISMATCH, {
        recognizedGesture: 'SUBJECT_YOU', sentenceTokens: ['SUBJECT_I', 'GRAB'],
      });
      expect(result.narrative).toContain('subject');
    });
  });
});
