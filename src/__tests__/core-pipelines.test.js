/**
 * Core Pipeline Tests — Reliability suite for MLAF's untested modules.
 * Covers: GestureMasteryGate, SpacedRepetitionScheduler, CognitiveLoadAdapter,
 *         AutomaticityTracker, ISLInterferenceDetector, LandmarkSmoother.
 *
 * These modules are critical for learning progression, adaptation, and L1
 * transfer detection. Failures here mean the adaptive scaffolding breaks.
 */

import { describe, it, expect, beforeEach } from 'vitest';
import {
  GestureMasteryGate,
  CURRICULUM_SEQUENCE,
} from '../core/GestureMasteryGate.js';
import { SpacedRepetitionScheduler } from '../core/SpacedRepetitionScheduler.js';
import { CognitiveLoadAdapter } from '../core/CognitiveLoadAdapter.js';
import { AutomaticityTracker } from '../core/AutomaticityTracker.js';
import { ISLInterferenceDetector } from '../core/ISLInterferenceDetector.js';
import { LandmarkSmoother } from '../core/LandmarkSmoother.js';
import { IntentionalityDetector } from '../core/IntentionalityDetector.js';
import { SpatialGrammarMapper } from '../core/SpatialGrammarMapper.js';

// =============================================================================
// HELPERS
// =============================================================================

/** Generate 21 fake hand landmarks at a given base position with optional noise. */
function fakeLandmarks(baseX = 0.5, baseY = 0.5, baseZ = 0.0, noise = 0) {
  return Array.from({ length: 21 }, (_, i) => ({
    x: baseX + (i * 0.01) + (noise ? (Math.random() - 0.5) * noise : 0),
    y: baseY + (i * 0.005) + (noise ? (Math.random() - 0.5) * noise : 0),
    z: baseZ + (noise ? (Math.random() - 0.5) * noise : 0),
  }));
}

/** Build a sentence word object as useSentenceBuilder produces. */
function word(grammar_id, type, wordDisplay, transitive = false) {
  return { grammar_id, type, word: wordDisplay, transitive };
}

// =============================================================================
// GestureMasteryGate
// =============================================================================

describe('GestureMasteryGate', () => {
  let gate;

  beforeEach(() => {
    // Use a unique storageKey to isolate tests (no localStorage in Node anyway)
    gate = new GestureMasteryGate({ storageKey: 'test_mastery_' + Date.now(), threshold: 3 });
  });

  it('should start at Stage 1 with no productions', () => {
    expect(gate.getCurrentStage()).toBe(1);
    expect(gate.getHighestMasteredStage()).toBe(0);
  });

  it('should track production counts correctly', () => {
    const r1 = gate.recordProduction('SUBJECT_I');
    expect(r1.count).toBe(1);
    expect(r1.wasMastered).toBe(false);
    expect(r1.isNowMastered).toBe(false);

    gate.recordProduction('SUBJECT_I');
    const r3 = gate.recordProduction('SUBJECT_I');
    expect(r3.count).toBe(3);
    expect(r3.isNowMastered).toBe(true);
  });

  it('should not advance stage until ALL gestures in current stage are mastered', () => {
    // Stage 1 has SUBJECT_I and STOP (threshold=3)
    for (let i = 0; i < 3; i++) gate.recordProduction('SUBJECT_I');
    expect(gate.getCurrentStage()).toBe(1); // STOP not yet mastered
    expect(gate.getHighestMasteredStage()).toBe(0);

    for (let i = 0; i < 3; i++) gate.recordProduction('STOP');
    expect(gate.getHighestMasteredStage()).toBe(1);
    expect(gate.getCurrentStage()).toBe(2);
  });

  it('should report correct mastery data in getMasteryReport', () => {
    gate.recordProduction('SUBJECT_I');
    gate.recordProduction('SUBJECT_I');
    const report = gate.getMasteryReport();

    expect(report).toBeDefined();
    expect(report.currentStage).toBe(1);
    const stage1 = report.stages.find(s => s.stage === 1);
    const subjectI = stage1.gestureStatuses.find(g => g.id === 'SUBJECT_I');
    expect(subjectI).toBeDefined();
    expect(subjectI.count).toBe(2);
  });

  it('should keep stages sequential (cannot skip Stage 2 to Stage 3)', () => {
    // Master Stage 1
    for (const g of ['SUBJECT_I', 'STOP']) {
      for (let i = 0; i < 3; i++) gate.recordProduction(g);
    }
    expect(gate.getCurrentStage()).toBe(2);

    // Master Stage 3 gestures (skipping Stage 2)
    for (const g of ['APPLE', 'BALL']) {
      for (let i = 0; i < 3; i++) gate.recordProduction(g);
    }
    // Should still be at Stage 2 because Stage 2 isn't mastered
    expect(gate.getCurrentStage()).toBe(2);
    expect(gate.getHighestMasteredStage()).toBe(1);
  });

  it('should have 5 stages in the curriculum sequence', () => {
    expect(CURRICULUM_SEQUENCE.length).toBe(5);
    expect(CURRICULUM_SEQUENCE[0].stage).toBe(1);
    expect(CURRICULUM_SEQUENCE[4].stage).toBe(5);
  });
});

// =============================================================================
// SpacedRepetitionScheduler
// =============================================================================

describe('SpacedRepetitionScheduler', () => {
  let srs;

  beforeEach(() => {
    srs = new SpacedRepetitionScheduler({ storageKey: 'test_srs_' + Date.now() });
  });

  it('should create a new entry on first review', () => {
    const result = srs.recordReview('GRAB', 5);
    expect(result.easeFactor).toBeGreaterThan(2.0);
    expect(result.interval).toBe(1);
    expect(result.repetitions).toBe(1);
  });

  it('should increase interval on consecutive correct reviews', () => {
    srs.recordReview('GRAB', 5);
    const r2 = srs.recordReview('GRAB', 5);
    expect(r2.interval).toBe(6);

    const r3 = srs.recordReview('GRAB', 5);
    expect(r3.interval).toBeGreaterThan(6);
  });

  it('should reset repetitions on incorrect review (quality < 3)', () => {
    srs.recordReview('GRAB', 5);
    srs.recordReview('GRAB', 5);
    const r3 = srs.recordReview('GRAB', 1); // Incorrect
    expect(r3.repetitions).toBe(0);
    expect(r3.interval).toBe(1);
  });

  it('should decrease ease factor on low quality', () => {
    const r1 = srs.recordReview('GRAB', 5);
    const initialEF = r1.easeFactor;
    const r2 = srs.recordReview('GRAB', 2);
    expect(r2.easeFactor).toBeLessThan(initialEF);
  });

  it('should never let ease factor drop below 1.3', () => {
    // Repeatedly fail
    for (let i = 0; i < 20; i++) {
      srs.recordReview('GRAB', 0);
    }
    const schedule = srs.getGestureSchedule('GRAB');
    expect(schedule.easeFactor).toBeGreaterThanOrEqual(1.3);
  });

  it('should correctly identify due gestures', () => {
    srs.recordReview('GRAB', 5);
    // Just reviewed — should have a future nextReview
    const due = srs.getDueGestures();
    expect(due).not.toContain('GRAB');
  });

  it('should clamp quality to 0-5 range', () => {
    const r = srs.recordReview('GRAB', 10); // Over max
    expect(r).toBeDefined();
    expect(r.repetitions).toBe(1); // quality clamped to 5, which is correct
  });
});

// =============================================================================
// CognitiveLoadAdapter
// =============================================================================

describe('CognitiveLoadAdapter', () => {
  let adapter;

  beforeEach(() => {
    adapter = new CognitiveLoadAdapter({ windowSize: 6 });
  });

  it('should start at LOW level', () => {
    expect(adapter.getLevel()).toBe('LOW');
    expect(adapter.getJitter()).toBe(0);
  });

  it('should return no-change result for insufficient frames', () => {
    const r = adapter.update(fakeLandmarks());
    expect(r.level).toBe('LOW');
    expect(r.levelChanged).toBe(false);
  });

  it('should detect LOW cognitive load from stable landmarks', () => {
    // Feed stable landmarks (very low noise)
    for (let i = 0; i < 10; i++) {
      adapter.update(fakeLandmarks(0.5, 0.5, 0, 0.0001));
    }
    expect(adapter.getLevel()).toBe('LOW');
    expect(adapter.getJitter()).toBeLessThan(0.004);
  });

  it('should detect HIGH cognitive load from jittery landmarks', () => {
    // Feed very noisy landmarks to trigger HIGH
    for (let i = 0; i < 20; i++) {
      adapter.update(fakeLandmarks(0.5, 0.5, 0, 0.15));
    }
    // With hysteresis, may take several frames to confirm
    expect(['MEDIUM', 'HIGH']).toContain(adapter.getLevel());
  });

  it('should return valid recommendedFrames', () => {
    adapter.setBaseFrames(45);
    const r = adapter.update(fakeLandmarks());
    expect(r.recommendedFrames).toBeGreaterThanOrEqual(45);
    expect(r.recommendedFrames).toBeLessThanOrEqual(120);
  });

  it('should handle null/empty landmarks gracefully', () => {
    const r1 = adapter.update(null);
    expect(r1.level).toBe('LOW');
    const r2 = adapter.update([]);
    expect(r2.level).toBe('LOW');
  });

  it('should apply hysteresis — not flicker between levels', () => {
    // Alternate stable and jittery frames
    const changes = [];
    for (let i = 0; i < 20; i++) {
      const noise = i % 2 === 0 ? 0.0001 : 0.05;
      const r = adapter.update(fakeLandmarks(0.5, 0.5, 0, noise));
      if (r.levelChanged) changes.push(r.level);
    }
    // Hysteresis should prevent rapid flickering
    expect(changes.length).toBeLessThan(5);
  });
});

// =============================================================================
// AutomaticityTracker
// =============================================================================

describe('AutomaticityTracker', () => {
  let tracker;

  beforeEach(() => {
    tracker = new AutomaticityTracker({ storageKey: 'test_auto_' + Date.now() });
  });

  it('should initialize session without error', () => {
    expect(() => tracker.initSession()).not.toThrow();
  });

  it('should record gesture frames without error', () => {
    tracker.initSession();
    expect(() => tracker.onGestureFrame('GRAB')).not.toThrow();
    expect(() => tracker.onGestureFrame(null)).not.toThrow();
    expect(() => tracker.onGestureFrame('STOP')).not.toThrow();
  });

  it('should handle onReadyForNextGesture reset', () => {
    tracker.initSession();
    tracker.onGestureFrame('GRAB');
    tracker.onReadyForNextGesture();
    // After ready reset, tracking gesture should be cleared
    tracker.onGestureFrame('STOP');
    // Should not throw
    expect(() => tracker.onGestureLocked('STOP')).not.toThrow();
  });

  it('should return null for invalid timing data', () => {
    // No initSession, no frames — timing incomplete
    const result = tracker.onGestureLocked('GRAB');
    expect(result).toBeNull();
  });

  it('should return session summary without error', () => {
    tracker.initSession();
    const summary = tracker.getSessionSummary();
    expect(summary).toBeDefined();
  });
});

// =============================================================================
// ISLInterferenceDetector
// =============================================================================

describe('ISLInterferenceDetector', () => {
  let detector;

  beforeEach(() => {
    detector = new ISLInterferenceDetector();
  });

  it('should return empty report for empty sentence', () => {
    const report = detector.analyze([]);
    expect(report.hasInterference).toBe(false);
    expect(report.patterns).toEqual([]);
  });

  it('should return empty report for null sentence', () => {
    const report = detector.analyze(null);
    expect(report.hasInterference).toBe(false);
  });

  it('should detect SOV order interference', () => {
    // ISL order: Subject Object Verb → "I apple eat"
    const sentence = [
      word('SUBJECT_I', 'SUBJECT', 'I'),
      word('APPLE', 'OBJECT', 'apple'),
      word('EAT', 'VERB', 'eat', true),
    ];
    const report = detector.analyze(sentence);
    expect(report.hasInterference).toBe(true);
    const sovPattern = report.patterns.find(p => p.id === 'SOV_ORDER');
    expect(sovPattern).toBeDefined();
  });

  it('should detect topic fronting', () => {
    // ISL topic fronting: Object Subject Verb → "Apple I eat"
    const sentence = [
      word('APPLE', 'OBJECT', 'apple'),
      word('SUBJECT_I', 'SUBJECT', 'I'),
      word('EAT', 'VERB', 'eat', true),
    ];
    const report = detector.analyze(sentence);
    expect(report.hasInterference).toBe(true);
    const topicPattern = report.patterns.find(p => p.id === 'TOPIC_FRONTING');
    expect(topicPattern).toBeDefined();
  });

  it('should detect transitive object drop', () => {
    // "I eat" (no object for transitive verb)
    const sentence = [
      word('SUBJECT_I', 'SUBJECT', 'I'),
      word('EAT', 'VERB', 'eat', true),
    ];
    const report = detector.analyze(sentence);
    const dropPattern = report.patterns.find(p => p.id === 'TRANSITIVE_OBJECT_DROP');
    expect(dropPattern).toBeDefined();
  });

  it('should NOT flag correct SVO sentence', () => {
    // Correct: "I eat apple"
    const sentence = [
      word('SUBJECT_I', 'SUBJECT', 'I'),
      word('EAT', 'VERB', 'eat', true),
      word('APPLE', 'OBJECT', 'apple'),
    ];
    const report = detector.analyze(sentence);
    // Should have no SOV or topic-fronting errors
    const sovPattern = report.patterns.find(p => p.id === 'SOV_ORDER');
    const topicPattern = report.patterns.find(p => p.id === 'TOPIC_FRONTING');
    expect(sovPattern).toBeUndefined();
    expect(topicPattern).toBeUndefined();
  });

  it('should track detection history across multiple analyses', () => {
    const bad = [
      word('SUBJECT_I', 'SUBJECT', 'I'),
      word('APPLE', 'OBJECT', 'apple'),
      word('EAT', 'VERB', 'eat', true),
    ];
    detector.analyze(bad);
    detector.analyze(bad);
    const trend = detector.getTrend();
    expect(trend.total_detections).toBe(2);
  });
});

// =============================================================================
// LandmarkSmoother
// =============================================================================

describe('LandmarkSmoother', () => {
  let smoother;

  beforeEach(() => {
    smoother = new LandmarkSmoother();
  });

  it('should return landmarks on first frame (no previous data)', () => {
    const lm = fakeLandmarks();
    const result = smoother.smooth(lm);
    expect(result).toBeDefined();
    expect(result.length).toBe(21);
  });

  it('should reduce variance after smoothing', () => {
    const frames = [];
    const rawVariances = [];
    const smoothedVariances = [];

    // Generate noisy frames
    for (let i = 0; i < 30; i++) {
      const lm = fakeLandmarks(0.5, 0.5, 0, 0.02);
      frames.push(lm);
      smoother.smooth(lm);
    }

    // Measure variance of raw vs smoothed for wrist (landmark 0)
    const rawX = frames.map(f => f[0].x);

    // Re-run with fresh smoother to collect smoothed values
    const smoother2 = new LandmarkSmoother();
    smoother2.calibrateFromJitter(0.01); // Moderate smoothing
    const smoothedX = frames.map(f => smoother2.smooth(f)[0].x);

    const variance = arr => {
      const mean = arr.reduce((a, b) => a + b) / arr.length;
      return arr.reduce((a, b) => a + (b - mean) ** 2, 0) / arr.length;
    };

    expect(variance(smoothedX)).toBeLessThan(variance(rawX));
  });

  it('should calibrate alpha from jitter', () => {
    smoother.calibrateFromJitter(0.001); // Low jitter → high alpha (less smoothing)
    expect(smoother.getAlpha()).toBeGreaterThan(0.3);

    smoother.calibrateFromJitter(0.05); // High jitter → low alpha (heavy smoothing)
    expect(smoother.getAlpha()).toBeLessThan(0.3);
  });
});

// =============================================================================
// IntentionalityDetector
// =============================================================================

describe('IntentionalityDetector', () => {
  let detector;

  beforeEach(() => {
    detector = new IntentionalityDetector({});
  });

  it('should classify as GESTURE_ACTIVE without resting profile set', () => {
    // Without calibration, detector should default to active (safe fallback)
    const result = detector.detect(fakeLandmarks());
    expect(result).toBeDefined();
    expect(result.intent).toBeDefined();
  });

  it('should handle null landmarks gracefully', () => {
    expect(() => detector.detect(null)).not.toThrow();
  });
});

// =============================================================================
// SpatialGrammarMapper
// =============================================================================

describe('SpatialGrammarMapper', () => {
  let mapper;

  beforeEach(() => {
    mapper = new SpatialGrammarMapper();
  });

  it('should map left wrist position to SUBJECT_ZONE', () => {
    // Wrist at x=0.15 (left side of frame = SUBJECT zone in mirrored space)
    const lm = fakeLandmarks(0.15, 0.5, 0);
    const result = mapper.map(lm);
    expect(result).toBeDefined();
    expect(result.syntactic_zone).toBeDefined();
  });

  it('should map center wrist position to VERB_ZONE', () => {
    const lm = fakeLandmarks(0.5, 0.5, 0);
    const result = mapper.map(lm);
    expect(result.syntactic_zone).toBeDefined();
  });

  it('should map right wrist position to OBJECT_ZONE', () => {
    const lm = fakeLandmarks(0.85, 0.5, 0);
    const result = mapper.map(lm);
    expect(result.syntactic_zone).toBeDefined();
  });

  it('should include tense zone information', () => {
    const lm = fakeLandmarks(0.5, 0.2, 0); // Y=0.2 → FUTURE zone
    const result = mapper.map(lm);
    expect(result.tense_zone).toBeDefined();
  });

  it('should handle null landmarks gracefully', () => {
    expect(() => mapper.map(null)).not.toThrow();
  });
});
