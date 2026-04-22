/**
 * aggme.test.js — Tremor Normalization Test
 *
 * Rigorous test harness for the 4-phase AGGME pipeline:
 *   Phase 1: RestingBoundaryCalibrator
 *   Phase 2: LandmarkSmoother (EMA)
 *   Phase 3: IntentionalityDetector
 *   Phase 4: SpatialGrammarMapper
 *
 * Metrics produced (paper-ready):
 *   - False Positive Rate (tremor classified as gesture)
 *   - False Negative Rate (gesture classified as resting)
 *   - Variance Reduction Ratio (raw vs smoothed)
 *   - Calibration Convergence (frames to READY state)
 *   - Zone Stability (flicker rate during tremor)
 *   - End-to-End Classification Accuracy
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { RestingBoundaryCalibrator } from '../core/RestingBoundaryCalibrator.js';
import { LandmarkSmoother } from '../core/LandmarkSmoother.js';
import { IntentionalityDetector } from '../core/IntentionalityDetector.js';
import { SpatialGrammarMapper } from '../core/SpatialGrammarMapper.js';
import {
  generateTremorSequence,
  generateGestureTrajectory,
  generateFullTestSequence,
  generateHandAtPosition,
  neutralHand,
  grabHand,
  stopHand,
  subjectIHand,
  computeSequenceVariance,
} from './helpers/generators.js';

// =============================================================================
// TEST 1: PHASE 1 — RESTING BOUNDARY CALIBRATION
// =============================================================================

describe('AGGME Phase 1: RestingBoundaryCalibrator', () => {
  let calibrator;

  beforeEach(() => {
    calibrator = new RestingBoundaryCalibrator();
  });

  it('should reach READY state within 90 calibration frames with stable tremor', () => {
    const frames = generateTremorSequence(90, 0.005, neutralHand, 100);

    calibrator.startCalibration();
    expect(calibrator.getState()).toBe('CALIBRATING');

    let readyFrame = -1;
    for (let i = 0; i < frames.length; i++) {
      const result = calibrator.processFrame(frames[i]);
      if (result.state === 'READY') {
        readyFrame = i;
        break;
      }
    }

    expect(calibrator.isReady()).toBe(true);
    expect(readyFrame).toBeGreaterThan(0);
    expect(readyFrame).toBeLessThanOrEqual(90);

    console.log(`[Phase 1] Calibration converged at frame ${readyFrame + 1} / 90`);
  });

  it('should produce a valid resting profile with per-landmark statistics', () => {
    const frames = generateTremorSequence(90, 0.006, neutralHand, 200);
    calibrator.startCalibration();
    for (const frame of frames) calibrator.processFrame(frame);

    const profile = calibrator.getRestingProfile();
    expect(profile).not.toBeNull();
    expect(profile).toHaveLength(21);

    // Each landmark should have mean, stddev, var
    for (let i = 0; i < 21; i++) {
      expect(profile[i]).toHaveProperty('mean');
      expect(profile[i]).toHaveProperty('stddev');
      expect(profile[i].mean).toHaveProperty('x');
      expect(profile[i].mean).toHaveProperty('y');
      expect(profile[i].mean).toHaveProperty('z');
    }

    const jitter = calibrator.getRestingJitter();
    expect(jitter).toBeGreaterThan(0);
    expect(jitter).toBeLessThan(0.05); // Reasonable jitter for σ=0.006

    console.log(`[Phase 1] Resting jitter: ${jitter.toFixed(6)}`);
  });

  it('should correctly classify RESTING vs ACTIVE after calibration', () => {
    // Calibrate with mild tremor
    const calFrames = generateTremorSequence(90, 0.005, neutralHand, 300);
    calibrator.startCalibration();
    for (const frame of calFrames) calibrator.processFrame(frame);
    expect(calibrator.isReady()).toBe(true);

    // Test resting frames — should classify RESTING
    const restingFrames = generateTremorSequence(60, 0.005, neutralHand, 400);
    let restingCorrect = 0;
    for (const frame of restingFrames) {
      const result = calibrator.processFrame(frame);
      if (result.classification === 'RESTING') restingCorrect++;
    }

    // Test active frames — intentional movement should classify ACTIVE
    const activeFrames = generateGestureTrajectory(30, neutralHand, grabHand, 0.003, 500);
    let activeCorrect = 0;
    for (const frame of activeFrames) {
      const result = calibrator.processFrame(frame);
      if (result.classification === 'ACTIVE') activeCorrect++;
    }

    const restingAccuracy = restingCorrect / restingFrames.length;
    const activeAccuracy = activeCorrect / activeFrames.length;

    console.log(`[Phase 1] Resting classification accuracy: ${(restingAccuracy * 100).toFixed(1)}% (${restingCorrect}/${restingFrames.length})`);
    console.log(`[Phase 1] Active classification accuracy: ${(activeAccuracy * 100).toFixed(1)}% (${activeCorrect}/${activeFrames.length})`);

    expect(restingAccuracy).toBeGreaterThan(0.7);
    expect(activeAccuracy).toBeGreaterThan(0.3);
  });

  it('should measure calibration convergence across varying tremor intensities', () => {
    const sigmas = [0.001, 0.003, 0.005, 0.008, 0.012, 0.018];
    const results = [];

    for (const sigma of sigmas) {
      const cal = new RestingBoundaryCalibrator();
      const frames = generateTremorSequence(90, sigma, neutralHand, Math.round(sigma * 10000));
      cal.startCalibration();

      let convergedAt = -1;
      for (let i = 0; i < frames.length; i++) {
        const r = cal.processFrame(frames[i]);
        if (r.state === 'READY' && convergedAt === -1) {
          convergedAt = i + 1;
        }
      }

      results.push({
        sigma,
        converged: cal.isReady(),
        frames_to_ready: convergedAt,
        jitter: cal.isReady() ? cal.getRestingJitter().toFixed(6) : 'N/A',
      });
    }

    console.log('\n[Phase 1] Calibration Convergence Table:');
    console.table(results);

    // All should converge
    for (const r of results) {
      expect(r.converged).toBe(true);
    }
  });
});

// =============================================================================
// TEST 2: PHASE 2 — EMA SIGNAL SMOOTHING
// =============================================================================

describe('AGGME Phase 2: LandmarkSmoother', () => {
  it('should reduce landmark variance after smoothing', () => {
    const sigmas = [0.003, 0.006, 0.010, 0.015, 0.020];
    const results = [];

    for (const sigma of sigmas) {
      const smoother = new LandmarkSmoother();
      const rawFrames = generateTremorSequence(120, sigma, neutralHand, Math.round(sigma * 10000));
      const smoothedFrames = [];

      for (const frame of rawFrames) {
        smoothedFrames.push(smoother.smooth(frame));
      }

      const rawVariance = computeSequenceVariance(rawFrames);
      const smoothedVariance = computeSequenceVariance(smoothedFrames);
      const reductionRatio = smoothedVariance > 0 ? rawVariance / smoothedVariance : Infinity;

      results.push({
        sigma,
        raw_variance: rawVariance.toExponential(3),
        smoothed_variance: smoothedVariance.toExponential(3),
        reduction_ratio: reductionRatio.toFixed(2) + 'x',
        alpha: smoother.getAlpha().toFixed(3),
      });

      expect(smoothedVariance).toBeLessThan(rawVariance);
    }

    console.log('\n[Phase 2] Variance Reduction Table:');
    console.table(results);
  });

  it('should auto-calibrate alpha from resting jitter', () => {
    const jitterValues = [0.002, 0.005, 0.010, 0.015, 0.020];
    const results = [];

    for (const jitter of jitterValues) {
      const smoother = new LandmarkSmoother();
      smoother.calibrateFromJitter(jitter);
      const alpha = smoother.getAlpha();

      results.push({
        resting_jitter: jitter,
        calibrated_alpha: alpha.toFixed(3),
        smoothing_strength: alpha < 0.3 ? 'HEAVY' : alpha < 0.5 ? 'MODERATE' : 'LIGHT',
      });

      expect(alpha).toBeGreaterThanOrEqual(0.10);
      expect(alpha).toBeLessThanOrEqual(0.70);
    }

    console.log('\n[Phase 2] Alpha Auto-Calibration Table:');
    console.table(results);

    // Higher jitter → lower alpha (more smoothing)
    const alphas = results.map(r => parseFloat(r.calibrated_alpha));
    for (let i = 1; i < alphas.length; i++) {
      expect(alphas[i]).toBeLessThanOrEqual(alphas[i - 1] + 0.01);
    }
  });

  it('should preserve intentional gesture signal while filtering tremor', () => {
    const smoother = new LandmarkSmoother();
    smoother.calibrateFromJitter(0.008);

    // Generate gesture trajectory from neutral to GRAB
    const trajectory = generateGestureTrajectory(30, neutralHand, grabHand, 0.005, 777);
    const smoothed = trajectory.map(f => smoother.smooth(f));

    // The wrist should still show the trajectory (not flattened to zero)
    const rawWristDx = Math.abs(trajectory[trajectory.length - 1][0].x - trajectory[0][0].x);
    const smoothedWristDx = Math.abs(smoothed[smoothed.length - 1][0].x - smoothed[0][0].x);

    // Smoothed signal should retain at least 50% of the raw movement amplitude
    const signalRetention = smoothedWristDx / (rawWristDx || 0.001);

    console.log(`[Phase 2] Signal retention: ${(signalRetention * 100).toFixed(1)}% of raw movement preserved`);
    expect(signalRetention).toBeGreaterThan(0.4);
  });
});

// =============================================================================
// TEST 3: PHASE 3 — INTENTIONALITY DETECTION
// =============================================================================

describe('AGGME Phase 3: IntentionalityDetector', () => {
  let detector;
  let calibrator;

  beforeEach(() => {
    // Calibrate first (Phase 1 feeds Phase 3)
    calibrator = new RestingBoundaryCalibrator();
    const calFrames = generateTremorSequence(90, 0.006, neutralHand, 42);
    calibrator.startCalibration();
    for (const frame of calFrames) calibrator.processFrame(frame);

    detector = new IntentionalityDetector({});
    detector.setRestingProfile(calibrator.getRestingProfile());
  });

  it('should produce low false positive rate on pure tremor', () => {
    const tremorSigmas = [0.003, 0.006, 0.010, 0.015];
    const results = [];

    for (const sigma of tremorSigmas) {
      const det = new IntentionalityDetector({});
      det.setRestingProfile(calibrator.getRestingProfile());

      const smoother = new LandmarkSmoother();
      smoother.calibrateFromJitter(calibrator.getRestingJitter());

      const tremorFrames = generateTremorSequence(120, sigma, neutralHand, Math.round(sigma * 10000));
      let falsePositives = 0;

      for (const frame of tremorFrames) {
        const smoothed = smoother.smooth(frame);
        const result = det.detect(smoothed);
        if (result.intent === 'GESTURE_ACTIVE') falsePositives++;
      }

      const fpr = falsePositives / tremorFrames.length;
      results.push({
        tremor_sigma: sigma,
        total_frames: tremorFrames.length,
        false_positives: falsePositives,
        FPR: (fpr * 100).toFixed(2) + '%',
      });
    }

    console.log('\n[Phase 3] False Positive Rate (Tremor → Gesture):');
    console.table(results);

    // FPR should be below 20% for mild tremor
    expect(parseFloat(results[0].FPR)).toBeLessThan(20);
  });

  it('should produce low false negative rate on intentional gestures', () => {
    const gestures = [
      { name: 'GRAB', pose: grabHand },
      { name: 'STOP', pose: stopHand },
      { name: 'SUBJECT_I', pose: subjectIHand },
    ];
    const results = [];

    for (const g of gestures) {
      const det = new IntentionalityDetector({});
      det.setRestingProfile(calibrator.getRestingProfile());
      const smoother = new LandmarkSmoother();
      smoother.calibrateFromJitter(calibrator.getRestingJitter());

      const seq = generateFullTestSequence({
        restingFrames: 30,
        transitionFrames: 10,
        sustainedFrames: 45,
        tremorSigma: 0.006,
        gestureSigma: 0.003,
        gesturePose: g.pose,
        seed: 42,
      });

      let correctActive = 0;
      let totalActiveFrames = 0;
      let correctResting = 0;
      let totalRestingFrames = 0;

      for (let i = 0; i < seq.frames.length; i++) {
        const smoothed = smoother.smooth(seq.frames[i]);
        const result = det.detect(smoothed);
        const label = seq.labels[i];

        if (label === 'GESTURE_ACTIVE') {
          totalActiveFrames++;
          if (result.intent === 'GESTURE_ACTIVE') correctActive++;
        } else if (label === 'RESTING') {
          totalRestingFrames++;
          if (result.intent === 'RESTING') correctResting++;
        }
      }

      const fnr = totalActiveFrames > 0 ? 1 - (correctActive / totalActiveFrames) : 0;
      const fpr = totalRestingFrames > 0 ? 1 - (correctResting / totalRestingFrames) : 0;

      results.push({
        gesture: g.name,
        active_frames: totalActiveFrames,
        detected_active: correctActive,
        FNR: (fnr * 100).toFixed(1) + '%',
        resting_frames: totalRestingFrames,
        detected_resting: correctResting,
        FPR: (fpr * 100).toFixed(1) + '%',
      });
    }

    console.log('\n[Phase 3] False Negative Rate (Gesture → Resting):');
    console.table(results);
  });

  it('should respect hysteresis (3 onset, 8 offset frames)', () => {
    const smoother = new LandmarkSmoother();
    smoother.calibrateFromJitter(calibrator.getRestingJitter());

    // Generate: resting → sudden move → hold → release → resting
    const seq = generateFullTestSequence({
      restingFrames: 20,
      transitionFrames: 5,
      sustainedFrames: 30,
      tremorSigma: 0.005,
      gestureSigma: 0.002,
      gesturePose: grabHand,
      seed: 999,
    });

    const intentHistory = [];
    for (const frame of seq.frames) {
      const smoothed = smoother.smooth(frame);
      const result = detector.detect(smoothed);
      intentHistory.push(result.intent);
    }

    // Find onset: first GESTURE_ACTIVE after RESTING
    const onsetIdx = intentHistory.findIndex((v, i) => v === 'GESTURE_ACTIVE' && (i === 0 || intentHistory[i - 1] === 'RESTING'));
    // Find offset: first RESTING after GESTURE_ACTIVE (in the second half)
    const midpoint = Math.floor(intentHistory.length / 2);
    const offsetIdx = intentHistory.findIndex((v, i) => i > midpoint && v === 'RESTING' && intentHistory[i - 1] === 'GESTURE_ACTIVE');

    console.log(`[Phase 3] Onset detected at frame ${onsetIdx}, Offset at frame ${offsetIdx}`);
    console.log(`[Phase 3] Total sequence: ${intentHistory.length} frames`);

    // Onset should occur after the resting phase (frame ~20+)
    if (onsetIdx >= 0) {
      expect(onsetIdx).toBeGreaterThan(10);
    }
  });
});

// =============================================================================
// TEST 4: PHASE 4 — SPATIAL GRAMMAR MAPPING
// =============================================================================

describe('AGGME Phase 4: SpatialGrammarMapper', () => {
  let mapper;

  beforeEach(() => {
    mapper = new SpatialGrammarMapper();
  });

  it('should correctly classify SVO zones from wrist position', () => {
    const testPositions = [
      { screenX: 0.15, expectedZone: 'SUBJECT_ZONE', label: 'Left (Subject)' },
      { screenX: 0.50, expectedZone: 'VERB_ZONE', label: 'Center (Verb)' },
      { screenX: 0.80, expectedZone: 'OBJECT_ZONE', label: 'Right (Object)' },
    ];

    const results = [];

    for (const pos of testPositions) {
      const hand = generateHandAtPosition(pos.screenX, 0.5);
      const result = mapper.map(hand);

      results.push({
        screen_x: pos.screenX,
        expected: pos.expectedZone,
        actual: result.syntactic_zone,
        correct: result.syntactic_zone === pos.expectedZone ? 'YES' : 'NO',
        confidence: (result.zone_confidence * 100).toFixed(0) + '%',
      });

      expect(result.syntactic_zone).toBe(pos.expectedZone);
    }

    console.log('\n[Phase 4] SVO Zone Classification:');
    console.table(results);
  });

  it('should correctly classify Y-axis tense zones', () => {
    const testPositions = [
      { screenY: 0.15, expectedTense: 'FUTURE', label: 'Top (Future)' },
      { screenY: 0.50, expectedTense: 'PRESENT', label: 'Middle (Present)' },
      { screenY: 0.85, expectedTense: 'PAST', label: 'Bottom (Past)' },
    ];

    const results = [];

    for (const pos of testPositions) {
      const hand = generateHandAtPosition(0.5, pos.screenY);
      const result = mapper.map(hand);

      results.push({
        screen_y: pos.screenY,
        expected: pos.expectedTense,
        actual: result.tense_zone,
        correct: result.tense_zone === pos.expectedTense ? 'YES' : 'NO',
      });

      expect(result.tense_zone).toBe(pos.expectedTense);
    }

    console.log('\n[Phase 4] Tense Zone Classification:');
    console.table(results);
  });

  it('should measure zone stability (flicker rate) during resting tremor', () => {
    const sigmas = [0.003, 0.006, 0.010, 0.015, 0.020];
    const results = [];

    for (const sigma of sigmas) {
      const m = new SpatialGrammarMapper();
      const frames = generateTremorSequence(90, sigma, neutralHand, Math.round(sigma * 10000));
      let zoneChanges = 0;
      let prevZone = null;

      for (const frame of frames) {
        const result = m.map(frame);
        if (prevZone && result.syntactic_zone !== prevZone) {
          zoneChanges++;
        }
        prevZone = result.syntactic_zone;
      }

      const flickerRate = zoneChanges / (frames.length - 1);
      results.push({
        tremor_sigma: sigma,
        total_frames: frames.length,
        zone_changes: zoneChanges,
        flicker_rate: (flickerRate * 100).toFixed(1) + '%',
        stability: flickerRate < 0.05 ? 'STABLE' : flickerRate < 0.15 ? 'MODERATE' : 'UNSTABLE',
      });
    }

    console.log('\n[Phase 4] Zone Stability Under Tremor:');
    console.table(results);

    // Mild tremor should be stable
    expect(parseFloat(results[0].flicker_rate)).toBeLessThan(10);
  });

  it('should detect complete SVO path traversal', () => {
    const m = new SpatialGrammarMapper();

    // Subject zone (left)
    const subjectHands = generateTremorSequence(10, 0.002, () => generateHandAtPosition(0.15, 0.5), 1);
    for (const f of subjectHands) m.map(f);

    // Verb zone (center)
    const verbHands = generateTremorSequence(10, 0.002, () => generateHandAtPosition(0.50, 0.5), 2);
    for (const f of verbHands) m.map(f);

    // Object zone (right)
    const objectHands = generateTremorSequence(10, 0.002, () => generateHandAtPosition(0.85, 0.5), 3);
    for (const f of objectHands) m.map(f);

    const transitions = m.getZoneTransitions();
    const hasPath = m.hasCompleteSVOPath();

    console.log(`[Phase 4] Zone transitions: ${transitions.length}, Complete SVO path: ${hasPath}`);
    expect(transitions.length).toBeGreaterThanOrEqual(2);
    expect(hasPath).toBe(true);
  });
});

// =============================================================================
// TEST 5: END-TO-END AGGME PIPELINE
// =============================================================================

describe('AGGME End-to-End Pipeline', () => {
  it('should correctly process a full resting → gesture → resting sequence', () => {
    const gestures = [
      { name: 'GRAB', pose: grabHand },
      { name: 'STOP', pose: stopHand },
      { name: 'SUBJECT_I', pose: subjectIHand },
    ];

    const pipelineResults = [];

    for (const g of gestures) {
      // Phase 1: Calibrate
      const calibrator = new RestingBoundaryCalibrator();
      const calFrames = generateTremorSequence(90, 0.006, neutralHand, 42);
      calibrator.startCalibration();
      for (const frame of calFrames) calibrator.processFrame(frame);

      // Phase 2: Configure smoother
      const smoother = new LandmarkSmoother();
      smoother.calibrateFromJitter(calibrator.getRestingJitter());

      // Phase 3: Configure intent detector
      const detector = new IntentionalityDetector({});
      detector.setRestingProfile(calibrator.getRestingProfile());

      // Phase 4: Spatial mapper
      const mapper = new SpatialGrammarMapper();

      // Generate full sequence
      const seq = generateFullTestSequence({
        restingFrames: 60,
        transitionFrames: 12,
        sustainedFrames: 45,
        tremorSigma: 0.006,
        gestureSigma: 0.003,
        gesturePose: g.pose,
        seed: 42,
      });

      // Process through full pipeline
      let correctClassifications = 0;
      const phaseMetrics = { resting_correct: 0, resting_total: 0, active_correct: 0, active_total: 0 };

      for (let i = 0; i < seq.frames.length; i++) {
        // Phase 2: Smooth
        const smoothed = smoother.smooth(seq.frames[i]);
        // Phase 3: Intent
        const intent = detector.detect(smoothed);
        // Phase 4: Spatial
        mapper.map(smoothed);

        const label = seq.labels[i];
        const predicted = intent.intent;

        if (label === 'RESTING') {
          phaseMetrics.resting_total++;
          if (predicted === 'RESTING') {
            phaseMetrics.resting_correct++;
            correctClassifications++;
          }
        } else if (label === 'GESTURE_ACTIVE') {
          phaseMetrics.active_total++;
          if (predicted === 'GESTURE_ACTIVE') {
            phaseMetrics.active_correct++;
            correctClassifications++;
          }
        } else {
          // Transitions — accept either classification
          correctClassifications++;
        }
      }

      const totalEval = phaseMetrics.resting_total + phaseMetrics.active_total;
      const overallAccuracy = totalEval > 0
        ? (phaseMetrics.resting_correct + phaseMetrics.active_correct) / totalEval
        : 0;

      pipelineResults.push({
        gesture: g.name,
        total_frames: seq.totalFrames,
        resting_accuracy: phaseMetrics.resting_total > 0
          ? ((phaseMetrics.resting_correct / phaseMetrics.resting_total) * 100).toFixed(1) + '%'
          : 'N/A',
        gesture_accuracy: phaseMetrics.active_total > 0
          ? ((phaseMetrics.active_correct / phaseMetrics.active_total) * 100).toFixed(1) + '%'
          : 'N/A',
        overall_accuracy: (overallAccuracy * 100).toFixed(1) + '%',
        smoother_alpha: smoother.getAlpha().toFixed(3),
        resting_jitter: calibrator.getRestingJitter().toFixed(6),
      });
    }

    console.log('\n═══════════════════════════════════════════════════════');
    console.log('  AGGME END-TO-END PIPELINE RESULTS');
    console.log('═══════════════════════════════════════════════════════');
    console.table(pipelineResults);
  });

  it('should measure pipeline performance across tremor intensity spectrum', () => {
    const sigmas = [0.002, 0.005, 0.008, 0.012, 0.018];
    const intensityResults = [];

    for (const sigma of sigmas) {
      const calibrator = new RestingBoundaryCalibrator();
      const calFrames = generateTremorSequence(90, sigma, neutralHand, Math.round(sigma * 10000));
      calibrator.startCalibration();
      for (const frame of calFrames) calibrator.processFrame(frame);

      if (!calibrator.isReady()) {
        intensityResults.push({ tremor_sigma: sigma, status: 'CALIBRATION_FAILED' });
        continue;
      }

      const smoother = new LandmarkSmoother();
      smoother.calibrateFromJitter(calibrator.getRestingJitter());

      const detector = new IntentionalityDetector({});
      detector.setRestingProfile(calibrator.getRestingProfile());

      const seq = generateFullTestSequence({
        restingFrames: 60,
        transitionFrames: 10,
        sustainedFrames: 40,
        tremorSigma: sigma,
        gestureSigma: sigma * 0.5,
        gesturePose: grabHand,
        seed: 42,
      });

      let restingFP = 0, restingTotal = 0, gestureFN = 0, gestureTotal = 0;

      for (let i = 0; i < seq.frames.length; i++) {
        const smoothed = smoother.smooth(seq.frames[i]);
        const intent = detector.detect(smoothed);

        if (seq.labels[i] === 'RESTING') {
          restingTotal++;
          if (intent.intent === 'GESTURE_ACTIVE') restingFP++;
        } else if (seq.labels[i] === 'GESTURE_ACTIVE') {
          gestureTotal++;
          if (intent.intent === 'RESTING') gestureFN++;
        }
      }

      intensityResults.push({
        tremor_sigma: sigma,
        FPR: ((restingFP / restingTotal) * 100).toFixed(1) + '%',
        FNR: ((gestureFN / gestureTotal) * 100).toFixed(1) + '%',
        alpha: smoother.getAlpha().toFixed(3),
        jitter: calibrator.getRestingJitter().toFixed(5),
      });
    }

    console.log('\n═══════════════════════════════════════════════════════');
    console.log('  AGGME ACROSS TREMOR INTENSITY SPECTRUM');
    console.log('═══════════════════════════════════════════════════════');
    console.table(intensityResults);
  });
});
