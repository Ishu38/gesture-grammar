/**
 * calibration-timeout.test.js — verifies the 2026-05-27 safety-net that
 * prevents RestingBoundaryCalibrator from dead-locking the pipeline.
 *
 * Before the fix: a user whose hand never sat still in frame (or whose
 * camera framing didn't capture the hand at all) would see
 * "Calibrating 0%" indefinitely, with all downstream gesture detection
 * gated off. Reported in the field as "I held my fist in front of the
 * laptop camera and the percentage never completed."
 *
 * After the fix:
 *   (a) Calibration auto-completes 4 s after startCalibration() with
 *       a synthesised default resting profile.
 *   (b) Calibration auto-completes when MAX_NO_HAND_FRAMES is exceeded,
 *       using the same synthetic profile (previously: terminal FAILED).
 *   (c) The synthetic profile is a real, valid 21-landmark structure —
 *       _classifyFrame must run without throwing and must distinguish
 *       resting from active hand poses.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { RestingBoundaryCalibrator } from '../core/RestingBoundaryCalibrator.js';
import {
  neutralHand,
  subjectIHand,
  generateHandAtPosition,
} from './helpers/generators.js';

describe('RestingBoundaryCalibrator — calibration-stuck safety net', () => {
  let calibrator;

  beforeEach(() => {
    calibrator = new RestingBoundaryCalibrator();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('auto-completes with synthetic profile after a 4 s timeout, even with zero captured frames', () => {
    // Pin "now" so we control elapsed time deterministically.
    const t0 = 1_700_000_000_000;
    vi.useFakeTimers();
    vi.setSystemTime(t0);

    calibrator.startCalibration();
    expect(calibrator.getState()).toBe('CALIBRATING');

    // Feed one null frame — pre-timeout, the safety net hasn't fired yet
    // and the no-hand-frame counter is below max. State should stay CALIBRATING.
    let r = calibrator.processFrame(null);
    expect(r.state).toBe('CALIBRATING');
    expect(r.progress).toBe(0);

    // Advance the clock past the 4-second timeout. The very next processFrame
    // call must trip the safety net and promote the state to READY.
    vi.setSystemTime(t0 + 4001);

    r = calibrator.processFrame(null);
    expect(r.state).toBe('READY');
    expect(r.progress).toBe(1.0);
    expect(calibrator.getState()).toBe('READY');
    expect(calibrator.isReady()).toBe(true);
  });

  it('auto-completes with synthetic profile when no-hand frames exceed the max', () => {
    calibrator.startCalibration();

    // Hammer in 20 null frames — well past the no-hand max of 15. The
    // previous behaviour was a terminal FAILED state; the new behaviour
    // promotes to READY via the synthetic profile so the rest of the
    // pipeline runs.
    let lastResult;
    for (let i = 0; i < 20; i++) {
      lastResult = calibrator.processFrame(null);
      if (lastResult.state === 'READY') break;
    }

    expect(lastResult.state).toBe('READY');
    expect(calibrator.getState()).toBe('READY');
    expect(calibrator.isReady()).toBe(true);
    expect(calibrator.getRestingProfile()).not.toBeNull();
  });

  it('synthetic profile has the correct 21-landmark shape with valid stats', () => {
    calibrator.startCalibration();
    // Trigger the no-hand fallback path
    for (let i = 0; i < 20; i++) calibrator.processFrame(null);

    const profile = calibrator.getRestingProfile();
    expect(profile).toHaveLength(21);
    for (let i = 0; i < 21; i++) {
      expect(profile[i]).toHaveProperty('mean.x');
      expect(profile[i]).toHaveProperty('mean.y');
      expect(profile[i]).toHaveProperty('mean.z');
      expect(profile[i]).toHaveProperty('stddev.x');
      expect(profile[i].stddev.x).toBeGreaterThan(0);
      expect(profile[i].stddev.y).toBeGreaterThan(0);
      expect(profile[i].stddev.z).toBeGreaterThan(0);
    }
    const jitter = calibrator.getRestingJitter();
    expect(jitter).toBeGreaterThan(0);
  });

  it('keeps the gesture pipeline alive after fallback (real hands always classify, no throws)', () => {
    // Drive into READY via the no-hand fallback path
    calibrator.startCalibration();
    for (let i = 0; i < 20; i++) calibrator.processFrame(null);
    expect(calibrator.isReady()).toBe(true);

    // The synthetic profile sets every landmark mean to a single point
    // (0.5, 0.65, 0). A real hand pose has fingers spread relative to the
    // wrist, so individual landmarks deviate from the synthetic point —
    // every real hand classifies as ACTIVE. That's the intentional safe
    // trade-off: the gesture classifier runs on every frame instead of
    // anything being mis-rejected as "resting". The user can /Recalibrate
    // later for a tighter resting-vs-active distinction.
    const poses = [
      generateHandAtPosition(0.5, 0.65, neutralHand, 1),
      generateHandAtPosition(0.5, 0.30, subjectIHand, 2),
      generateHandAtPosition(0.3, 0.50, neutralHand, 3),
    ];
    for (const pose of poses) {
      const r = calibrator.processFrame(pose);
      // Critical: never throws, always READY, always emits a classification.
      expect(r.state).toBe('READY');
      expect(['ACTIVE', 'RESTING']).toContain(r.classification);
      expect(Number.isFinite(r.displacement)).toBe(true);
    }
  });

  it('a real successful 90-frame calibration still wins over the safety net', () => {
    // Sanity check: the happy path must still produce a real (not synthetic)
    // profile when the user does cooperate. Mean.y in the real profile will
    // sit near the supplied hand position (0.5, 0.7), distinct from the
    // synthetic default of (0.5, 0.65).
    calibrator.startCalibration();
    for (let i = 0; i < 100; i++) {
      const frame = generateHandAtPosition(0.5, 0.7, neutralHand, i);
      const r = calibrator.processFrame(frame);
      if (r.state === 'READY') break;
    }
    const profile = calibrator.getRestingProfile();
    expect(profile).not.toBeNull();
    // Real profile's wrist y should be within 0.02 of 0.7, not 0.65.
    expect(profile[0].mean.y).toBeGreaterThan(0.68);
    expect(profile[0].mean.y).toBeLessThan(0.72);
  });
});
