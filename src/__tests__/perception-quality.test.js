/**
 * Reliability #3 — perception-base input-quality gate.
 * Verifies the gate passes a normal in-frame hand and rejects clearly-broken
 * input (missing points, hand too far, hand mostly off-screen) so the classifier
 * never emits confident garbage.
 */
import { describe, it, expect } from 'vitest';
import { isLandmarkQualityOk } from '../utils/gestureDetection.js';

// 21 points spread across ~0.3 of the frame, centred — a normal hand.
const goodHand = Array.from({ length: 21 }, (_, i) => ({
  x: 0.40 + (i % 5) * 0.06,
  y: 0.40 + Math.floor(i / 5) * 0.06,
  z: 0,
}));

describe('Fix #3 — landmark quality gate', () => {
  it('passes a normal in-frame hand', () => {
    expect(isLandmarkQualityOk(goodHand)).toBe(true);
  });

  it('rejects missing / incomplete landmarks', () => {
    expect(isLandmarkQualityOk(null)).toBe(false);
    expect(isLandmarkQualityOk(goodHand.slice(0, 10))).toBe(false);
  });

  it('rejects non-finite coordinates', () => {
    const bad = goodHand.map((p) => ({ ...p }));
    bad[8] = { x: NaN, y: 0.5, z: 0 };
    expect(isLandmarkQualityOk(bad)).toBe(false);
  });

  it('rejects a hand too small/far (unreadable finger geometry)', () => {
    const tiny = Array.from({ length: 21 }, (_, i) => ({ x: 0.50 + (i % 5) * 0.008, y: 0.50 + Math.floor(i / 5) * 0.008, z: 0 }));
    expect(isLandmarkQualityOk(tiny)).toBe(false);
  });

  it('rejects a hand mostly clipped off-screen', () => {
    const clipped = Array.from({ length: 21 }, (_, i) => ({ x: i < 14 ? 0.0 : 0.5, y: 0.5, z: 0 }));
    expect(isLandmarkQualityOk(clipped)).toBe(false);
  });
});
