/**
 * Reliability Program guardrails (Fix #1 single-source-of-truth, Fix #2 separability).
 *
 * These tests make two whole classes of bug IMPOSSIBLE to ship:
 *   1. Drift — the finger-matcher must agree with the canonical gestureCatalog,
 *      and every catalogued gesture must exist in the grammar LEXICON.
 *   2. Collision — no two gestures may share the same signature (finger pattern
 *      + discriminator), so "I → BALL" can never recur silently.
 */
import { describe, it, expect } from 'vitest';
import { GESTURE_CATALOG, signatureOf } from '../data/gestureCatalog.js';
import { GESTURE_FINGER_PATTERNS } from '../utils/fingerAnalysis.js';
import { LEXICON } from '../data/Lexicon.js';

describe('Fix #1 — single source of truth (no drift)', () => {
  it('every catalogued gesture exists in the grammar LEXICON', () => {
    const missing = Object.keys(GESTURE_CATALOG).filter((id) => !LEXICON[id]);
    expect(missing, `not in LEXICON: ${missing.join(', ')}`).toEqual([]);
  });

  it('the finger-matcher patterns match the canonical catalog exactly', () => {
    for (const [id, pattern] of Object.entries(GESTURE_FINGER_PATTERNS)) {
      expect(GESTURE_CATALOG[id], `matcher has "${id}" but catalog does not`).toBeDefined();
      expect(pattern, `finger-pattern drift for "${id}"`).toEqual(GESTURE_CATALOG[id].fingers);
    }
  });
});

describe('Fix #2 — separability (no two gestures collide)', () => {
  it('every gesture signature is unique (except declared aliases)', () => {
    const seen = {};
    const collisions = [];
    for (const [id, entry] of Object.entries(GESTURE_CATALOG)) {
      if (entry.aliasOf) continue; // SHE is an intentional alias of HE
      const sig = signatureOf(entry);
      if (seen[sig]) collisions.push(`${seen[sig]} ⟷ ${id}  [${sig}]`);
      else seen[sig] = id;
    }
    expect(collisions, `colliding gestures:\n  ${collisions.join('\n  ')}`).toEqual([]);
  });

  it('finger-curl-only would collide — proving the discriminator is necessary', () => {
    // Documents WHY signatures need the extra dimension: many gestures share a
    // finger pattern. This is the measured fact behind "I → BALL".
    const byFingers = {};
    for (const [id, e] of Object.entries(GESTURE_CATALOG)) {
      if (e.aliasOf) continue;
      const k = `${e.fingers.thumb}${e.fingers.index}${e.fingers.middle}${e.fingers.ring}${e.fingers.pinky}`;
      (byFingers[k] ||= []).push(id);
    }
    const fingerCollisions = Object.values(byFingers).filter((g) => g.length > 1);
    expect(fingerCollisions.length).toBeGreaterThan(0); // they DO collide on fingers alone
  });
});
