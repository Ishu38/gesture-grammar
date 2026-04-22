/**
 * Formal Systems Tests — Mathematical foundations for MLAF.
 * Covers: GestureLifecycleDFA, SemanticTypeSystem, verifyTransitionTable.
 *
 * These modules implement formal methods from Partee, ter Meulen & Wall:
 *   - DFA (Ch 17): Finite automata for gesture lifecycle
 *   - Type Theory (Ch 13): Typed lambda calculus for semantic validation
 */

import { describe, it, expect, beforeEach, vi } from 'vitest';
import {
  GestureLifecycleDFA,
  DFA_STATES,
  DFA_INPUTS,
  verifyTransitionTable,
} from '../core/GestureLifecycleDFA.js';
import {
  SemanticTypeSystem,
  typeEquals,
  typeToString,
  isFunctionType,
  applyType,
  GESTURE_TYPES,
  getSemanticTypeSystem,
} from '../core/SemanticTypeSystem.js';

// =============================================================================
// GESTURE LIFECYCLE DFA
// =============================================================================

describe('GestureLifecycleDFA', () => {
  let dfa;

  beforeEach(() => {
    dfa = new GestureLifecycleDFA({ confirmationFrames: 5, cooldownMs: 100 });
  });

  // ---------------------------------------------------------------------------
  // Transition table totality (Ch 17: total function requirement)
  // ---------------------------------------------------------------------------

  describe('Transition Table Verification', () => {
    it('transition table is total — every (state, input) pair defined', () => {
      const result = verifyTransitionTable();
      expect(result.valid).toBe(true);
      expect(result.missing).toHaveLength(0);
    });

    it('all transitions map to valid states', () => {
      const result = verifyTransitionTable();
      const states = Object.values(DFA_STATES);
      expect(result.totalTransitions).toBe(states.length * Object.values(DFA_INPUTS).length);
      expect(result.definedTransitions).toBe(result.totalTransitions);
    });
  });

  // ---------------------------------------------------------------------------
  // Start state (q0 = IDLE)
  // ---------------------------------------------------------------------------

  describe('Initial State', () => {
    it('starts in IDLE state', () => {
      expect(dfa.getState().state).toBe(DFA_STATES.IDLE);
    });

    it('has zero confirmation count initially', () => {
      expect(dfa.getState().confirmationCount).toBe(0);
    });
  });

  // ---------------------------------------------------------------------------
  // IDLE transitions
  // ---------------------------------------------------------------------------

  describe('IDLE State Transitions', () => {
    it('IDLE + NO_HAND = IDLE', () => {
      const result = dfa.process({ handPresent: false, intentState: 'RESTING', gestureId: null });
      expect(result.state).toBe(DFA_STATES.IDLE);
    });

    it('IDLE + HAND_RESTING = IDLE', () => {
      const result = dfa.process({ handPresent: true, intentState: 'RESTING', gestureId: null });
      expect(result.state).toBe(DFA_STATES.IDLE);
    });

    it('IDLE + gesture detected = DETECTING', () => {
      const result = dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      expect(result.state).toBe(DFA_STATES.DETECTING);
      expect(result.gestureId).toBe('EAT');
    });
  });

  // ---------------------------------------------------------------------------
  // DETECTING → CONFIRMING → LOCKED flow
  // ---------------------------------------------------------------------------

  describe('Full Lock Lifecycle', () => {
    it('DETECTING + same gesture = CONFIRMING', () => {
      // First frame: IDLE → DETECTING
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      // Second frame: DETECTING → CONFIRMING
      const result = dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      expect(result.state).toBe(DFA_STATES.CONFIRMING);
    });

    it('reaches LOCKED after confirmationFrames', () => {
      const lockCallback = vi.fn();
      const testDfa = new GestureLifecycleDFA({
        confirmationFrames: 3,
        cooldownMs: 100,
        onLock: lockCallback,
      });

      // Frame 1: IDLE → DETECTING
      testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'SUBJECT_I' });
      // Frame 2: DETECTING → CONFIRMING (count=2)
      testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'SUBJECT_I' });
      // Frame 3: CONFIRMING (count=3 = threshold) → LOCKED
      const result = testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'SUBJECT_I' });

      expect(result.state).toBe(DFA_STATES.LOCKED);
      expect(result.isLocked).toBe(true);
      expect(lockCallback).toHaveBeenCalledWith('SUBJECT_I');
    });

    it('LOCKED transitions to COOLDOWN on next frame', () => {
      const testDfa = new GestureLifecycleDFA({ confirmationFrames: 2, cooldownMs: 500 });

      testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'GO' });
      testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'GO' });
      // Now LOCKED
      const afterLock = testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'GO' });
      expect(afterLock.state).toBe(DFA_STATES.COOLDOWN);
      expect(afterLock.isCoolingDown).toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // Reset and interruption paths
  // ---------------------------------------------------------------------------

  describe('Interruption Paths', () => {
    it('DETECTING + different gesture = DETECTING (reset to new)', () => {
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      const result = dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'GO' });
      expect(result.state).toBe(DFA_STATES.DETECTING);
      expect(result.gestureId).toBe('GO');
    });

    it('CONFIRMING + hand lost = IDLE', () => {
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      const result = dfa.process({ handPresent: false, intentState: 'RESTING', gestureId: null });
      expect(result.state).toBe(DFA_STATES.IDLE);
    });

    it('CONFIRMING + hand resting = IDLE', () => {
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      const result = dfa.process({ handPresent: true, intentState: 'RESTING', gestureId: null });
      expect(result.state).toBe(DFA_STATES.IDLE);
    });

    it('COOLDOWN + hand dropped = IDLE (skip cooldown)', () => {
      const testDfa = new GestureLifecycleDFA({ confirmationFrames: 2, cooldownMs: 10000 });
      testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      // LOCKED → next frame → COOLDOWN
      testDfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      // Hand dropped during cooldown → IDLE
      const result = testDfa.process({ handPresent: false, intentState: 'RESTING', gestureId: null });
      expect(result.state).toBe(DFA_STATES.IDLE);
    });
  });

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  describe('Configuration', () => {
    it('setConfirmationFrames clamps to valid range', () => {
      dfa.setConfirmationFrames(0);
      expect(dfa.getConfirmationFrames()).toBe(1);
      dfa.setConfirmationFrames(200);
      expect(dfa.getConfirmationFrames()).toBe(120);
      dfa.setConfirmationFrames(50);
      expect(dfa.getConfirmationFrames()).toBe(50);
    });

    it('reset returns to IDLE', () => {
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      dfa.reset();
      expect(dfa.getState().state).toBe(DFA_STATES.IDLE);
      expect(dfa.getState().confirmationCount).toBe(0);
    });

    it('getFormalSpec returns complete DFA tuple', () => {
      const spec = dfa.getFormalSpec();
      expect(spec.Q).toEqual(Object.values(DFA_STATES));
      expect(spec.Sigma).toEqual(Object.values(DFA_INPUTS));
      expect(spec.q0).toBe(DFA_STATES.IDLE);
      expect(spec.F).toContain(DFA_STATES.LOCKED);
      expect(spec.delta).toBeDefined();
    });
  });

  // ---------------------------------------------------------------------------
  // Transition logging
  // ---------------------------------------------------------------------------

  describe('Diagnostics', () => {
    it('tracks transition count', () => {
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      expect(dfa.getState().transitionCount).toBe(1); // IDLE → DETECTING
    });

    it('getTransitionLog returns transition history', () => {
      dfa.process({ handPresent: true, intentState: 'GESTURE_ACTIVE', gestureId: 'EAT' });
      dfa.process({ handPresent: false, intentState: 'RESTING', gestureId: null });
      const log = dfa.getTransitionLog();
      expect(log.length).toBe(2);
      expect(log[0].from).toBe(DFA_STATES.IDLE);
      expect(log[0].to).toBe(DFA_STATES.DETECTING);
      expect(log[1].from).toBe(DFA_STATES.DETECTING);
      expect(log[1].to).toBe(DFA_STATES.IDLE);
    });
  });
});

// =============================================================================
// SEMANTIC TYPE SYSTEM
// =============================================================================

describe('SemanticTypeSystem', () => {
  let sts;

  beforeEach(() => {
    sts = new SemanticTypeSystem();
  });

  // ---------------------------------------------------------------------------
  // Type operations (Ch 13: Type Theory)
  // ---------------------------------------------------------------------------

  describe('Type Operations', () => {
    it('typeEquals: simple types', () => {
      expect(typeEquals('e', 'e')).toBe(true);
      expect(typeEquals('t', 't')).toBe(true);
      expect(typeEquals('e', 't')).toBe(false);
    });

    it('typeEquals: function types', () => {
      const et = { from: 'e', to: 't' };
      const et2 = { from: 'e', to: 't' };
      const te = { from: 't', to: 'e' };
      expect(typeEquals(et, et2)).toBe(true);
      expect(typeEquals(et, te)).toBe(false);
    });

    it('typeEquals: nested function types', () => {
      const eet = { from: 'e', to: { from: 'e', to: 't' } };
      const eet2 = { from: 'e', to: { from: 'e', to: 't' } };
      expect(typeEquals(eet, eet2)).toBe(true);
    });

    it('typeToString: formats types correctly', () => {
      expect(typeToString('e')).toBe('e');
      expect(typeToString('t')).toBe('t');
      expect(typeToString({ from: 'e', to: 't' })).toBe('<e, t>');
      expect(typeToString({ from: 'e', to: { from: 'e', to: 't' } })).toBe('<e, <e, t>>');
    });

    it('isFunctionType: distinguishes simple from complex', () => {
      expect(isFunctionType('e')).toBe(false);
      expect(isFunctionType({ from: 'e', to: 't' })).toBe(true);
    });

    it('applyType: functional application', () => {
      // <e, t> applied to e gives t
      const result1 = applyType({ from: 'e', to: 't' }, 'e');
      expect(result1).toBe('t');

      // <e, <e, t>> applied to e gives <e, t>
      const result2 = applyType({ from: 'e', to: { from: 'e', to: 't' } }, 'e');
      expect(typeEquals(result2, { from: 'e', to: 't' })).toBe(true);

      // Type mismatch: <e, t> applied to t gives null
      expect(applyType({ from: 'e', to: 't' }, 't')).toBeNull();

      // Non-function applied gives null
      expect(applyType('e', 'e')).toBeNull();
    });
  });

  // ---------------------------------------------------------------------------
  // Gesture type assignments
  // ---------------------------------------------------------------------------

  describe('Gesture Type Assignments', () => {
    it('pronouns have type e', () => {
      const subjects = ['SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_HE', 'SUBJECT_SHE', 'SUBJECT_WE', 'SUBJECT_THEY'];
      for (const s of subjects) {
        expect(sts.getType(s)).toBe('e');
        expect(sts.getGestureType(s).slot).toBe('SUBJECT');
      }
    });

    it('transitive verbs have type <e, <e, t>>', () => {
      const verbs = ['WANT', 'EAT', 'SEE', 'GRAB', 'DRINK'];
      for (const v of verbs) {
        expect(sts.getTypeString(v)).toBe('<e, <e, t>>');
        expect(sts.getGestureType(v).transitive).toBe(true);
      }
    });

    it('intransitive verbs have type <e, t>', () => {
      const verbs = ['GO', 'STOP'];
      for (const v of verbs) {
        expect(sts.getTypeString(v)).toBe('<e, t>');
        expect(sts.getGestureType(v).transitive).toBe(false);
      }
    });

    it('object nouns have type e', () => {
      const objects = ['FOOD', 'WATER', 'BOOK', 'APPLE', 'BALL', 'HOUSE'];
      for (const o of objects) {
        expect(sts.getType(o)).toBe('e');
        expect(sts.getGestureType(o).slot).toBe('OBJECT');
      }
    });

    it('all 19 gestures have type assignments', () => {
      expect(Object.keys(GESTURE_TYPES).length).toBe(19);
    });

    it('returns null for unknown gesture', () => {
      expect(sts.getGestureType('NONEXISTENT')).toBeNull();
      expect(sts.getType('NONEXISTENT')).toBeNull();
    });
  });

  // ---------------------------------------------------------------------------
  // Slot validation
  // ---------------------------------------------------------------------------

  describe('Slot Validation', () => {
    it('pronoun validates for SUBJECT slot', () => {
      const result = sts.validateSlot('SUBJECT_I', 'SUBJECT');
      expect(result.valid).toBe(true);
    });

    it('verb validates for VERB slot', () => {
      const result = sts.validateSlot('EAT', 'VERB');
      expect(result.valid).toBe(true);
    });

    it('noun validates for OBJECT slot', () => {
      const result = sts.validateSlot('FOOD', 'OBJECT');
      expect(result.valid).toBe(true);
    });

    it('noun in VERB slot is invalid', () => {
      const result = sts.validateSlot('FOOD', 'VERB');
      expect(result.valid).toBe(false);
      expect(result.error).toBe('SLOT_MISMATCH');
    });

    it('verb in SUBJECT slot is invalid', () => {
      const result = sts.validateSlot('EAT', 'SUBJECT');
      expect(result.valid).toBe(false);
    });

    it('unknown gesture returns UNKNOWN_GESTURE error', () => {
      const result = sts.validateSlot('NONEXISTENT', 'SUBJECT');
      expect(result.valid).toBe(false);
      expect(result.error).toBe('UNKNOWN_GESTURE');
    });
  });

  // ---------------------------------------------------------------------------
  // Sentence composition (Frege's Principle)
  // ---------------------------------------------------------------------------

  describe('Sentence Composition', () => {
    it('empty sentence expects SUBJECT', () => {
      const result = sts.composeSentence([]);
      expect(result.complete).toBe(false);
      expect(result.expectsNext).toContain('SUBJECT');
    });

    it('subject only: incomplete, expects VERB', () => {
      const result = sts.composeSentence(['SUBJECT_SHE']);
      expect(result.complete).toBe(false);
      expect(result.expectsNext).toContain('VERB');
      expect(result.slots.subject).toBe('She');
    });

    it('subject + intransitive verb = complete sentence (type t)', () => {
      const result = sts.composeSentence(['SUBJECT_I', 'GO']);
      expect(result.complete).toBe(true);
      expect(result.resultTypeString).toBe('t');
      expect(result.errors).toHaveLength(0);
    });

    it('subject + transitive verb: incomplete, expects OBJECT', () => {
      const result = sts.composeSentence(['SUBJECT_SHE', 'EAT']);
      expect(result.complete).toBe(false);
      expect(result.expectsNext).toContain('OBJECT');
    });

    it('subject + transitive verb + object = complete SVO (type t)', () => {
      const result = sts.composeSentence(['SUBJECT_SHE', 'EAT', 'FOOD']);
      expect(result.complete).toBe(true);
      expect(result.resultTypeString).toBe('t');
      expect(result.slots.subject).toBe('She');
      expect(result.slots.verb).toBe('eat');
      expect(result.slots.object).toBe('food');
    });

    it('intransitive verb + object = EXTRA_ARGUMENT error', () => {
      const result = sts.composeSentence(['SUBJECT_I', 'GO', 'FOOD']);
      expect(result.errors.some(e => e.error === 'EXTRA_ARGUMENT')).toBe(true);
    });
  });

  // ---------------------------------------------------------------------------
  // Next gesture suggestions
  // ---------------------------------------------------------------------------

  describe('Valid Next Gestures', () => {
    it('from empty: suggests all subjects', () => {
      const suggestions = sts.getValidNextGestures([]);
      const slots = suggestions.map(s => s.slot);
      expect(slots.every(s => s === 'SUBJECT')).toBe(true);
      expect(suggestions.length).toBe(6); // 6 pronouns
    });

    it('after subject: suggests verbs', () => {
      const suggestions = sts.getValidNextGestures(['SUBJECT_I']);
      const slots = suggestions.map(s => s.slot);
      expect(slots.every(s => s === 'VERB')).toBe(true);
      expect(suggestions.length).toBe(7); // 7 verbs
    });

    it('after subject + transitive verb: suggests objects', () => {
      const suggestions = sts.getValidNextGestures(['SUBJECT_SHE', 'EAT']);
      const slots = suggestions.map(s => s.slot);
      expect(slots.every(s => s === 'OBJECT')).toBe(true);
      expect(suggestions.length).toBe(6); // 6 nouns
    });

    it('after subject + intransitive verb: suggests nothing', () => {
      const suggestions = sts.getValidNextGestures(['SUBJECT_I', 'GO']);
      expect(suggestions).toHaveLength(0);
    });
  });

  // ---------------------------------------------------------------------------
  // Singleton
  // ---------------------------------------------------------------------------

  describe('Singleton', () => {
    it('getSemanticTypeSystem returns same instance', () => {
      const a = getSemanticTypeSystem();
      const b = getSemanticTypeSystem();
      expect(a).toBe(b);
    });
  });

  // ---------------------------------------------------------------------------
  // Type catalog
  // ---------------------------------------------------------------------------

  describe('Type Catalog', () => {
    it('returns all 19 gestures', () => {
      const catalog = sts.getTypeCatalog();
      expect(catalog.length).toBe(19);
      expect(catalog.every(c => c.gestureId && c.type && c.category)).toBe(true);
    });
  });
});
