/**
 * GestureLifecycleDFA.js — Formal Finite Automaton for Gesture Recognition
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Implements a mathematically rigorous DFA (Deterministic Finite Automaton)
 * for gesture lifecycle management, replacing the ad-hoc threshold/counter
 * approach with a provably correct state machine.
 *
 * Formal Definition (Partee, ter Meulen & Wall, Ch 17):
 *   M = <Q, Sigma, delta, q0, F>
 *
 *   Q (States):
 *     IDLE         — No gesture detected, hand at rest or absent
 *     DETECTING    — Hand active, gesture classifier returning a candidate
 *     CONFIRMING   — Same gesture held for consecutive frames, building confidence
 *     LOCKED       — Gesture confirmed, emitted to sentence builder
 *     COOLDOWN     — Post-lock debounce, preventing double-fires
 *
 *   Sigma (Input Alphabet):
 *     NO_HAND        — No hand landmarks detected
 *     HAND_RESTING   — Hand present but IntentionalityDetector says RESTING
 *     GESTURE_NEW    — New gesture detected (different from previous)
 *     GESTURE_SAME   — Same gesture detected as previous frame
 *     GESTURE_NONE   — Hand active but classifier returned null
 *     LOCK_THRESHOLD — Confirmation counter reached threshold (internal)
 *     COOLDOWN_DONE  — Cooldown timer expired (internal)
 *
 *   delta (Transition Function): See TRANSITION_TABLE below
 *   q0 (Start State): IDLE
 *   F (Accept States): { LOCKED } — the state that emits a gesture lock event
 *
 * Properties:
 *   - Deterministic: every (state, input) pair maps to exactly one next state
 *   - Total: every input is handled in every state (no undefined transitions)
 *   - Observable: emits events on state transitions for UI feedback
 *
 * Peak-Capture Mode (for athetoid CP):
 *   When config.peakCaptureMode is true, the DFA uses a fundamentally different
 *   confirmation strategy designed for children with athetoid CP who can only
 *   hold a correct gesture shape for ~200ms before involuntary movement distorts it.
 *
 *   In peak-capture mode:
 *     - confirmationFrames is set very low (4–8 frames ≈ 130–260ms)
 *     - CONFIRMING tolerates brief dropouts (up to peakGracePeriod frames of
 *       GESTURE_NONE) without resetting — the child's involuntary movements
 *       may briefly interrupt detection
 *     - Once the gesture is detected for confirmationFrames TOTAL (not necessarily
 *       consecutive) within a peakWindow, it locks
 *     - After lock, a longer cooldown prevents re-fire during post-gesture distortion
 *
 * Integration:
 *   Replaces ConfidenceLock in gestureDetection.js. Called per-frame from the
 *   AGGME pipeline after IntentionalityDetector and gesture classification.
 */

import { DFA_MODES } from './AccessibilityProfile.js';

// =============================================================================
// STATES (Q)
// =============================================================================

export const DFA_STATES = {
  IDLE:       'IDLE',
  DETECTING:  'DETECTING',
  CONFIRMING: 'CONFIRMING',
  LOCKED:     'LOCKED',
  COOLDOWN:   'COOLDOWN',
};

// =============================================================================
// INPUT ALPHABET (Sigma)
// =============================================================================

export const DFA_INPUTS = {
  NO_HAND:        'NO_HAND',
  HAND_RESTING:   'HAND_RESTING',
  GESTURE_NEW:    'GESTURE_NEW',
  GESTURE_SAME:   'GESTURE_SAME',
  GESTURE_NONE:   'GESTURE_NONE',
  LOCK_THRESHOLD: 'LOCK_THRESHOLD',
  COOLDOWN_DONE:  'COOLDOWN_DONE',
};

// =============================================================================
// TRANSITION TABLE (delta)
// =============================================================================
// Format: TRANSITION_TABLE[currentState][input] = nextState
// This is a total function — every (state, input) pair is defined.

const TRANSITION_TABLE = {
  [DFA_STATES.IDLE]: {
    [DFA_INPUTS.NO_HAND]:        DFA_STATES.IDLE,
    [DFA_INPUTS.HAND_RESTING]:   DFA_STATES.IDLE,
    [DFA_INPUTS.GESTURE_NEW]:    DFA_STATES.DETECTING,
    [DFA_INPUTS.GESTURE_SAME]:   DFA_STATES.DETECTING,  // first frame counts as new
    [DFA_INPUTS.GESTURE_NONE]:   DFA_STATES.IDLE,
    [DFA_INPUTS.LOCK_THRESHOLD]: DFA_STATES.IDLE,       // impossible, ignore
    [DFA_INPUTS.COOLDOWN_DONE]:  DFA_STATES.IDLE,       // impossible, ignore
  },

  [DFA_STATES.DETECTING]: {
    [DFA_INPUTS.NO_HAND]:        DFA_STATES.IDLE,
    [DFA_INPUTS.HAND_RESTING]:   DFA_STATES.IDLE,
    [DFA_INPUTS.GESTURE_NEW]:    DFA_STATES.DETECTING,   // reset to new gesture
    [DFA_INPUTS.GESTURE_SAME]:   DFA_STATES.CONFIRMING,  // start confirming
    [DFA_INPUTS.GESTURE_NONE]:   DFA_STATES.IDLE,
    [DFA_INPUTS.LOCK_THRESHOLD]: DFA_STATES.LOCKED,      // immediate lock if threshold=1
    [DFA_INPUTS.COOLDOWN_DONE]:  DFA_STATES.DETECTING,   // ignore
  },

  [DFA_STATES.CONFIRMING]: {
    [DFA_INPUTS.NO_HAND]:        DFA_STATES.IDLE,
    [DFA_INPUTS.HAND_RESTING]:   DFA_STATES.IDLE,
    [DFA_INPUTS.GESTURE_NEW]:    DFA_STATES.DETECTING,   // different gesture, restart
    [DFA_INPUTS.GESTURE_SAME]:   DFA_STATES.CONFIRMING,  // keep counting
    [DFA_INPUTS.GESTURE_NONE]:   DFA_STATES.IDLE,        // lost gesture
    [DFA_INPUTS.LOCK_THRESHOLD]: DFA_STATES.LOCKED,      // threshold reached!
    [DFA_INPUTS.COOLDOWN_DONE]:  DFA_STATES.CONFIRMING,  // ignore
  },

  [DFA_STATES.LOCKED]: {
    // LOCKED is transient — immediately transitions to COOLDOWN on next frame
    [DFA_INPUTS.NO_HAND]:        DFA_STATES.COOLDOWN,
    [DFA_INPUTS.HAND_RESTING]:   DFA_STATES.COOLDOWN,
    [DFA_INPUTS.GESTURE_NEW]:    DFA_STATES.COOLDOWN,
    [DFA_INPUTS.GESTURE_SAME]:   DFA_STATES.COOLDOWN,
    [DFA_INPUTS.GESTURE_NONE]:   DFA_STATES.COOLDOWN,
    [DFA_INPUTS.LOCK_THRESHOLD]: DFA_STATES.COOLDOWN,
    [DFA_INPUTS.COOLDOWN_DONE]:  DFA_STATES.IDLE,
  },

  [DFA_STATES.COOLDOWN]: {
    [DFA_INPUTS.NO_HAND]:        DFA_STATES.IDLE,        // hand dropped, skip cooldown
    [DFA_INPUTS.HAND_RESTING]:   DFA_STATES.IDLE,        // returned to rest, skip cooldown
    [DFA_INPUTS.GESTURE_NEW]:    DFA_STATES.COOLDOWN,    // ignore during cooldown
    [DFA_INPUTS.GESTURE_SAME]:   DFA_STATES.COOLDOWN,    // ignore during cooldown
    [DFA_INPUTS.GESTURE_NONE]:   DFA_STATES.COOLDOWN,    // still cooling down
    [DFA_INPUTS.LOCK_THRESHOLD]: DFA_STATES.COOLDOWN,    // impossible, ignore
    [DFA_INPUTS.COOLDOWN_DONE]:  DFA_STATES.IDLE,        // cooldown expired
  },
};

// =============================================================================
// DFA IMPLEMENTATION
// =============================================================================

export class GestureLifecycleDFA {
  /**
   * @param {object} config
   * @param {number}   [config.confirmationFrames=30] — frames to hold before lock
   * @param {number}   [config.cooldownMs=1500] — post-lock cooldown duration
   * @param {boolean}  [config.peakCaptureMode=false] — enable peak-capture for athetoid CP
   * @param {number}   [config.peakGracePeriod=3] — frames of dropout tolerated in peak mode
   * @param {number}   [config.peakWindow=15] — max frames to accumulate confirmations in peak mode
   * @param {function} [config.onLock] — callback when gesture locks: (gestureId) => void
   * @param {function} [config.onStateChange] — callback on any state change: (newState, prevState) => void
   */
  constructor(config = {}) {
    // Configuration
    this._confirmationFrames = config.confirmationFrames || 30;
    this._cooldownMs = config.cooldownMs || 1500;
    this._onLock = config.onLock || null;
    this._onStateChange = config.onStateChange || null;

    // Peak-capture mode (athetoid CP)
    this._peakCaptureMode = config.peakCaptureMode || false;
    this._peakGracePeriod = config.peakGracePeriod || 3;   // tolerate 3 dropout frames
    this._peakWindow = config.peakWindow || 15;            // 500ms window at 30fps
    this._peakDropoutCount = 0;    // consecutive dropout frames in peak mode
    this._peakWindowCount = 0;     // total frames since peak tracking started
    this._peakAccumulatedHits = 0; // total confirmed frames (non-consecutive)

    // DFA state (q)
    this._state = DFA_STATES.IDLE;

    // Augmented state (not part of DFA formalism, but needed for counting)
    this._currentGestureId = null;
    this._confirmationCount = 0;
    this._cooldownTimer = null;
    this._lastLockedGesture = null;
    this._lockTimestamp = null;

    // Diagnostics
    this._transitionCount = 0;
    this._transitionLog = []; // last N transitions for debug
    this._maxLogSize = 50;
  }

  // ===========================================================================
  // PUBLIC — Per-frame processing
  // ===========================================================================

  /**
   * Process one frame of input. Call this every frame from the AGGME pipeline.
   *
   * @param {object} frame
   * @param {boolean} frame.handPresent — is a hand detected?
   * @param {string}  frame.intentState — 'RESTING' | 'GESTURE_ACTIVE' from IntentionalityDetector
   * @param {string|null} frame.gestureId — classified gesture ID or null
   * @returns {DFAResult}
   */
  process(frame) {
    const { handPresent, intentState, gestureId } = frame;

    // Classify the input symbol
    const input = this._classifyInput(handPresent, intentState, gestureId);

    let effectiveInput = input;

    if (this._peakCaptureMode) {
      // ═══════════════════════════════════════════════════════════════════
      // PEAK-CAPTURE MODE (athetoid CP)
      // Tolerates brief dropouts; accumulates hits over a window.
      // ═══════════════════════════════════════════════════════════════════
      if (
        (this._state === DFA_STATES.DETECTING || this._state === DFA_STATES.CONFIRMING)
      ) {
        this._peakWindowCount++;

        if (input === DFA_INPUTS.GESTURE_SAME) {
          // Valid frame — count it
          this._peakAccumulatedHits++;
          this._confirmationCount++;
          this._peakDropoutCount = 0;

          if (this._peakAccumulatedHits >= this._confirmationFrames) {
            effectiveInput = DFA_INPUTS.LOCK_THRESHOLD;
          }
        } else if (
          input === DFA_INPUTS.GESTURE_NONE ||
          input === DFA_INPUTS.GESTURE_NEW
        ) {
          // Dropout frame — tolerate up to peakGracePeriod
          this._peakDropoutCount++;

          if (this._peakDropoutCount <= this._peakGracePeriod) {
            // Stay in current state — override the transition
            effectiveInput = DFA_INPUTS.GESTURE_SAME;
            // Don't increment confirmation count for dropout frames
          } else {
            // Grace period exceeded — reset
            this._peakDropoutCount = 0;
            this._peakWindowCount = 0;
            this._peakAccumulatedHits = 0;
          }
        }

        // If the peak window has elapsed without enough hits, reset
        if (this._peakWindowCount > this._peakWindow &&
            this._peakAccumulatedHits < this._confirmationFrames) {
          this._peakDropoutCount = 0;
          this._peakWindowCount = 0;
          this._peakAccumulatedHits = 0;
          effectiveInput = DFA_INPUTS.GESTURE_NONE; // force reset to IDLE
        }
      } else {
        // Not in detecting/confirming — reset peak tracking
        this._peakDropoutCount = 0;
        this._peakWindowCount = 0;
        this._peakAccumulatedHits = 0;
      }
    } else {
      // ═══════════════════════════════════════════════════════════════════
      // SUSTAINED-HOLD MODE (default)
      // Requires N consecutive frames of the same gesture.
      // ═══════════════════════════════════════════════════════════════════
      if (
        (this._state === DFA_STATES.DETECTING || this._state === DFA_STATES.CONFIRMING) &&
        input === DFA_INPUTS.GESTURE_SAME
      ) {
        this._confirmationCount++;
        if (this._confirmationCount >= this._confirmationFrames) {
          effectiveInput = DFA_INPUTS.LOCK_THRESHOLD;
        }
      }
    }

    // Look up transition
    const prevState = this._state;
    const nextState = TRANSITION_TABLE[prevState][effectiveInput];

    // Execute transition
    this._executeTransition(prevState, nextState, effectiveInput, gestureId);

    // Build result
    const progress = this._state === DFA_STATES.CONFIRMING || this._state === DFA_STATES.DETECTING
      ? Math.min(
          (this._peakCaptureMode ? this._peakAccumulatedHits : this._confirmationCount)
            / this._confirmationFrames, 1)
      : this._state === DFA_STATES.LOCKED ? 1 : 0;

    return {
      state: this._state,
      gestureId: this._currentGestureId,
      isLocked: this._state === DFA_STATES.LOCKED,
      isCoolingDown: this._state === DFA_STATES.COOLDOWN,
      progress,
      confirmationCount: this._peakCaptureMode ? this._peakAccumulatedHits : this._confirmationCount,
      lastLockedGesture: this._lastLockedGesture,
      input: effectiveInput,
      transition: prevState !== this._state ? `${prevState} -> ${this._state}` : null,
      peakCapture: this._peakCaptureMode ? {
        accumulatedHits: this._peakAccumulatedHits,
        windowCount: this._peakWindowCount,
        dropoutCount: this._peakDropoutCount,
      } : null,
    };
  }

  // ===========================================================================
  // PUBLIC — Configuration
  // ===========================================================================

  /**
   * Update the confirmation frame threshold (used by CognitiveLoadAdapter).
   * @param {number} frames
   */
  setConfirmationFrames(frames) {
    this._confirmationFrames = Math.max(1, Math.min(120, Math.round(frames)));
  }

  /**
   * Get the current confirmation frame threshold.
   * @returns {number}
   */
  getConfirmationFrames() {
    return this._confirmationFrames;
  }

  /**
   * Enable or disable peak-capture mode (for athetoid CP).
   * @param {boolean} enabled
   * @param {object} [options]
   * @param {number} [options.gracePeriod=3] — dropout frames tolerated
   * @param {number} [options.window=15] — total window frames
   */
  setPeakCaptureMode(enabled, options = {}) {
    this._peakCaptureMode = enabled;
    if (options.gracePeriod !== undefined) {
      this._peakGracePeriod = Math.max(1, Math.min(10, options.gracePeriod));
    }
    if (options.window !== undefined) {
      this._peakWindow = Math.max(5, Math.min(60, options.window));
    }
    this._peakDropoutCount = 0;
    this._peakWindowCount = 0;
    this._peakAccumulatedHits = 0;
  }

  /**
   * Check if peak-capture mode is active.
   * @returns {boolean}
   */
  isPeakCaptureMode() {
    return this._peakCaptureMode;
  }

  /**
   * Reset the DFA to IDLE state.
   */
  reset() {
    this._clearCooldownTimer();
    this._state = DFA_STATES.IDLE;
    this._currentGestureId = null;
    this._confirmationCount = 0;
    this._lastLockedGesture = null;
    this._lockTimestamp = null;
    this._peakDropoutCount = 0;
    this._peakWindowCount = 0;
    this._peakAccumulatedHits = 0;
  }

  /**
   * Get current state for debug display.
   * @returns {object}
   */
  getState() {
    return {
      state: this._state,
      gestureId: this._currentGestureId,
      confirmationCount: this._confirmationCount,
      confirmationFrames: this._confirmationFrames,
      lastLockedGesture: this._lastLockedGesture,
      transitionCount: this._transitionCount,
      peakCaptureMode: this._peakCaptureMode,
      peakAccumulatedHits: this._peakAccumulatedHits,
    };
  }

  /**
   * Get recent transition log for debug panel.
   * @returns {Array}
   */
  getTransitionLog() {
    return [...this._transitionLog];
  }

  /**
   * Get the formal DFA specification as a serializable object.
   * Useful for documentation and verification.
   * @returns {object}
   */
  getFormalSpec() {
    return {
      Q: Object.values(DFA_STATES),
      Sigma: Object.values(DFA_INPUTS),
      delta: TRANSITION_TABLE,
      q0: DFA_STATES.IDLE,
      F: [DFA_STATES.LOCKED],
      description: 'Gesture Lifecycle DFA — M = <Q, Sigma, delta, q0, F>',
    };
  }

  // ===========================================================================
  // PRIVATE — Input classification
  // ===========================================================================

  /**
   * Map raw frame data to a DFA input symbol.
   * This is the "input encoding" function: RawFrame -> Sigma
   */
  _classifyInput(handPresent, intentState, gestureId) {
    if (!handPresent) {
      return DFA_INPUTS.NO_HAND;
    }

    if (intentState === 'RESTING') {
      return DFA_INPUTS.HAND_RESTING;
    }

    // Hand is active (GESTURE_ACTIVE)
    if (!gestureId) {
      return DFA_INPUTS.GESTURE_NONE;
    }

    // Gesture detected — is it the same as what we're tracking?
    if (gestureId === this._currentGestureId) {
      return DFA_INPUTS.GESTURE_SAME;
    }

    return DFA_INPUTS.GESTURE_NEW;
  }

  // ===========================================================================
  // PRIVATE — Transition execution
  // ===========================================================================

  /**
   * Execute side effects of a state transition.
   */
  _executeTransition(prevState, nextState, input, gestureId) {
    // Log transition
    if (prevState !== nextState) {
      this._transitionCount++;
      this._transitionLog.push({
        from: prevState,
        to: nextState,
        input,
        gestureId,
        timestamp: Date.now(),
      });
      if (this._transitionLog.length > this._maxLogSize) {
        this._transitionLog.shift();
      }
    }

    // State-specific entry actions
    switch (nextState) {
      case DFA_STATES.IDLE:
        this._currentGestureId = null;
        this._confirmationCount = 0;
        this._clearCooldownTimer();
        break;

      case DFA_STATES.DETECTING:
        if (input === DFA_INPUTS.GESTURE_NEW || prevState === DFA_STATES.IDLE) {
          // New gesture — reset counter and start tracking
          this._currentGestureId = gestureId;
          this._confirmationCount = 1;
        }
        break;

      case DFA_STATES.CONFIRMING:
        // Counter already incremented in process()
        break;

      case DFA_STATES.LOCKED:
        this._lastLockedGesture = this._currentGestureId;
        this._lockTimestamp = Date.now();

        // Fire lock callback
        if (this._onLock) {
          try {
            this._onLock(this._currentGestureId);
          } catch (e) {
            console.error('[GestureLifecycleDFA] onLock callback error:', e);
          }
        }

        // Reset counter
        this._confirmationCount = 0;
        break;

      case DFA_STATES.COOLDOWN:
        if (prevState !== DFA_STATES.COOLDOWN) {
          // Start cooldown timer
          this._clearCooldownTimer();
          this._cooldownTimer = setTimeout(() => {
            // Internal event: cooldown done
            // We can't call process() from inside a timer (no frame data),
            // so we directly transition to IDLE
            const prev = this._state;
            this._state = DFA_STATES.IDLE;
            this._currentGestureId = null;
            this._confirmationCount = 0;
            this._cooldownTimer = null;

            if (this._onStateChange && prev !== DFA_STATES.IDLE) {
              this._onStateChange(DFA_STATES.IDLE, prev);
            }
          }, this._cooldownMs);
        }
        break;
    }

    // Update state
    this._state = nextState;

    // Fire state change callback
    if (prevState !== nextState && this._onStateChange) {
      try {
        this._onStateChange(nextState, prevState);
      } catch (e) {
        console.error('[GestureLifecycleDFA] onStateChange callback error:', e);
      }
    }
  }

  _clearCooldownTimer() {
    if (this._cooldownTimer) {
      clearTimeout(this._cooldownTimer);
      this._cooldownTimer = null;
    }
  }
}

// =============================================================================
// DFA VERIFICATION UTILITY
// =============================================================================

/**
 * Verify that the transition table is total (every state × input pair is defined).
 * Call this during development/testing to ensure no undefined transitions.
 * @returns {{ valid: boolean, missing: Array }}
 */
export function verifyTransitionTable() {
  const states = Object.values(DFA_STATES);
  const inputs = Object.values(DFA_INPUTS);
  const missing = [];

  for (const state of states) {
    if (!TRANSITION_TABLE[state]) {
      missing.push({ state, input: '*', error: 'state not in table' });
      continue;
    }
    for (const input of inputs) {
      if (TRANSITION_TABLE[state][input] === undefined) {
        missing.push({ state, input, error: 'transition undefined' });
      } else if (!states.includes(TRANSITION_TABLE[state][input])) {
        missing.push({ state, input, error: `invalid target: ${TRANSITION_TABLE[state][input]}` });
      }
    }
  }

  return {
    valid: missing.length === 0,
    missing,
    totalTransitions: states.length * inputs.length,
    definedTransitions: states.length * inputs.length - missing.length,
  };
}

export default GestureLifecycleDFA;
