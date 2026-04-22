/**
 * PerseverationDetector.js — Script Loop / Perseveration Detection for ASD
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Monitors completed sentence sequences and detects when a child enters a
 * rigid repetitive loop — producing the same gesture sequence compulsively
 * regardless of the prompt. This is a common ASD presentation where the
 * child finds a "safe" pattern and perseverates on it.
 *
 * When perseveration is detected:
 *   1. Pauses the current task (stops accepting gesture input)
 *   2. Emits a callback with a gentle redirection suggestion
 *   3. Optionally suggests a new target sentence that shares one element
 *      with the perseverated sequence (scaffolded novelty)
 *
 * Detection algorithm:
 *   - Maintains a ring buffer of the last N completed sentence token sequences
 *   - Compares each new completion against the buffer
 *   - If K consecutive identical sequences are detected, triggers perseveration
 *   - Partial matches (same structure, different content) count as 0.5 toward K
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const DEFAULT_BUFFER_SIZE = 10;
const DEFAULT_THRESHOLD = 3;

// =============================================================================
// PERSEVERATION DETECTOR
// =============================================================================

export class PerseverationDetector {
  /**
   * @param {object} [config]
   * @param {number} [config.threshold=3] — consecutive identical sequences to trigger
   * @param {number} [config.bufferSize=10] — max completed sentences to remember
   * @param {function} [config.onPerseverationDetected] — callback: (info) => void
   * @param {function} [config.onPerseverationResolved] — callback: () => void
   */
  constructor(config = {}) {
    this._threshold = config.threshold || DEFAULT_THRESHOLD;
    this._bufferSize = config.bufferSize || DEFAULT_BUFFER_SIZE;
    this._onDetected = config.onPerseverationDetected || null;
    this._onResolved = config.onPerseverationResolved || null;

    /** @type {Array<string[]>} Ring buffer of completed sentence token arrays */
    this._completedSequences = [];

    /** @type {number} Count of consecutive identical sequences */
    this._consecutiveIdentical = 0;

    /** @type {string[]|null} The perseverated sequence (if active) */
    this._perseveratedSequence = null;

    /** @type {boolean} Whether perseveration is currently active */
    this._isActive = false;

    /** @type {number} Total perseveration events this session */
    this._totalEvents = 0;
  }

  // ===========================================================================
  // PUBLIC
  // ===========================================================================

  /**
   * Record a completed sentence. Call this when a valid sentence is formed.
   *
   * @param {string[]} tokenSequence — array of grammar IDs (e.g., ['SUBJECT_I', 'GRAB', 'APPLE'])
   * @returns {PerseverationResult}
   */
  recordCompletion(tokenSequence) {
    if (!tokenSequence || tokenSequence.length === 0) {
      return this._result();
    }

    const sequenceKey = tokenSequence.join('|');
    const lastSequence = this._completedSequences.length > 0
      ? this._completedSequences[this._completedSequences.length - 1]
      : null;
    const lastKey = lastSequence ? lastSequence.join('|') : null;

    // Add to buffer
    this._completedSequences.push([...tokenSequence]);
    if (this._completedSequences.length > this._bufferSize) {
      this._completedSequences.shift();
    }

    // Check for consecutive identical
    if (lastKey && sequenceKey === lastKey) {
      this._consecutiveIdentical++;
    } else {
      // Different sequence — reset count
      if (this._isActive) {
        // Perseveration broken
        this._isActive = false;
        this._perseveratedSequence = null;
        if (this._onResolved) {
          try { this._onResolved(); } catch (e) { /* swallow */ }
        }
      }
      this._consecutiveIdentical = 1;
    }

    // Check threshold
    if (this._consecutiveIdentical >= this._threshold && !this._isActive) {
      this._isActive = true;
      this._perseveratedSequence = [...tokenSequence];
      this._totalEvents++;

      const suggestion = this._generateRedirection(tokenSequence);

      if (this._onDetected) {
        try {
          this._onDetected({
            sequence: tokenSequence,
            repetitions: this._consecutiveIdentical,
            suggestion,
          });
        } catch (e) { /* swallow */ }
      }
    }

    return this._result();
  }

  /**
   * Check if perseveration is currently active.
   * @returns {boolean}
   */
  isActive() {
    return this._isActive;
  }

  /**
   * Get the perseverated sequence.
   * @returns {string[]|null}
   */
  getPerseveratedSequence() {
    return this._perseveratedSequence;
  }

  /**
   * Manually acknowledge/dismiss perseveration (e.g., educator presses continue).
   */
  acknowledge() {
    this._isActive = false;
    this._perseveratedSequence = null;
    this._consecutiveIdentical = 0;
  }

  /**
   * Get diagnostic status.
   * @returns {object}
   */
  getStatus() {
    return {
      isActive: this._isActive,
      consecutiveIdentical: this._consecutiveIdentical,
      threshold: this._threshold,
      perseveratedSequence: this._perseveratedSequence,
      totalEvents: this._totalEvents,
      bufferSize: this._completedSequences.length,
    };
  }

  /**
   * Reset all state.
   */
  reset() {
    this._completedSequences = [];
    this._consecutiveIdentical = 0;
    this._perseveratedSequence = null;
    this._isActive = false;
  }

  /**
   * Update threshold at runtime.
   * @param {number} threshold
   */
  setThreshold(threshold) {
    this._threshold = Math.max(2, Math.min(10, threshold));
  }

  // ===========================================================================
  // PRIVATE
  // ===========================================================================

  /**
   * Generate a gentle redirection suggestion.
   * Scaffolded novelty: suggest a sentence that shares one element with
   * the perseverated sequence, making the transition less jarring.
   */
  _generateRedirection(sequence) {
    // Alternative subjects, verbs, and objects for variation
    const subjects = ['SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_HE', 'SUBJECT_SHE', 'SUBJECT_WE', 'SUBJECT_THEY'];
    const verbs = ['GRAB', 'EAT', 'WANT', 'DRINK', 'GO', 'STOP'];
    const objects = ['APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE'];

    const currentSubject = sequence[0];
    const currentVerb = sequence.length > 1 ? sequence[1] : null;
    const currentObject = sequence.length > 2 ? sequence[2] : null;

    // Keep the subject, change the verb/object (least disruptive change)
    const altVerbs = verbs.filter(v => v !== currentVerb && v !== currentVerb?.replace(/S$/, ''));
    const altObjects = objects.filter(o => o !== currentObject);

    const suggestion = {
      message: 'Great job with that sentence! Can you try a new one?',
      keepElement: currentSubject,
      tryVerb: altVerbs.length > 0 ? altVerbs[0] : null,
      tryObject: altObjects.length > 0 ? altObjects[0] : null,
    };

    if (suggestion.tryVerb && suggestion.tryObject) {
      suggestion.targetSentence = [currentSubject, suggestion.tryVerb, suggestion.tryObject];
    }

    return suggestion;
  }

  _result() {
    return {
      isActive: this._isActive,
      consecutiveIdentical: this._consecutiveIdentical,
      perseveratedSequence: this._perseveratedSequence,
    };
  }
}

export default PerseverationDetector;
