/**
 * AccessibleFeedbackEngine.js — Audio + Haptic Correction Feedback for Low Vision
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Provides two non-visual feedback pathways for low-vision users:
 *
 *   1. TTS Correction Relay — speaks the worst correction_instruction aloud
 *      via Web Speech API, throttled to avoid rapid-fire interruptions.
 *
 *   2. Haptic Vibration — navigator.vibrate() patterns mapped to error severity:
 *        - Correct (all within tolerance): double short pulse  [50, 50, 50]
 *        - Close   (worst error < 2.0):    single medium buzz  [150]
 *        - Far     (worst error >= 2.0):   long sustained buzz [400]
 *
 * Both pathways are gated by AccessibilityProfile feedbackModes.
 * The engine is designed to be called every frame but internally throttles
 * to avoid sensory overload.
 */

// =============================================================================
// CONSTANTS
// =============================================================================

/** Minimum interval between spoken corrections (ms). */
const TTS_THROTTLE_MS = 3000;

/** Minimum interval between haptic pulses (ms). */
const HAPTIC_THROTTLE_MS = 1000;

/** Haptic patterns by severity. */
const HAPTIC_PATTERNS = {
  good:  [50, 50, 50],   // Double short pulse — "correct"
  close: [150],           // Single medium buzz — "almost"
  far:   [400],           // Long sustained buzz — "adjust needed"
};

// =============================================================================
// ACCESSIBLE FEEDBACK ENGINE
// =============================================================================

export class AccessibleFeedbackEngine {
  /**
   * @param {object} config
   * @param {boolean} [config.audioEnabled=false] — enable TTS correction relay
   * @param {boolean} [config.hapticEnabled=false] — enable vibration feedback
   * @param {number}  [config.ttsRate=0.9] — speech rate for corrections
   * @param {number}  [config.ttsThrottleMs=3000] — min interval between spoken corrections
   * @param {number}  [config.hapticThrottleMs=1000] — min interval between vibrations
   */
  constructor(config = {}) {
    this._audioEnabled = config.audioEnabled || false;
    this._hapticEnabled = config.hapticEnabled || false;
    this._ttsRate = config.ttsRate || 0.9;
    this._ttsThrottleMs = config.ttsThrottleMs || TTS_THROTTLE_MS;
    this._hapticThrottleMs = config.hapticThrottleMs || HAPTIC_THROTTLE_MS;

    // Throttle state
    this._lastSpokenTime = 0;
    this._lastSpokenMessage = '';
    this._lastHapticTime = 0;

    // TTS voice selection
    this._selectedVoice = null;
    this._voicesLoaded = false;

    // Availability checks
    this._hasTTS = typeof window !== 'undefined' && 'speechSynthesis' in window;
    this._hasVibration = typeof navigator !== 'undefined' && 'vibrate' in navigator;

    if (this._hasTTS && this._audioEnabled) {
      this._loadVoices();
    }
  }

  // ===========================================================================
  // PUBLIC
  // ===========================================================================

  /**
   * Process an error vector frame and emit non-visual feedback.
   * Call this every frame — the engine handles throttling internally.
   *
   * @param {object|null} errorData — from AbductiveFeedbackLoop / ErrorOverlay
   *   Expected shape: {
   *     aggregate_error: number,
   *     per_constraint_errors: [{ within_tolerance, normalized_error, correction_instruction }],
   *     is_within_tolerance: boolean
   *   }
   */
  processFrame(errorData) {
    if (!errorData || !errorData.per_constraint_errors) return;

    const severity = this._computeSeverity(errorData);
    const now = Date.now();

    // Haptic feedback
    if (this._hapticEnabled && this._hasVibration) {
      if (now - this._lastHapticTime >= this._hapticThrottleMs) {
        this._emitHaptic(severity);
        this._lastHapticTime = now;
      }
    }

    // Audio correction feedback
    if (this._audioEnabled && this._hasTTS) {
      if (now - this._lastSpokenTime >= this._ttsThrottleMs) {
        this._speakCorrection(errorData, severity);
      }
    }
  }

  /**
   * Speak a custom message (e.g., sentence completion, achievement).
   * Bypasses throttle.
   * @param {string} message
   */
  speakImmediate(message) {
    if (!this._audioEnabled || !this._hasTTS || !message) return;
    this._speak(message);
    this._lastSpokenTime = Date.now();
  }

  /**
   * Enable or disable audio feedback at runtime.
   * @param {boolean} enabled
   */
  setAudioEnabled(enabled) {
    this._audioEnabled = enabled;
    if (enabled && this._hasTTS && !this._voicesLoaded) {
      this._loadVoices();
    }
  }

  /**
   * Enable or disable haptic feedback at runtime.
   * @param {boolean} enabled
   */
  setHapticEnabled(enabled) {
    this._hapticEnabled = enabled;
  }

  /**
   * Cancel any ongoing speech.
   */
  cancel() {
    if (this._hasTTS) {
      try { speechSynthesis.cancel(); } catch (_) { /* ignore */ }
    }
  }

  /**
   * Clean up resources.
   */
  destroy() {
    this.cancel();
  }

  /**
   * Check feature availability.
   * @returns {{ ttsAvailable: boolean, vibrationAvailable: boolean, audioEnabled: boolean, hapticEnabled: boolean }}
   */
  getStatus() {
    return {
      ttsAvailable: this._hasTTS,
      vibrationAvailable: this._hasVibration,
      audioEnabled: this._audioEnabled,
      hapticEnabled: this._hapticEnabled,
    };
  }

  // ===========================================================================
  // PRIVATE
  // ===========================================================================

  _computeSeverity(errorData) {
    const errors = errorData.per_constraint_errors;
    const allGood = errors.every(e => e.within_tolerance);
    if (allGood) return 'good';

    const worstError = errors
      .filter(e => !e.within_tolerance)
      .sort((a, b) => b.normalized_error - a.normalized_error)[0];

    if (!worstError) return 'good';
    return worstError.normalized_error < 2.0 ? 'close' : 'far';
  }

  _emitHaptic(severity) {
    const pattern = HAPTIC_PATTERNS[severity];
    if (pattern) {
      try { navigator.vibrate(pattern); } catch (_) { /* ignore */ }
    }
  }

  _speakCorrection(errorData, severity) {
    if (severity === 'good') {
      // Only announce "correct" once, not repeatedly
      if (this._lastSpokenMessage !== '__GOOD__') {
        this._speak('Good form. Hold steady.');
        this._lastSpokenMessage = '__GOOD__';
        this._lastSpokenTime = Date.now();
      }
      return;
    }

    // Find the worst constraint violation
    const worstError = errorData.per_constraint_errors
      .filter(e => !e.within_tolerance)
      .sort((a, b) => b.normalized_error - a.normalized_error)[0];

    if (!worstError) return;

    const message = worstError.correction_instruction || 'Adjust your hand position.';

    // Don't repeat the exact same message
    if (message === this._lastSpokenMessage) return;

    this._speak(message);
    this._lastSpokenMessage = message;
    this._lastSpokenTime = Date.now();
  }

  _speak(text) {
    try {
      speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(text);
      if (this._selectedVoice) utterance.voice = this._selectedVoice;
      utterance.rate = this._ttsRate;
      utterance.pitch = 1.0;
      utterance.volume = 1.0;
      speechSynthesis.speak(utterance);
    } catch (_) { /* ignore */ }
  }

  _loadVoices() {
    if (!this._hasTTS) return;
    const loadFn = () => {
      const voices = speechSynthesis.getVoices();
      // Prefer an English voice
      const english = voices.find(v => v.lang.startsWith('en'));
      if (english) this._selectedVoice = english;
      this._voicesLoaded = true;
    };
    loadFn();
    if (speechSynthesis.onvoiceschanged !== undefined) {
      speechSynthesis.addEventListener('voiceschanged', loadFn, { once: true });
    }
  }
}

export default AccessibleFeedbackEngine;
