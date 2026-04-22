/**
 * SessionTimer.js — Attention-Window Session Architecture for ASD/ADHD
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Provides structured session management with configurable attention windows,
 * micro-break prompts, and progress segmentation. Designed for autistic
 * children and children with ADHD co-presentation who have functional
 * attention windows of 8-12 minutes.
 *
 * Session architecture:
 *   [Work segment] → [Micro-break] → [Work segment] → [Micro-break] → [Session end]
 *
 * The timer emits callbacks at key transitions:
 *   - onMicroBreakStart: time to take a short break
 *   - onMicroBreakEnd: break is over, resume work
 *   - onSessionWarning: approaching session end (2 min before)
 *   - onSessionEnd: session duration reached, suggest stopping
 *
 * All timing is configurable per AccessibilityProfile.
 */

// =============================================================================
// SESSION STATES
// =============================================================================

export const SESSION_STATES = {
  NOT_STARTED: 'NOT_STARTED',
  ACTIVE:      'ACTIVE',
  MICRO_BREAK: 'MICRO_BREAK',
  WARNING:     'WARNING',
  ENDED:       'ENDED',
  PAUSED:      'PAUSED',
};

// =============================================================================
// SESSION TIMER
// =============================================================================

export class SessionTimer {
  /**
   * @param {object} config
   * @param {number} [config.sessionDurationMinutes=0] — total session length (0 = unlimited)
   * @param {number} [config.microBreakIntervalMinutes=0] — interval between breaks (0 = no breaks)
   * @param {number} [config.microBreakDurationSeconds=30] — break duration
   * @param {number} [config.warningBeforeEndMinutes=2] — warning before session ends
   * @param {function} [config.onMicroBreakStart] — callback
   * @param {function} [config.onMicroBreakEnd] — callback
   * @param {function} [config.onSessionWarning] — callback
   * @param {function} [config.onSessionEnd] — callback
   * @param {function} [config.onStateChange] — callback: (newState, info) => void
   */
  constructor(config = {}) {
    this._sessionDurationMs = (config.sessionDurationMinutes || 0) * 60000;
    this._microBreakIntervalMs = (config.microBreakIntervalMinutes || 0) * 60000;
    this._microBreakDurationMs = (config.microBreakDurationSeconds || 30) * 1000;
    this._warningBeforeEndMs = (config.warningBeforeEndMinutes || 2) * 60000;

    this._onMicroBreakStart = config.onMicroBreakStart || null;
    this._onMicroBreakEnd = config.onMicroBreakEnd || null;
    this._onSessionWarning = config.onSessionWarning || null;
    this._onSessionEnd = config.onSessionEnd || null;
    this._onStateChange = config.onStateChange || null;

    // State
    this._state = SESSION_STATES.NOT_STARTED;
    this._sessionStartTime = null;
    this._lastBreakTime = null;
    this._breakEndTime = null;
    this._warningFired = false;
    this._tickInterval = null;

    // Counters
    this._totalBreaksTaken = 0;
    this._totalWorkSegments = 0;
    this._pauseStartTime = null;
    this._totalPausedMs = 0;
  }

  // ===========================================================================
  // PUBLIC — Lifecycle
  // ===========================================================================

  /**
   * Start the session timer. Call when the camera starts / session begins.
   */
  start() {
    if (this._state !== SESSION_STATES.NOT_STARTED && this._state !== SESSION_STATES.ENDED) {
      return;
    }

    this._sessionStartTime = Date.now();
    this._lastBreakTime = Date.now();
    this._warningFired = false;
    this._totalBreaksTaken = 0;
    this._totalWorkSegments = 1;
    this._totalPausedMs = 0;
    this._setState(SESSION_STATES.ACTIVE);

    // Tick every second for time checks
    this._tickInterval = setInterval(() => this._tick(), 1000);
  }

  /**
   * Pause the session (e.g., camera lost, user navigated away).
   */
  pause() {
    if (this._state === SESSION_STATES.ACTIVE || this._state === SESSION_STATES.WARNING) {
      this._pauseStartTime = Date.now();
      this._setState(SESSION_STATES.PAUSED);
    }
  }

  /**
   * Resume from pause.
   */
  resume() {
    if (this._state === SESSION_STATES.PAUSED) {
      if (this._pauseStartTime) {
        this._totalPausedMs += Date.now() - this._pauseStartTime;
        this._pauseStartTime = null;
      }
      this._setState(SESSION_STATES.ACTIVE);
    }
  }

  /**
   * Acknowledge a micro-break prompt (child/educator pressed "I'm ready").
   */
  endBreak() {
    if (this._state === SESSION_STATES.MICRO_BREAK) {
      this._lastBreakTime = Date.now();
      this._totalWorkSegments++;
      this._setState(SESSION_STATES.ACTIVE);

      if (this._onMicroBreakEnd) {
        try { this._onMicroBreakEnd(); } catch (e) { /* swallow */ }
      }
    }
  }

  /**
   * Explicitly end the session.
   */
  end() {
    this._setState(SESSION_STATES.ENDED);
    if (this._tickInterval) {
      clearInterval(this._tickInterval);
      this._tickInterval = null;
    }
  }

  /**
   * Destroy and clean up.
   */
  destroy() {
    if (this._tickInterval) {
      clearInterval(this._tickInterval);
      this._tickInterval = null;
    }
  }

  // ===========================================================================
  // PUBLIC — Query
  // ===========================================================================

  /**
   * Get current session state.
   * @returns {string} SESSION_STATES value
   */
  getState() {
    return this._state;
  }

  /**
   * Get active work time (excluding pauses and breaks) in milliseconds.
   * @returns {number}
   */
  getActiveWorkTimeMs() {
    if (!this._sessionStartTime) return 0;
    const total = Date.now() - this._sessionStartTime;
    return Math.max(0, total - this._totalPausedMs);
  }

  /**
   * Get time remaining in the session (ms). Returns Infinity if no limit.
   * @returns {number}
   */
  getTimeRemainingMs() {
    if (!this._sessionDurationMs || !this._sessionStartTime) return Infinity;
    const elapsed = this.getActiveWorkTimeMs();
    return Math.max(0, this._sessionDurationMs - elapsed);
  }

  /**
   * Get time until next micro-break (ms). Returns Infinity if no breaks.
   * @returns {number}
   */
  getTimeUntilBreakMs() {
    if (!this._microBreakIntervalMs || !this._lastBreakTime) return Infinity;
    const sinceLast = Date.now() - this._lastBreakTime;
    return Math.max(0, this._microBreakIntervalMs - sinceLast);
  }

  /**
   * Get session progress as 0-1.
   * @returns {number}
   */
  getProgress() {
    if (!this._sessionDurationMs) return 0;
    const elapsed = this.getActiveWorkTimeMs();
    return Math.min(1, elapsed / this._sessionDurationMs);
  }

  /**
   * Get comprehensive session timing info.
   * @returns {object}
   */
  getTimingInfo() {
    return {
      state: this._state,
      activeWorkTimeMs: this.getActiveWorkTimeMs(),
      timeRemainingMs: this.getTimeRemainingMs(),
      timeUntilBreakMs: this.getTimeUntilBreakMs(),
      progress: this.getProgress(),
      totalBreaksTaken: this._totalBreaksTaken,
      totalWorkSegments: this._totalWorkSegments,
      sessionDurationMs: this._sessionDurationMs,
      microBreakIntervalMs: this._microBreakIntervalMs,
      isUnlimited: !this._sessionDurationMs,
    };
  }

  // ===========================================================================
  // PRIVATE
  // ===========================================================================

  _tick() {
    if (this._state === SESSION_STATES.PAUSED || this._state === SESSION_STATES.ENDED) {
      return;
    }

    // Handle micro-break state: check if break duration elapsed
    if (this._state === SESSION_STATES.MICRO_BREAK) {
      if (this._breakEndTime && Date.now() >= this._breakEndTime) {
        // Auto-end break after duration
        this.endBreak();
      }
      return;
    }

    const activeTime = this.getActiveWorkTimeMs();

    // Check session end
    if (this._sessionDurationMs && activeTime >= this._sessionDurationMs) {
      this._setState(SESSION_STATES.ENDED);
      if (this._tickInterval) {
        clearInterval(this._tickInterval);
        this._tickInterval = null;
      }
      if (this._onSessionEnd) {
        try { this._onSessionEnd({ activeTimeMs: activeTime }); } catch (e) { /* swallow */ }
      }
      return;
    }

    // Check session warning
    if (
      this._sessionDurationMs &&
      !this._warningFired &&
      activeTime >= (this._sessionDurationMs - this._warningBeforeEndMs)
    ) {
      this._warningFired = true;
      this._setState(SESSION_STATES.WARNING);
      if (this._onSessionWarning) {
        try {
          this._onSessionWarning({
            remainingMs: this._sessionDurationMs - activeTime,
          });
        } catch (e) { /* swallow */ }
      }
      return;
    }

    // Check micro-break interval (cache Date.now() to avoid tick drift)
    if (this._microBreakIntervalMs && this._lastBreakTime) {
      const now = Date.now();
      const sinceLast = now - this._lastBreakTime;
      if (sinceLast >= this._microBreakIntervalMs) {
        this._totalBreaksTaken++;
        this._breakEndTime = now + this._microBreakDurationMs;
        this._setState(SESSION_STATES.MICRO_BREAK);
        if (this._onMicroBreakStart) {
          try {
            this._onMicroBreakStart({
              breakNumber: this._totalBreaksTaken,
              durationMs: this._microBreakDurationMs,
            });
          } catch (e) { /* swallow */ }
        }
      }
    }
  }

  _setState(newState) {
    const prev = this._state;
    if (prev === newState) return;
    this._state = newState;
    if (this._onStateChange) {
      try {
        this._onStateChange(newState, {
          previousState: prev,
          activeWorkTimeMs: this.getActiveWorkTimeMs(),
          progress: this.getProgress(),
        });
      } catch (e) { /* swallow */ }
    }
  }
}

export default SessionTimer;
