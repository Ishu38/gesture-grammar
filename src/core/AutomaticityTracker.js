/**
 * AutomaticityTracker.js — Gesture Fluency & Automaticity Measurement
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Measures the development of gestural automaticity by tracking production
 * latency across practice sessions. As learners internalize gesture-grammar
 * mappings, response time decreases — this is the motor-kinesthetic analog
 * of reading fluency development.
 *
 * Theoretical grounding:
 *   LaBerge & Samuels (1974): Automaticity theory — fluent performance is
 *     characterized by fast, accurate, effortless processing that no longer
 *     requires conscious attention.
 *   National Reading Panel (NICHD, 2000): Fluency identified as one of the
 *     "Big Five" essential reading components. Fluency = accuracy × rate.
 *   Birsh & Carreker (2018):
 *     Ch. 1, p. 56: Fluency as NRP Big Five essential component.
 *     Ch. 12, "Fluency in Learning to Read: Conceptions, Misconceptions,
 *     Learning Disabilities, and Instructional Moves" (Garnett) —
 *     automaticity as the reduction of cognitive load through practice.
 *   Birsh & Carreker (2018), Preface, p. 44–45:
 *     Effective instructional apps Criterion 5: "Practice activities develop
 *     automaticity."
 *
 * Transfer to MLAF:
 *   Gestural automaticity = fast, effortless gesture-to-grammar retrieval.
 *   As in reading fluency, reduced production latency frees working memory
 *   for higher-order syntactic processing (sentence assembly, tense selection).
 *   AutomaticityTracker makes this development measurable and visible,
 *   providing empirical evidence of skill acquisition over time.
 *
 * Metrics tracked per gesture:
 *   onset_latency_ms   — time from "ready" signal to first gesture detection
 *                         (reflects lexical-motor retrieval speed)
 *   hold_duration_ms   — time from first detection to confidence lock
 *                         (reflects motor precision and stability)
 *   total_ms           — onset + hold (overall production time)
 *   automaticity_score — 0.0–1.0, normalized inverse of mean total_ms
 *
 * Storage:
 *   Persisted to localStorage under STORAGE_KEY.
 *   Format: { [gesture_id]: [{ onset_ms, hold_ms, total_ms, ts }, ...] }
 *   Maximum MAX_RECORDS_PER_GESTURE records retained per gesture.
 *
 * Patent claim:
 *   "A method for measuring gestural automaticity in a multimodal language
 *   acquisition system by tracking response latency reduction over practice
 *   sessions, operationalizing reading fluency theory (LaBerge & Samuels,
 *   1974; NICHD, 2000) in a gesture-based syntax instruction interface
 *   running locally on a mobile device under the Edge Constraint."
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'mlaf_automaticity_v1';

// Latency benchmarks for score normalization (milliseconds).
// BASELINE_MIN: physically impossible to produce faster (just the hold time).
// BASELINE_MAX: a novice taking a long time to recall + form the gesture.
const BASELINE_MIN_MS = 1500;
const BASELINE_MAX_MS = 7000;

// Maximum records stored per gesture (oldest are dropped).
const MAX_RECORDS_PER_GESTURE = 50;

// Trend thresholds: ≥10% latency reduction = improving; ≥10% increase = declining.
const TREND_IMPROVEMENT_THRESHOLD = 0.10;
const TREND_DECLINE_THRESHOLD = 0.10;

// Minimum records needed for a valid trend calculation.
const MIN_RECORDS_FOR_TREND = 4;

// =============================================================================
// AUTOMATICITY TRACKER
// =============================================================================

export class AutomaticityTracker {
  /**
   * @param {object} config
   * @param {string} config.storageKey — override the localStorage key
   */
  constructor(config = {}) {
    this.storageKey = config.storageKey || STORAGE_KEY;

    // Timing anchors for the current gesture being formed
    this._readyTime = null;           // When user became ready (debounce ended)
    this._firstDetectionTime = null;  // When correct gesture was first detected
    this._trackingGesture = null;     // Gesture currently being timed

    // In-session production records: [{ gesture_id, onset_ms, hold_ms, total_ms, ts }]
    this.sessionLog = [];

    // Persistent history loaded from localStorage:
    // { [gesture_id]: [{ onset_ms, hold_ms, total_ms, ts }, ...] }
    this._history = this._loadHistory();
  }

  // ===========================================================================
  // PUBLIC — Timing event hooks (called from SandboxMode.jsx)
  // ===========================================================================

  /**
   * Call once when the camera session starts.
   * Sets the initial ready time so the first gesture of the session can be timed.
   */
  initSession() {
    this._readyTime = performance.now();
    this._firstDetectionTime = null;
    this._trackingGesture = null;
  }

  /**
   * Call when isLocked transitions true → false (debounce ends).
   * The user is now free to produce the next gesture.
   */
  onReadyForNextGesture() {
    this._readyTime = performance.now();
    this._firstDetectionTime = null;
    this._trackingGesture = null;
  }

  /**
   * Call every frame with the currently detected gesture ID (or null).
   * Records the timestamp of first detection for the gesture that will lock.
   *
   * @param {string|null} gestureId — current frame's detected gesture
   */
  onGestureFrame(gestureId) {
    if (!gestureId) {
      // Hand dropped — if user was forming a gesture, reset detection time.
      // (They may be starting over with a different gesture.)
      if (this._trackingGesture !== null) {
        this._firstDetectionTime = null;
        this._trackingGesture = null;
      }
      return;
    }

    if (gestureId !== this._trackingGesture) {
      // New gesture detected — start timing from this frame.
      this._trackingGesture = gestureId;
      this._firstDetectionTime = performance.now();
    }
    // If same gesture as before: do nothing, we already have the start time.
  }

  /**
   * Call when a gesture locks (a new word is added to the sentence).
   * Computes and records the full production record.
   *
   * @param {string} gestureId — the grammar ID of the word that just locked
   * @returns {object|null} the production record, or null if timing was incomplete
   */
  onGestureLocked(gestureId) {
    const lockTime = performance.now();

    const onset_ms = (this._readyTime && this._firstDetectionTime)
      ? Math.round(this._firstDetectionTime - this._readyTime)
      : null;

    const hold_ms = this._firstDetectionTime
      ? Math.round(lockTime - this._firstDetectionTime)
      : null;

    const total_ms = (onset_ms !== null && hold_ms !== null)
      ? onset_ms + hold_ms
      : null;

    // Only record if we have valid timing data
    if (total_ms === null || total_ms < 100 || total_ms > 30000) {
      // Invalid timing — reset and move on
      this._readyTime = performance.now();
      this._firstDetectionTime = null;
      this._trackingGesture = null;
      return null;
    }

    const record = {
      gesture_id: gestureId,
      onset_ms,
      hold_ms,
      total_ms,
      ts: Date.now(),
    };

    // Add to in-session log
    this.sessionLog.push(record);

    // Persist to history
    this._appendToHistory(gestureId, { onset_ms, hold_ms, total_ms, ts: record.ts });

    // Set ready time for next gesture
    this._readyTime = performance.now();
    this._firstDetectionTime = null;
    this._trackingGesture = null;

    return record;
  }

  // ===========================================================================
  // PUBLIC — Score and trend queries
  // ===========================================================================

  /**
   * Get the automaticity score for a gesture (0.0 to 1.0).
   * Uses full cross-session history if available, falls back to session-only.
   *
   * 1.0 = fully automatic (near-minimum production time)
   * 0.0 = very slow (near-maximum baseline time)
   *
   * @param {string} gestureId
   * @returns {number} score 0.0–1.0
   */
  getAutomaticityScore(gestureId) {
    const records = this._getAllRecordsFor(gestureId);
    if (records.length === 0) return null;

    const meanMs = this._mean(records.map(r => r.total_ms));
    const score = 1 - Math.max(0, Math.min(1,
      (meanMs - BASELINE_MIN_MS) / (BASELINE_MAX_MS - BASELINE_MIN_MS)
    ));
    return Math.round(score * 100) / 100; // Round to 2 decimal places
  }

  /**
   * Get the trend direction for a gesture based on latency change over time.
   *
   * @param {string} gestureId
   * @returns {'improving' | 'stable' | 'declining' | 'new'}
   */
  getTrend(gestureId) {
    const records = this._getAllRecordsFor(gestureId);
    if (records.length < MIN_RECORDS_FOR_TREND) return 'new';

    const half = Math.floor(records.length / 2);
    const firstHalf = records.slice(0, half);
    const secondHalf = records.slice(-half);

    const earlyMean = this._mean(firstHalf.map(r => r.total_ms));
    const recentMean = this._mean(secondHalf.map(r => r.total_ms));

    const change = (recentMean - earlyMean) / earlyMean;

    if (change < -TREND_IMPROVEMENT_THRESHOLD) return 'improving';
    if (change > TREND_DECLINE_THRESHOLD) return 'declining';
    return 'stable';
  }

  /**
   * Get mean production time for a gesture (milliseconds).
   *
   * @param {string} gestureId
   * @returns {number|null}
   */
  getMeanProductionTime(gestureId) {
    const records = this._getAllRecordsFor(gestureId);
    if (records.length === 0) return null;
    return Math.round(this._mean(records.map(r => r.total_ms)));
  }

  /**
   * Get a summary of the current session and cross-session history.
   * Used to populate the fluency display panel in SandboxMode.jsx.
   *
   * @returns {AutomaticitySummary}
   */
  getSessionSummary() {
    // Collect unique gestures seen this session
    const sessionGestureIds = [...new Set(this.sessionLog.map(r => r.gesture_id))];

    const per_gesture = {};
    for (const id of sessionGestureIds) {
      const sessionRecords = this.sessionLog.filter(r => r.gesture_id === id);
      const allRecords = this._getAllRecordsFor(id);

      per_gesture[id] = {
        count: sessionRecords.length,
        session_mean_ms: Math.round(this._mean(sessionRecords.map(r => r.total_ms))),
        all_time_mean_ms: allRecords.length > 0
          ? Math.round(this._mean(allRecords.map(r => r.total_ms)))
          : null,
        score: this.getAutomaticityScore(id),
        trend: this.getTrend(id),
      };
    }

    return {
      session_gestures: this.sessionLog.length,
      unique_gestures: sessionGestureIds.length,
      per_gesture,
      // Sorted by count (most practiced first)
      sorted_ids: sessionGestureIds.sort(
        (a, b) => per_gesture[b].count - per_gesture[a].count
      ),
    };
  }

  /**
   * Get cross-session progress report for all tracked gestures.
   * Intended for teacher/therapist review.
   *
   * @returns {object}
   */
  getCrossSessionReport() {
    const allIds = Object.keys(this._history);
    const report = {};
    for (const id of allIds) {
      report[id] = {
        total_attempts: this._history[id].length,
        score: this.getAutomaticityScore(id),
        trend: this.getTrend(id),
        mean_ms: this.getMeanProductionTime(id),
        earliest_ms: this._history[id][0]?.total_ms || null,
        latest_ms: this._history[id][this._history[id].length - 1]?.total_ms || null,
      };
    }
    return report;
  }

  /**
   * Clear all stored history from localStorage and reset session.
   */
  clearHistory() {
    this._history = {};
    this.sessionLog = [];
    try {
      localStorage.removeItem(this.storageKey);
    } catch (e) {
      // localStorage unavailable
    }
  }

  // ===========================================================================
  // PRIVATE — Storage helpers
  // ===========================================================================

  _getAllRecordsFor(gestureId) {
    return this._history[gestureId] || [];
  }

  _appendToHistory(gestureId, record) {
    if (!this._history[gestureId]) {
      this._history[gestureId] = [];
    }
    this._history[gestureId].push(record);

    // Enforce max records per gesture
    if (this._history[gestureId].length > MAX_RECORDS_PER_GESTURE) {
      this._history[gestureId] = this._history[gestureId].slice(-MAX_RECORDS_PER_GESTURE);
    }

    this._saveHistory();
  }

  _loadHistory() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      // Validate structure
      if (typeof parsed !== 'object' || Array.isArray(parsed)) return {};
      return parsed;
    } catch (e) {
      return {};
    }
  }

  _saveHistory() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this._history));
    } catch (e) {
      // localStorage full or unavailable — fail silently
    }
  }

  _mean(arr) {
    if (arr.length === 0) return null;
    return arr.reduce((a, b) => a + b, 0) / arr.length;
  }
}

// =============================================================================
// SCORE LABEL HELPERS (used in UI)
// =============================================================================

/**
 * Convert a 0–1 automaticity score to a human-readable label.
 * @param {number|null} score
 * @returns {string}
 */
export function scoreToLabel(score) {
  if (score === null) return 'No data';
  if (score >= 0.80) return 'Fluent';
  if (score >= 0.55) return 'Developing';
  if (score >= 0.30) return 'Emerging';
  return 'Novice';
}

/**
 * Convert trend string to a display arrow character.
 * @param {string} trend
 * @returns {string}
 */
export function trendToArrow(trend) {
  switch (trend) {
    case 'improving': return '↓';   // Latency going down = improving
    case 'declining': return '↑';   // Latency going up = declining
    case 'stable':    return '→';
    case 'new':       return '·';
    default:          return '·';
  }
}

/**
 * Convert trend string to a CSS color class name.
 * @param {string} trend
 * @returns {string}
 */
export function trendToColor(trend) {
  switch (trend) {
    case 'improving': return '#4ade80';   // green
    case 'declining': return '#f87171';   // red
    case 'stable':    return '#facc15';   // yellow
    default:          return '#64748b';   // grey
  }
}

export default AutomaticityTracker;
