/**
 * SessionDataLogger.js — Diagnostic-Prescriptive Data Collection Engine
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Collects comprehensive event-level data during gesture practice sessions
 * to enable diagnostic-prescriptive instruction — one of the four core
 * Structured Literacy principles. Every gesture production, error, cognitive
 * load shift, and interference detection is timestamped and logged.
 *
 * Theoretical grounding:
 *   Birsh & Carreker (2018):
 *     Ch. 1, p. 52: Structured Literacy is "explicit, systematic, cumulative,
 *       and DIAGNOSTIC-PRESCRIPTIVE." Instruction must be informed by ongoing
 *       assessment data; the teacher (or system) adjusts based on measured
 *       performance, not assumptions.
 *     Ch. 2, "Structured Literacy Instruction" (Coyne, Carnine, Kameenui):
 *       "Progress monitoring through frequent assessment enables teachers to
 *       identify skills that need reteaching." SessionDataLogger makes this
 *       monitoring continuous and automatic.
 *     Preface, p. 44–45: Criterion 7 for effective instructional apps:
 *       "The app includes progress monitoring and data reporting."
 *     Ch. 12 (Garnett): Fluency assessment requires repeated measurement
 *       across sessions — the cross-session persistence layer enables this.
 *     Ch. 19, "Language and Literacy Development Among English Language Learners"
 *       (Helman): L2 assessment must capture specific error categories
 *       (transfer errors, morphological errors) not just overall accuracy —
 *       this is why we log ISL interference events and S-V agreement errors
 *       as distinct event types.
 *
 * Data architecture:
 *   Events (in-session): timestamped array of typed events
 *   Session summaries (cross-session): persisted aggregates per session
 *   Export: full session log as downloadable JSON
 *
 * Event types:
 *   GESTURE_LOCK        — a gesture was held long enough to add a word
 *   SENTENCE_COMPLETE   — a grammatically valid sentence was formed
 *   SENTENCE_CLEAR      — user cleared the sentence strip
 *   ISL_INTERFERENCE    — ISL transfer pattern detected
 *   COGNITIVE_LOAD      — cognitive load level changed (LOW/MEDIUM/HIGH)
 *   MASTERY_ACHIEVED    — a gesture crossed the mastery threshold
 *   AGREEMENT_ERROR     — subject-verb agreement mismatch detected
 *   SESSION_START       — camera started, session began
 *   SESSION_END         — session explicitly ended or exported
 *
 * Patent claim:
 *   "A diagnostic-prescriptive data collection system for gesture-based language
 *   instruction that continuously records gesture production events, syntactic
 *   transfer errors, cognitive load state transitions, and morphological accuracy
 *   metrics, operationalizing Structured Literacy assessment principles (Birsh &
 *   Carreker, 2018, Ch.1 p.52; Ch.2; Preface Criterion 7) to enable evidence-based
 *   instructional adaptation in a real-time gesture recognition interface."
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'mlaf_session_history_v1';

// Maximum session summaries retained in localStorage.
const MAX_SESSION_HISTORY = 50;

// =============================================================================
// SESSION DATA LOGGER
// =============================================================================

export class SessionDataLogger {
  /**
   * @param {object} config
   * @param {string} [config.storageKey] — override localStorage key
   * @param {string} [config.profileType] — accessibility profile active at session start
   */
  constructor(config = {}) {
    this.storageKey = config.storageKey || STORAGE_KEY;

    /** @type {string} Unique session identifier */
    this.sessionId = this._generateSessionId();

    /** @type {number} Session start timestamp (ms since epoch) */
    this.startTime = Date.now();

    /** @type {string} Active accessibility profile */
    this.profileType = config.profileType || 'default';

    /** @type {Array} Ordered event log for this session */
    this.events = [];

    /** @type {object} Running counters updated on each event */
    this._counters = {
      totalGestures: 0,
      sentencesCompleted: 0,
      sentencesCleared: 0,
      islInterferences: 0,
      agreementErrors: 0,
      cognitiveLoadChanges: 0,
      masteryAchievements: 0,
      gesturesByType: {},        // { [grammar_id]: count }
      agreementErrorDetails: [], // [{ subject, verb, expected_verb, ts }]
      islPatterns: {},           // { [pattern_type]: count }
      cognitiveLoadTimeline: [], // [{ level, ts }]
    };

    /** @type {Array} Cross-session history loaded from localStorage */
    this._history = this._loadHistory();
  }

  // ===========================================================================
  // PUBLIC — Event logging methods (called from SandboxMode.jsx)
  // ===========================================================================

  /**
   * Log the start of a camera/practice session.
   */
  logSessionStart() {
    this._pushEvent('SESSION_START', {
      profile: this.profileType,
    });
    this._counters.cognitiveLoadTimeline.push({ level: 'LOW', ts: Date.now() });
  }

  /**
   * Log a gesture lock (word added to sentence).
   *
   * @param {object} wordData — the word object from useSentenceBuilder
   * @param {object} meta — optional metadata
   * @param {number} [meta.productionTime] — total_ms from AutomaticityTracker
   * @param {string} [meta.cognitiveLoad] — current cognitive load level
   * @param {number} [meta.jitter] — current jitter value
   * @param {number} [meta.lockProgress] — confidence progress at lock (should be 1.0)
   */
  logGestureLock(wordData, meta = {}) {
    const grammarId = wordData.grammar_id || wordData.grammarId || 'UNKNOWN';

    this._counters.totalGestures++;
    this._counters.gesturesByType[grammarId] =
      (this._counters.gesturesByType[grammarId] || 0) + 1;

    this._pushEvent('GESTURE_LOCK', {
      grammar_id: grammarId,
      word: wordData.word,
      type: wordData.type,
      tense: wordData.tense || null,
      production_time_ms: meta.productionTime || null,
      cognitive_load: meta.cognitiveLoad || null,
      jitter: meta.jitter || null,
    });
  }

  /**
   * Log a complete, grammatically valid sentence.
   *
   * @param {Array} sentence — the full sentence array from useSentenceBuilder
   * @param {object} validation — the validation result from GrammarEngine
   */
  logSentenceComplete(sentence, validation) {
    this._counters.sentencesCompleted++;

    this._pushEvent('SENTENCE_COMPLETE', {
      sentence: sentence.map(w => w.word).join(' '),
      grammar_ids: sentence.map(w => w.grammar_id),
      word_count: sentence.length,
      parse_tree: validation?.parseTree ? true : false,
    });
  }

  /**
   * Log a sentence clear (user pressed Clear or started over).
   *
   * @param {Array} sentence — the sentence at time of clearing
   * @param {string} reason — 'user_clear' | 'user_undo' | 'auto'
   */
  logSentenceClear(sentence, reason = 'user_clear') {
    this._counters.sentencesCleared++;

    this._pushEvent('SENTENCE_CLEAR', {
      sentence_at_clear: sentence.map(w => w.word).join(' '),
      word_count: sentence.length,
      reason,
    });
  }

  /**
   * Log an ISL interference detection event.
   *
   * @param {object} interferenceReport — from ISLInterferenceDetector.analyze()
   */
  logISLInterference(interferenceReport) {
    this._counters.islInterferences++;

    interferenceReport.patterns.forEach(p => {
      const patternType = p.pattern_type || p.title;
      this._counters.islPatterns[patternType] =
        (this._counters.islPatterns[patternType] || 0) + 1;
    });

    this._pushEvent('ISL_INTERFERENCE', {
      severity: interferenceReport.severity,
      pattern_count: interferenceReport.patterns.length,
      patterns: interferenceReport.patterns.map(p => ({
        type: p.pattern_type || p.title,
        description: p.description,
      })),
      sentence: interferenceReport.sentence_display,
    });
  }

  /**
   * Log a cognitive load level change.
   *
   * @param {string} newLevel — 'LOW' | 'MEDIUM' | 'HIGH'
   * @param {number} jitter — the jitter value that triggered the change
   * @param {number} recommendedFrames — the new confidence threshold
   */
  logCognitiveLoadChange(newLevel, jitter, recommendedFrames) {
    this._counters.cognitiveLoadChanges++;
    this._counters.cognitiveLoadTimeline.push({ level: newLevel, ts: Date.now() });

    this._pushEvent('COGNITIVE_LOAD', {
      level: newLevel,
      jitter,
      recommended_frames: recommendedFrames,
    });
  }

  /**
   * Log a mastery achievement (gesture crossed mastery threshold).
   *
   * @param {string} gestureId — grammar ID that was just mastered
   * @param {number} count — total production count at time of mastery
   */
  logMasteryAchieved(gestureId, count) {
    this._counters.masteryAchievements++;

    this._pushEvent('MASTERY_ACHIEVED', {
      grammar_id: gestureId,
      production_count: count,
    });
  }

  /**
   * Log a subject-verb agreement error.
   *
   * @param {string} subject — the subject grammar_id
   * @param {string} verb — the verb grammar_id the user produced
   * @param {string} expectedVerb — the correct verb form
   */
  logAgreementError(subject, verb, expectedVerb) {
    this._counters.agreementErrors++;
    this._counters.agreementErrorDetails.push({
      subject, verb, expected_verb: expectedVerb, ts: Date.now(),
    });

    this._pushEvent('AGREEMENT_ERROR', {
      subject,
      verb_produced: verb,
      verb_expected: expectedVerb,
    });
  }

  // ===========================================================================
  // PUBLIC — Session statistics and reporting
  // ===========================================================================

  /**
   * Get a live summary of the current session.
   * Used to populate the session stats panel in SandboxMode debug view.
   *
   * @returns {SessionSummary}
   */
  getSessionSummary() {
    const elapsed = Date.now() - this.startTime;
    const elapsedMin = elapsed / 60000;

    return {
      session_id: this.sessionId,
      profile: this.profileType,
      duration_ms: elapsed,
      duration_display: this._formatDuration(elapsed),
      total_events: this.events.length,
      total_gestures: this._counters.totalGestures,
      sentences_completed: this._counters.sentencesCompleted,
      sentences_cleared: this._counters.sentencesCleared,
      isl_interferences: this._counters.islInterferences,
      agreement_errors: this._counters.agreementErrors,
      cognitive_load_changes: this._counters.cognitiveLoadChanges,
      mastery_achievements: this._counters.masteryAchievements,
      gestures_per_minute: elapsedMin > 0
        ? Math.round((this._counters.totalGestures / elapsedMin) * 10) / 10
        : 0,
      accuracy_rate: this._computeAccuracyRate(),
      sv_agreement_error_rate: this._computeSVAgreementErrorRate(),
      most_common_gesture: this._getMostCommonGesture(),
      cognitive_load_distribution: this._computeLoadDistribution(),
    };
  }

  /**
   * End the session and persist a summary to cross-session history.
   *
   * @returns {SessionSummary} the final session summary
   */
  endSession() {
    this._pushEvent('SESSION_END', {});

    const summary = this.getSessionSummary();

    // Persist to cross-session history
    this._history.push({
      ...summary,
      gestures_by_type: { ...this._counters.gesturesByType },
      isl_patterns: { ...this._counters.islPatterns },
      ended_at: Date.now(),
    });

    // Enforce max history
    if (this._history.length > MAX_SESSION_HISTORY) {
      this._history = this._history.slice(-MAX_SESSION_HISTORY);
    }

    this._saveHistory();
    return summary;
  }

  /**
   * Export the full session log as a JSON-serializable object.
   * Intended for download by the teacher/therapist/researcher.
   *
   * @returns {object} complete session data
   */
  exportSession() {
    return {
      meta: {
        framework: 'MLAF — Multimodal Language Acquisition Framework',
        version: '1.0.0',
        export_timestamp: new Date().toISOString(),
        session_id: this.sessionId,
        profile: this.profileType,
        duration_ms: Date.now() - this.startTime,
        book_reference: 'Birsh & Carreker (2018), Ch.1 p.52: diagnostic-prescriptive instruction.',
      },
      summary: this.getSessionSummary(),
      events: this.events,
      counters: {
        gestures_by_type: this._counters.gesturesByType,
        isl_patterns: this._counters.islPatterns,
        agreement_error_details: this._counters.agreementErrorDetails,
        cognitive_load_timeline: this._counters.cognitiveLoadTimeline,
      },
    };
  }

  /**
   * Trigger a browser download of the session data as JSON.
   */
  downloadSessionJSON() {
    const data = this.exportSession();
    const json = JSON.stringify(data, null, 2);
    const blob = new Blob([json], { type: 'application/json' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `mlaf-session-${this.sessionId}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  // ===========================================================================
  // PUBLIC — Cross-session history queries
  // ===========================================================================

  /**
   * Get all persisted session summaries for trend analysis.
   *
   * @returns {Array} array of session summaries, oldest first
   */
  getSessionHistory() {
    return this._history;
  }

  /**
   * Compute cross-session progress metrics.
   * This is the data that supports the patent's efficacy claims.
   *
   * @returns {object} cross-session trend report
   */
  getCrossSessionReport() {
    if (this._history.length < 2) {
      return { sessions: this._history.length, trend: 'insufficient_data' };
    }

    const half = Math.floor(this._history.length / 2);
    const early = this._history.slice(0, half);
    const recent = this._history.slice(-half);

    const earlyAccuracy = this._meanOf(early, 'accuracy_rate');
    const recentAccuracy = this._meanOf(recent, 'accuracy_rate');
    const accuracyChange = recentAccuracy - earlyAccuracy;

    const earlySVErrorRate = this._meanOf(early, 'sv_agreement_error_rate');
    const recentSVErrorRate = this._meanOf(recent, 'sv_agreement_error_rate');
    const svErrorChange = recentSVErrorRate - earlySVErrorRate;

    const earlyGPM = this._meanOf(early, 'gestures_per_minute');
    const recentGPM = this._meanOf(recent, 'gestures_per_minute');
    const gpmChange = recentGPM - earlyGPM;

    return {
      sessions: this._history.length,
      accuracy: {
        early: Math.round(earlyAccuracy * 100),
        recent: Math.round(recentAccuracy * 100),
        change_pct: Math.round(accuracyChange * 100),
        trend: accuracyChange > 0.05 ? 'improving' : accuracyChange < -0.05 ? 'declining' : 'stable',
      },
      sv_agreement_errors: {
        early_rate: Math.round(earlySVErrorRate * 100),
        recent_rate: Math.round(recentSVErrorRate * 100),
        change_pct: Math.round(svErrorChange * 100),
        trend: svErrorChange < -0.05 ? 'improving' : svErrorChange > 0.05 ? 'declining' : 'stable',
      },
      fluency: {
        early_gpm: Math.round(earlyGPM * 10) / 10,
        recent_gpm: Math.round(recentGPM * 10) / 10,
        change: Math.round(gpmChange * 10) / 10,
        trend: gpmChange > 0.5 ? 'improving' : gpmChange < -0.5 ? 'declining' : 'stable',
      },
    };
  }

  /**
   * Clear all cross-session history.
   */
  clearHistory() {
    this._history = [];
    try { localStorage.removeItem(this.storageKey); } catch (e) { /* ignore */ }
  }

  // ===========================================================================
  // PRIVATE — Event management
  // ===========================================================================

  _pushEvent(type, data) {
    this.events.push({
      type,
      ts: Date.now(),
      elapsed_ms: Date.now() - this.startTime,
      ...data,
    });
  }

  _generateSessionId() {
    const now = new Date();
    const date = now.toISOString().slice(0, 10).replace(/-/g, '');
    const time = now.toISOString().slice(11, 19).replace(/:/g, '');
    const rand = Math.random().toString(36).slice(2, 6);
    return `${date}-${time}-${rand}`;
  }

  // ===========================================================================
  // PRIVATE — Statistical computations
  // ===========================================================================

  /**
   * Accuracy rate = sentences_completed / (sentences_completed + sentences_cleared).
   * A cleared sentence is treated as an incomplete attempt.
   */
  _computeAccuracyRate() {
    const total = this._counters.sentencesCompleted + this._counters.sentencesCleared;
    if (total === 0) return 0;
    return Math.round((this._counters.sentencesCompleted / total) * 100) / 100;
  }

  /**
   * S-V agreement error rate = agreement_errors / total_gestures.
   * This is the metric that the patent claims MLAF reduces by ≥15%.
   */
  _computeSVAgreementErrorRate() {
    if (this._counters.totalGestures === 0) return 0;
    return Math.round((this._counters.agreementErrors / this._counters.totalGestures) * 1000) / 1000;
  }

  _getMostCommonGesture() {
    const entries = Object.entries(this._counters.gesturesByType);
    if (entries.length === 0) return null;
    entries.sort((a, b) => b[1] - a[1]);
    return { id: entries[0][0], count: entries[0][1] };
  }

  /**
   * Compute time-weighted distribution of cognitive load levels.
   * Returns { LOW: %, MEDIUM: %, HIGH: % }.
   */
  _computeLoadDistribution() {
    const timeline = this._counters.cognitiveLoadTimeline;
    if (timeline.length === 0) return { LOW: 100, MEDIUM: 0, HIGH: 0 };

    const totals = { LOW: 0, MEDIUM: 0, HIGH: 0 };
    const now = Date.now();

    for (let i = 0; i < timeline.length; i++) {
      const start = timeline[i].ts;
      const end = (i + 1 < timeline.length) ? timeline[i + 1].ts : now;
      const duration = end - start;
      totals[timeline[i].level] = (totals[timeline[i].level] || 0) + duration;
    }

    const total = totals.LOW + totals.MEDIUM + totals.HIGH;
    if (total === 0) return { LOW: 100, MEDIUM: 0, HIGH: 0 };

    return {
      LOW: Math.round((totals.LOW / total) * 100),
      MEDIUM: Math.round((totals.MEDIUM / total) * 100),
      HIGH: Math.round((totals.HIGH / total) * 100),
    };
  }

  _formatDuration(ms) {
    const sec = Math.floor(ms / 1000);
    const min = Math.floor(sec / 60);
    const remSec = sec % 60;
    return min > 0 ? `${min}m ${remSec}s` : `${remSec}s`;
  }

  _meanOf(arr, key) {
    if (arr.length === 0) return 0;
    return arr.reduce((sum, item) => sum + (item[key] || 0), 0) / arr.length;
  }

  // ===========================================================================
  // PRIVATE — Storage helpers
  // ===========================================================================

  _loadHistory() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      if (!Array.isArray(parsed)) return [];
      return parsed;
    } catch (e) {
      return [];
    }
  }

  _saveHistory() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this._history));
    } catch (e) {
      // localStorage full or unavailable — fail silently
    }
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/**
 * Format an accuracy rate (0–1) as a percentage string.
 * @param {number} rate
 * @returns {string}
 */
export function formatAccuracy(rate) {
  if (rate === 0) return '--';
  return `${Math.round(rate * 100)}%`;
}

/**
 * Get a CSS color for an accuracy rate.
 * @param {number} rate — 0.0 to 1.0
 * @returns {string}
 */
export function accuracyColor(rate) {
  if (rate >= 0.8) return '#4ade80';   // green
  if (rate >= 0.5) return '#facc15';   // yellow
  if (rate > 0)    return '#f87171';   // red
  return '#64748b';                     // grey (no data)
}

/**
 * Get a trend arrow and color for a cross-session trend.
 * @param {string} trend — 'improving' | 'stable' | 'declining' | 'insufficient_data'
 * @returns {{ arrow: string, color: string }}
 */
export function trendDisplay(trend) {
  switch (trend) {
    case 'improving':         return { arrow: '↑', color: '#4ade80' };
    case 'declining':         return { arrow: '↓', color: '#f87171' };
    case 'stable':            return { arrow: '→', color: '#facc15' };
    case 'insufficient_data': return { arrow: '·', color: '#64748b' };
    default:                  return { arrow: '·', color: '#64748b' };
  }
}

export default SessionDataLogger;
