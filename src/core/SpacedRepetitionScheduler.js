/**
 * SpacedRepetitionScheduler.js — SM-2 Spaced Repetition for Gesture Review
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Implements the SuperMemo SM-2 algorithm to optimally schedule gesture
 * review intervals. Each gesture tracked independently with:
 *   - easeFactor (EF): difficulty multiplier, starts at 2.5
 *   - interval: days until next review
 *   - repetitions: consecutive correct reviews
 *   - nextReview: timestamp of next due date
 *
 * Quality scale (0–5):
 *   5 = perfect response, no hesitation
 *   4 = correct after brief hesitation
 *   3 = correct with difficulty
 *   2 = incorrect but close (remembered after seeing answer)
 *   1 = incorrect, vague memory
 *   0 = complete blackout
 *
 * Integration:
 *   - On gesture lock (correct production): quality = 5
 *   - On guided practice correct: quality = 4–5 (based on hints used)
 *   - On guided practice incorrect: quality = 1–2
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'mlaf_srs_v1';

/** Initial ease factor for new items */
const INITIAL_EF = 2.5;

/** Minimum ease factor (prevents items from becoming "too hard") */
const MIN_EF = 1.3;

/** Milliseconds in one day */
const MS_PER_DAY = 86400000;

// =============================================================================
// SPACED REPETITION SCHEDULER
// =============================================================================

export class SpacedRepetitionScheduler {
  /**
   * @param {object} config
   * @param {string} [config.storageKey] — localStorage key
   */
  constructor(config = {}) {
    this._storageKey = config.storageKey || STORAGE_KEY;
    this._data = this._load();
  }

  // ===========================================================================
  // PUBLIC — Core SM-2
  // ===========================================================================

  /**
   * Record a review for a gesture and update its schedule.
   *
   * @param {string} gestureId — gesture identifier (e.g., 'GRAB', 'SUBJECT_I')
   * @param {number} quality — review quality 0–5
   * @returns {{ easeFactor: number, interval: number, nextReview: number, repetitions: number }}
   */
  recordReview(gestureId, quality) {
    quality = Math.max(0, Math.min(5, Math.round(quality)));

    let item = this._data[gestureId];
    if (!item) {
      item = {
        easeFactor: INITIAL_EF,
        interval: 0,
        repetitions: 0,
        nextReview: 0,
        lastReview: null,
        firstSeen: Date.now(),
        totalReviews: 0,
        correctReviews: 0,
      };
      this._data[gestureId] = item;
    }

    item.totalReviews++;
    item.lastReview = Date.now();

    if (quality >= 3) {
      // Correct response
      item.correctReviews++;

      if (item.repetitions === 0) {
        item.interval = 1; // 1 day
      } else if (item.repetitions === 1) {
        item.interval = 6; // 6 days
      } else {
        item.interval = Math.round(item.interval * item.easeFactor);
      }
      item.repetitions++;
    } else {
      // Incorrect — reset to beginning
      item.repetitions = 0;
      item.interval = 1;
    }

    // Update ease factor using SM-2 formula
    item.easeFactor = Math.max(
      MIN_EF,
      item.easeFactor + (0.1 - (5 - quality) * (0.08 + (5 - quality) * 0.02))
    );

    // Schedule next review
    item.nextReview = Date.now() + item.interval * MS_PER_DAY;

    this._save();

    return {
      easeFactor: item.easeFactor,
      interval: item.interval,
      nextReview: item.nextReview,
      repetitions: item.repetitions,
    };
  }

  // ===========================================================================
  // PUBLIC — Query
  // ===========================================================================

  /**
   * Check if a gesture is due for review.
   * @param {string} gestureId
   * @returns {boolean}
   */
  isDue(gestureId) {
    const item = this._data[gestureId];
    if (!item) return false;
    return Date.now() >= item.nextReview;
  }

  /**
   * Get all gesture IDs currently due for review.
   * @returns {string[]}
   */
  getDueGestures() {
    const now = Date.now();
    return Object.keys(this._data).filter(gId => now >= this._data[gId].nextReview);
  }

  /**
   * Get the number of gestures currently due.
   * @returns {number}
   */
  getDueCount() {
    return this.getDueGestures().length;
  }

  /**
   * Get the schedule for a specific gesture.
   * @param {string} gestureId
   * @returns {object|null}
   */
  getGestureSchedule(gestureId) {
    const item = this._data[gestureId];
    if (!item) return null;

    return {
      easeFactor: item.easeFactor,
      interval: item.interval,
      nextReview: item.nextReview,
      lastReview: item.lastReview,
      repetitions: item.repetitions,
      totalReviews: item.totalReviews,
      correctReviews: item.correctReviews,
      accuracy: item.totalReviews > 0
        ? Math.round((item.correctReviews / item.totalReviews) * 100)
        : 0,
      isDue: Date.now() >= item.nextReview,
    };
  }

  /**
   * Get the full schedule for all tracked gestures.
   * @returns {Object} Map<gestureId, schedule>
   */
  getFullSchedule() {
    const schedule = {};
    for (const gId of Object.keys(this._data)) {
      schedule[gId] = this.getGestureSchedule(gId);
    }
    return schedule;
  }

  /**
   * Get a review report for the UI panel.
   * @returns {object}
   */
  getReviewReport() {
    const now = Date.now();
    const endOfDay = new Date();
    endOfDay.setHours(23, 59, 59, 999);
    const endOfTomorrow = new Date(endOfDay.getTime() + MS_PER_DAY);

    const allIds = Object.keys(this._data);
    const dueNow = allIds.filter(gId => now >= this._data[gId].nextReview);
    const dueToday = allIds.filter(gId =>
      this._data[gId].nextReview <= endOfDay.getTime() &&
      now < this._data[gId].nextReview
    );
    const dueTomorrow = allIds.filter(gId =>
      this._data[gId].nextReview > endOfDay.getTime() &&
      this._data[gId].nextReview <= endOfTomorrow.getTime()
    );

    // Per-gesture summary sorted by next due date
    const perGesture = allIds
      .map(gId => ({
        gestureId: gId,
        ...this.getGestureSchedule(gId),
      }))
      .sort((a, b) => a.nextReview - b.nextReview);

    // Average ease factor
    const avgEF = allIds.length > 0
      ? allIds.reduce((s, gId) => s + this._data[gId].easeFactor, 0) / allIds.length
      : INITIAL_EF;

    return {
      totalTracked: allIds.length,
      dueNow: dueNow.length,
      dueNowIds: dueNow,
      dueToday: dueToday.length,
      dueTomorrow: dueTomorrow.length,
      averageEaseFactor: Math.round(avgEF * 100) / 100,
      perGesture,
    };
  }

  // ===========================================================================
  // PUBLIC — Lifecycle
  // ===========================================================================

  /**
   * Reset all scheduling data.
   */
  reset() {
    this._data = {};
    this._save();
  }

  /**
   * Clear history and remove from storage.
   */
  clearHistory() {
    this._data = {};
    try {
      localStorage.removeItem(this._storageKey);
    } catch { /* ignore */ }
  }

  /**
   * Get serializable descriptor for logging/export.
   * @returns {object}
   */
  toDescriptor() {
    return {
      totalTracked: Object.keys(this._data).length,
      dueCount: this.getDueCount(),
      schedule: this.getFullSchedule(),
    };
  }

  // ===========================================================================
  // PRIVATE — Persistence
  // ===========================================================================

  _load() {
    try {
      const raw = localStorage.getItem(this._storageKey);
      if (raw) return JSON.parse(raw);
    } catch { /* ignore */ }
    return {};
  }

  _save() {
    try {
      localStorage.setItem(this._storageKey, JSON.stringify(this._data));
    } catch { /* ignore */ }
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/**
 * Convert an interval in days to a human-readable label.
 * @param {number} days
 * @returns {string}
 */
export function intervalToLabel(days) {
  if (days <= 0) return 'Now';
  if (days === 1) return '1d';
  if (days < 7) return `${days}d`;
  if (days < 14) return '1w';
  if (days < 30) return `${Math.round(days / 7)}w`;
  if (days < 60) return '1mo';
  return `${Math.round(days / 30)}mo`;
}

/**
 * Get color for an ease factor value.
 * @param {number} ef — ease factor
 * @returns {string}
 */
export function easeFactorColor(ef) {
  if (ef >= 2.5) return '#4ade80';   // green — easy
  if (ef >= 1.8) return '#facc15';   // yellow — moderate
  return '#f87171';                   // red — hard
}

/**
 * Get color for due status.
 * @param {boolean} isDue
 * @returns {string}
 */
export function dueStatusColor(isDue) {
  return isDue ? '#f87171' : '#4ade80'; // red if due, green if not
}

export default SpacedRepetitionScheduler;
