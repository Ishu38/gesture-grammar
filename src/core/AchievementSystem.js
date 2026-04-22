/**
 * AchievementSystem.js — Achievement & Streak Tracking for Gamification
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Tracks:
 *   - 11 defined achievements across 4 categories (sentence, mastery, streak, quality)
 *   - Daily practice streaks (current + longest)
 *   - Running counters for each achievement metric
 *
 * Each event method returns { newlyUnlocked: Achievement[] } so the UI
 * can display toast notifications on unlock.
 *
 * Persists to localStorage under 'mlaf_achievements_v1'.
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'mlaf_achievements_v1';

/** Milliseconds in one day */
const MS_PER_DAY = 86400000;

// =============================================================================
// ACHIEVEMENT DEFINITIONS
// =============================================================================

export const ACHIEVEMENTS = [
  // Sentence milestones
  { id: 'FIRST_SENTENCE',    category: 'sentence', title: 'First Words',       description: 'Complete your first sentence',                 icon: '⭐', threshold: 1,  metric: 'sentences' },
  { id: 'FIVE_SENTENCES',    category: 'sentence', title: 'Sentence Builder',  description: 'Complete 5 sentences',                         icon: '🏆', threshold: 5,  metric: 'sentences' },
  { id: 'TWENTY_SENTENCES',  category: 'sentence', title: 'Grammar Pro',       description: 'Complete 20 sentences',                        icon: '👑', threshold: 20, metric: 'sentences' },

  // Mastery milestones
  { id: 'FIRST_MASTERY',     category: 'mastery',  title: 'First Mastery',     description: 'Master your first gesture',                    icon: '🥇', threshold: 1,  metric: 'mastered_gestures' },
  { id: 'STAGE_COMPLETE',    category: 'mastery',  title: 'Stage Clear',       description: 'Complete a full curriculum stage',              icon: '🚩', threshold: 1,  metric: 'stages_complete' },
  { id: 'ALL_MASTERED',      category: 'mastery',  title: 'Grand Master',      description: 'Master all 18 gestures',                       icon: '💎', threshold: 18, metric: 'mastered_gestures' },

  // Streak milestones
  { id: 'STREAK_3',          category: 'streak',   title: 'Getting Going',     description: 'Practice 3 days in a row',                     icon: '🔥', threshold: 3,  metric: 'streak' },
  { id: 'STREAK_7',          category: 'streak',   title: 'Week Warrior',      description: 'Practice 7 days in a row',                     icon: '🔥', threshold: 7,  metric: 'streak' },
  { id: 'STREAK_30',         category: 'streak',   title: 'Monthly Champion',  description: 'Practice 30 days in a row',                    icon: '🔥', threshold: 30, metric: 'streak' },

  // Quality milestones
  { id: 'NO_ERROR_SESSION',  category: 'quality',  title: 'Flawless',          description: 'Complete a session with no grammar errors',     icon: '🛡️', threshold: 1,  metric: 'flawless_sessions' },
  { id: 'NO_ISL_TRANSFER',   category: 'quality',  title: 'English Thinker',   description: 'Build 5 sentences with no ISL transfer errors', icon: '🧠', threshold: 5,  metric: 'no_isl_sentences' },
];

// =============================================================================
// ACHIEVEMENT SYSTEM
// =============================================================================

export class AchievementSystem {
  /**
   * @param {object} config
   * @param {string} [config.storageKey]
   */
  constructor(config = {}) {
    this._storageKey = config.storageKey || STORAGE_KEY;
    this._state = this._load();
  }

  // ===========================================================================
  // PUBLIC — Event Handlers (called from SandboxMode)
  // ===========================================================================

  /**
   * Called when a sentence is completed successfully.
   * @param {{ hasISLError: boolean }} sentenceData
   * @returns {{ newlyUnlocked: Array }}
   */
  onSentenceComplete(sentenceData = {}) {
    this._state.counters.sentences++;

    if (!sentenceData.hasISLError) {
      this._state.counters.no_isl_sentences++;
    } else {
      this._state.counters.no_isl_sentences = 0; // Reset consecutive count
    }

    return this._checkAndUnlock();
  }

  /**
   * Called when a gesture is mastered (threshold reached in GestureMasteryGate).
   * @param {string} gestureId
   * @returns {{ newlyUnlocked: Array }}
   */
  onGestureMastered(gestureId) {
    if (!this._state.masteredGestures.includes(gestureId)) {
      this._state.masteredGestures.push(gestureId);
    }
    this._state.counters.mastered_gestures = this._state.masteredGestures.length;
    return this._checkAndUnlock();
  }

  /**
   * Called when a curriculum stage is completed.
   * @param {number} stageNumber
   * @returns {{ newlyUnlocked: Array }}
   */
  onStageComplete(stageNumber) {
    if (!this._state.completedStages.includes(stageNumber)) {
      this._state.completedStages.push(stageNumber);
    }
    this._state.counters.stages_complete = this._state.completedStages.length;
    return this._checkAndUnlock();
  }

  /**
   * Called when a session ends.
   * @param {{ isl_interferences: number, agreement_errors: number }} sessionSummary
   * @returns {{ newlyUnlocked: Array }}
   */
  onSessionEnd(sessionSummary = {}) {
    const errors = (sessionSummary.isl_interferences || 0) + (sessionSummary.agreement_errors || 0);
    if (errors === 0 && this._state.counters.sentences > 0) {
      this._state.counters.flawless_sessions++;
    }
    return this._checkAndUnlock();
  }

  // ===========================================================================
  // PUBLIC — Streak Management
  // ===========================================================================

  /**
   * Record daily activity. Call on session start.
   * Updates the practice streak.
   */
  recordDailyActivity() {
    const today = this._dateKey(Date.now());
    const lastDate = this._state.streak.lastDate;

    if (lastDate === today) {
      return; // Already recorded today
    }

    const yesterday = this._dateKey(Date.now() - MS_PER_DAY);

    if (lastDate === yesterday) {
      // Consecutive day — extend streak
      this._state.streak.current++;
    } else {
      // Streak broken — reset
      this._state.streak.current = 1;
    }

    this._state.streak.lastDate = today;

    if (this._state.streak.current > this._state.streak.longest) {
      this._state.streak.longest = this._state.streak.current;
    }

    this._state.counters.streak = this._state.streak.current;
    this._save();
  }

  /**
   * Get the current streak data.
   * @returns {{ current: number, longest: number, lastDate: string }}
   */
  getStreak() {
    return { ...this._state.streak };
  }

  // ===========================================================================
  // PUBLIC — Query
  // ===========================================================================

  /**
   * Get all unlocked achievements with their unlock timestamps.
   * @returns {Array}
   */
  getUnlockedAchievements() {
    return ACHIEVEMENTS
      .filter(a => this._state.unlocked[a.id])
      .map(a => ({
        ...a,
        unlockedAt: this._state.unlocked[a.id],
      }));
  }

  /**
   * Get all locked (not yet earned) achievements.
   * @returns {Array}
   */
  getLockedAchievements() {
    return ACHIEVEMENTS.filter(a => !this._state.unlocked[a.id]);
  }

  /**
   * Get progress toward a specific achievement.
   * @param {string} achievementId
   * @returns {{ current: number, threshold: number, percent: number, unlocked: boolean }}
   */
  getProgress(achievementId) {
    const achievement = ACHIEVEMENTS.find(a => a.id === achievementId);
    if (!achievement) return null;

    const current = this._state.counters[achievement.metric] || 0;
    const unlocked = !!this._state.unlocked[achievement.id];

    return {
      current,
      threshold: achievement.threshold,
      percent: Math.min(100, Math.round((current / achievement.threshold) * 100)),
      unlocked,
    };
  }

  /**
   * Get full achievement report for the UI panel.
   * @returns {object}
   */
  getAchievementReport() {
    const unlocked = this.getUnlockedAchievements();
    const locked = this.getLockedAchievements();

    // Find close-to-unlocking (>= 50% progress)
    const closeToUnlock = locked
      .map(a => ({
        ...a,
        progress: this.getProgress(a.id),
      }))
      .filter(a => a.progress.percent >= 50)
      .sort((a, b) => b.progress.percent - a.progress.percent);

    return {
      totalAchievements: ACHIEVEMENTS.length,
      unlockedCount: unlocked.length,
      lockedCount: locked.length,
      unlocked,
      locked,
      closeToUnlock,
      streak: this.getStreak(),
      counters: { ...this._state.counters },
    };
  }

  // ===========================================================================
  // PUBLIC — Lifecycle
  // ===========================================================================

  /**
   * Reset all achievement data.
   */
  reset() {
    this._state = this._defaultState();
    this._save();
  }

  /**
   * Get serializable descriptor.
   * @returns {object}
   */
  toDescriptor() {
    return {
      unlockedCount: Object.keys(this._state.unlocked).length,
      totalAchievements: ACHIEVEMENTS.length,
      streak: this._state.streak,
      counters: this._state.counters,
    };
  }

  // ===========================================================================
  // PRIVATE
  // ===========================================================================

  /** Check all achievements and unlock any that meet their threshold. */
  _checkAndUnlock() {
    const newlyUnlocked = [];

    for (const achievement of ACHIEVEMENTS) {
      if (this._state.unlocked[achievement.id]) continue; // Already unlocked

      const current = this._state.counters[achievement.metric] || 0;
      if (current >= achievement.threshold) {
        this._state.unlocked[achievement.id] = Date.now();
        newlyUnlocked.push({ ...achievement, unlockedAt: Date.now() });
      }
    }

    if (newlyUnlocked.length > 0) {
      this._save();
    }

    return { newlyUnlocked };
  }

  /** Convert timestamp to YYYY-MM-DD string for date comparison. */
  _dateKey(timestamp) {
    const d = new Date(timestamp);
    return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
  }

  _defaultState() {
    return {
      unlocked: {},               // { [achievementId]: unlockTimestamp }
      counters: {
        sentences: 0,
        mastered_gestures: 0,
        stages_complete: 0,
        streak: 0,
        flawless_sessions: 0,
        no_isl_sentences: 0,      // Consecutive sentences with no ISL errors
      },
      streak: {
        current: 0,
        longest: 0,
        lastDate: null,
      },
      masteredGestures: [],
      completedStages: [],
    };
  }

  _load() {
    try {
      const raw = localStorage.getItem(this._storageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        // Merge with defaults to handle schema evolution
        const defaults = this._defaultState();
        return {
          unlocked: { ...defaults.unlocked, ...parsed.unlocked },
          counters: { ...defaults.counters, ...parsed.counters },
          streak: { ...defaults.streak, ...parsed.streak },
          masteredGestures: parsed.masteredGestures || [],
          completedStages: parsed.completedStages || [],
        };
      }
    } catch { /* ignore */ }
    return this._defaultState();
  }

  _save() {
    try {
      localStorage.setItem(this._storageKey, JSON.stringify(this._state));
    } catch { /* ignore */ }
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/**
 * Get color for achievement unlock status.
 * @param {boolean} unlocked
 * @returns {string}
 */
export function achievementColor(unlocked) {
  return unlocked ? '#fbbf24' : '#4a4a6a'; // gold if unlocked, grey if locked
}

/**
 * Get color for streak days.
 * @param {number} days
 * @returns {string}
 */
export function streakColor(days) {
  if (days >= 30) return '#ef4444';  // red-hot
  if (days >= 7) return '#f97316';   // orange
  if (days >= 3) return '#facc15';   // yellow
  if (days >= 1) return '#60a5fa';   // blue
  return '#64748b';                   // grey
}

export default AchievementSystem;
