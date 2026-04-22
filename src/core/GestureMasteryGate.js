/**
 * GestureMasteryGate.js — Cumulative Mastery-Gated Curriculum Sequencer
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Implements the Structured Literacy cumulative principle:
 * new gesture-grammar forms are introduced only after earlier forms are mastered.
 * Mastery = MASTERY_THRESHOLD correct productions of a gesture.
 *
 * Theoretical grounding:
 *   Birsh & Carreker (2018):
 *     Ch. 1, p. 52: Structured Literacy characteristics — "explicit, systematic,
 *       cumulative, diagnostic-prescriptive instruction."
 *     Ch. 2, "Structured Literacy Instruction" (Coyne, Carnine, Kameenui):
 *       "Cumulative review and mastery learning — new skills are introduced only
 *       after previous skills are mastered."
 *   Preface, p. 44–45: Criterion 5 for effective instructional apps:
 *     "Practice activities develop automaticity." Automaticity is the goal of
 *     mastery-gated repetition; the gate enforces sufficient practice.
 *
 * Mechanism:
 *   1. CURRICULUM_SEQUENCE defines 5 ordered stages of gesture-grammar forms.
 *   2. Each production (gesture lock) increments a per-gesture counter.
 *   3. A gesture is "mastered" when its count reaches MASTERY_THRESHOLD.
 *   4. A stage is "complete" when ALL its gestures are mastered.
 *   5. The next stage unlocks only when the current stage is complete.
 *   6. State persists across sessions via localStorage.
 *
 * Patent claim:
 *   "A cumulative mastery-gated curriculum sequencer for gesture-based language
 *   instruction, implementing structured literacy progression principles (Birsh &
 *   Carreker, 2018, Ch.1, Ch.2) in a real-time gesture recognition system,
 *   wherein each gesture stage is unlocked only upon demonstrating mastery of
 *   prerequisite gesture-grammar forms, operationalizing Criterion 5 of the
 *   Preface (effective instructional apps must develop automaticity)."
 */

// =============================================================================
// CONSTANTS
// =============================================================================

const STORAGE_KEY = 'mlaf_mastery_v1';

// Correct productions required per gesture to consider it "mastered".
// 5 productions ≈ distributed practice across multiple sentence attempts.
const MASTERY_THRESHOLD = 5;

// =============================================================================
// CURRICULUM SEQUENCE
// =============================================================================

/**
 * CURRICULUM_SEQUENCE defines the staged progression of gesture-grammar forms.
 *
 * Organized by the Structured Literacy cumulative principle:
 *   Stage 1 — Foundation:          First-person subject + base-form verb (simplest unit)
 *   Stage 2 — Pronoun Expansion:   Third-person subjects + core action verbs
 *   Stage 3 — Object Introduction: Concrete nouns to enable complete SVO sentences
 *   Stage 4 — S-V Agreement:       Third-person singular -s inflection (English morphology)
 *   Stage 5 — Full Paradigm:       All remaining subject pronouns; complete inflectional set
 *
 * Book reference: Birsh & Carreker (2018), Ch.2 — "skills taught in a logical,
 * sequential manner so that each new skill builds on previously learned skills."
 */
export const CURRICULUM_SEQUENCE = [
  {
    stage: 1,
    label: 'Stage 1 — Foundation',
    description: 'First-person subject and base verb. Minimum viable sentence.',
    book_reference: 'Birsh & Carreker (2018), Ch.1, p.52: explicit, systematic, cumulative instruction.',
    gestures: ['SUBJECT_I', 'STOP'],
  },
  {
    stage: 2,
    label: 'Stage 2 — Pronoun Expansion',
    description: 'Third-person subjects and core action verbs (GRAB, EAT).',
    book_reference: 'Birsh & Carreker (2018), Ch.2: cumulative introduction of new morphological categories.',
    gestures: ['SUBJECT_HE', 'SUBJECT_SHE', 'GRAB', 'EAT'],
  },
  {
    stage: 3,
    label: 'Stage 3 — Object Introduction',
    description: 'Concrete noun objects — enables complete SVO sentences.',
    book_reference: 'Birsh & Carreker (2018), Ch.2: skills taught in a logical, sequential manner.',
    gestures: ['APPLE', 'BALL'],
  },
  {
    stage: 4,
    label: 'Stage 4 — Subject-Verb Agreement',
    description: 'Third-person singular inflection (-s). Core English morphosyntax.',
    book_reference: 'Birsh & Carreker (2018), Ch.19: L2 learners require explicit morphosyntax instruction.',
    gestures: ['GRABS', 'EATS', 'STOPS'],
  },
  {
    stage: 5,
    label: 'Stage 5 — Full Paradigm',
    description: 'Remaining subject pronouns and complete inflectional paradigm.',
    book_reference: 'Birsh & Carreker (2018), Ch.12 (Garnett): fluency through extended, cumulative practice.',
    gestures: ['SUBJECT_YOU', 'SUBJECT_WE', 'SUBJECT_THEY'],
  },
];

// Build a flat map: gesture_id → stage number (for O(1) lookups)
const GESTURE_STAGE_MAP = {};
CURRICULUM_SEQUENCE.forEach(({ stage, gestures }) => {
  gestures.forEach(g => { GESTURE_STAGE_MAP[g] = stage; });
});

// =============================================================================
// GESTURE MASTERY GATE
// =============================================================================

export class GestureMasteryGate {
  /**
   * @param {object} config
   * @param {string} [config.storageKey] — override localStorage key
   * @param {number} [config.threshold]  — productions required for mastery
   */
  constructor(config = {}) {
    this.storageKey = config.storageKey || STORAGE_KEY;
    this.threshold  = config.threshold  || MASTERY_THRESHOLD;

    /**
     * Per-gesture data:
     * { [gesture_id]: { count: number, mastered: boolean, firstSeen: number, masteredAt: number|null } }
     */
    this._data = this._load();
  }

  // ===========================================================================
  // PUBLIC — Recording productions
  // ===========================================================================

  /**
   * Record one correct production of a gesture.
   *
   * Called every time a word locks into the sentence (in SandboxMode) or when
   * a curriculum challenge is answered correctly (via LessonManager integration).
   *
   * @param {string} gestureId — grammar ID of the gesture that was produced
   * @returns {{ wasMastered: boolean, isNowMastered: boolean, count: number }}
   */
  recordProduction(gestureId) {
    if (!this._data[gestureId]) {
      this._data[gestureId] = {
        count:      0,
        mastered:   false,
        firstSeen:  Date.now(),
        masteredAt: null,
      };
    }

    const wasMastered = this._data[gestureId].mastered;
    this._data[gestureId].count++;

    const isNowMastered = this._data[gestureId].count >= this.threshold;
    if (isNowMastered && !wasMastered) {
      this._data[gestureId].mastered   = true;
      this._data[gestureId].masteredAt = Date.now();
    }

    this._save();

    return {
      wasMastered,
      isNowMastered,
      count: this._data[gestureId].count,
    };
  }

  // ===========================================================================
  // PUBLIC — Stage and unlock queries
  // ===========================================================================

  /**
   * Get the highest stage that is fully mastered (all gestures at that stage mastered).
   * Returns 0 if no stage is fully complete yet.
   *
   * @returns {number}
   */
  getHighestMasteredStage() {
    let highest = 0;
    for (const { stage, gestures } of CURRICULUM_SEQUENCE) {
      const allMastered = gestures.every(g => this._data[g]?.mastered);
      if (allMastered) {
        highest = stage;
      } else {
        break; // stages must be mastered sequentially
      }
    }
    return highest;
  }

  /**
   * Get the current active stage (lowest stage not yet fully mastered).
   *
   * @returns {number} 1–5
   */
  getCurrentStage() {
    const highest = this.getHighestMasteredStage();
    return Math.min(highest + 1, CURRICULUM_SEQUENCE.length);
  }

  /**
   * Check whether a gesture is "unlocked" — its stage is ≤ the current active stage.
   * Gestures not in the curriculum sequence are always unlocked.
   *
   * @param {string} gestureId
   * @returns {boolean}
   */
  isGestureUnlocked(gestureId) {
    const gestureStage = GESTURE_STAGE_MAP[gestureId];
    if (gestureStage === undefined) return true; // not gated
    return gestureStage <= this.getCurrentStage();
  }

  /**
   * Check whether a gesture is mastered.
   *
   * @param {string} gestureId
   * @returns {boolean}
   */
  isGestureMastered(gestureId) {
    return this._data[gestureId]?.mastered || false;
  }

  /**
   * Get the raw production count for a gesture.
   *
   * @param {string} gestureId
   * @returns {number}
   */
  getProductionCount(gestureId) {
    return this._data[gestureId]?.count || 0;
  }

  // ===========================================================================
  // PUBLIC — Report for UI
  // ===========================================================================

  /**
   * Get the full mastery report for the SandboxMode debug panel.
   *
   * @returns {object} MasteryReport with per-stage and per-gesture detail
   */
  getMasteryReport() {
    const currentStage    = this.getCurrentStage();
    const highestMastered = this.getHighestMasteredStage();

    const stages = CURRICULUM_SEQUENCE.map(({ stage, label, description, gestures }) => {
      const isUnlocked = stage <= currentStage;
      const isComplete = gestures.every(g => this._data[g]?.mastered);

      const gestureStatuses = gestures.map(g => ({
        id:       g,
        count:    this._data[g]?.count    || 0,
        mastered: this._data[g]?.mastered || false,
        progress: Math.min(1, (this._data[g]?.count || 0) / this.threshold),
      }));

      return {
        stage,
        label,
        description,
        isUnlocked,
        isComplete,
        gestureStatuses,
      };
    });

    return {
      currentStage,
      highestMastered,
      threshold: this.threshold,
      stages,
      totalMastered: Object.values(this._data).filter(d => d.mastered).length,
      totalTracked:  Object.keys(this._data).length,
    };
  }

  /**
   * Clear all mastery data and reset to initial state.
   */
  clearHistory() {
    this._data = {};
    try { localStorage.removeItem(this.storageKey); } catch (e) { /* ignore */ }
  }

  // ===========================================================================
  // PRIVATE — Storage helpers
  // ===========================================================================

  _load() {
    try {
      const raw = localStorage.getItem(this.storageKey);
      if (!raw) return {};
      const parsed = JSON.parse(raw);
      if (typeof parsed !== 'object' || Array.isArray(parsed)) return {};
      return parsed;
    } catch (e) {
      return {};
    }
  }

  _save() {
    try {
      localStorage.setItem(this.storageKey, JSON.stringify(this._data));
    } catch (e) {
      // localStorage full or unavailable — fail silently
    }
  }
}

// =============================================================================
// DISPLAY HELPERS (used in SandboxMode mastery panel)
// =============================================================================

/**
 * Get a CSS color for a gesture's mastery progress fraction.
 *
 * @param {number} progress — 0.0 to 1.0
 * @returns {string} hex color
 */
export function masteryProgressColor(progress) {
  if (progress >= 1.0) return '#4ade80';  // green  — mastered
  if (progress >= 0.6) return '#facc15';  // yellow — approaching mastery
  return '#60a5fa';                        // blue   — in progress
}

/**
 * Get a short label for a gesture's mastery status.
 *
 * @param {boolean} mastered
 * @param {number}  count
 * @param {number}  threshold
 * @returns {string}
 */
export function masteryStatusLabel(mastered, count, threshold) {
  if (mastered) return 'Mastered';
  if (count === 0) return 'Not started';
  return `${count}/${threshold}`;
}

export default GestureMasteryGate;
