/**
 * CurriculumEngine.js
 * Manages lesson loading, progress tracking, and challenge validation
 */

// =============================================================================
// TYPE DEFINITIONS (JSDoc)
// =============================================================================

/**
 * @typedef {Object} Challenge
 * @property {string} challenge_id
 * @property {string} prompt_display
 * @property {string} target_concept
 * @property {string[]} required_sequence
 * @property {Object} error_feedback_map
 */

/**
 * @typedef {Object} Lesson
 * @property {string} level_id
 * @property {string} title
 * @property {number} difficulty
 * @property {Challenge[]} challenges
 */

/**
 * @typedef {Object} ChallengeResult
 * @property {boolean} isCorrect
 * @property {boolean} isPartial
 * @property {string|null} feedback
 * @property {string|null} errorType
 * @property {number} score
 */

// =============================================================================
// CHALLENGE VALIDATOR
// =============================================================================

/**
 * Validates a user's answer against the required sequence
 * @param {Challenge} challenge - The challenge being attempted
 * @param {string[]} userSequence - Array of Grammar IDs the user input
 * @returns {ChallengeResult}
 */
export function validateChallenge(challenge, userSequence) {
  const { required_sequence, acceptable_alternatives, error_feedback_map } = challenge;

  // Check for exact match with required sequence
  if (arraysEqual(userSequence, required_sequence)) {
    return {
      isCorrect: true,
      isPartial: false,
      feedback: 'Correct! ' + (challenge.target_concept_explanation || ''),
      errorType: null,
      score: challenge.scoring?.points || 100,
    };
  }

  // Check acceptable alternatives
  if (acceptable_alternatives) {
    for (const alt of acceptable_alternatives) {
      if (arraysEqual(userSequence, alt.sequence)) {
        return {
          isCorrect: true,
          isPartial: false,
          feedback: `Correct! ${alt.note || ''}`,
          errorType: null,
          score: challenge.scoring?.points || 100,
        };
      }
    }
  }

  // Check for partial match (user is on the right track)
  const isPartial = isPartialMatch(userSequence, required_sequence);

  // Look up specific error feedback
  const errorKey = userSequence.join('|');
  const specificFeedback = error_feedback_map?.[errorKey];

  if (specificFeedback) {
    return {
      isCorrect: false,
      isPartial,
      feedback: specificFeedback.feedback,
      errorType: specificFeedback.error_type,
      ruleReference: specificFeedback.rule_reference,
      gestureHint: specificFeedback.gesture_hint,
      score: 0,
    };
  }

  // Generate generic feedback based on error analysis
  const analysis = analyzeError(userSequence, required_sequence, challenge);

  return {
    isCorrect: false,
    isPartial,
    feedback: analysis.feedback,
    errorType: analysis.errorType,
    score: 0,
  };
}

/**
 * Check if user is on the right track (partial match)
 */
function isPartialMatch(userSeq, requiredSeq) {
  if (userSeq.length === 0) return false;
  if (userSeq.length >= requiredSeq.length) return false;

  for (let i = 0; i < userSeq.length; i++) {
    if (userSeq[i] !== requiredSeq[i]) return false;
  }
  return true;
}

/**
 * Analyze what went wrong
 */
function analyzeError(userSeq, requiredSeq, challenge) {
  // Empty input
  if (userSeq.length === 0) {
    return {
      errorType: 'NO_INPUT',
      feedback: 'Start building your sentence! ' + (challenge.hints?.[0] || ''),
    };
  }

  // Check for order problems
  const userTypes = userSeq.map(id => getWordType(id));
  const requiredTypes = requiredSeq.map(id => getWordType(id));

  // Started with wrong type
  if (userTypes[0] !== requiredTypes[0]) {
    if (requiredTypes[0] === 'SUBJECT' && userTypes[0] === 'VERB') {
      return {
        errorType: 'ORDER_ERROR',
        feedback: 'Sentences must start with a Subject, not a Verb!',
      };
    }
  }

  // Wrong verb form (agreement error)
  if (userSeq.length >= 2) {
    const userVerb = userSeq.find(id => getWordType(id) === 'VERB');
    const reqVerb = requiredSeq.find(id => getWordType(id) === 'VERB');

    if (userVerb && reqVerb && userVerb !== reqVerb) {
      const userHasS = userVerb.endsWith('S') || userVerb.includes('_S');
      const reqHasS = reqVerb.endsWith('S') || reqVerb.includes('_S');

      if (userHasS !== reqHasS) {
        return {
          errorType: 'AGREEMENT_ERROR',
          feedback: userHasS
            ? "This subject doesn't need the '-s' form. Use the base verb."
            : "This subject needs the '-s' form. He/She/It requires verb+s.",
        };
      }
    }
  }

  // Incomplete sentence
  if (userSeq.length < requiredSeq.length) {
    const nextExpected = requiredTypes[userSeq.length];
    return {
      errorType: 'INCOMPLETE',
      feedback: `Keep going! Add a ${nextExpected} next.`,
    };
  }

  // Default
  return {
    errorType: 'UNKNOWN',
    feedback: 'Not quite right. Check your sequence and try again.',
  };
}

/**
 * Get word type from Grammar ID
 */
const OBJECT_IDS = new Set(['APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE']);

function getWordType(grammarId) {
  if (grammarId.startsWith('SUBJECT')) return 'SUBJECT';
  if (grammarId.startsWith('OBJECT') || OBJECT_IDS.has(grammarId)) return 'OBJECT';
  return 'VERB';
}

/**
 * Compare two arrays for equality
 */
function arraysEqual(a, b) {
  if (a.length !== b.length) return false;
  return a.every((val, idx) => val === b[idx]);
}

// =============================================================================
// PROGRESS TRACKER
// =============================================================================

export class ProgressTracker {
  constructor(storageKey = 'mlaf_progress') {
    this.storageKey = storageKey;
    this.progress = this.load();
  }

  load() {
    try {
      const saved = localStorage.getItem(this.storageKey);
      return saved ? JSON.parse(saved) : this.getDefaultProgress();
    } catch {
      return this.getDefaultProgress();
    }
  }

  save() {
    localStorage.setItem(this.storageKey, JSON.stringify(this.progress));
  }

  getDefaultProgress() {
    return {
      completedLessons: [],
      challengeAttempts: {},
      totalScore: 0,
      achievements: [],
      currentStreak: 0,
      lastPlayedDate: null,
    };
  }

  recordChallengeAttempt(lessonId, challengeId, result) {
    const key = `${lessonId}:${challengeId}`;

    if (!this.progress.challengeAttempts[key]) {
      this.progress.challengeAttempts[key] = {
        attempts: 0,
        bestScore: 0,
        completed: false,
        errors: [],
      };
    }

    const record = this.progress.challengeAttempts[key];
    record.attempts++;

    if (result.isCorrect) {
      record.completed = true;
      record.bestScore = Math.max(record.bestScore, result.score);
      this.progress.totalScore += result.score;
    } else {
      record.errors.push(result.errorType);
    }

    this.progress.lastPlayedDate = new Date().toISOString();
    this.save();

    return record;
  }

  markLessonComplete(lessonId) {
    if (!this.progress.completedLessons.includes(lessonId)) {
      this.progress.completedLessons.push(lessonId);
      this.save();
    }
  }

  isLessonComplete(lessonId) {
    return this.progress.completedLessons.includes(lessonId);
  }

  getLessonProgress(lessonId) {
    const lessonAttempts = Object.entries(this.progress.challengeAttempts)
      .filter(([key]) => key.startsWith(lessonId + ':'))
      .map(([key, data]) => ({ challengeId: key.split(':')[1], ...data }));

    return {
      isComplete: this.isLessonComplete(lessonId),
      challengeProgress: lessonAttempts,
      completedCount: lessonAttempts.filter(c => c.completed).length,
    };
  }

  awardAchievement(achievementId) {
    if (!this.progress.achievements.includes(achievementId)) {
      this.progress.achievements.push(achievementId);
      this.save();
      return true;
    }
    return false;
  }

  getStats() {
    return {
      totalScore: this.progress.totalScore,
      lessonsCompleted: this.progress.completedLessons.length,
      achievementsUnlocked: this.progress.achievements.length,
      currentStreak: this.progress.currentStreak,
    };
  }

  reset() {
    this.progress = this.getDefaultProgress();
    this.save();
  }
}

// =============================================================================
// HINT SYSTEM
// =============================================================================

export class HintSystem {
  constructor(challenge) {
    this.challenge = challenge;
    this.hintsUsed = 0;
    this.maxHints = challenge.hints?.length || 0;
  }

  getNextHint() {
    if (this.hintsUsed >= this.maxHints) {
      return {
        hint: 'No more hints available. Look at the required sequence structure: Subject → Verb → Object',
        hintsRemaining: 0,
        penalty: 0,
      };
    }

    const hint = this.challenge.hints[this.hintsUsed];
    this.hintsUsed++;

    return {
      hint,
      hintsRemaining: this.maxHints - this.hintsUsed,
      penalty: this.challenge.scoring?.hint_penalty || 10,
    };
  }

  getHintCount() {
    return {
      used: this.hintsUsed,
      remaining: this.maxHints - this.hintsUsed,
      total: this.maxHints,
    };
  }

  reset() {
    this.hintsUsed = 0;
  }
}

// =============================================================================
// LESSON MANAGER
// =============================================================================

export class LessonManager {
  constructor() {
    this.currentLesson = null;
    this.currentChallengeIndex = 0;
    this.hintSystem = null;
    this.progressTracker = new ProgressTracker();
    this.masteryGate = null;  // Injected by SandboxMode via setMasteryGate()
    this.sessionErrors = [];
    this.sessionScore = 0;
  }

  /**
   * Inject a GestureMasteryGate instance so that correct challenge answers
   * also record mastery productions (cumulative principle: Ch.2, Birsh & Carreker).
   * @param {import('../core/GestureMasteryGate').GestureMasteryGate} gate
   */
  setMasteryGate(gate) {
    this.masteryGate = gate;
  }

  loadLesson(lesson) {
    this.currentLesson = lesson;
    this.currentChallengeIndex = 0;
    this.sessionErrors = [];
    this.sessionScore = 0;

    if (lesson.challenges?.length > 0) {
      this.hintSystem = new HintSystem(lesson.challenges[0]);
    }

    return this.getCurrentChallenge();
  }

  getCurrentChallenge() {
    if (!this.currentLesson) return null;

    const challenge = this.currentLesson.challenges[this.currentChallengeIndex];
    return {
      challenge,
      index: this.currentChallengeIndex,
      total: this.currentLesson.challenges.length,
      isLast: this.currentChallengeIndex === this.currentLesson.challenges.length - 1,
    };
  }

  submitAnswer(userSequence) {
    if (!this.currentLesson) return { isCorrect: false, feedback: 'No lesson loaded.' };
    const challenge = this.currentLesson.challenges[this.currentChallengeIndex];
    const result = validateChallenge(challenge, userSequence);

    // Record attempt
    this.progressTracker.recordChallengeAttempt(
      this.currentLesson.level_id,
      challenge.challenge_id,
      result
    );

    if (result.isCorrect) {
      this.sessionScore += result.score;
      // Record mastery production for each gesture in the correct sequence
      if (this.masteryGate) {
        userSequence.forEach(grammarId => {
          this.masteryGate.recordProduction(grammarId);
        });
      }
    } else {
      this.sessionErrors.push(result.errorType);
    }

    return result;
  }

  nextChallenge() {
    if (!this.currentLesson) return null;
    if (this.currentChallengeIndex < this.currentLesson.challenges.length - 1) {
      this.currentChallengeIndex++;
      this.hintSystem = new HintSystem(this.currentLesson.challenges[this.currentChallengeIndex]);
      return this.getCurrentChallenge();
    }

    // Lesson complete
    this.progressTracker.markLessonComplete(this.currentLesson.level_id);
    return null;
  }

  getHint() {
    return this.hintSystem?.getNextHint() || null;
  }

  getLessonSummary() {
    return {
      lesson: this.currentLesson?.title,
      totalChallenges: this.currentLesson?.challenges.length || 0,
      completedChallenges: this.currentChallengeIndex + 1,
      sessionScore: this.sessionScore,
      errorCount: this.sessionErrors.length,
      commonErrors: this.getMostCommonErrors(),
    };
  }

  getMostCommonErrors() {
    const counts = {};
    this.sessionErrors.forEach(err => {
      counts[err] = (counts[err] || 0) + 1;
    });
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([error, count]) => ({ error, count }));
  }
}

// =============================================================================
// EXPORTS
// =============================================================================

export const defaultProgressTracker = new ProgressTracker();
export const defaultLessonManager = new LessonManager();
