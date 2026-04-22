/**
 * lessonIndex.js — Lesson Registry for Guided Practice Mode
 * Maps lesson IDs to their JSON files, enabling dynamic lesson loading.
 */

// NOTE: Current lessons cover present tense agreement only.
// Past/future tense is handled by spatial tense zones (wrist Y position)
// and applied as a display transformation in useSentenceBuilder.js.
// Tense-specific challenges require extending the challenge engine to
// validate spatial zone state alongside required_sequence.
export const LESSON_INDEX = [
  {
    id: 'LVL_01_SV_AGREEMENT',
    title: 'Subject-Verb Agreement',
    stage: 4,
    difficulty: 1,
    file: () => import('./lessons/Level_01_Subject_Verb_Agreement.json'),
  },
  {
    id: 'LVL_02_SVO_ORDER',
    title: 'SVO Word Order',
    stage: 1,
    difficulty: 1,
    file: () => import('./lessons/Level_02_SVO_Order.json'),
  },
  {
    id: 'LVL_03_PRONOUN_VERBS',
    title: 'Pronouns & Verbs',
    stage: 2,
    difficulty: 2,
    file: () => import('./lessons/Level_03_Pronoun_Verbs.json'),
  },
  {
    id: 'LVL_04_COMPLETE',
    title: 'Complete Sentences',
    stage: 3,
    difficulty: 2,
    file: () => import('./lessons/Level_04_Complete_Sentences.json'),
  },
  {
    id: 'LVL_05_VERB_AGREE',
    title: 'Advanced Agreement',
    stage: 4,
    difficulty: 3,
    file: () => import('./lessons/Level_05_Verb_Agreement.json'),
  },
];

/**
 * Get lessons available for a given mastery stage.
 * @param {number} stage — mastery stage (1–5)
 * @returns {Array}
 */
export function getLessonsForStage(stage) {
  return LESSON_INDEX.filter(l => l.stage <= stage);
}

/**
 * Get the next recommended lesson based on completed lesson IDs.
 * @param {string[]} completedIds — IDs of completed lessons
 * @returns {object|null}
 */
export function getNextLesson(completedIds = []) {
  return LESSON_INDEX.find(l => !completedIds.includes(l.id)) || null;
}
