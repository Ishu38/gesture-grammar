/**
 * GrammarEngine.js
 * Grammar validation via Earley parser (Nearley.js) with formal CFG.
 * LEXICON and helpers remain here; validateSentence is delegated to EarleyParser.
 */

// Re-export validateSentence from the Earley parser module
export { validateSentence } from '../grammar/EarleyParser.js';

// =============================================================================
// 1. DATA STRUCTURE: THE LEXICON (Strict Typing)
// =============================================================================

export const LEXICON = {
  // -------------------------------------------------------------------------
  // SUBJECTS (Pronouns)
  // -------------------------------------------------------------------------
  'SUBJECT_I': {
    type: 'SUBJECT',
    display: 'I',
    person: 1,
    number: 'singular',
  },
  'SUBJECT_YOU': {
    type: 'SUBJECT',
    display: 'You',
    person: 2,
    number: 'singular',
  },
  'SUBJECT_HE': {
    type: 'SUBJECT',
    display: 'He',
    person: 3,
    number: 'singular',
  },
  'SUBJECT_SHE': {
    type: 'SUBJECT',
    display: 'She',
    person: 3,
    number: 'singular',
  },
  'SUBJECT_WE': {
    type: 'SUBJECT',
    display: 'We',
    person: 1,
    number: 'plural',
  },
  'SUBJECT_THEY': {
    type: 'SUBJECT',
    display: 'They',
    person: 3,
    number: 'plural',
  },

  // -------------------------------------------------------------------------
  // VERBS (Base Form - used with I, You, We, They)
  // -------------------------------------------------------------------------
  'GRAB': {
    type: 'VERB',
    display: 'grab',
    transitive: true,
    requires_s_form: false,
    s_form_pair: 'GRABS',
  },
  'GO': {
    type: 'VERB',
    display: 'go',
    transitive: false,
    requires_s_form: false,
    s_form_pair: 'GOES',
  },
  'EAT': {
    type: 'VERB',
    display: 'eat',
    transitive: true,
    requires_s_form: false,
    s_form_pair: 'EATS',
  },
  'WANT': {
    type: 'VERB',
    display: 'want',
    transitive: true,
    requires_s_form: false,
    s_form_pair: 'WANTS',
  },
  'STOP': {
    type: 'VERB',
    display: 'stop',
    transitive: false,
    requires_s_form: false,
    s_form_pair: 'STOPS',
  },
  'DRINK': {
    type: 'VERB',
    display: 'drink',
    transitive: true,
    requires_s_form: false,
    s_form_pair: 'DRINKS',
  },

  // -------------------------------------------------------------------------
  // VERBS (S-Form - used with He, She, It)
  // -------------------------------------------------------------------------
  'GRABS': {
    type: 'VERB',
    display: 'grabs',
    transitive: true,
    requires_s_form: true,
    base_form_pair: 'GRAB',
  },
  'GOES': {
    type: 'VERB',
    display: 'goes',
    transitive: false,
    requires_s_form: true,
    base_form_pair: 'GO',
  },
  'EATS': {
    type: 'VERB',
    display: 'eats',
    transitive: true,
    requires_s_form: true,
    base_form_pair: 'EAT',
  },
  'WANTS': {
    type: 'VERB',
    display: 'wants',
    transitive: true,
    requires_s_form: true,
    base_form_pair: 'WANT',
  },
  'STOPS': {
    type: 'VERB',
    display: 'stops',
    transitive: false,
    requires_s_form: true,
    base_form_pair: 'STOP',
  },
  'DRINKS': {
    type: 'VERB',
    display: 'drinks',
    transitive: true,
    requires_s_form: true,
    base_form_pair: 'DRINK',
  },

  // -------------------------------------------------------------------------
  // OBJECTS (Nouns)
  // -------------------------------------------------------------------------
  'APPLE': {
    type: 'OBJECT',
    display: 'apple',
  },
  'BALL': {
    type: 'OBJECT',
    display: 'ball',
  },
  'WATER': {
    type: 'OBJECT',
    display: 'water',
  },
  'FOOD': {
    type: 'OBJECT',
    display: 'food',
  },
  'BOOK': {
    type: 'OBJECT',
    display: 'book',
  },
  'HOUSE': {
    type: 'OBJECT',
    display: 'house',
  },
};

// =============================================================================
// 2. HELPER FUNCTIONS
// =============================================================================

/**
 * Get words from lexicon by type
 */
export function getWordsByType(type) {
  return Object.entries(LEXICON)
    .filter(([_, entry]) => entry.type === type)
    .map(([key, entry]) => ({ key, ...entry }));
}

/**
 * Get the appropriate verb form based on subject
 */
export function getCorrectVerbForm(verbKey, subjectKey) {
  const verb = LEXICON[verbKey];
  const subject = LEXICON[subjectKey];

  if (!verb || !subject || verb.type !== 'VERB' || subject.type !== 'SUBJECT') {
    return verbKey;
  }

  const needsSForm = subject.person === 3 && subject.number === 'singular';

  if (needsSForm && !verb.requires_s_form && verb.s_form_pair) {
    return verb.s_form_pair;
  }

  if (!needsSForm && verb.requires_s_form && verb.base_form_pair) {
    return verb.base_form_pair;
  }

  return verbKey;
}
