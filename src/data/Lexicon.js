/**
 * Lexicon.js — Canonical LEXICON data definition
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Separated from GrammarEngine.js to break the circular dependency:
 *   GrammarEngine.js re-exports validateSentence from EarleyParser.js
 *   EarleyParser.js needs LEXICON
 *   → Both now import LEXICON from this file instead of from each other.
 */

// =============================================================================
// THE LEXICON (Strict Typing)
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
    // "You" is grammatically 2nd-person but takes plural verb forms (no -s suffix).
    // Agreement check keys on person !== 3, so 'singular' vs 'plural' doesn't affect
    // conjugation. We use 'singular' to reflect the gesture's intended referent (one person).
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
  'SEE': {
    type: 'VERB',
    display: 'see',
    transitive: true,
    requires_s_form: false,
    s_form_pair: 'SEES',
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
  'SEES': {
    type: 'VERB',
    display: 'sees',
    transitive: true,
    requires_s_form: true,
    base_form_pair: 'SEE',
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
