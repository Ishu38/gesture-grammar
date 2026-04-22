/**
 * GrammarEngine.js
 * Grammar validation via Earley parser (Nearley.js) with formal CFG.
 * LEXICON is defined in src/data/Lexicon.js (canonical source) and re-exported here.
 * validateSentence is delegated to EarleyParser.
 */

// Re-export validateSentence from the Earley parser module
export { validateSentence } from '../grammar/EarleyParser.js';

// Re-export LEXICON from canonical source (breaks circular dep with EarleyParser)
export { LEXICON } from '../data/Lexicon.js';

// =============================================================================
// HELPER FUNCTIONS
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
