/**
 * GrammarRules.js
 * Implements the Grammar Logic Matrix for Subject-Verb Agreement
 * Pure deterministic rule-based system - no AI inference
 */

// =============================================================================
// DECISION MATRIX: The Truth Table
// =============================================================================

/**
 * Condensed lookup table: MATRIX[person][number][tense] -> agreement rule
 *
 * Visual Truth Table for Present Tense:
 * ┌─────────┬──────────┬──────────┬─────────────┬──────────────────────────┐
 * │ Person  │ Number   │ Tense    │ Verb Form   │ Example                  │
 * ├─────────┼──────────┼──────────┼─────────────┼──────────────────────────┤
 * │ 1       │ singular │ present  │ BASE        │ I walk / I eat           │
 * │ 2       │ singular │ present  │ BASE        │ You walk / You eat       │
 * │ 3       │ singular │ present  │ S-FORM (+s) │ He walks / She eats      │
 * │ 1       │ plural   │ present  │ BASE        │ We walk / We eat         │
 * │ 2       │ plural   │ present  │ BASE        │ You walk / You eat       │
 * │ 3       │ plural   │ present  │ BASE        │ They walk / They eat     │
 * └─────────┴──────────┴──────────┴─────────────┴──────────────────────────┘
 *
 * Key Insight: ONLY 3rd person singular present requires S-form
 */

export const AGREEMENT_MATRIX = {
  1: {
    singular: {
      present:             { form: 'base',    aux: null,   requires_s: false },
      past:                { form: 'past',    aux: null,   requires_s: false },
      future:              { form: 'base',    aux: 'will', requires_s: false },
      present_progressive: { form: 'ing',     aux: 'am',   requires_s: false },
      past_progressive:    { form: 'ing',     aux: 'was',  requires_s: false },
    },
    plural: {
      present:             { form: 'base',    aux: null,   requires_s: false },
      past:                { form: 'past',    aux: null,   requires_s: false },
      future:              { form: 'base',    aux: 'will', requires_s: false },
      present_progressive: { form: 'ing',     aux: 'are',  requires_s: false },
      past_progressive:    { form: 'ing',     aux: 'were', requires_s: false },
    },
  },
  2: {
    singular: {
      present:             { form: 'base',    aux: null,   requires_s: false },
      past:                { form: 'past',    aux: null,   requires_s: false },
      future:              { form: 'base',    aux: 'will', requires_s: false },
      present_progressive: { form: 'ing',     aux: 'are',  requires_s: false },
      past_progressive:    { form: 'ing',     aux: 'were', requires_s: false },
    },
    plural: {
      present:             { form: 'base',    aux: null,   requires_s: false },
      past:                { form: 'past',    aux: null,   requires_s: false },
      future:              { form: 'base',    aux: 'will', requires_s: false },
      present_progressive: { form: 'ing',     aux: 'are',  requires_s: false },
      past_progressive:    { form: 'ing',     aux: 'were', requires_s: false },
    },
  },
  3: {
    singular: {
      present:             { form: 's_form',  aux: null,   requires_s: true  }, // THE KEY RULE
      past:                { form: 'past',    aux: null,   requires_s: false },
      future:              { form: 'base',    aux: 'will', requires_s: false },
      present_progressive: { form: 'ing',     aux: 'is',   requires_s: false },
      past_progressive:    { form: 'ing',     aux: 'was',  requires_s: false },
    },
    plural: {
      present:             { form: 'base',    aux: null,   requires_s: false },
      past:                { form: 'past',    aux: null,   requires_s: false },
      future:              { form: 'base',    aux: 'will', requires_s: false },
      present_progressive: { form: 'ing',     aux: 'are',  requires_s: false },
      past_progressive:    { form: 'ing',     aux: 'were', requires_s: false },
    },
  },
};

// =============================================================================
// SUFFIX RULES
// =============================================================================

export const SUFFIX_RULES = {
  s_form: [
    { pattern: /[sxz]$|[cs]h$/, suffix: 'es', example: 'watch → watches' },
    { pattern: /[^aeiou]y$/, replace: 'y', with: 'ies', example: 'fly → flies' },
    { pattern: /.*/, suffix: 's', example: 'walk → walks' }, // default
  ],
  past: [
    { pattern: /e$/, suffix: 'd', example: 'love → loved' },
    { pattern: /[^aeiou]y$/, replace: 'y', with: 'ied', example: 'try → tried' },
    { pattern: /[^aeiou][aeiou][^aeiouwxy]$/, double: true, suffix: 'ed', example: 'stop → stopped' },
    { pattern: /.*/, suffix: 'ed', example: 'walk → walked' }, // default
  ],
  ing: [
    { pattern: /ie$/, replace: 'ie', with: 'ying', example: 'die → dying' },
    { pattern: /e$/, replace: 'e', with: 'ing', example: 'make → making' },
    { pattern: /[^aeiou][aeiou][^aeiouwxy]$/, double: true, suffix: 'ing', example: 'run → running' },
    { pattern: /.*/, suffix: 'ing', example: 'walk → walking' }, // default
  ],
};

// =============================================================================
// IRREGULAR VERBS DATABASE
// =============================================================================

export const IRREGULAR_VERBS = {
  be:    { base: 'be',    s_form: 'is',    past: 'was/were', ing: 'being'   },
  have:  { base: 'have',  s_form: 'has',   past: 'had',      ing: 'having'  },
  do:    { base: 'do',    s_form: 'does',  past: 'did',      ing: 'doing'   },
  go:    { base: 'go',    s_form: 'goes',  past: 'went',     ing: 'going'   },
  eat:   { base: 'eat',   s_form: 'eats',  past: 'ate',      ing: 'eating'  },
  get:   { base: 'get',   s_form: 'gets',  past: 'got',      ing: 'getting' },
  give:  { base: 'give',  s_form: 'gives', past: 'gave',     ing: 'giving'  },
  take:  { base: 'take',  s_form: 'takes', past: 'took',     ing: 'taking'  },
  make:  { base: 'make',  s_form: 'makes', past: 'made',     ing: 'making'  },
  see:   { base: 'see',   s_form: 'sees',  past: 'saw',      ing: 'seeing'  },
  come:  { base: 'come',  s_form: 'comes', past: 'came',     ing: 'coming'  },
  know:  { base: 'know',  s_form: 'knows', past: 'knew',     ing: 'knowing' },
  think: { base: 'think', s_form: 'thinks',past: 'thought',  ing: 'thinking'},
  want:  { base: 'want',  s_form: 'wants', past: 'wanted',   ing: 'wanting' },
  grab:  { base: 'grab',  s_form: 'grabs', past: 'grabbed',  ing: 'grabbing'},
  stop:  { base: 'stop',  s_form: 'stops', past: 'stopped',  ing: 'stopping'},
};

// =============================================================================
// LOOKUP FUNCTIONS
// =============================================================================

/**
 * Get the agreement rule for a given subject and tense
 * @param {number} person - 1, 2, or 3
 * @param {string} number - 'singular' or 'plural'
 * @param {string} tense - 'present', 'past', 'future', etc.
 * @returns {Object} The agreement rule
 */
export function getAgreementRule(person, number, tense = 'present') {
  const rule = AGREEMENT_MATRIX[person]?.[number]?.[tense];

  if (!rule) {
    return {
      form: 'base',
      aux: null,
      requires_s: false,
      error: `No rule found for person=${person}, number=${number}, tense=${tense}`,
    };
  }

  return rule;
}

/**
 * Check if a subject requires S-form verbs
 * @param {number} person - 1, 2, or 3
 * @param {string} number - 'singular' or 'plural'
 * @param {string} tense - defaults to 'present'
 * @returns {boolean}
 */
export function requiresSForm(person, number, tense = 'present') {
  return person === 3 && number === 'singular' && tense === 'present';
}

/**
 * Apply suffix to a verb base form
 * @param {string} verb - Base form of the verb
 * @param {string} formType - 's_form', 'past', or 'ing'
 * @returns {string} The conjugated verb
 */
export function applyVerbSuffix(verb, formType) {
  // Check for irregular verb first
  const irregular = IRREGULAR_VERBS[verb.toLowerCase()];
  if (irregular && irregular[formType]) {
    return irregular[formType];
  }

  // Apply regular suffix rules
  const rules = SUFFIX_RULES[formType];
  if (!rules) return verb;

  for (const rule of rules) {
    if (rule.pattern.test(verb)) {
      if (rule.replace) {
        return verb.replace(new RegExp(rule.replace + '$'), rule.with);
      }
      if (rule.double) {
        return verb + verb.slice(-1) + rule.suffix;
      }
      return verb + rule.suffix;
    }
  }

  return verb;
}

/**
 * Get the correct verb form based on subject properties
 * @param {string} verbBase - Base form of verb (e.g., 'walk')
 * @param {Object} subject - Subject with person and number
 * @param {string} tense - Tense to use
 * @returns {Object} { auxiliary, verb, full }
 */
export function conjugateVerb(verbBase, subject, tense = 'present') {
  const rule = getAgreementRule(subject.person, subject.number, tense);

  let conjugatedVerb = verbBase;

  switch (rule.form) {
    case 's_form':
      conjugatedVerb = applyVerbSuffix(verbBase, 's_form');
      break;
    case 'past':
      conjugatedVerb = applyVerbSuffix(verbBase, 'past');
      break;
    case 'ing':
      conjugatedVerb = applyVerbSuffix(verbBase, 'ing');
      break;
    default:
      // base form - no change
      break;
  }

  return {
    auxiliary: rule.aux,
    verb: conjugatedVerb,
    full: rule.aux ? `${rule.aux} ${conjugatedVerb}` : conjugatedVerb,
  };
}

/**
 * Validate subject-verb agreement
 * @param {Object} subject - { person, number }
 * @param {Object} verb - { requires_s_form }
 * @param {string} tense - defaults to 'present'
 * @returns {Object} { isValid, error }
 */
export function validateAgreement(subject, verb, tense = 'present') {
  const needsSForm = requiresSForm(subject.person, subject.number, tense);

  if (needsSForm && !verb.requires_s_form) {
    return {
      isValid: false,
      error: `Agreement Error: 3rd person singular requires S-form verb.`,
      suggestion: `Use "${verb.s_form_pair || 'S-form'}" instead of "${verb.display || 'base form'}"`,
    };
  }

  if (!needsSForm && verb.requires_s_form) {
    return {
      isValid: false,
      error: `Agreement Error: Only 3rd person singular uses S-form verb.`,
      suggestion: `Use "${verb.base_form_pair || 'base form'}" instead of "${verb.display || 'S-form'}"`,
    };
  }

  return { isValid: true, error: null };
}

// =============================================================================
// VISUAL DECISION TABLE (for debugging/logging)
// =============================================================================

export function printDecisionTable() {
  console.log(`
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    SUBJECT-VERB AGREEMENT DECISION TABLE                      ║
╠═════════╦══════════╦══════════╦═════════════╦════════════════════════════════╣
║ Person  ║ Number   ║ Tense    ║ Verb Form   ║ Example                        ║
╠═════════╬══════════╬══════════╬═════════════╬════════════════════════════════╣
║ 1st     ║ Singular ║ Present  ║ BASE        ║ I walk, I eat                  ║
║ 2nd     ║ Singular ║ Present  ║ BASE        ║ You walk, You eat              ║
║ 3rd     ║ Singular ║ Present  ║ S-FORM (+s) ║ He walks, She eats ⚠️ KEY RULE ║
║ 1st     ║ Plural   ║ Present  ║ BASE        ║ We walk, We eat                ║
║ 2nd     ║ Plural   ║ Present  ║ BASE        ║ You walk, You eat              ║
║ 3rd     ║ Plural   ║ Present  ║ BASE        ║ They walk, They eat            ║
╠═════════╬══════════╬══════════╬═════════════╬════════════════════════════════╣
║ ALL     ║ ALL      ║ Past     ║ PAST (-ed)  ║ I/He/They walked               ║
║ ALL     ║ ALL      ║ Future   ║ will + BASE ║ I/He/They will walk            ║
╚═════════╩══════════╩══════════╩═════════════╩════════════════════════════════╝

Key Rule: ONLY [3rd Person] + [Singular] + [Present Tense] = S-FORM required
  `);
}
