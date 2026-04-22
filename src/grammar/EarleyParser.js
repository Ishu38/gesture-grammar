/**
 * EarleyParser.js — Parser wrapper using Nearley.js (Earley algorithm)
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Two-phase validation:
 *   Phase 1 (Syntactic): CFG enforces word order (S → NP VP) and transitivity
 *   Phase 2 (Semantic): Post-parse agreement check (3rd person singular → S-form)
 *
 * Exports validateSentence(sentenceArray) with the same return shape as the
 * old GrammarEngine.validateSentence.
 */

import nearley from 'nearley';
import grammarDef from './GestureGrammar.js';
import { LEXICON } from '../data/Lexicon.js';

// ── Human-readable category labels ─────────────────────────────────────────────

const CATEGORY_LABELS = {
  SUBJECT_I: 'Subject', SUBJECT_YOU: 'Subject', SUBJECT_HE: 'Subject',
  SUBJECT_SHE: 'Subject', SUBJECT_WE: 'Subject', SUBJECT_THEY: 'Subject',
  GRAB: 'Verb', EAT: 'Verb', WANT: 'Verb', DRINK: 'Verb', SEE: 'Verb',
  GRABS: 'Verb', EATS: 'Verb', WANTS: 'Verb', DRINKS: 'Verb', SEES: 'Verb',
  GO: 'Verb', STOP: 'Verb', GOES: 'Verb', STOPS: 'Verb',
  APPLE: 'Object', BALL: 'Object', WATER: 'Object',
  FOOD: 'Object', BOOK: 'Object', HOUSE: 'Object',
};

// Maps non-terminal names to human-readable suggestions
const NONTERMINAL_SUGGESTIONS = {
  NP: 'a Subject (I, You, He, She, We, They)',
  VP: 'a Verb',
  VT: 'a Verb',
  VI: 'a Verb',
  OBJ: 'an Object (Apple, Ball, Water, Food, Book, House)',
};

// ── Helper: build readable sentence from parse tree ────────────────────────────

function buildReadableSentence(parseTree) {
  if (!parseTree) return '';
  const words = [];
  function walk(node) {
    if (!node) return;
    if (node.value) {
      const entry = LEXICON[node.value];
      words.push(entry ? entry.display : node.value);
    }
    if (node.children) {
      node.children.forEach(walk);
    }
  }
  walk(parseTree);
  if (words.length > 0) {
    words[0] = words[0].charAt(0).toUpperCase() + words[0].slice(1);
  }
  return words.join(' ') + '.';
}

// ── Helper: extract NP and verb node from parse tree ───────────────────────────

function extractParts(parseTree) {
  const children = parseTree?.children || [];
  const np = children[0] || null;
  const vp = children[1] || null;
  const verbNode = vp?.children?.[0] || null;
  return { np, vp, verbNode };
}

// ── Helper: check subject-verb agreement (Phase 2) ─────────────────────────────

function checkAgreement(parseTree) {
  const { np, verbNode } = extractParts(parseTree);
  if (!np || !verbNode) return { valid: true };
  const needsSForm = np.person === 3 && np.number === 'singular';
  const verbEntry = LEXICON[verbNode.value];

  if (needsSForm && !verbNode.sForm) {
    const suggestedVerb = verbEntry?.s_form_pair
      ? LEXICON[verbEntry.s_form_pair]?.display
      : 'S-form';
    return {
      valid: false,
      error: "Subject-Verb Agreement Error: 'He/She' needs the S-form.",
      suggestion: `Use "${suggestedVerb}" instead of "${verbEntry?.display}" with "${LEXICON[np.value]?.display}".`,
      expectedNext: ['VERB_S_FORM'],
    };
  }

  if (!needsSForm && verbNode.sForm) {
    const suggestedVerb = verbEntry?.base_form_pair
      ? LEXICON[verbEntry.base_form_pair]?.display
      : 'base form';
    return {
      valid: false,
      error: "Subject-Verb Agreement Error: Only 'He/She' uses the S-form.",
      suggestion: `Use "${suggestedVerb}" instead of "${verbEntry?.display}" with "${LEXICON[np.value]?.display}".`,
      expectedNext: ['VERB_BASE_FORM'],
    };
  }

  return { valid: true };
}

// ── Helper: get expected tokens from Earley chart ──────────────────────────────

/**
 * Inspects the Earley chart's last column for states with the dot before
 * a terminal symbol, extracting what tokens can validly come next.
 *
 * This is a natural capability of the Earley chart — each state in the chart
 * tracks { rule, dot position, origin }. When the dot is before a terminal,
 * that terminal is a valid next token.
 */
function getExpectedTokens(parser) {
  const expected = new Set();

  // The parser's table is the Earley chart — an array of columns (one per input position + 1)
  // Each column is a Column object with a `scannable` array containing states
  // where the dot is before a terminal symbol.
  const lastColumn = parser.table[parser.table.length - 1];
  if (!lastColumn) return [];

  for (const state of lastColumn.scannable) {
    const nextSymbol = state.rule.symbols[state.dot];

    // Terminal symbols have a `type` property (from our tok() helper)
    if (nextSymbol && typeof nextSymbol === 'object' && nextSymbol.type) {
      expected.add(nextSymbol.type);
    }
  }

  return [...expected];
}

/**
 * Convert raw token type names to human-readable category
 * e.g., ['GRAB', 'EAT', 'GO'] → 'Verb'
 * e.g., ['APPLE', 'BALL'] → 'Object'
 */
function summarizeExpected(tokenTypes) {
  const categories = new Set();
  for (const t of tokenTypes) {
    categories.add(CATEGORY_LABELS[t] || t);
  }
  return [...categories];
}

/**
 * Build a suggestion string from expected categories.
 */
function buildSuggestion(expectedCategories, subjectValue) {
  if (expectedCategories.length === 0) return null;

  if (expectedCategories.includes('Subject')) {
    return 'Start with a Subject (e.g., I, He, She)';
  }
  if (expectedCategories.includes('Verb')) {
    if (subjectValue) {
      const subjectEntry = LEXICON[subjectValue];
      const needsSForm = subjectEntry?.person === 3 && subjectEntry?.number === 'singular';
      return needsSForm
        ? 'Add a Verb (S-form needed: Grabs, Goes, Eats...)'
        : 'Add a Verb (e.g., Grab, Go, Eat...)';
    }
    return 'Add a Verb';
  }
  if (expectedCategories.includes('Object')) {
    return 'Add an Object (e.g., Apple, Ball, Food)';
  }

  return `Expected: ${expectedCategories.join(' or ')}`;
}

// ── Helper: find the failure point by incremental parsing ──────────────────────

function findFailurePoint(sentenceArray) {
  for (let i = 0; i < sentenceArray.length; i++) {
    const parser = new nearley.Parser(nearley.Grammar.fromCompiled(grammarDef));
    try {
      parser.feed(sentenceArray.slice(0, i + 1));
    } catch {
      // Failed at position i — get what was expected at position i
      const prevParser = new nearley.Parser(nearley.Grammar.fromCompiled(grammarDef));
      if (i > 0) {
        prevParser.feed(sentenceArray.slice(0, i));
      }
      const expected = getExpectedTokens(prevParser);
      return { failIndex: i, expected };
    }
  }
  return null;
}

// ── Main export ────────────────────────────────────────────────────────────────

/**
 * Validates a sentence array against the MLAF grammar CFG.
 *
 * @param {string[]} sentenceArray - Array of grammar ID strings (e.g., ['SUBJECT_I', 'GRAB', 'APPLE'])
 * @returns {object} Validation result with shape:
 *   { isValid, isComplete, error, suggestion, expectedNext, parseTree, readableSentence }
 */
export function validateSentence(sentenceArray) {
  // ── Empty input ──────────────────────────────────────────────────────────────
  if (!Array.isArray(sentenceArray) || sentenceArray.length === 0) {
    return {
      isValid: false,
      isComplete: false,
      error: null,
      suggestion: 'Start with a Subject (e.g., I, He, She)',
      expectedNext: ['SUBJECT'],
    };
  }

  // ── Check for unknown words ──────────────────────────────────────────────────
  for (const word of sentenceArray) {
    if (!LEXICON[word]) {
      return {
        isValid: false,
        isComplete: false,
        error: `Unknown word: "${word}" is not in the lexicon.`,
        suggestion: 'Use a recognized word from the gesture vocabulary.',
        expectedNext: [],
      };
    }
  }

  // ── Phase 1: Syntactic parse via Earley algorithm ────────────────────────────
  const parser = new nearley.Parser(nearley.Grammar.fromCompiled(grammarDef));

  try {
    parser.feed(sentenceArray);
  } catch {
    // Parse error — find the failure point
    const failure = findFailurePoint(sentenceArray);

    if (failure) {
      const failedToken = sentenceArray[failure.failIndex];
      const failedEntry = LEXICON[failedToken];
      const expectedCategories = summarizeExpected(failure.expected);

      // Special case: first token isn't a subject
      if (failure.failIndex === 0) {
        return {
          isValid: false,
          isComplete: false,
          error: 'Sentence must start with a Subject.',
          suggestion: `"${failedEntry?.display || failedToken}" is a ${failedEntry?.type || 'unknown'}. Start with: I, You, He, She, We, They.`,
          expectedNext: ['SUBJECT'],
        };
      }

      // Check if the sentence was already complete before this token
      const prevParser = new nearley.Parser(nearley.Grammar.fromCompiled(grammarDef));
      prevParser.feed(sentenceArray.slice(0, failure.failIndex));
      if (prevParser.results.length > 0) {
        // Previous tokens formed a complete sentence — extra tokens
        const prevVerb = extractParts(prevParser.results[0]).verbNode;
        if (prevVerb?.value) {
          const verbEntry = LEXICON[prevVerb.value];
          if (prevVerb.type === 'VI') {
            return {
              isValid: false,
              isComplete: false,
              error: 'Intransitive verb cannot take an Object.',
              suggestion: `"${verbEntry?.display}" doesn't need an Object. Remove "${failedEntry?.display}" or use a transitive verb.`,
              expectedNext: [],
            };
          }
        }
        return {
          isValid: false,
          isComplete: false,
          error: 'Sentence is already complete.',
          suggestion: `Remove "${failedEntry?.display}" — the sentence was already complete.`,
          expectedNext: [],
        };
      }

      // Generic error at this position
      return {
        isValid: false,
        isComplete: false,
        error: `Unexpected "${failedEntry?.display || failedToken}" at position ${failure.failIndex + 1}.`,
        suggestion: `Expected ${expectedCategories.join(' or ')} but got "${failedEntry?.display || failedToken}".`,
        expectedNext: expectedCategories,
      };
    }

    // Shouldn't reach here, but fallback
    return {
      isValid: false,
      isComplete: false,
      error: 'Invalid sentence structure.',
      suggestion: 'Follow Subject-Verb-Object order.',
      expectedNext: [],
    };
  }

  // ── Complete parse? ──────────────────────────────────────────────────────────
  if (parser.results.length > 0) {
    const parseTree = parser.results[0];

    // Phase 2: Semantic agreement check
    const agreement = checkAgreement(parseTree);
    if (!agreement.valid) {
      return {
        isValid: false,
        isComplete: false,
        error: agreement.error,
        suggestion: agreement.suggestion,
        expectedNext: agreement.expectedNext,
        parseTree,
      };
    }

    // Valid complete sentence
    return {
      isValid: true,
      isComplete: true,
      error: null,
      suggestion: 'Sentence complete! You can clear and start a new one.',
      expectedNext: [],
      parseTree,
      readableSentence: buildReadableSentence(parseTree),
    };
  }

  // ── Incomplete but valid prefix ──────────────────────────────────────────────
  const expectedTokens = getExpectedTokens(parser);
  const expectedCategories = summarizeExpected(expectedTokens);

  // Extract the subject value for context-aware suggestions
  let subjectValue = null;
  if (sentenceArray.length >= 1 && LEXICON[sentenceArray[0]]?.type === 'SUBJECT') {
    subjectValue = sentenceArray[0];
  }

  const suggestion = buildSuggestion(expectedCategories, subjectValue);

  return {
    isValid: true,
    isComplete: false,
    error: null,
    suggestion: suggestion || 'Continue building the sentence.',
    expectedNext: expectedCategories,
  };
}
