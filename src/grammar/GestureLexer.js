/**
 * GestureLexer.js — Custom Nearley-compatible lexer
 *
 * Implements Nearley's lexer interface to iterate over an array of
 * Grammar ID strings. Each string becomes a token with
 * { type: grammarId, value: grammarId }.
 *
 * Required interface methods: reset, next, save, formatError, has
 */

// All terminal symbols the grammar recognizes
const KNOWN_TOKENS = new Set([
  // Subjects
  'SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_HE', 'SUBJECT_SHE', 'SUBJECT_WE', 'SUBJECT_THEY',
  // Base verbs (transitive)
  'GRAB', 'EAT', 'WANT', 'DRINK',
  // S-form verbs (transitive)
  'GRABS', 'EATS', 'WANTS', 'DRINKS',
  // Base verbs (intransitive)
  'GO', 'STOP',
  // S-form verbs (intransitive)
  'GOES', 'STOPS',
  // Objects
  'APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE',
]);

export class GestureLexer {
  constructor() {
    this.buffer = [];
    this.index = 0;
  }

  /**
   * Reset the lexer with a new input.
   * Nearley calls this before parsing with the input data.
   * @param {string[]} data - Array of grammar ID strings
   * @param {object} [state] - Optional saved state to restore
   */
  reset(data, state) {
    this.buffer = data;
    this.index = state ? state.index : 0;
  }

  /**
   * Return the next token or undefined if done.
   * Each token: { type, value, offset, lineBreaks, line, col }
   * @returns {object|undefined}
   */
  next() {
    if (this.index >= this.buffer.length) {
      return undefined;
    }
    const value = this.buffer[this.index];
    const token = {
      type: value,
      value: value,
      offset: this.index,
      lineBreaks: 0,
      line: 1,
      col: this.index + 1,
    };
    this.index++;
    return token;
  }

  /**
   * Save current lexer state (for backtracking).
   * @returns {object}
   */
  save() {
    return { index: this.index };
  }

  /**
   * Format an error message for a token.
   * @param {object} token
   * @returns {string}
   */
  formatError(token) {
    return `Unexpected token "${token.value}" at position ${token.offset}`;
  }

  /**
   * Check whether this lexer can emit a token of the given type.
   * Nearley calls this to determine if a terminal symbol is valid.
   * @param {string} tokenType
   * @returns {boolean}
   */
  has(tokenType) {
    return KNOWN_TOKENS.has(tokenType);
  }
}
