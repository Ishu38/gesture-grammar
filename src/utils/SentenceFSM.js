/**
 * SentenceFSM.js
 * Finite State Machine for SVO Sentence Construction
 * Enforces grammatical word order through state transitions
 */

// =============================================================================
// STATE DEFINITIONS
// =============================================================================

export const STATES = {
  EMPTY: 'STATE_EMPTY',
  SUBJECT_LOCKED: 'STATE_SUBJECT_LOCKED',
  VERB_LOCKED_TRANS: 'STATE_VERB_LOCKED_TRANS',
  VERB_LOCKED_INTRANS: 'STATE_VERB_LOCKED_INTRANS',
  OBJECT_LOCKED: 'STATE_OBJECT_LOCKED',
  ERROR: 'STATE_ERROR',
};

export const STATE_INFO = {
  [STATES.EMPTY]: {
    description: 'Empty sentence - awaiting subject',
    isAccepting: false,
    hint: 'Start with a Subject (I, You, He, She, We, They)',
  },
  [STATES.SUBJECT_LOCKED]: {
    description: 'Subject added - awaiting verb',
    isAccepting: false,
    hint: 'Add a Verb (grab, go, eat, stop...)',
  },
  [STATES.VERB_LOCKED_TRANS]: {
    description: 'Transitive verb added - awaiting object',
    isAccepting: false,
    hint: 'Add an Object (apple, ball, food...)',
  },
  [STATES.VERB_LOCKED_INTRANS]: {
    description: 'Intransitive verb added - sentence complete',
    isAccepting: true,
    hint: 'Sentence complete! Clear to start new.',
  },
  [STATES.OBJECT_LOCKED]: {
    description: 'Object added - sentence complete',
    isAccepting: true,
    hint: 'Sentence complete! Clear to start new.',
  },
};

// =============================================================================
// INPUT TYPES (Word Categories)
// =============================================================================

export const INPUT_TYPES = {
  SUBJECT: 'SUBJECT',
  VERB_TRANSITIVE: 'VERB_TRANSITIVE',
  VERB_INTRANSITIVE: 'VERB_INTRANSITIVE',
  VERB: 'VERB', // Generic - will be resolved to TRANS or INTRANS
  OBJECT: 'OBJECT',
  MODIFIER: 'MODIFIER',
  CLEAR: 'CLEAR',
};

// =============================================================================
// ERROR CODES
// =============================================================================

export const ERRORS = {
  ERR_VERB_BEFORE_SUBJECT: {
    code: 'E001',
    message: 'Sentence must start with a Subject',
    suggestion: 'Add a Subject first (I, You, He, She, We, They)',
  },
  ERR_OBJECT_BEFORE_SUBJECT: {
    code: 'E002',
    message: 'Cannot start sentence with an Object',
    suggestion: 'Add a Subject first, then Verb, then Object',
  },
  ERR_OBJECT_BEFORE_VERB: {
    code: 'E003',
    message: 'Object must come after a Verb',
    suggestion: 'Add a Verb first (grab, eat, want...)',
  },
  ERR_DUPLICATE_SUBJECT: {
    code: 'E004',
    message: 'Sentence already has a Subject',
    suggestion: 'Add a Verb next, not another Subject',
  },
  ERR_DUPLICATE_VERB: {
    code: 'E005',
    message: 'Sentence already has a Verb',
    suggestion: 'Add an Object (if needed) or clear to start over',
  },
  ERR_SUBJECT_AFTER_VERB: {
    code: 'E006',
    message: 'Cannot add Subject after Verb',
    suggestion: 'Add an Object instead, or clear to start over',
  },
  ERR_OBJECT_AFTER_INTRANSITIVE: {
    code: 'E007',
    message: "This verb doesn't take an Object",
    suggestion: 'Sentence is complete. Clear to start a new one.',
  },
  ERR_MODIFIER_NO_TARGET: {
    code: 'E008',
    message: 'Modifier needs something to modify',
    suggestion: 'Add a Subject first, then you can modify it',
  },
  ERR_SENTENCE_COMPLETE: {
    code: 'E009',
    message: 'Sentence is already complete',
    suggestion: 'Clear to start a new sentence',
  },
  ERR_AGREEMENT_VIOLATION: {
    code: 'E010',
    message: 'Subject-Verb agreement error',
    suggestion: 'Use the correct verb form for this subject',
  },
};

// =============================================================================
// TRANSITION TABLE
// =============================================================================

/**
 * Transition matrix: TRANSITIONS[currentState][inputType] → result
 * Result: { nextState, error?, validate? }
 */
const TRANSITIONS = {
  [STATES.EMPTY]: {
    [INPUT_TYPES.SUBJECT]:          { nextState: STATES.SUBJECT_LOCKED },
    [INPUT_TYPES.VERB]:             { nextState: STATES.ERROR, error: 'ERR_VERB_BEFORE_SUBJECT' },
    [INPUT_TYPES.VERB_TRANSITIVE]:  { nextState: STATES.ERROR, error: 'ERR_VERB_BEFORE_SUBJECT' },
    [INPUT_TYPES.VERB_INTRANSITIVE]:{ nextState: STATES.ERROR, error: 'ERR_VERB_BEFORE_SUBJECT' },
    [INPUT_TYPES.OBJECT]:           { nextState: STATES.ERROR, error: 'ERR_OBJECT_BEFORE_SUBJECT' },
    [INPUT_TYPES.MODIFIER]:         { nextState: STATES.ERROR, error: 'ERR_MODIFIER_NO_TARGET' },
  },

  [STATES.SUBJECT_LOCKED]: {
    [INPUT_TYPES.SUBJECT]:          { nextState: STATES.ERROR, error: 'ERR_DUPLICATE_SUBJECT' },
    [INPUT_TYPES.VERB_TRANSITIVE]:  { nextState: STATES.VERB_LOCKED_TRANS, validate: 'AGREEMENT' },
    [INPUT_TYPES.VERB_INTRANSITIVE]:{ nextState: STATES.VERB_LOCKED_INTRANS, validate: 'AGREEMENT' },
    [INPUT_TYPES.OBJECT]:           { nextState: STATES.ERROR, error: 'ERR_OBJECT_BEFORE_VERB' },
    [INPUT_TYPES.MODIFIER]:         { nextState: STATES.SUBJECT_LOCKED },
  },

  [STATES.VERB_LOCKED_TRANS]: {
    [INPUT_TYPES.SUBJECT]:          { nextState: STATES.ERROR, error: 'ERR_SUBJECT_AFTER_VERB' },
    [INPUT_TYPES.VERB]:             { nextState: STATES.ERROR, error: 'ERR_DUPLICATE_VERB' },
    [INPUT_TYPES.VERB_TRANSITIVE]:  { nextState: STATES.ERROR, error: 'ERR_DUPLICATE_VERB' },
    [INPUT_TYPES.VERB_INTRANSITIVE]:{ nextState: STATES.ERROR, error: 'ERR_DUPLICATE_VERB' },
    [INPUT_TYPES.OBJECT]:           { nextState: STATES.OBJECT_LOCKED },
    [INPUT_TYPES.MODIFIER]:         { nextState: STATES.VERB_LOCKED_TRANS },
  },

  [STATES.VERB_LOCKED_INTRANS]: {
    [INPUT_TYPES.SUBJECT]:          { nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.VERB]:             { nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.VERB_TRANSITIVE]:  { nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.VERB_INTRANSITIVE]:{ nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.OBJECT]:           { nextState: STATES.ERROR, error: 'ERR_OBJECT_AFTER_INTRANSITIVE' },
    [INPUT_TYPES.MODIFIER]:         { nextState: STATES.VERB_LOCKED_INTRANS },
  },

  [STATES.OBJECT_LOCKED]: {
    [INPUT_TYPES.SUBJECT]:          { nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.VERB]:             { nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.VERB_TRANSITIVE]:  { nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.VERB_INTRANSITIVE]:{ nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.OBJECT]:           { nextState: STATES.ERROR, error: 'ERR_SENTENCE_COMPLETE' },
    [INPUT_TYPES.MODIFIER]:         { nextState: STATES.OBJECT_LOCKED },
  },
};

// =============================================================================
// FSM CLASS
// =============================================================================

export class SentenceFSM {
  constructor() {
    this.currentState = STATES.EMPTY;
    this.sentence = [];
    this.history = [];
  }

  /**
   * Get current state information
   */
  getState() {
    return {
      state: this.currentState,
      info: STATE_INFO[this.currentState],
      sentence: [...this.sentence],
      isComplete: STATE_INFO[this.currentState]?.isAccepting || false,
    };
  }

  /**
   * Check if a transition is valid without executing it
   * @param {string} inputType - The type of word being added
   * @param {Object} wordData - Additional word data for validation
   * @returns {Object} { isValid, error, nextState }
   */
  canTransition(inputType, wordData = {}) {
    // Handle CLEAR specially
    if (inputType === INPUT_TYPES.CLEAR) {
      return { isValid: true, error: null, nextState: STATES.EMPTY };
    }

    // Resolve generic VERB to specific type
    let resolvedInput = inputType;
    if (inputType === INPUT_TYPES.VERB && wordData.transitive !== undefined) {
      resolvedInput = wordData.transitive
        ? INPUT_TYPES.VERB_TRANSITIVE
        : INPUT_TYPES.VERB_INTRANSITIVE;
    }

    // Look up transition
    const stateTransitions = TRANSITIONS[this.currentState];
    if (!stateTransitions) {
      return {
        isValid: false,
        error: ERRORS.ERR_SENTENCE_COMPLETE,
        nextState: STATES.ERROR,
      };
    }

    const transition = stateTransitions[resolvedInput];
    if (!transition) {
      return {
        isValid: false,
        error: { code: 'E999', message: 'Unknown input type', suggestion: 'Try a different gesture' },
        nextState: STATES.ERROR,
      };
    }

    // Check if transition leads to error
    if (transition.error) {
      return {
        isValid: false,
        error: ERRORS[transition.error],
        nextState: STATES.ERROR,
        errorCode: transition.error,
      };
    }

    return {
      isValid: true,
      error: null,
      nextState: transition.nextState,
      requiresValidation: transition.validate,
    };
  }

  /**
   * Execute a transition
   * @param {string} inputType - The type of word being added
   * @param {Object} wordData - The word data to add
   * @returns {Object} { success, error, state }
   */
  transition(inputType, wordData = {}) {
    // Handle CLEAR
    if (inputType === INPUT_TYPES.CLEAR) {
      this.reset();
      return {
        success: true,
        error: null,
        state: this.getState(),
      };
    }

    // Check transition validity
    const check = this.canTransition(inputType, wordData);

    if (!check.isValid) {
      return {
        success: false,
        error: check.error,
        errorCode: check.errorCode,
        state: this.getState(),
      };
    }

    // Execute transition
    this.history.push({
      fromState: this.currentState,
      input: inputType,
      toState: check.nextState,
      word: wordData,
      timestamp: Date.now(),
    });

    this.currentState = check.nextState;
    this.sentence.push(wordData);

    return {
      success: true,
      error: null,
      state: this.getState(),
      requiresValidation: check.requiresValidation,
    };
  }

  /**
   * Reset FSM to initial state
   */
  reset() {
    this.currentState = STATES.EMPTY;
    this.sentence = [];
    this.history = [];
  }

  /**
   * Get allowed input types for current state
   */
  getAllowedInputs() {
    const stateTransitions = TRANSITIONS[this.currentState];
    if (!stateTransitions) return [];

    return Object.entries(stateTransitions)
      .filter(([_, transition]) => !transition.error)
      .map(([inputType, _]) => inputType);
  }

  /**
   * Get visual representation of current state
   */
  getStateVisualization() {
    const states = [
      { id: STATES.EMPTY, label: 'Empty', symbol: '○' },
      { id: STATES.SUBJECT_LOCKED, label: 'Subject', symbol: 'S' },
      { id: STATES.VERB_LOCKED_TRANS, label: 'Verb (T)', symbol: 'V' },
      { id: STATES.VERB_LOCKED_INTRANS, label: 'Verb (I)', symbol: 'V' },
      { id: STATES.OBJECT_LOCKED, label: 'Object', symbol: 'O' },
    ];

    return states.map(s => ({
      ...s,
      isCurrent: s.id === this.currentState,
      isComplete: STATE_INFO[s.id]?.isAccepting || false,
    }));
  }
}

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

/**
 * Determine input type from LEXICON entry
 */
export function getInputTypeFromLexicon(lexiconEntry) {
  if (!lexiconEntry) return null;

  switch (lexiconEntry.type) {
    case 'SUBJECT':
      return INPUT_TYPES.SUBJECT;
    case 'VERB':
      return lexiconEntry.transitive
        ? INPUT_TYPES.VERB_TRANSITIVE
        : INPUT_TYPES.VERB_INTRANSITIVE;
    case 'OBJECT':
      return INPUT_TYPES.OBJECT;
    case 'MODIFIER':
      return INPUT_TYPES.MODIFIER;
    default:
      return null;
  }
}

/**
 * Print FSM diagram to console
 */
export function printFSMDiagram() {
  console.log(`
╔═══════════════════════════════════════════════════════════════════════════════╗
║                     SENTENCE CONSTRUCTION FSM                                  ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║                                                                               ║
║                              ┌─────────┐                                      ║
║                              │ (START) │                                      ║
║                              │  EMPTY  │                                      ║
║                              └────┬────┘                                      ║
║                                   │ SUBJECT                                   ║
║                                   ▼                                           ║
║                              ┌─────────┐                                      ║
║                              │ SUBJECT │                                      ║
║                              │ LOCKED  │                                      ║
║                              └────┬────┘                                      ║
║                        ┌──────────┴──────────┐                                ║
║                        │ VERB                │ VERB                           ║
║                        │ (transitive)        │ (intransitive)                 ║
║                        ▼                     ▼                                ║
║                   ┌─────────┐           ┌─────────┐                           ║
║                   │  VERB   │           │  VERB   │                           ║
║                   │ (TRANS) │           │(INTRANS)│                           ║
║                   └────┬────┘           └─────────┘                           ║
║                        │ OBJECT              ║                                ║
║                        ▼                 [COMPLETE]                           ║
║                   ┌─────────┐                                                 ║
║                   │ OBJECT  │                                                 ║
║                   │ LOCKED  │                                                 ║
║                   └─────────┘                                                 ║
║                       ║                                                       ║
║                   [COMPLETE]                                                  ║
║                                                                               ║
╠═══════════════════════════════════════════════════════════════════════════════╣
║  Legend: ─── Valid transition    ║ Accepting (complete) state                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝
  `);
}

// =============================================================================
// SINGLETON INSTANCE (optional)
// =============================================================================

export const defaultFSM = new SentenceFSM();
