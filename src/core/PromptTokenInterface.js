/**
 * PromptTokenInterface.js — LLM Middleware Layer
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * THE KEY DIFFERENTIATOR: Converts gesture tokens into structured prompt-tokens
 * for ANY Large Language Model. This is NOT a chatbot — this is the INPUT
 * INTERFACE that controls a chatbot.
 *
 * Patent claim: "An interface layer that converts non-verbal spatial inputs
 * into prompt-tokens for Large Language Models."
 */

import { SyntacticGestureSpace } from './SyntacticGesture';

// =============================================================================
// GRAMMAR CATEGORY TRANSITIONS (model-agnostic replacement for Earley chart)
// =============================================================================

/**
 * Valid category transitions for building grammatically correct sentences.
 * This replaces the fixed Earley chart with a flexible, model-agnostic system.
 */
const CATEGORY_TRANSITIONS = {
  // Empty buffer → must start with a noun phrase
  START: ['NP'],
  // After a noun phrase → must have a verb phrase
  NP: ['VP'],
  // After a verb → can have an object (transitive) or end (intransitive)
  VP: ['OBJ', 'END'],
  // After an object → sentence is complete
  OBJ: ['END'],
};

/**
 * Agreement rules that any LLM should enforce.
 */
const AGREEMENT_RULES = {
  SUBJECT_VERB: {
    rule: 'Third person singular subjects require S-form verbs in present tense',
    applies_when: (tokens) => {
      const subject = tokens.find(t => t.category === 'NP');
      return subject && subject.features.person === 3 && subject.features.number === 'singular';
    },
    description: 'He/She/It → grabs, eats, wants (not grab, eat, want)',
  },
  TRANSITIVITY: {
    rule: 'Transitive verbs require an object; intransitive verbs must not have one',
    applies_when: (tokens) => {
      return tokens.some(t => t.category === 'VP');
    },
    description: 'Grab [something] vs. Stop (no object)',
  },
};

// =============================================================================
// PROMPT TOKEN INTERFACE
// =============================================================================

/**
 * Converts a sequence of SyntacticGesture recognitions into a structured
 * prompt that any LLM can consume.
 */
export class PromptTokenInterface {
  /**
   * @param {object} config
   * @param {string} config.model — target LLM: 'gpt-4', 'llama', 'claude', 'any'
   * @param {string} config.mode — interaction mode: 'sentence_completion', 'error_correction', 'translation'
   * @param {object} config.userProfile — accessibility and learner profile
   */
  constructor(config = {}) {
    this.modelTarget = config.model || 'any';
    this.mode = config.mode || 'sentence_completion';
    this.userProfile = config.userProfile || null;
    this.tokenBuffer = [];
    this.contextWindow = [];
    this.maxContextSize = config.maxContextSize || 10;
  }

  /**
   * Add a recognized gesture token to the buffer.
   * Each token carries morphosyntactic features from the SyntacticGesture.
   *
   * @param {import('./SyntacticGesture').SyntacticGesture} syntacticGesture
   */
  addToken(syntacticGesture) {
    const token = syntacticGesture.toPromptToken();
    this.tokenBuffer.push(token);

    // Maintain context window
    if (this.contextWindow.length >= this.maxContextSize) {
      this.contextWindow.shift();
    }
    this.contextWindow.push(token);
  }

  /**
   * Build a structured prompt from the token buffer.
   * The prompt includes token sequence, constraints, user profile, and instructions.
   *
   * @returns {object} structured prompt object
   */
  buildPrompt() {
    return {
      system: this._buildSystemMessage(),
      tokens: this.tokenBuffer.map(t => ({
        value: t.value,
        category: t.category,
        features: t.features,
        confidence: t.confidence,
        error_vector: t.error_vector,
        temporal: t.temporal,
        tense: t.tense,
      })),
      constraints: {
        valid_next_categories: this.getValidNextCategories(),
        agreement_rules: this.getActiveAgreementRules(),
      },
      user_profile: this.getUserProfile(),
      instruction: this.mode,
      context: {
        previous_sentences: this.contextWindow.length > this.tokenBuffer.length
          ? this.contextWindow.slice(0, -this.tokenBuffer.length)
          : [],
        sentence_position: this.tokenBuffer.length,
      },
    };
  }

  /**
   * Get valid next token categories.
   * This is model-agnostic — works with any LLM.
   * Replaces the Earley chart prediction with a flexible transition system.
   *
   * @returns {string[]} array of valid category names
   */
  getValidNextCategories() {
    if (this.tokenBuffer.length === 0) {
      return CATEGORY_TRANSITIONS.START;
    }

    const lastToken = this.tokenBuffer[this.tokenBuffer.length - 1];
    const transitions = CATEGORY_TRANSITIONS[lastToken.category];

    if (!transitions) return ['END'];

    // Filter based on transitivity
    if (lastToken.category === 'VP') {
      const isTransitive = lastToken.features.requires_object || lastToken.features.transitivity === 'transitive';
      if (isTransitive) {
        return ['OBJ']; // Must have object
      } else {
        return ['END']; // Cannot have object
      }
    }

    return transitions;
  }

  /**
   * Get agreement rules that are currently active based on the token buffer.
   *
   * @returns {object[]} active agreement rules
   */
  getActiveAgreementRules() {
    const active = [];

    for (const [ruleId, rule] of Object.entries(AGREEMENT_RULES)) {
      if (rule.applies_when(this.tokenBuffer)) {
        active.push({
          rule_id: ruleId,
          description: rule.description,
          rule_text: rule.rule,
        });
      }
    }

    return active;
  }

  /**
   * Get the user profile for the prompt.
   *
   * @returns {object|null} user profile data
   */
  getUserProfile() {
    if (!this.userProfile) {
      return {
        type: 'default',
        l1_language: 'unknown',
        proficiency_level: 'beginner',
        accessibility: null,
      };
    }

    return {
      type: this.userProfile.type || 'default',
      l1_language: this.userProfile.l1Language || 'unknown',
      proficiency_level: this.userProfile.proficiencyLevel || 'beginner',
      accessibility: this.userProfile.accessibility || null,
      feedback_preferences: this.userProfile.feedbackPreferences || ['visual', 'audio'],
    };
  }

  /**
   * Format the prompt for a specific LLM API.
   *
   * @param {string} model — target model: 'gpt-4', 'claude', 'llama', 'any'
   * @returns {object} model-specific prompt format
   */
  formatForModel(model) {
    const basePrompt = this.buildPrompt();
    const targetModel = model || this.modelTarget;

    switch (targetModel) {
      case 'gpt-4':
      case 'gpt-3.5':
        return this._formatOpenAI(basePrompt);
      case 'claude':
        return this._formatClaude(basePrompt);
      case 'llama':
        return this._formatLlama(basePrompt);
      case 'any':
      default:
        return this._formatGeneric(basePrompt);
    }
  }

  /**
   * Clear the token buffer (start a new sentence).
   */
  clearBuffer() {
    this.tokenBuffer = [];
  }

  /**
   * Remove the last token from the buffer.
   */
  undoLastToken() {
    this.tokenBuffer.pop();
  }

  /**
   * Get the current sentence as a readable string.
   *
   * @returns {string} human-readable sentence from current tokens
   */
  getReadableSentence() {
    return this.tokenBuffer.map(t => t.value).join(' ');
  }

  /**
   * Check if the current token buffer forms a complete sentence.
   *
   * @returns {boolean}
   */
  isComplete() {
    const validNext = this.getValidNextCategories();
    return validNext.length === 1 && validNext[0] === 'END';
  }

  // ===========================================================================
  // PRIVATE — System message and model-specific formatters
  // ===========================================================================

  _buildSystemMessage() {
    return [
      'You are a language acquisition assistant integrated with MLAF (Multimodal Language Acquisition Framework).',
      'You receive structured gesture-tokens that represent words formed by hand gestures.',
      'Each token carries grammatical features (person, number, transitivity) and confidence scores.',
      '',
      'Your role is to:',
      '1. Validate the grammatical structure of the gesture sequence',
      '2. Suggest corrections if agreement rules are violated',
      '3. Complete partial sentences when requested',
      '4. Provide feedback appropriate to the user\'s accessibility profile',
      '5. Explain grammar rules in context when errors occur',
      '',
      'IMPORTANT: You are receiving structured data, not free text. Parse the token objects.',
    ].join('\n');
  }

  _formatOpenAI(prompt) {
    return {
      model: 'gpt-4',
      messages: [
        { role: 'system', content: prompt.system },
        {
          role: 'user',
          content: JSON.stringify({
            gesture_tokens: prompt.tokens,
            constraints: prompt.constraints,
            user_profile: prompt.user_profile,
            instruction: prompt.instruction,
          }),
        },
      ],
      temperature: 0.3,
      max_tokens: 500,
    };
  }

  _formatClaude(prompt) {
    return {
      model: 'claude-sonnet-4-5-20250929',
      system: prompt.system,
      messages: [
        {
          role: 'user',
          content: JSON.stringify({
            gesture_tokens: prompt.tokens,
            constraints: prompt.constraints,
            user_profile: prompt.user_profile,
            instruction: prompt.instruction,
          }),
        },
      ],
      max_tokens: 500,
    };
  }

  _formatLlama(prompt) {
    return {
      prompt: `<|system|>\n${prompt.system}\n<|user|>\n${JSON.stringify({
        gesture_tokens: prompt.tokens,
        constraints: prompt.constraints,
        user_profile: prompt.user_profile,
        instruction: prompt.instruction,
      })}\n<|assistant|>\n`,
      max_tokens: 500,
      temperature: 0.3,
    };
  }

  _formatGeneric(prompt) {
    return {
      system: prompt.system,
      input: {
        tokens: prompt.tokens,
        constraints: prompt.constraints,
        user_profile: prompt.user_profile,
        instruction: prompt.instruction,
        context: prompt.context,
      },
      parameters: {
        temperature: 0.3,
        max_tokens: 500,
      },
    };
  }

  /**
   * Async method: query the Prolog grammar engine for valid next gestures.
   * Falls back to the synchronous CATEGORY_TRANSITIONS state machine when
   * the engine is unreachable.
   * @returns {Promise<Object>} { valid_next, current_state, parse_progress }
   */
  async getValidNextFromEngine() {
    try {
      const res = await fetch('/grammar/predict-next', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gestures: this.tokenBuffer.map(t => t.value) }),
      });
      if (!res.ok) throw new Error('Grammar engine unavailable');
      return await res.json();
    } catch {
      return { valid_next: this.getValidNextCategories().map(c => ({ category: c })) };
    }
  }

  /**
   * Async method: get full grammaticality validation from the Prolog engine.
   * Falls back to null when the engine is unreachable.
   * @returns {Promise<Object|null>} ValidationResponse or null
   */
  async getValidationFromEngine() {
    try {
      const res = await fetch('/grammar/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ gestures: this.tokenBuffer.map(t => t.value) }),
      });
      if (!res.ok) throw new Error('Grammar engine unavailable');
      return await res.json();
    } catch {
      return null;
    }
  }
}
