/**
 * SemanticTypeSystem.js — Type-Theoretic Gesture Validation
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Implements a typed semantic system based on Montague Grammar / TL schema
 * (Partee, ter Meulen & Wall, Ch 13: Compositionality & Type Theory).
 *
 * Every gesture in the MLAF vocabulary is assigned a semantic type:
 *   - e         : entity (pronouns, nouns)
 *   - t         : truth value (complete sentences)
 *   - <a, b>    : function from type a to type b
 *
 * The type system catches slot incompatibilities BEFORE sending to the Prolog
 * grammar engine, reducing round-trip latency and providing instant feedback.
 *
 * Type Assignments (following Montague's PTQ):
 *   Pronouns:   type e                     (I, You, He, She, We, They)
 *   Trans Verbs: type <e, <e, t>>          (want, eat, see, grab, drink)
 *   Intrans Verbs: type <e, t>             (go, stop)
 *   Nouns:      type e                     (food, water, book, apple, ball, house)
 *
 * Sentence composition:
 *   [Subject:e] + [Verb:<e,<e,t>>] + [Object:e]  =>  t   (SVO complete)
 *   [Subject:e] + [Verb:<e,t>]                    =>  t   (SV complete)
 *
 * The Principle of Compositionality (Frege's Principle):
 *   "The meaning of a complex expression is a function of the meanings
 *    of its parts and the way they are combined."
 *
 * In MLAF, this means: the grammaticality of a gesture sentence is determined
 * entirely by the types of its parts and their syntactic combination.
 */

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

/**
 * Semantic types are represented as strings for simple types
 * and objects for complex (function) types.
 *
 * Simple: 'e', 't'
 * Complex: { from: Type, to: Type }
 *
 * Examples:
 *   'e'                              — entity
 *   't'                              — truth value
 *   { from: 'e', to: 't' }          — <e, t> (intransitive verb / predicate)
 *   { from: 'e', to: { from: 'e', to: 't' } }  — <e, <e, t>> (transitive verb)
 */

// Shorthand constructors
const e = 'e';
const t = 't';
const fn = (from, to) => ({ from, to });

// Common type patterns
const PREDICATE = fn(e, t);             // <e, t> — 1-place predicate
const TRANSITIVE = fn(e, fn(e, t));     // <e, <e, t>> — 2-place relation

// =============================================================================
// GESTURE TYPE ASSIGNMENTS
// =============================================================================

/**
 * Maps each gesture ID to its semantic type, syntactic category,
 * and expected slot in a sentence.
 */
const GESTURE_TYPES = {
  // --- Pronouns (type e) ---
  SUBJECT_I:     { type: e, category: 'NP', slot: 'SUBJECT', label: 'I' },
  SUBJECT_YOU:   { type: e, category: 'NP', slot: 'SUBJECT', label: 'You' },
  SUBJECT_HE:    { type: e, category: 'NP', slot: 'SUBJECT', label: 'He' },
  SUBJECT_SHE:   { type: e, category: 'NP', slot: 'SUBJECT', label: 'She' },
  SUBJECT_WE:    { type: e, category: 'NP', slot: 'SUBJECT', label: 'We' },
  SUBJECT_THEY:  { type: e, category: 'NP', slot: 'SUBJECT', label: 'They' },

  // --- Transitive Verbs (type <e, <e, t>>) ---
  // IDs match LEXICON/parser convention: unprefixed (GRAB, not VERB_GRAB)
  WANT:     { type: TRANSITIVE, category: 'VP', slot: 'VERB', label: 'want', transitive: true },
  EAT:      { type: TRANSITIVE, category: 'VP', slot: 'VERB', label: 'eat', transitive: true },
  SEE:      { type: TRANSITIVE, category: 'VP', slot: 'VERB', label: 'see', transitive: true },
  GRAB:     { type: TRANSITIVE, category: 'VP', slot: 'VERB', label: 'grab', transitive: true },
  DRINK:    { type: TRANSITIVE, category: 'VP', slot: 'VERB', label: 'drink', transitive: true },

  // --- Intransitive Verbs (type <e, t>) ---
  GO:       { type: PREDICATE, category: 'VP', slot: 'VERB', label: 'go', transitive: false },
  STOP:     { type: PREDICATE, category: 'VP', slot: 'VERB', label: 'stop', transitive: false },

  // --- Object Nouns (type e) ---
  FOOD:   { type: e, category: 'NP', slot: 'OBJECT', label: 'food' },
  WATER:  { type: e, category: 'NP', slot: 'OBJECT', label: 'water' },
  BOOK:   { type: e, category: 'NP', slot: 'OBJECT', label: 'book' },
  APPLE:  { type: e, category: 'NP', slot: 'OBJECT', label: 'apple' },
  BALL:   { type: e, category: 'NP', slot: 'OBJECT', label: 'ball' },
  HOUSE:  { type: e, category: 'NP', slot: 'OBJECT', label: 'house' },
};

// =============================================================================
// TYPE OPERATIONS
// =============================================================================

/**
 * Check if two types are equal.
 * @param {Type} a
 * @param {Type} b
 * @returns {boolean}
 */
function typeEquals(a, b) {
  if (typeof a === 'string' && typeof b === 'string') return a === b;
  if (typeof a === 'object' && typeof b === 'object' && a !== null && b !== null) {
    return typeEquals(a.from, b.from) && typeEquals(a.to, b.to);
  }
  return false;
}

/**
 * Pretty-print a type for display.
 * @param {Type} type
 * @returns {string}
 */
function typeToString(type) {
  if (typeof type === 'string') return type;
  if (typeof type === 'object' && type !== null) {
    const from = typeToString(type.from);
    const to = typeToString(type.to);
    return `<${from}, ${to}>`;
  }
  return '?';
}

/**
 * Check if a type is a function type.
 * @param {Type} type
 * @returns {boolean}
 */
function isFunctionType(type) {
  return typeof type === 'object' && type !== null && 'from' in type && 'to' in type;
}

/**
 * Apply functional application: if f has type <a, b> and x has type a,
 * then f(x) has type b.
 * @param {Type} fnType — the function type
 * @param {Type} argType — the argument type
 * @returns {Type|null} result type, or null if type mismatch
 */
function applyType(fnType, argType) {
  if (!isFunctionType(fnType)) return null;
  if (!typeEquals(fnType.from, argType)) return null;
  return fnType.to;
}

// =============================================================================
// SEMANTIC TYPE SYSTEM (Main class)
// =============================================================================

export class SemanticTypeSystem {
  constructor() {
    this._gestureTypes = { ...GESTURE_TYPES };
  }

  // ===========================================================================
  // PUBLIC — Type lookup
  // ===========================================================================

  /**
   * Get the semantic type info for a gesture ID.
   * @param {string} gestureId — e.g., 'SUBJECT_I', 'EAT', 'FOOD'
   * @returns {GestureTypeInfo|null}
   */
  getGestureType(gestureId) {
    return this._gestureTypes[gestureId] || null;
  }

  /**
   * Get the raw semantic type for a gesture.
   * @param {string} gestureId
   * @returns {Type|null}
   */
  getType(gestureId) {
    const info = this._gestureTypes[gestureId];
    return info ? info.type : null;
  }

  /**
   * Get a human-readable type string for a gesture.
   * @param {string} gestureId
   * @returns {string}
   */
  getTypeString(gestureId) {
    const type = this.getType(gestureId);
    return type ? typeToString(type) : 'unknown';
  }

  // ===========================================================================
  // PUBLIC — Slot validation (pre-Prolog check)
  // ===========================================================================

  /**
   * Validate whether a gesture can fill a given syntactic slot.
   * Called at gesture lock time, BEFORE sending to Prolog.
   *
   * @param {string} gestureId — the gesture being locked
   * @param {string} targetSlot — 'SUBJECT' | 'VERB' | 'OBJECT'
   * @returns {SlotValidation}
   */
  validateSlot(gestureId, targetSlot) {
    const info = this.getGestureType(gestureId);
    if (!info) {
      return {
        valid: false,
        gestureId,
        targetSlot,
        error: 'UNKNOWN_GESTURE',
        message: `Unknown gesture: ${gestureId}`,
      };
    }

    // Check slot compatibility
    const expectedSlot = info.slot;
    const slotMatch = expectedSlot === targetSlot;

    // Also check if type is compatible with slot expectations
    const slotTypeExpectation = SLOT_TYPE_EXPECTATIONS[targetSlot];
    const typeMatch = slotTypeExpectation
      ? slotTypeExpectation.some(expected => typeEquals(info.type, expected))
      : true;

    if (!slotMatch && !typeMatch) {
      return {
        valid: false,
        gestureId,
        targetSlot,
        expectedSlot,
        gestureType: typeToString(info.type),
        error: 'SLOT_MISMATCH',
        message: `"${info.label}" (type ${typeToString(info.type)}) cannot fill ${targetSlot} slot. Expected in ${expectedSlot} slot.`,
      };
    }

    return {
      valid: true,
      gestureId,
      targetSlot,
      gestureType: typeToString(info.type),
      category: info.category,
    };
  }

  // ===========================================================================
  // PUBLIC — Sentence type composition
  // ===========================================================================

  /**
   * Compute the semantic type of a partial or complete sentence.
   * Implements compositional type checking (Frege's Principle).
   *
   * @param {Array<string>} gestureIds — ordered list of gesture IDs in the sentence
   * @returns {CompositionResult}
   */
  composeSentence(gestureIds) {
    if (!gestureIds || gestureIds.length === 0) {
      return {
        complete: false,
        resultType: null,
        resultTypeString: 'empty',
        errors: [],
        slots: { subject: null, verb: null, object: null },
        expectsNext: ['SUBJECT'],
      };
    }

    const errors = [];
    let subject = null;
    let verb = null;
    let object = null;

    // Parse gesture sequence into slots
    for (const gid of gestureIds) {
      const info = this.getGestureType(gid);
      if (!info) {
        errors.push({ gestureId: gid, error: 'UNKNOWN_GESTURE' });
        continue;
      }

      if (info.slot === 'SUBJECT' && !subject) {
        subject = info;
      } else if (info.slot === 'VERB' && !verb) {
        verb = info;
      } else if (info.slot === 'OBJECT' && !object) {
        object = info;
      } else {
        errors.push({
          gestureId: gid,
          error: 'DUPLICATE_SLOT',
          message: `Duplicate ${info.slot} — already filled`,
        });
      }
    }

    // Compose types step by step
    let resultType = null;
    let complete = false;

    if (subject && verb) {
      // Apply verb to subject: verb(subject)
      const afterSubject = applyType(verb.type, subject.type);

      if (afterSubject === null) {
        errors.push({
          error: 'TYPE_MISMATCH',
          message: `Verb "${verb.label}" (${typeToString(verb.type)}) cannot take subject "${subject.label}" (${typeToString(subject.type)})`,
        });
      } else if (typeof afterSubject === 'string' && afterSubject === t) {
        // Intransitive: <e, t>(e) = t — sentence complete
        resultType = t;
        complete = !object; // complete only if no spurious object
        if (object) {
          errors.push({
            error: 'EXTRA_ARGUMENT',
            message: `"${verb.label}" is intransitive — does not take an object`,
          });
        }
      } else if (isFunctionType(afterSubject)) {
        // Transitive: <e, <e, t>>(e) = <e, t> — needs object
        if (object) {
          const finalType = applyType(afterSubject, object.type);
          if (finalType === null) {
            errors.push({
              error: 'TYPE_MISMATCH',
              message: `"${verb.label}" cannot take object "${object.label}" (${typeToString(object.type)})`,
            });
          } else {
            resultType = finalType;
            complete = typeof finalType === 'string' && finalType === t;
          }
        } else {
          resultType = afterSubject;
        }
      }
    } else if (subject && !verb) {
      resultType = subject.type;
    } else if (verb && !subject) {
      resultType = verb.type;
    }

    // Determine what's expected next
    const expectsNext = [];
    if (!subject) expectsNext.push('SUBJECT');
    if (!verb) expectsNext.push('VERB');
    if (verb && verb.transitive && !object) expectsNext.push('OBJECT');

    return {
      complete,
      resultType,
      resultTypeString: resultType ? typeToString(resultType) : 'incomplete',
      errors,
      slots: {
        subject: subject ? subject.label : null,
        verb: verb ? verb.label : null,
        object: object ? object.label : null,
      },
      expectsNext,
    };
  }

  // ===========================================================================
  // PUBLIC — Next gesture suggestions (type-driven)
  // ===========================================================================

  /**
   * Given the current sentence, return which gesture IDs are type-valid next.
   * This filters the full vocabulary to only show compatible options.
   *
   * @param {Array<string>} currentGestureIds — gestures already in sentence
   * @returns {Array<{gestureId: string, label: string, type: string, slot: string}>}
   */
  getValidNextGestures(currentGestureIds) {
    const composition = this.composeSentence(currentGestureIds);
    const validNext = [];

    for (const [gid, info] of Object.entries(this._gestureTypes)) {
      if (composition.expectsNext.includes(info.slot)) {
        validNext.push({
          gestureId: gid,
          label: info.label,
          type: typeToString(info.type),
          slot: info.slot,
        });
      }
    }

    return validNext;
  }

  // ===========================================================================
  // PUBLIC — Diagnostics
  // ===========================================================================

  /**
   * Get the full type catalog for debug display.
   * @returns {Array<{gestureId: string, label: string, type: string, category: string, slot: string}>}
   */
  getTypeCatalog() {
    return Object.entries(this._gestureTypes).map(([gid, info]) => ({
      gestureId: gid,
      label: info.label,
      type: typeToString(info.type),
      category: info.category,
      slot: info.slot,
      transitive: info.transitive || false,
    }));
  }
}

// =============================================================================
// SLOT TYPE EXPECTATIONS
// =============================================================================
// What types are valid in each syntactic slot

const SLOT_TYPE_EXPECTATIONS = {
  SUBJECT: [e],                            // entities only
  VERB:    [PREDICATE, TRANSITIVE],        // predicates and relations
  OBJECT:  [e],                            // entities only
};

// =============================================================================
// SINGLETON EXPORT
// =============================================================================

let _instance = null;

/**
 * Get the singleton SemanticTypeSystem instance.
 * @returns {SemanticTypeSystem}
 */
export function getSemanticTypeSystem() {
  if (!_instance) {
    _instance = new SemanticTypeSystem();
  }
  return _instance;
}

// Export utilities for testing
export { typeEquals, typeToString, isFunctionType, applyType, GESTURE_TYPES };
export default SemanticTypeSystem;
