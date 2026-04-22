/**
 * MLAFKnowledgeGraph.js — In-Browser Knowledge Graph for MLAF
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ARCHITECTURE: COMPOSITIONAL GRAPH RAG
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Goes beyond neuro-symbolic AI by implementing three reasoning layers:
 *
 *   Layer 1 — ONTOLOGICAL (Graph Structure)
 *     Entities and typed edges encoding linguistic ontology.
 *     Traversal = deductive inference. No LLM needed.
 *
 *   Layer 2 — COMPOSITIONAL (Type-Theoretic)
 *     Montague-style semantic types on nodes enable compositional
 *     reasoning: f: <e, t> applied to arg: e yields t.
 *     The graph COMPUTES meaning, not just retrieves it.
 *
 *   Layer 3 — ABDUCTIVE (Learner Model)
 *     Given observed errors, the graph traverses interference patterns,
 *     mastery state, and curriculum position to ABDUCE the most likely
 *     cause and prescribe the optimal next intervention.
 *
 * All three layers run in-browser via graphology. Zero network calls.
 * Complete patent protection — no data leaves the device.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHY THIS IS BEYOND NEURO-SYMBOLIC
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Standard neuro-symbolic: neural perception + symbolic reasoning
 * MLAF's approach:        neural perception (MediaPipe + RF classifier)
 *                       + symbolic grammar  (Earley parser + CFG)
 *                       + compositional semantics (Montague types in graph)
 *                       + abductive diagnosis (error cause inference)
 *                       + Bayesian fusion (UMCE trimodal posterior)
 *                       + grounded retrieval (Graph RAG for LLM context)
 *
 * The knowledge graph doesn't just STORE facts — it REASONS over them
 * via typed edges, compositional application, and abductive traversal.
 */

import Graph from 'graphology';

// =============================================================================
// NODE TYPES (Ontological Categories)
// =============================================================================

export const NODE_TYPES = Object.freeze({
  GESTURE:       'GESTURE',
  GRAMMAR_RULE:  'GRAMMAR_RULE',
  LEXICAL_ENTRY: 'LEXICAL_ENTRY',
  SEMANTIC_TYPE: 'SEMANTIC_TYPE',
  AGREEMENT:     'AGREEMENT',
  INTERFERENCE:  'INTERFERENCE',
  CURRICULUM:    'CURRICULUM',
  ERROR_CLASS:   'ERROR_CLASS',
  MODALITY:      'MODALITY',
});

// =============================================================================
// EDGE TYPES (Relational Predicates)
// =============================================================================

export const EDGE_TYPES = Object.freeze({
  // Grammatical relations
  PRODUCES:        'PRODUCES',         // GESTURE → LEXICAL_ENTRY
  TRIGGERS_FORM:   'TRIGGERS_FORM',    // AGREEMENT → LEXICAL_ENTRY (S-form selection)
  REQUIRES_OBJECT: 'REQUIRES_OBJECT',  // LEXICAL_ENTRY → constraint
  FORBIDS_OBJECT:  'FORBIDS_OBJECT',   // LEXICAL_ENTRY → constraint
  HAS_S_FORM:      'HAS_S_FORM',       // base verb → s-form verb
  HAS_BASE_FORM:   'HAS_BASE_FORM',    // s-form → base verb
  BELONGS_TO:      'BELONGS_TO',        // LEXICAL_ENTRY → GRAMMAR_RULE (NP, VP, OBJ)
  EXPANDS_TO:      'EXPANDS_TO',        // GRAMMAR_RULE → child rules

  // Semantic types (Montague)
  HAS_TYPE:        'HAS_TYPE',          // LEXICAL_ENTRY → SEMANTIC_TYPE
  APPLIES_TO:      'APPLIES_TO',        // function type → argument type
  YIELDS:          'YIELDS',            // function application → result type

  // Disambiguation
  AMBIGUOUS_WITH:  'AMBIGUOUS_WITH',    // GESTURE ↔ GESTURE (confusion pair)
  DISAMBIGUATED_BY:'DISAMBIGUATED_BY',  // GESTURE → distinguishing feature

  // L1 Interference
  PRONE_TO:        'PRONE_TO',          // learner profile → INTERFERENCE pattern
  CORRECTED_BY:    'CORRECTED_BY',      // INTERFERENCE → correction strategy
  MANIFESTS_AS:    'MANIFESTS_AS',      // INTERFERENCE → observable error sequence

  // Curriculum
  PREREQUISITE:    'PREREQUISITE',      // CURRICULUM stage → prior stage
  CONTAINS:        'CONTAINS',          // CURRICULUM stage → GESTURE
  UNLOCKS:         'UNLOCKS',           // mastery → next stage

  // Error diagnosis (abductive)
  CAUSED_BY:       'CAUSED_BY',         // ERROR → root cause
  INDICATES:       'INDICATES',         // ERROR → interference pattern
  REMEDIATED_BY:   'REMEDIATED_BY',     // ERROR → curriculum stage or exercise

  // Multimodal
  DETECTED_BY:     'DETECTED_BY',       // GESTURE → MODALITY (visual, acoustic, gaze)
  FUSED_VIA:       'FUSED_VIA',         // MODALITY → fusion method (UMCE Bayesian)
});

// =============================================================================
// KNOWLEDGE GRAPH BUILDER
// =============================================================================

/**
 * Build the complete MLAF knowledge graph.
 * All linguistic, pedagogical, and diagnostic knowledge encoded as
 * nodes + typed edges. Runs entirely in-browser via graphology.
 *
 * @returns {Graph} The populated knowledge graph
 */
export function buildMLAFKnowledgeGraph() {
  const g = new Graph({ multi: true, type: 'directed' });

  // ─── LAYER 1: ONTOLOGICAL — Entities & Relations ────────────────────────

  _addGrammarRules(g);
  _addLexicalEntries(g);
  _addGestures(g);
  _addAgreementRules(g);
  _addSemanticTypes(g);
  _addInterferencePatterns(g);
  _addCurriculum(g);
  _addErrorClasses(g);
  _addModalities(g);
  _addDisambiguationEdges(g);

  return g;
}

// =============================================================================
// GRAMMAR RULES (CFG Productions)
// =============================================================================

function _addGrammarRules(g) {
  // Non-terminals
  g.addNode('RULE:S',   { type: NODE_TYPES.GRAMMAR_RULE, label: 'S → NP VP', production: 'S → NP VP' });
  g.addNode('RULE:NP',  { type: NODE_TYPES.GRAMMAR_RULE, label: 'NP → Subject', category: 'NP' });
  g.addNode('RULE:VP',  { type: NODE_TYPES.GRAMMAR_RULE, label: 'VP → VT OBJ | VI', category: 'VP' });
  g.addNode('RULE:VT',  { type: NODE_TYPES.GRAMMAR_RULE, label: 'VT → transitive verb', category: 'VT' });
  g.addNode('RULE:VI',  { type: NODE_TYPES.GRAMMAR_RULE, label: 'VI → intransitive verb', category: 'VI' });
  g.addNode('RULE:OBJ', { type: NODE_TYPES.GRAMMAR_RULE, label: 'OBJ → object noun', category: 'OBJ' });

  // Production hierarchy
  g.addEdge('RULE:S', 'RULE:NP', { type: EDGE_TYPES.EXPANDS_TO, position: 0 });
  g.addEdge('RULE:S', 'RULE:VP', { type: EDGE_TYPES.EXPANDS_TO, position: 1 });
  g.addEdge('RULE:VP', 'RULE:VT', { type: EDGE_TYPES.EXPANDS_TO, variant: 'transitive' });
  g.addEdge('RULE:VP', 'RULE:OBJ', { type: EDGE_TYPES.EXPANDS_TO, variant: 'transitive' });
  g.addEdge('RULE:VP', 'RULE:VI', { type: EDGE_TYPES.EXPANDS_TO, variant: 'intransitive' });
}

// =============================================================================
// LEXICAL ENTRIES (Terminal Symbols)
// =============================================================================

function _addLexicalEntries(g) {
  // ── Subjects ──
  const subjects = [
    { id: 'SUBJECT_I',    display: 'I',    person: 1, number: 'singular' },
    { id: 'SUBJECT_YOU',  display: 'You',  person: 2, number: 'singular' },
    { id: 'SUBJECT_HE',   display: 'He',   person: 3, number: 'singular' },
    { id: 'SUBJECT_SHE',  display: 'She',  person: 3, number: 'singular' },
    { id: 'SUBJECT_WE',   display: 'We',   person: 1, number: 'plural' },
    { id: 'SUBJECT_THEY', display: 'They', person: 3, number: 'plural' },
  ];

  for (const s of subjects) {
    g.addNode(`LEX:${s.id}`, {
      type: NODE_TYPES.LEXICAL_ENTRY,
      grammar_id: s.id,
      display: s.display,
      category: 'SUBJECT',
      person: s.person,
      number: s.number,
    });
    g.addEdge(`LEX:${s.id}`, 'RULE:NP', { type: EDGE_TYPES.BELONGS_TO });
  }

  // ── Base Verbs ──
  const verbs = [
    { id: 'GRAB',  display: 'grab',  transitive: true,  sForm: 'GRABS' },
    { id: 'EAT',   display: 'eat',   transitive: true,  sForm: 'EATS' },
    { id: 'WANT',  display: 'want',  transitive: true,  sForm: 'WANTS' },
    { id: 'DRINK', display: 'drink', transitive: true,  sForm: 'DRINKS' },
    { id: 'SEE',   display: 'see',   transitive: true,  sForm: 'SEES' },
    { id: 'GO',    display: 'go',    transitive: false, sForm: 'GOES' },
    { id: 'STOP',  display: 'stop',  transitive: false, sForm: 'STOPS' },
  ];

  for (const v of verbs) {
    const ruleKey = v.transitive ? 'RULE:VT' : 'RULE:VI';
    g.addNode(`LEX:${v.id}`, {
      type: NODE_TYPES.LEXICAL_ENTRY,
      grammar_id: v.id,
      display: v.display,
      category: 'VERB',
      transitive: v.transitive,
      is_s_form: false,
    });
    g.addEdge(`LEX:${v.id}`, ruleKey, { type: EDGE_TYPES.BELONGS_TO });

    if (v.transitive) {
      g.addEdge(`LEX:${v.id}`, 'RULE:OBJ', { type: EDGE_TYPES.REQUIRES_OBJECT });
    } else {
      g.addNode(`CONSTRAINT:no_obj_${v.id}`, {
        type: 'CONSTRAINT',
        rule: `${v.display} is intransitive — cannot take an object`,
      });
      g.addEdge(`LEX:${v.id}`, `CONSTRAINT:no_obj_${v.id}`, { type: EDGE_TYPES.FORBIDS_OBJECT });
    }
  }

  // ── S-Form Verbs ──
  const sForms = [
    { id: 'GRABS',  display: 'grabs',  base: 'GRAB',  transitive: true },
    { id: 'EATS',   display: 'eats',   base: 'EAT',   transitive: true },
    { id: 'WANTS',  display: 'wants',  base: 'WANT',  transitive: true },
    { id: 'DRINKS', display: 'drinks', base: 'DRINK', transitive: true },
    { id: 'SEES',   display: 'sees',   base: 'SEE',   transitive: true },
    { id: 'GOES',   display: 'goes',   base: 'GO',    transitive: false },
    { id: 'STOPS',  display: 'stops',  base: 'STOP',  transitive: false },
  ];

  for (const sf of sForms) {
    const ruleKey = sf.transitive ? 'RULE:VT' : 'RULE:VI';
    g.addNode(`LEX:${sf.id}`, {
      type: NODE_TYPES.LEXICAL_ENTRY,
      grammar_id: sf.id,
      display: sf.display,
      category: 'VERB',
      transitive: sf.transitive,
      is_s_form: true,
      base_form: sf.base,
    });
    g.addEdge(`LEX:${sf.id}`, ruleKey, { type: EDGE_TYPES.BELONGS_TO });
    g.addEdge(`LEX:${sf.base}`, `LEX:${sf.id}`, { type: EDGE_TYPES.HAS_S_FORM });
    g.addEdge(`LEX:${sf.id}`, `LEX:${sf.base}`, { type: EDGE_TYPES.HAS_BASE_FORM });
  }

  // ── Objects ──
  const objects = [
    { id: 'APPLE', display: 'apple', countable: true },
    { id: 'BALL',  display: 'ball',  countable: true },
    { id: 'WATER', display: 'water', countable: false },
    { id: 'FOOD',  display: 'food',  countable: true },
    { id: 'BOOK',  display: 'book',  countable: true },
    { id: 'HOUSE', display: 'house', countable: true },
  ];

  for (const o of objects) {
    g.addNode(`LEX:${o.id}`, {
      type: NODE_TYPES.LEXICAL_ENTRY,
      grammar_id: o.id,
      display: o.display,
      category: 'OBJECT',
      countable: o.countable,
    });
    g.addEdge(`LEX:${o.id}`, 'RULE:OBJ', { type: EDGE_TYPES.BELONGS_TO });
  }
}

// =============================================================================
// GESTURES (Physical Hand Configurations)
// =============================================================================

function _addGestures(g) {
  const gestureMap = [
    { id: 'SUBJECT_I',    gesture: 'FIST',                 lex: 'SUBJECT_I' },
    { id: 'SUBJECT_YOU',  gesture: 'INDEX_POINT',          lex: 'SUBJECT_YOU' },
    { id: 'SUBJECT_HE',   gesture: 'THUMB_UP',             lex: 'SUBJECT_HE' },
    { id: 'SUBJECT_SHE',  gesture: 'THUMB_DOWN',           lex: 'SUBJECT_SHE' },
    { id: 'SUBJECT_WE',   gesture: 'TWO_FINGERS_TOGETHER', lex: 'SUBJECT_WE' },
    { id: 'SUBJECT_THEY', gesture: 'FOUR_FINGERS_SPREAD',  lex: 'SUBJECT_THEY' },
    { id: 'GRAB',         gesture: 'CLAW',                 lex: 'GRAB' },
    { id: 'EAT',          gesture: 'BUNCHED_TO_MOUTH',     lex: 'EAT' },
    { id: 'WANT',         gesture: 'CURVED_FINGERS_PULL',  lex: 'WANT' },
    { id: 'DRINK',        gesture: 'C_SHAPE',              lex: 'DRINK' },
    { id: 'SEE',          gesture: 'V_SHAPE_SPREAD',       lex: 'SEE' },
    { id: 'GO',           gesture: 'INDEX_POINT_FORWARD',  lex: 'GO' },
    { id: 'STOP',         gesture: 'OPEN_PALM',            lex: 'STOP' },
    { id: 'APPLE',        gesture: 'CUPPED_HAND',          lex: 'APPLE' },
    { id: 'BALL',         gesture: 'CURVED_SPREAD',        lex: 'BALL' },
    { id: 'WATER',        gesture: 'W_SHAPE',              lex: 'WATER' },
    { id: 'FOOD',         gesture: 'FLAT_FINGERS_FORWARD',lex: 'FOOD' },
    { id: 'BOOK',         gesture: 'FLAT_PALM_UP',         lex: 'BOOK' },
    { id: 'HOUSE',        gesture: 'ROOF_SHAPE',           lex: 'HOUSE' },
  ];

  for (const gm of gestureMap) {
    g.addNode(`GST:${gm.id}`, {
      type: NODE_TYPES.GESTURE,
      gesture_name: gm.gesture,
      grammar_id: gm.id,
    });
    g.addEdge(`GST:${gm.id}`, `LEX:${gm.lex}`, { type: EDGE_TYPES.PRODUCES });
  }
}

// =============================================================================
// AGREEMENT RULES (Morphosyntactic Constraints)
// =============================================================================

function _addAgreementRules(g) {
  // Third-person singular requires S-form
  g.addNode('AGREE:3SG_S_FORM', {
    type: NODE_TYPES.AGREEMENT,
    rule: 'Third-person singular subjects (He, She) require S-form verbs',
    person: 3,
    number: 'singular',
    requires: 's_form',
  });

  // Non-3sg requires base form
  g.addNode('AGREE:NON_3SG_BASE', {
    type: NODE_TYPES.AGREEMENT,
    rule: 'Non-third-singular subjects (I, You, We, They) require base-form verbs',
    requires: 'base_form',
  });

  // Connect 3sg subjects to agreement rule
  for (const subj of ['SUBJECT_HE', 'SUBJECT_SHE']) {
    g.addEdge(`LEX:${subj}`, 'AGREE:3SG_S_FORM', { type: EDGE_TYPES.TRIGGERS_FORM });
  }

  // Connect non-3sg subjects
  for (const subj of ['SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_WE', 'SUBJECT_THEY']) {
    g.addEdge(`LEX:${subj}`, 'AGREE:NON_3SG_BASE', { type: EDGE_TYPES.TRIGGERS_FORM });
  }

  // Connect agreement rules to verb forms
  const baseVerbs = ['GRAB', 'EAT', 'WANT', 'DRINK', 'SEE', 'GO', 'STOP'];
  const sFormVerbs = ['GRABS', 'EATS', 'WANTS', 'DRINKS', 'SEES', 'GOES', 'STOPS'];

  for (const v of sFormVerbs) {
    g.addEdge('AGREE:3SG_S_FORM', `LEX:${v}`, { type: EDGE_TYPES.TRIGGERS_FORM, selects: v });
  }
  for (const v of baseVerbs) {
    g.addEdge('AGREE:NON_3SG_BASE', `LEX:${v}`, { type: EDGE_TYPES.TRIGGERS_FORM, selects: v });
  }
}

// =============================================================================
// SEMANTIC TYPES (Montague — Compositional Layer)
// =============================================================================

function _addSemanticTypes(g) {
  // Primitive types
  g.addNode('TYPE:e', { type: NODE_TYPES.SEMANTIC_TYPE, notation: 'e', description: 'Entity type' });
  g.addNode('TYPE:t', { type: NODE_TYPES.SEMANTIC_TYPE, notation: 't', description: 'Truth-value type' });

  // Function types
  g.addNode('TYPE:<e,t>', {
    type: NODE_TYPES.SEMANTIC_TYPE,
    notation: '<e, t>',
    description: 'Predicate (intransitive verb): takes entity, yields truth value',
  });
  g.addNode('TYPE:<e,<e,t>>', {
    type: NODE_TYPES.SEMANTIC_TYPE,
    notation: '<e, <e, t>>',
    description: 'Relation (transitive verb): takes two entities, yields truth value',
  });

  // Compositional application edges
  g.addEdge('TYPE:<e,t>', 'TYPE:e', { type: EDGE_TYPES.APPLIES_TO, argument: 'subject' });
  g.addEdge('TYPE:<e,t>', 'TYPE:t', { type: EDGE_TYPES.YIELDS, result: 'sentence truth value' });
  g.addEdge('TYPE:<e,<e,t>>', 'TYPE:e', { type: EDGE_TYPES.APPLIES_TO, argument: 'object' });
  g.addEdge('TYPE:<e,<e,t>>', 'TYPE:<e,t>', { type: EDGE_TYPES.YIELDS, result: 'predicate after object saturation' });

  // Link lexical entries to their types
  const entityNodes = [
    'SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_HE', 'SUBJECT_SHE', 'SUBJECT_WE', 'SUBJECT_THEY',
    'APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE',
  ];
  for (const e of entityNodes) {
    g.addEdge(`LEX:${e}`, 'TYPE:e', { type: EDGE_TYPES.HAS_TYPE });
  }

  // Transitive verbs → <e, <e, t>>
  for (const v of ['GRAB', 'EAT', 'WANT', 'DRINK', 'SEE', 'GRABS', 'EATS', 'WANTS', 'DRINKS', 'SEES']) {
    g.addEdge(`LEX:${v}`, 'TYPE:<e,<e,t>>', { type: EDGE_TYPES.HAS_TYPE });
  }

  // Intransitive verbs → <e, t>
  for (const v of ['GO', 'STOP', 'GOES', 'STOPS']) {
    g.addEdge(`LEX:${v}`, 'TYPE:<e,t>', { type: EDGE_TYPES.HAS_TYPE });
  }
}

// =============================================================================
// ISL INTERFERENCE PATTERNS (Abductive Layer)
// =============================================================================

function _addInterferencePatterns(g) {
  g.addNode('INTF:SOV_ORDER', {
    type: NODE_TYPES.INTERFERENCE,
    id: 'SOV_ORDER',
    severity: 'error',
    title: 'ISL Word Order (SOV)',
    l1_structure: 'Subject → Object → Verb',
    l2_target: 'Subject → Verb → Object',
    example_l1: 'I apple eat',
    example_l2: 'I eat apple',
    linguistic_basis: 'Odlin (1989): L1 word order transfer; ISL is SOV, English is SVO',
  });

  g.addNode('INTF:TOPIC_FRONTING', {
    type: NODE_TYPES.INTERFERENCE,
    id: 'TOPIC_FRONTING',
    severity: 'error',
    title: 'ISL Topic Fronting',
    l1_structure: 'Object → Subject → Verb (topic-prominent)',
    l2_target: 'Subject → Verb → Object',
    example_l1: 'Apple, I eat',
    example_l2: 'I eat apple',
    linguistic_basis: 'Cummins (1979): Linguistic Interdependence Hypothesis — L1 topic-prominence transfers',
  });

  g.addNode('INTF:OBJECT_DROP', {
    type: NODE_TYPES.INTERFERENCE,
    id: 'TRANSITIVE_OBJECT_DROP',
    severity: 'warning',
    title: 'Missing Object (Pro-drop)',
    l1_structure: 'Subject → Verb (object implied)',
    l2_target: 'Subject → Verb → Object (explicit)',
    example_l1: 'I eat',
    example_l2: 'I eat apple',
    linguistic_basis: 'ISL pro-drop: objects contextually inferred from signing environment',
  });

  // Correction strategies
  g.addNode('CORRECT:REORDER_SVO', {
    type: 'CORRECTION',
    strategy: 'Reorder tokens to Subject → Verb → Object',
    explanation: 'In English, the verb comes between subject and object',
  });
  g.addNode('CORRECT:ADD_OBJECT', {
    type: 'CORRECTION',
    strategy: 'Prompt learner to add an explicit object noun',
    explanation: 'English transitive verbs require an explicit object',
  });

  g.addEdge('INTF:SOV_ORDER', 'CORRECT:REORDER_SVO', { type: EDGE_TYPES.CORRECTED_BY });
  g.addEdge('INTF:TOPIC_FRONTING', 'CORRECT:REORDER_SVO', { type: EDGE_TYPES.CORRECTED_BY });
  g.addEdge('INTF:OBJECT_DROP', 'CORRECT:ADD_OBJECT', { type: EDGE_TYPES.CORRECTED_BY });

  // What patterns manifest as (observable error sequences)
  g.addNode('SEQ:S_O_V', { type: 'ERROR_SEQUENCE', pattern: ['SUBJECT', 'OBJECT', 'VERB'] });
  g.addNode('SEQ:O_S_V', { type: 'ERROR_SEQUENCE', pattern: ['OBJECT', 'SUBJECT', 'VERB'] });
  g.addNode('SEQ:S_VT',  { type: 'ERROR_SEQUENCE', pattern: ['SUBJECT', 'TRANSITIVE_VERB_WITHOUT_OBJ'] });

  g.addEdge('INTF:SOV_ORDER', 'SEQ:S_O_V', { type: EDGE_TYPES.MANIFESTS_AS });
  g.addEdge('INTF:TOPIC_FRONTING', 'SEQ:O_S_V', { type: EDGE_TYPES.MANIFESTS_AS });
  g.addEdge('INTF:OBJECT_DROP', 'SEQ:S_VT', { type: EDGE_TYPES.MANIFESTS_AS });
}

// =============================================================================
// CURRICULUM (Mastery-Gated Stages)
// =============================================================================

function _addCurriculum(g) {
  const stages = [
    { id: 'STAGE:1', label: 'Foundation',              gestures: ['SUBJECT_I', 'STOP'] },
    { id: 'STAGE:2', label: 'Pronoun Expansion',       gestures: ['SUBJECT_HE', 'SUBJECT_SHE', 'GRAB', 'EAT'] },
    { id: 'STAGE:3', label: 'Object Introduction',     gestures: ['APPLE', 'BALL'] },
    { id: 'STAGE:4', label: 'Subject-Verb Agreement',  gestures: ['GRABS', 'EATS', 'STOPS'] },
    { id: 'STAGE:5', label: 'Full Paradigm',           gestures: ['SUBJECT_YOU', 'SUBJECT_WE', 'SUBJECT_THEY'] },
  ];

  for (let i = 0; i < stages.length; i++) {
    const s = stages[i];
    g.addNode(s.id, {
      type: NODE_TYPES.CURRICULUM,
      stage: i + 1,
      label: s.label,
    });

    // Connect stage → gestures it contains
    for (const gestureId of s.gestures) {
      g.addEdge(s.id, `LEX:${gestureId}`, { type: EDGE_TYPES.CONTAINS });
    }

    // Prerequisite chain
    if (i > 0) {
      g.addEdge(stages[i].id, stages[i - 1].id, { type: EDGE_TYPES.PREREQUISITE });
      g.addEdge(stages[i - 1].id, stages[i].id, { type: EDGE_TYPES.UNLOCKS });
    }
  }
}

// =============================================================================
// ERROR CLASSES (Abductive Diagnosis)
// =============================================================================

function _addErrorClasses(g) {
  g.addNode('ERR:WRONG_VERB_FORM', {
    type: NODE_TYPES.ERROR_CLASS,
    title: 'Wrong Verb Form',
    description: 'Used base form with 3sg subject (or S-form with non-3sg)',
  });
  g.addNode('ERR:WRONG_WORD_ORDER', {
    type: NODE_TYPES.ERROR_CLASS,
    title: 'Wrong Word Order',
    description: 'Tokens not in SVO order',
  });
  g.addNode('ERR:MISSING_OBJECT', {
    type: NODE_TYPES.ERROR_CLASS,
    title: 'Missing Object',
    description: 'Transitive verb used without an object',
  });
  g.addNode('ERR:EXTRA_OBJECT', {
    type: NODE_TYPES.ERROR_CLASS,
    title: 'Extra Object',
    description: 'Intransitive verb followed by an object',
  });

  // Abductive: errors → possible causes
  g.addEdge('ERR:WRONG_VERB_FORM', 'AGREE:3SG_S_FORM', { type: EDGE_TYPES.CAUSED_BY, reason: 'Agreement rule not applied' });
  g.addEdge('ERR:WRONG_VERB_FORM', 'STAGE:4', { type: EDGE_TYPES.REMEDIATED_BY, reason: 'Stage 4 teaches S-V agreement' });

  g.addEdge('ERR:WRONG_WORD_ORDER', 'INTF:SOV_ORDER', { type: EDGE_TYPES.INDICATES, reason: 'L1 ISL SOV transfer' });
  g.addEdge('ERR:WRONG_WORD_ORDER', 'INTF:TOPIC_FRONTING', { type: EDGE_TYPES.INDICATES, reason: 'L1 ISL topic prominence' });

  g.addEdge('ERR:MISSING_OBJECT', 'INTF:OBJECT_DROP', { type: EDGE_TYPES.INDICATES, reason: 'L1 ISL pro-drop' });
  g.addEdge('ERR:MISSING_OBJECT', 'STAGE:3', { type: EDGE_TYPES.REMEDIATED_BY, reason: 'Stage 3 introduces objects' });
}

// =============================================================================
// MODALITIES (Multimodal Fusion Topology)
// =============================================================================

function _addModalities(g) {
  g.addNode('MOD:VISUAL',   { type: NODE_TYPES.MODALITY, name: 'Visual (AGGME)',   source: 'MediaPipe HandLandmarker' });
  g.addNode('MOD:ACOUSTIC', { type: NODE_TYPES.MODALITY, name: 'Acoustic (UASAM)', source: 'Web Audio API' });
  g.addNode('MOD:GAZE',     { type: NODE_TYPES.MODALITY, name: 'Gaze (EyeGaze)',   source: 'MediaPipe FaceLandmarker' });
  g.addNode('MOD:FUSION',   { type: NODE_TYPES.MODALITY, name: 'UMCE Bayesian Fusion', formula: 'P(S|A,V,G) ∝ P(A|S)·P(V|S)·P(G|S)·P(S)' });

  g.addEdge('MOD:VISUAL',   'MOD:FUSION', { type: EDGE_TYPES.FUSED_VIA, weight_symbol: 'w_V', provides: 'P(V|S)' });
  g.addEdge('MOD:ACOUSTIC', 'MOD:FUSION', { type: EDGE_TYPES.FUSED_VIA, weight_symbol: 'w_A', provides: 'P(A|S)' });
  g.addEdge('MOD:GAZE',     'MOD:FUSION', { type: EDGE_TYPES.FUSED_VIA, weight_symbol: 'w_G', provides: 'P(G|S)' });

  // All gestures detected by visual modality
  for (const nodeId of ['SUBJECT_I','SUBJECT_YOU','SUBJECT_HE','SUBJECT_SHE','SUBJECT_WE','SUBJECT_THEY',
                         'GRAB','EAT','WANT','DRINK','SEE','GO','STOP',
                         'APPLE','BALL','WATER','FOOD','BOOK','HOUSE']) {
    g.addEdge(`GST:${nodeId}`, 'MOD:VISUAL', { type: EDGE_TYPES.DETECTED_BY });
  }
}

// =============================================================================
// DISAMBIGUATION EDGES (Confusion Pairs)
// =============================================================================

function _addDisambiguationEdges(g) {
  // Known confusion pairs from gestureDetection.js
  const confusionPairs = [
    { a: 'SUBJECT_YOU', b: 'DRINK', feature: 'Thumb MCP angle: YOU > 140° (tucked), DRINK < 120° (C-spread)' },
    { a: 'SUBJECT_YOU', b: 'GO',    feature: 'YOU: hand vertical + thumb tucked; GO: hand not vertical + thumb relaxed' },
    { a: 'SEE',         b: 'SUBJECT_WE', feature: 'SEE: fingers spread apart > 0.06; WE: fingers together < 0.04' },
    { a: 'SUBJECT_HE',  b: 'SUBJECT_SHE', feature: 'HE: thumb above wrist; SHE: thumb below wrist' },
  ];

  for (const pair of confusionPairs) {
    g.addEdge(`GST:${pair.a}`, `GST:${pair.b}`, {
      type: EDGE_TYPES.AMBIGUOUS_WITH,
      distinguishing_feature: pair.feature,
    });
    g.addEdge(`GST:${pair.b}`, `GST:${pair.a}`, {
      type: EDGE_TYPES.AMBIGUOUS_WITH,
      distinguishing_feature: pair.feature,
    });
  }
}

// =============================================================================
// GRAPH STATISTICS (for debugging/display)
// =============================================================================

/**
 * Get summary statistics of the knowledge graph.
 * @param {Graph} g
 * @returns {object}
 */
export function getGraphStats(g) {
  const nodesByType = {};
  g.forEachNode((id, attrs) => {
    const t = attrs.type || 'unknown';
    nodesByType[t] = (nodesByType[t] || 0) + 1;
  });

  const edgesByType = {};
  g.forEachEdge((id, attrs) => {
    const t = attrs.type || 'unknown';
    edgesByType[t] = (edgesByType[t] || 0) + 1;
  });

  return {
    totalNodes: g.order,
    totalEdges: g.size,
    nodesByType,
    edgesByType,
  };
}
