/**
 * graph-rag.test.js — Tests for MLAF Knowledge Graph + Graph RAG Engine
 *
 * Validates all 4 reasoning layers:
 *   Layer 1: Deductive Traversal
 *   Layer 2: Compositional Type Application (Montague)
 *   Layer 3: Abductive Diagnosis
 *   Layer 4: LLM Context Enrichment
 */

import { describe, it, expect, beforeAll } from 'vitest';
import { buildMLAFKnowledgeGraph, getGraphStats, NODE_TYPES, EDGE_TYPES } from '../core/MLAFKnowledgeGraph.js';
import { GraphRAG } from '../core/GraphRAG.js';

let graph;
let rag;

beforeAll(() => {
  graph = buildMLAFKnowledgeGraph();
  rag = new GraphRAG(graph);
});

// =============================================================================
// GRAPH STRUCTURE INTEGRITY
// =============================================================================

describe('Knowledge Graph Structure', () => {
  it('should build a non-empty graph', () => {
    const stats = getGraphStats(graph);
    expect(stats.totalNodes).toBeGreaterThan(50);
    expect(stats.totalEdges).toBeGreaterThan(100);
  });

  it('should have all 19 gesture nodes', () => {
    const gestureIds = [
      'SUBJECT_I', 'SUBJECT_YOU', 'SUBJECT_HE', 'SUBJECT_SHE', 'SUBJECT_WE', 'SUBJECT_THEY',
      'GRAB', 'EAT', 'WANT', 'DRINK', 'SEE', 'GO', 'STOP',
      'APPLE', 'BALL', 'WATER', 'FOOD', 'BOOK', 'HOUSE',
    ];
    for (const id of gestureIds) {
      expect(graph.hasNode(`GST:${id}`), `Missing gesture node GST:${id}`).toBe(true);
      expect(graph.hasNode(`LEX:${id}`), `Missing lexical node LEX:${id}`).toBe(true);
    }
  });

  it('should have all 7 S-form verb nodes', () => {
    for (const sf of ['GRABS', 'EATS', 'WANTS', 'DRINKS', 'SEES', 'GOES', 'STOPS']) {
      expect(graph.hasNode(`LEX:${sf}`), `Missing S-form LEX:${sf}`).toBe(true);
    }
  });

  it('should have grammar rule nodes', () => {
    for (const rule of ['RULE:S', 'RULE:NP', 'RULE:VP', 'RULE:VT', 'RULE:VI', 'RULE:OBJ']) {
      expect(graph.hasNode(rule)).toBe(true);
    }
  });

  it('should have semantic type nodes', () => {
    for (const t of ['TYPE:e', 'TYPE:t', 'TYPE:<e,t>', 'TYPE:<e,<e,t>>']) {
      expect(graph.hasNode(t)).toBe(true);
    }
  });

  it('should have interference pattern nodes', () => {
    expect(graph.hasNode('INTF:SOV_ORDER')).toBe(true);
    expect(graph.hasNode('INTF:TOPIC_FRONTING')).toBe(true);
    expect(graph.hasNode('INTF:OBJECT_DROP')).toBe(true);
  });

  it('should have 5 curriculum stages', () => {
    for (let i = 1; i <= 5; i++) {
      expect(graph.hasNode(`STAGE:${i}`)).toBe(true);
    }
  });

  it('should have 3 modality nodes + fusion node', () => {
    expect(graph.hasNode('MOD:VISUAL')).toBe(true);
    expect(graph.hasNode('MOD:ACOUSTIC')).toBe(true);
    expect(graph.hasNode('MOD:GAZE')).toBe(true);
    expect(graph.hasNode('MOD:FUSION')).toBe(true);
  });
});

// =============================================================================
// LAYER 1: DEDUCTIVE TRAVERSAL
// =============================================================================

describe('Layer 1 — Deductive Traversal', () => {
  it('empty sentence → should return all subjects', () => {
    const result = rag.queryValidNext([]);
    expect(result.validNext.length).toBe(6);
    expect(result.validNext.every(v => v.category === 'SUBJECT')).toBe(true);
  });

  it('after SUBJECT_HE → should return S-form verbs only', () => {
    const result = rag.queryValidNext(['SUBJECT_HE']);
    expect(result.agreementRule).toContain('3sg');
    expect(result.validNext.every(v => v.is_s_form === true)).toBe(true);
    expect(result.validNext.some(v => v.grammar_id === 'GRABS')).toBe(true);
    expect(result.validNext.some(v => v.grammar_id === 'SEES')).toBe(true);
    // Should NOT include base forms
    expect(result.validNext.some(v => v.grammar_id === 'GRAB')).toBe(false);
  });

  it('after SUBJECT_I → should return base-form verbs only', () => {
    const result = rag.queryValidNext(['SUBJECT_I']);
    expect(result.validNext.every(v => v.is_s_form === false)).toBe(true);
    expect(result.validNext.some(v => v.grammar_id === 'GRAB')).toBe(true);
    expect(result.validNext.some(v => v.grammar_id === 'SEE')).toBe(true);
    // Should NOT include S-forms
    expect(result.validNext.some(v => v.grammar_id === 'GRABS')).toBe(false);
  });

  it('after transitive verb → should return objects', () => {
    const result = rag.queryValidNext(['SUBJECT_I', 'GRAB']);
    expect(result.validNext.every(v => v.category === 'OBJECT')).toBe(true);
    expect(result.validNext.length).toBe(6);
  });

  it('after intransitive verb → should be complete', () => {
    const result = rag.queryValidNext(['SUBJECT_I', 'GO']);
    expect(result.isComplete).toBe(true);
    expect(result.validNext.length).toBe(0);
  });

  it('after object → should be complete', () => {
    const result = rag.queryValidNext(['SUBJECT_HE', 'GRABS', 'APPLE']);
    expect(result.isComplete).toBe(true);
  });
});

// =============================================================================
// LAYER 1: VERB AGREEMENT QUERIES
// =============================================================================

describe('Layer 1 — Verb Agreement', () => {
  it('HE + GRAB → should return GRABS', () => {
    const result = rag.queryVerbAgreement('GRAB', 'SUBJECT_HE');
    expect(result.correctForm).toBe('GRABS');
  });

  it('I + GRABS → should return GRAB', () => {
    const result = rag.queryVerbAgreement('GRABS', 'SUBJECT_I');
    expect(result.correctForm).toBe('GRAB');
  });

  it('SHE + SEE → should return SEES', () => {
    const result = rag.queryVerbAgreement('SEE', 'SUBJECT_SHE');
    expect(result.correctForm).toBe('SEES');
  });

  it('THEY + GRAB → should keep GRAB', () => {
    const result = rag.queryVerbAgreement('GRAB', 'SUBJECT_THEY');
    expect(result.correctForm).toBe('GRAB');
  });
});

// =============================================================================
// LAYER 2: COMPOSITIONAL TYPE APPLICATION
// =============================================================================

describe('Layer 2 — Compositional Types (Montague)', () => {
  it('HE (e) + GRABS (<e,<e,t>>) → <e,t> (needs object)', () => {
    const result = rag.computeComposition(['SUBJECT_HE', 'GRABS']);
    expect(result.isWellTyped).toBe(true);
    expect(result.resultType).toBe('<e, t>');
    expect(result.isComplete).toBe(false);
  });

  it('HE + GRABS + APPLE → t (complete)', () => {
    const result = rag.computeComposition(['SUBJECT_HE', 'GRABS', 'APPLE']);
    expect(result.isWellTyped).toBe(true);
    expect(result.resultType).toBe('t');
    expect(result.isComplete).toBe(true);
  });

  it('I (e) + GO (<e,t>) → t (complete, intransitive)', () => {
    const result = rag.computeComposition(['SUBJECT_I', 'GO']);
    expect(result.isWellTyped).toBe(true);
    expect(result.resultType).toBe('t');
    expect(result.isComplete).toBe(true);
  });

  it('single token HE → e (partial)', () => {
    const result = rag.computeComposition(['SUBJECT_HE']);
    expect(result.resultType).toBe('e');
    expect(result.isComplete).toBe(false);
  });

  it('should generate a type trace for each token', () => {
    const result = rag.computeComposition(['SUBJECT_SHE', 'SEES', 'BOOK']);
    expect(result.typeTrace.length).toBe(3);
    expect(result.typeTrace[0].type).toBe('e');
    expect(result.typeTrace[1].type).toBe('<e, <e, t>>');
    expect(result.typeTrace[2].type).toBe('e');
    expect(result.isComplete).toBe(true);
  });
});

// =============================================================================
// LAYER 3: ABDUCTIVE DIAGNOSIS
// =============================================================================

describe('Layer 3 — Abductive Diagnosis', () => {
  it('WRONG_VERB_FORM → should identify agreement cause + curriculum remediation', () => {
    const result = rag.diagnoseError('WRONG_VERB_FORM');
    expect(result.causes.length).toBeGreaterThan(0);
    expect(result.remediation.length).toBeGreaterThan(0);
    expect(result.remediation[0].stage).toBe(4); // Stage 4 teaches agreement
  });

  it('WRONG_WORD_ORDER → should identify ISL interference patterns', () => {
    const result = rag.diagnoseError('WRONG_WORD_ORDER');
    expect(result.interferencePatterns.length).toBeGreaterThan(0);
    expect(result.interferencePatterns.some(p => p.id === 'SOV_ORDER')).toBe(true);
  });

  it('MISSING_OBJECT → should identify object drop + Stage 3 remediation', () => {
    const result = rag.diagnoseError('MISSING_OBJECT');
    expect(result.interferencePatterns.some(p => p.id === 'TRANSITIVE_OBJECT_DROP')).toBe(true);
    expect(result.remediation.some(r => r.stage === 3)).toBe(true);
  });

  it('should detect SOV interference in sentence order', () => {
    const sentence = [
      { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
      { type: 'OBJECT', grammar_id: 'APPLE' },
      { type: 'VERB', grammar_id: 'EAT', transitive: true },
    ];
    const result = rag.detectInterference(sentence);
    expect(result.isL1Transfer).toBe(true);
    expect(result.detected.some(d => d.id === 'SOV_ORDER')).toBe(true);
  });

  it('should detect topic fronting (object-first)', () => {
    const sentence = [
      { type: 'OBJECT', grammar_id: 'APPLE' },
      { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
      { type: 'VERB', grammar_id: 'EAT', transitive: true },
    ];
    const result = rag.detectInterference(sentence);
    expect(result.isL1Transfer).toBe(true);
    expect(result.detected.some(d => d.id === 'TOPIC_FRONTING')).toBe(true);
  });

  it('should NOT detect interference in correct SVO order', () => {
    const sentence = [
      { type: 'SUBJECT', grammar_id: 'SUBJECT_I' },
      { type: 'VERB', grammar_id: 'EAT', transitive: true },
      { type: 'OBJECT', grammar_id: 'APPLE' },
    ];
    const result = rag.detectInterference(sentence);
    expect(result.isL1Transfer).toBe(false);
  });
});

// =============================================================================
// LAYER 4: LLM CONTEXT ENRICHMENT
// =============================================================================

describe('Layer 4 — LLM Context Enrichment', () => {
  it('should build complete LLM context for partial sentence', () => {
    const context = rag.buildLLMContext(['SUBJECT_HE']);
    const ragCtx = context.graph_rag_context;

    // Valid next tokens
    expect(ragCtx.valid_next.tokens.length).toBeGreaterThan(0);
    expect(ragCtx.valid_next.agreement_rule).toContain('3sg');

    // Composition status
    expect(ragCtx.composition.current_type).toBe('e');
    expect(ragCtx.composition.is_complete).toBe(false);

    // Meta
    expect(ragCtx.meta.reasoning_layers).toEqual(['deductive', 'compositional', 'abductive']);
    expect(ragCtx.meta.retrieval_method).toBe('typed_edge_traversal');
  });

  it('should build complete context for full sentence', () => {
    const words = [
      { type: 'SUBJECT', grammar_id: 'SUBJECT_SHE' },
      { type: 'VERB', grammar_id: 'SEES', transitive: true },
      { type: 'OBJECT', grammar_id: 'BOOK' },
    ];
    const context = rag.buildLLMContext(['SUBJECT_SHE', 'SEES', 'BOOK'], words);
    const ragCtx = context.graph_rag_context;

    expect(ragCtx.composition.is_complete).toBe(true);
    expect(ragCtx.composition.current_type).toBe('t');
    expect(ragCtx.interference.detected).toBe(false);
  });

  it('should include curriculum stages', () => {
    const context = rag.buildLLMContext([]);
    const stages = context.graph_rag_context.curriculum.stages;
    expect(stages.length).toBe(5);
    expect(stages[0].stage).toBe(1);
    expect(stages[0].gestures).toContain('SUBJECT_I');
  });
});

// =============================================================================
// UTILITY QUERIES
// =============================================================================

describe('Utility Queries', () => {
  it('should return confusion pairs for YOU', () => {
    const pairs = rag.getConfusionPairs('SUBJECT_YOU');
    expect(pairs.length).toBeGreaterThan(0);
    expect(pairs.some(p => p.confusedWith === 'DRINK')).toBe(true);
  });

  it('should return full token profile', () => {
    const profile = rag.getTokenProfile('GRAB');
    expect(profile).not.toBeNull();
    expect(profile.category).toBe('VERB');
    expect(profile.transitive).toBe(true);
    expect(profile.semanticType).toBe('<e, <e, t>>');
    expect(profile.sFormPair).toBe('GRABS');
  });

  it('should return token profile for object', () => {
    const profile = rag.getTokenProfile('APPLE');
    expect(profile).not.toBeNull();
    expect(profile.category).toBe('OBJECT');
    expect(profile.semanticType).toBe('e');
  });

  it('should export graph snapshot', () => {
    const snapshot = rag.exportSnapshot();
    expect(snapshot.stats.totalNodes).toBeGreaterThan(50);
    expect(snapshot.stats.totalEdges).toBeGreaterThan(100);
  });
});

// =============================================================================
// PERFORMANCE (must be sub-millisecond for 30fps gesture loop)
// =============================================================================

describe('Performance', () => {
  it('queryValidNext should complete in < 1ms', () => {
    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      rag.queryValidNext(['SUBJECT_HE']);
    }
    const elapsed = performance.now() - start;
    const avgMs = elapsed / 1000;
    expect(avgMs).toBeLessThan(1);
    console.log(`queryValidNext avg: ${avgMs.toFixed(4)}ms`);
  });

  it('computeComposition should complete in < 1ms', () => {
    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      rag.computeComposition(['SUBJECT_HE', 'GRABS', 'APPLE']);
    }
    const elapsed = performance.now() - start;
    const avgMs = elapsed / 1000;
    expect(avgMs).toBeLessThan(1);
    console.log(`computeComposition avg: ${avgMs.toFixed(4)}ms`);
  });

  it('buildLLMContext should complete in < 2ms', () => {
    const start = performance.now();
    for (let i = 0; i < 1000; i++) {
      rag.buildLLMContext(['SUBJECT_HE', 'GRABS'], [
        { type: 'SUBJECT', grammar_id: 'SUBJECT_HE' },
        { type: 'VERB', grammar_id: 'GRABS', transitive: true },
      ]);
    }
    const elapsed = performance.now() - start;
    const avgMs = elapsed / 1000;
    expect(avgMs).toBeLessThan(2);
    console.log(`buildLLMContext avg: ${avgMs.toFixed(4)}ms`);
  });
});
