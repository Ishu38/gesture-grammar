@{%
/**
 * grammar.ne — Formal CFG for Gesture Grammar
 *
 * Production rules (Earley-parseable via Nearley.js):
 *   S   → NP VP
 *   NP  → SUBJECT_I | SUBJECT_YOU | SUBJECT_HE | SUBJECT_SHE | SUBJECT_WE | SUBJECT_THEY
 *   VP  → VT OBJ | VI
 *   VT  → GRAB | EAT | WANT | DRINK | GRABS | EATS | WANTS | DRINKS
 *   VI  → GO | STOP | GOES | STOPS
 *   OBJ → APPLE | BALL | WATER | FOOD | BOOK | HOUSE
 *
 * Two-phase validation:
 *   Phase 1 (here): Syntactic — word order and transitivity
 *   Phase 2 (EarleyParser.js): Semantic — subject-verb agreement
 */

const { GestureLexer } = require('./GestureLexer.js');
const lexer = new GestureLexer();
%}

@lexer lexer

# ── Top-level rule ──────────────────────────────────────────────────────────────
S -> NP VP  {%
  ([np, vp]) => ({ type: 'S', children: [np, vp] })
%}

# ── Noun Phrase (Subject) ──────────────────────────────────────────────────────
NP -> %SUBJECT_I   {% ([t]) => ({ type: 'NP', value: t.value, person: 1, number: 'singular' }) %}
    | %SUBJECT_YOU  {% ([t]) => ({ type: 'NP', value: t.value, person: 2, number: 'singular' }) %}
    | %SUBJECT_HE   {% ([t]) => ({ type: 'NP', value: t.value, person: 3, number: 'singular' }) %}
    | %SUBJECT_SHE  {% ([t]) => ({ type: 'NP', value: t.value, person: 3, number: 'singular' }) %}
    | %SUBJECT_WE   {% ([t]) => ({ type: 'NP', value: t.value, person: 1, number: 'plural' }) %}
    | %SUBJECT_THEY {% ([t]) => ({ type: 'NP', value: t.value, person: 3, number: 'plural' }) %}

# ── Verb Phrase ────────────────────────────────────────────────────────────────
VP -> VT OBJ  {% ([vt, obj]) => ({ type: 'VP', children: [vt, obj], transitive: true }) %}
   | VI       {% ([vi]) => ({ type: 'VP', children: [vi], transitive: false }) %}

# ── Transitive Verbs ──────────────────────────────────────────────────────────
VT -> %GRAB   {% ([t]) => ({ type: 'VT', value: t.value, sForm: false }) %}
   | %EAT     {% ([t]) => ({ type: 'VT', value: t.value, sForm: false }) %}
   | %WANT    {% ([t]) => ({ type: 'VT', value: t.value, sForm: false }) %}
   | %DRINK   {% ([t]) => ({ type: 'VT', value: t.value, sForm: false }) %}
   | %GRABS   {% ([t]) => ({ type: 'VT', value: t.value, sForm: true }) %}
   | %EATS    {% ([t]) => ({ type: 'VT', value: t.value, sForm: true }) %}
   | %WANTS   {% ([t]) => ({ type: 'VT', value: t.value, sForm: true }) %}
   | %DRINKS  {% ([t]) => ({ type: 'VT', value: t.value, sForm: true }) %}

# ── Intransitive Verbs ────────────────────────────────────────────────────────
VI -> %GO     {% ([t]) => ({ type: 'VI', value: t.value, sForm: false }) %}
   | %STOP    {% ([t]) => ({ type: 'VI', value: t.value, sForm: false }) %}
   | %GOES    {% ([t]) => ({ type: 'VI', value: t.value, sForm: true }) %}
   | %STOPS   {% ([t]) => ({ type: 'VI', value: t.value, sForm: true }) %}

# ── Objects (Nouns) ───────────────────────────────────────────────────────────
OBJ -> %APPLE  {% ([t]) => ({ type: 'OBJ', value: t.value }) %}
    | %BALL    {% ([t]) => ({ type: 'OBJ', value: t.value }) %}
    | %WATER   {% ([t]) => ({ type: 'OBJ', value: t.value }) %}
    | %FOOD    {% ([t]) => ({ type: 'OBJ', value: t.value }) %}
    | %BOOK    {% ([t]) => ({ type: 'OBJ', value: t.value }) %}
    | %HOUSE   {% ([t]) => ({ type: 'OBJ', value: t.value }) %}
