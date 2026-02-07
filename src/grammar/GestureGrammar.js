/**
 * GestureGrammar.js — Compiled grammar as ESM
 *
 * Hand-written ES module equivalent of what `nearleyc grammar.ne` would produce.
 * Uses Nearley's internal rule format: { name, symbols, postprocess }.
 *
 * This avoids CJS/ESM conversion issues with Vite while being 1:1 equivalent
 * to the .ne file.
 *
 * Production rules:
 *   S   → NP VP
 *   NP  → SUBJECT_I | SUBJECT_YOU | SUBJECT_HE | SUBJECT_SHE | SUBJECT_WE | SUBJECT_THEY
 *   VP  → VT OBJ | VI
 *   VT  → GRAB | EAT | WANT | DRINK | GRABS | EATS | WANTS | DRINKS
 *   VI  → GO | STOP | GOES | STOPS
 *   OBJ → APPLE | BALL | WATER | FOOD | BOOK | HOUSE
 */

import { GestureLexer } from './GestureLexer.js';

// Helper: create a terminal symbol reference for a token type
function tok(type) {
  return { type, test: (t) => t.type === type };
}

// ── Postprocessors ─────────────────────────────────────────────────────────────

function ppSentence([np, vp]) {
  return { type: 'S', children: [np, vp] };
}

function ppNP(person, number) {
  return ([t]) => ({ type: 'NP', value: t.value, person, number });
}

function ppVPTransitive([vt, obj]) {
  return { type: 'VP', children: [vt, obj], transitive: true };
}

function ppVPIntransitive([vi]) {
  return { type: 'VP', children: [vi], transitive: false };
}

function ppVerb(sForm) {
  return ([t]) => ({ type: t.type === undefined ? 'V' : (sForm ? 'VT' : 'VT'), value: t.value, sForm });
}

function ppVT(sForm) {
  return ([t]) => ({ type: 'VT', value: t.value, sForm });
}

function ppVI(sForm) {
  return ([t]) => ({ type: 'VI', value: t.value, sForm });
}

function ppOBJ([t]) {
  return { type: 'OBJ', value: t.value };
}

// ── Grammar rules (Nearley format) ─────────────────────────────────────────────

const rules = [
  // S → NP VP
  { name: 'S', symbols: ['NP', 'VP'], postprocess: ppSentence },

  // NP → subject terminals
  { name: 'NP', symbols: [tok('SUBJECT_I')],    postprocess: ppNP(1, 'singular') },
  { name: 'NP', symbols: [tok('SUBJECT_YOU')],   postprocess: ppNP(2, 'singular') },
  { name: 'NP', symbols: [tok('SUBJECT_HE')],    postprocess: ppNP(3, 'singular') },
  { name: 'NP', symbols: [tok('SUBJECT_SHE')],   postprocess: ppNP(3, 'singular') },
  { name: 'NP', symbols: [tok('SUBJECT_WE')],    postprocess: ppNP(1, 'plural') },
  { name: 'NP', symbols: [tok('SUBJECT_THEY')],  postprocess: ppNP(3, 'plural') },

  // VP → VT OBJ | VI
  { name: 'VP', symbols: ['VT', 'OBJ'], postprocess: ppVPTransitive },
  { name: 'VP', symbols: ['VI'],         postprocess: ppVPIntransitive },

  // VT → transitive verb terminals
  { name: 'VT', symbols: [tok('GRAB')],   postprocess: ppVT(false) },
  { name: 'VT', symbols: [tok('EAT')],    postprocess: ppVT(false) },
  { name: 'VT', symbols: [tok('WANT')],   postprocess: ppVT(false) },
  { name: 'VT', symbols: [tok('DRINK')],  postprocess: ppVT(false) },
  { name: 'VT', symbols: [tok('GRABS')],  postprocess: ppVT(true) },
  { name: 'VT', symbols: [tok('EATS')],   postprocess: ppVT(true) },
  { name: 'VT', symbols: [tok('WANTS')],  postprocess: ppVT(true) },
  { name: 'VT', symbols: [tok('DRINKS')], postprocess: ppVT(true) },

  // VI → intransitive verb terminals
  { name: 'VI', symbols: [tok('GO')],    postprocess: ppVI(false) },
  { name: 'VI', symbols: [tok('STOP')],  postprocess: ppVI(false) },
  { name: 'VI', symbols: [tok('GOES')],  postprocess: ppVI(true) },
  { name: 'VI', symbols: [tok('STOPS')], postprocess: ppVI(true) },

  // OBJ → object terminals
  { name: 'OBJ', symbols: [tok('APPLE')],  postprocess: ppOBJ },
  { name: 'OBJ', symbols: [tok('BALL')],   postprocess: ppOBJ },
  { name: 'OBJ', symbols: [tok('WATER')],  postprocess: ppOBJ },
  { name: 'OBJ', symbols: [tok('FOOD')],   postprocess: ppOBJ },
  { name: 'OBJ', symbols: [tok('BOOK')],   postprocess: ppOBJ },
  { name: 'OBJ', symbols: [tok('HOUSE')],  postprocess: ppOBJ },
];

// ── Exported grammar object (Nearley Grammar constructor format) ───────────────

const grammar = {
  Lexer: new GestureLexer(),
  ParserRules: rules,
  ParserStart: 'S',
};

export default grammar;
