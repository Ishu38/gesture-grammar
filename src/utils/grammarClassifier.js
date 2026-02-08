/**
 * grammarClassifier.js
 * Maps finger states to grammar tokens using deterministic pattern matching.
 */

import { analyzeFingerStates } from './fingerAnalysis';

/**
 * Token registry — each entry defines the required finger pattern.
 * Patterns are [thumb, index, middle, ring, pinky] where true = open/extended.
 */
const TOKEN_REGISTRY = [
  {
    token: 'Singular_Subject',
    label: 'Singular Subject',
    pattern: { thumb: false, index: true, middle: false, ring: false, pinky: false },
    description: 'Only index finger extended',
  },
  {
    token: 'Plural_Subject',
    label: 'Plural Subject',
    pattern: { thumb: false, index: true, middle: true, ring: false, pinky: false },
    description: 'Index + middle fingers (V-sign)',
  },
  {
    token: 'Negation',
    label: 'Negation',
    pattern: { thumb: true, index: true, middle: true, ring: true, pinky: true },
    description: 'All five fingers open (open palm)',
  },
  {
    token: 'Full_Stop',
    label: 'Full Stop',
    pattern: { thumb: false, index: false, middle: false, ring: false, pinky: false },
    description: 'All fingers curled (fist)',
  },
];

/**
 * Check if a finger state matches a pattern exactly.
 */
function matchesPattern(states, pattern) {
  return (
    states.thumb === pattern.thumb &&
    states.index === pattern.index &&
    states.middle === pattern.middle &&
    states.ring === pattern.ring &&
    states.pinky === pattern.pinky
  );
}

/**
 * Identify a grammar token from raw hand landmarks.
 * @param {Array<{x:number, y:number, z?:number}>} landmarks - raw 21-point landmarks
 * @returns {string} token name or 'UNKNOWN'
 */
export function identifyGrammarToken(landmarks) {
  const states = analyzeFingerStates(landmarks);
  return classifyFingerState(states);
}

/**
 * Classify a pre-computed finger state into a grammar token.
 * Avoids double computation when states are already available.
 * @param {{ thumb:boolean, index:boolean, middle:boolean, ring:boolean, pinky:boolean }} states
 * @returns {string} token name or 'UNKNOWN'
 */
export function classifyFingerState(states) {
  for (const entry of TOKEN_REGISTRY) {
    if (matchesPattern(states, entry.pattern)) {
      return entry.token;
    }
  }
  return 'UNKNOWN';
}

/**
 * Get the full token registry for UI gesture guides.
 * @returns {Array<{token:string, label:string, pattern:object, description:string}>}
 */
export function getGrammarTokens() {
  return TOKEN_REGISTRY;
}
