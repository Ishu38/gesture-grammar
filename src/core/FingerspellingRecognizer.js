/**
 * FingerspellingRecognizer.js
 * Recognizes ISL fingerspelling letters from hand landmarks.
 * Phase 1: 12 high-distinctiveness letters (A, B, C, D, E, I, L, O, U, V, W, Y).
 * Uses finger state analysis + geometric constraints for letter detection.
 * Includes a spelling buffer for building words that match against LEXICON.
 */

import { analyzeFingerStatesDetailed } from '../utils/fingerAnalysis';
import { euclideanDistance3D } from '../utils/vectorGeometry';
import { LEXICON } from '../utils/GrammarEngine';

// =============================================================================
// ISL LETTER DEFINITIONS
// =============================================================================
// Each letter defines:
//   fingerStates: { thumb, index, middle, ring, pinky } → true=open, false=curled, null=either
//   geometricChecks: optional array of functions(landmarks) → 0..1 score
//   description: human-readable hand shape description

const LETTER_DEFINITIONS = {
  A: {
    fingerStates: { thumb: true, index: false, middle: false, ring: false, pinky: false },
    description: 'Fist with thumb up alongside',
  },
  B: {
    fingerStates: { thumb: false, index: true, middle: true, ring: true, pinky: true },
    description: 'Four fingers extended, thumb tucked',
  },
  C: {
    fingerStates: { thumb: true, index: true, middle: true, ring: false, pinky: false },
    geometricChecks: [
      // Thumb and index tips should be separated (curved C shape)
      (lm) => {
        const dist = euclideanDistance3D(lm[4], lm[8]);
        return dist > 0.06 ? 1 : dist / 0.06;
      },
    ],
    description: 'Curved hand shape like holding a cup',
  },
  D: {
    fingerStates: { thumb: false, index: true, middle: false, ring: false, pinky: false },
    geometricChecks: [
      // Thumb tip touches middle finger MCP area
      (lm) => {
        const dist = euclideanDistance3D(lm[4], lm[10]);
        return dist < 0.06 ? 1 : Math.max(0, 1 - (dist - 0.06) / 0.06);
      },
    ],
    description: 'Index finger up, thumb touches middle finger',
  },
  E: {
    fingerStates: { thumb: false, index: false, middle: false, ring: false, pinky: false },
    description: 'All fingers curled into fist, thumb tucked',
  },
  I: {
    fingerStates: { thumb: false, index: false, middle: false, ring: false, pinky: true },
    description: 'Only pinky finger extended',
  },
  L: {
    fingerStates: { thumb: true, index: true, middle: false, ring: false, pinky: false },
    geometricChecks: [
      // Thumb and index should be roughly perpendicular (L shape)
      (lm) => {
        const thumbDir = { x: lm[4].x - lm[2].x, y: lm[4].y - lm[2].y };
        const indexDir = { x: lm[8].x - lm[5].x, y: lm[8].y - lm[5].y };
        const dot = thumbDir.x * indexDir.x + thumbDir.y * indexDir.y;
        const mag1 = Math.sqrt(thumbDir.x ** 2 + thumbDir.y ** 2) || 0.001;
        const mag2 = Math.sqrt(indexDir.x ** 2 + indexDir.y ** 2) || 0.001;
        const cosAngle = Math.abs(dot / (mag1 * mag2));
        // Should be near 0 (perpendicular), so low cosine = high score
        return 1 - cosAngle;
      },
    ],
    description: 'Thumb and index extended at right angle',
  },
  O: {
    fingerStates: { thumb: true, index: true, middle: false, ring: false, pinky: false },
    geometricChecks: [
      // Thumb and index tips should be touching (O shape)
      (lm) => {
        const dist = euclideanDistance3D(lm[4], lm[8]);
        return dist < 0.04 ? 1 : Math.max(0, 1 - (dist - 0.04) / 0.04);
      },
    ],
    description: 'Thumb and index tips touching in circle',
  },
  U: {
    fingerStates: { thumb: false, index: true, middle: true, ring: false, pinky: false },
    geometricChecks: [
      // Index and middle should be close together (parallel)
      (lm) => {
        const dist = euclideanDistance3D(lm[8], lm[12]);
        return dist < 0.04 ? 1 : Math.max(0, 1 - (dist - 0.04) / 0.04);
      },
    ],
    description: 'Index and middle fingers extended together',
  },
  V: {
    fingerStates: { thumb: false, index: true, middle: true, ring: false, pinky: false },
    geometricChecks: [
      // Index and middle should be spread apart (V shape)
      (lm) => {
        const dist = euclideanDistance3D(lm[8], lm[12]);
        return dist > 0.05 ? 1 : dist / 0.05;
      },
    ],
    description: 'Index and middle fingers spread in V',
  },
  W: {
    fingerStates: { thumb: false, index: true, middle: true, ring: true, pinky: false },
    description: 'Index, middle, and ring fingers extended',
  },
  Y: {
    fingerStates: { thumb: true, index: false, middle: false, ring: false, pinky: true },
    description: 'Thumb and pinky extended (shaka)',
  },
};

const LETTERS = Object.keys(LETTER_DEFINITIONS);
const FINGER_STATE_WEIGHT = 0.6;
const GEOMETRIC_WEIGHT = 0.4;
const CONFIDENCE_THRESHOLD = 0.7;
const CONFIRMATION_FRAMES = 20; // Fewer frames than gesture detection for faster spelling
const STORAGE_KEY = 'mlaf_fingerspelling_v1';

// =============================================================================
// FINGERSPELLING RECOGNIZER
// =============================================================================

export class FingerspellingRecognizer {
  constructor(config = {}) {
    this._confidenceThreshold = config.confidenceThreshold || CONFIDENCE_THRESHOLD;
    this._confirmationFrames = config.confirmationFrames || CONFIRMATION_FRAMES;

    // Detection state
    this._candidateLetter = null;
    this._candidateFrames = 0;
    this._lastConfirmedLetter = null;
    this._cooldownFrames = 0;

    // Spelling buffer
    this._spellingBuffer = [];
    this._isSpelling = false;

    // Letter mastery tracking
    this._letterMastery = {};

    this._load();
  }

  // ===========================================================================
  // LETTER DETECTION
  // ===========================================================================

  /**
   * Detect a fingerspelling letter from hand landmarks.
   * @param {Array} landmarks - 21-point hand landmarks
   * @returns {{ letter: string|null, confidence: number, allScores: Object, confirmed: boolean }}
   */
  detectLetter(landmarks) {
    if (!landmarks || landmarks.length < 21) {
      this._candidateLetter = null;
      this._candidateFrames = 0;
      return { letter: null, confidence: 0, allScores: {}, confirmed: false };
    }

    // Cooldown after confirming a letter (prevent double-detection)
    if (this._cooldownFrames > 0) {
      this._cooldownFrames--;
      return { letter: this._lastConfirmedLetter, confidence: 1, allScores: {}, confirmed: false };
    }

    const { states } = analyzeFingerStatesDetailed(landmarks);
    const allScores = {};
    let bestLetter = null;
    let bestScore = 0;

    for (const letter of LETTERS) {
      const def = LETTER_DEFINITIONS[letter];
      let score = this._scoreFingerStates(states, def.fingerStates);

      // Apply geometric checks if defined
      if (def.geometricChecks && def.geometricChecks.length > 0) {
        let geoScore = 0;
        for (const check of def.geometricChecks) {
          try {
            geoScore += check(landmarks);
          } catch {
            geoScore += 0.5; // Neutral on error
          }
        }
        geoScore /= def.geometricChecks.length;
        score = score * FINGER_STATE_WEIGHT + geoScore * GEOMETRIC_WEIGHT;
      }

      allScores[letter] = Math.round(score * 100) / 100;

      if (score > bestScore) {
        bestScore = score;
        bestLetter = letter;
      }
    }

    // Temporal confirmation
    let confirmed = false;
    if (bestScore >= this._confidenceThreshold) {
      if (bestLetter === this._candidateLetter) {
        this._candidateFrames++;
      } else {
        this._candidateLetter = bestLetter;
        this._candidateFrames = 1;
      }

      if (this._candidateFrames >= this._confirmationFrames) {
        confirmed = true;
        this._lastConfirmedLetter = bestLetter;
        this._candidateLetter = null;
        this._candidateFrames = 0;
        this._cooldownFrames = 10; // Brief cooldown

        // Auto-add to spelling buffer if active
        if (this._isSpelling) {
          this._spellingBuffer.push(bestLetter);
        }
      }
    } else {
      this._candidateLetter = null;
      this._candidateFrames = 0;
    }

    return {
      letter: bestLetter,
      confidence: bestScore,
      allScores,
      confirmed,
      candidateProgress: this._candidateFrames / this._confirmationFrames,
    };
  }

  /**
   * Score how well finger states match a letter definition.
   * @private
   */
  _scoreFingerStates(actual, expected) {
    let matches = 0;
    let total = 0;

    for (const finger of ['thumb', 'index', 'middle', 'ring', 'pinky']) {
      if (expected[finger] === null || expected[finger] === undefined) continue;
      total++;
      if (actual[finger] === expected[finger]) matches++;
    }

    return total > 0 ? matches / total : 0.5;
  }

  // ===========================================================================
  // SPELLING BUFFER
  // ===========================================================================

  startSpelling() {
    this._isSpelling = true;
    this._spellingBuffer = [];
  }

  stopSpelling() {
    this._isSpelling = false;
  }

  addLetter(letter) {
    if (LETTERS.includes(letter)) {
      this._spellingBuffer.push(letter);
    }
  }

  backspace() {
    this._spellingBuffer.pop();
  }

  clearSpelling() {
    this._spellingBuffer = [];
  }

  getSpelledWord() {
    return this._spellingBuffer.join('');
  }

  getSpellingBuffer() {
    return [...this._spellingBuffer];
  }

  /**
   * Check if the spelled word matches any LEXICON entry.
   * @param {string} [word] - word to check (defaults to current buffer)
   * @returns {{ matched: boolean, grammarId: string|null, suggestion: string|null }}
   */
  checkWord(word) {
    const target = (word || this.getSpelledWord()).toLowerCase();
    if (!target) return { matched: false, grammarId: null, suggestion: null };

    // Exact match against LEXICON display forms
    for (const [grammarId, entry] of Object.entries(LEXICON)) {
      if (entry.display.toLowerCase() === target) {
        return { matched: true, grammarId, suggestion: null };
      }
    }

    // Partial match / suggestion
    let bestMatch = null;
    let bestDist = Infinity;
    for (const [grammarId, entry] of Object.entries(LEXICON)) {
      const display = entry.display.toLowerCase();
      if (display.startsWith(target) || target.startsWith(display)) {
        const dist = Math.abs(display.length - target.length);
        if (dist < bestDist) {
          bestDist = dist;
          bestMatch = { grammarId, display: entry.display };
        }
      }
    }

    return {
      matched: false,
      grammarId: null,
      suggestion: bestMatch ? `Did you mean "${bestMatch.display}"?` : null,
    };
  }

  // ===========================================================================
  // LETTER MASTERY
  // ===========================================================================

  recordLetterAttempt(letter, success) {
    if (!this._letterMastery[letter]) {
      this._letterMastery[letter] = { attempts: 0, successes: 0, mastered: false };
    }
    const m = this._letterMastery[letter];
    m.attempts++;
    if (success) m.successes++;
    if (m.successes >= 5 && m.successes / m.attempts >= 0.8) {
      m.mastered = true;
    }
    this._save();
  }

  getLetterMastery() {
    const result = {};
    for (const letter of LETTERS) {
      result[letter] = this._letterMastery[letter] || { attempts: 0, successes: 0, mastered: false };
    }
    return result;
  }

  // ===========================================================================
  // REPORT & PERSISTENCE
  // ===========================================================================

  getReport() {
    const mastery = this.getLetterMastery();
    const totalLettersLearned = LETTERS.filter(l => mastery[l].attempts > 0).length;
    const masteredLetters = LETTERS.filter(l => mastery[l].mastered).length;

    return {
      totalLettersLearned,
      masteredLetters,
      totalLetters: LETTERS.length,
      currentWord: this.getSpelledWord(),
      isSpelling: this._isSpelling,
      letterMastery: mastery,
    };
  }

  reset() {
    this._letterMastery = {};
    this._spellingBuffer = [];
    this._isSpelling = false;
    this._candidateLetter = null;
    this._candidateFrames = 0;
    this._lastConfirmedLetter = null;
    this._cooldownFrames = 0;
    try { localStorage.removeItem(STORAGE_KEY); } catch {}
  }

  _save() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        letterMastery: this._letterMastery,
      }));
    } catch {}
  }

  _load() {
    try {
      const data = JSON.parse(localStorage.getItem(STORAGE_KEY));
      if (data?.letterMastery) this._letterMastery = data.letterMastery;
    } catch {}
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

export const SUPPORTED_LETTERS = LETTERS;

export function getLetterDefinition(letter) {
  return LETTER_DEFINITIONS[letter] || null;
}

export function letterMasteryColor(mastery) {
  if (mastery.mastered) return '#4ade80';
  if (mastery.successes > 0) return '#fbbf24';
  return '#4a4a6a';
}
