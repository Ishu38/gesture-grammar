/**
 * useSentenceBuilder.js
 * Custom React hook for managing sentence construction with gesture input
 * Handles locking, debouncing, confidence tracking, and tense modification
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { LEXICON, getCorrectVerbForm } from '../utils/GrammarEngine';
import { IRREGULAR_VERBS } from '../utils/GrammarRules';

// Default configuration constants
const DEFAULT_LOCK_THRESHOLD_FRAMES = 45; // ~1.5 seconds at 30fps
const DEBOUNCE_DURATION_MS = 2000; // 2 seconds cooldown after adding word
const NEUTRAL_TIMEOUT_MS = 500; // Time with no gesture to reset

// Tense zones based on wrist Y position
const TENSE_ZONES = {
  FUTURE: { min: 0, max: 0.3, label: 'Future', suffix: 'will' },
  PRESENT: { min: 0.3, max: 0.7, label: 'Present', suffix: null },
  PAST: { min: 0.7, max: 1.0, label: 'Past', suffix: 'ed' },
};

// Per-gesture adaptive lock thresholds (fraction of base threshold)
// Distinctive gestures need fewer frames; ambiguous ones need more
const GESTURE_LOCK_MULTIPLIER = {
  STOP: 0.5,          // Very distinctive (open palm) — 50% of base
  GRAB: 0.6,          // Distinctive (claw/pinch) — 60%
  HOUSE: 0.6,         // Distinctive (roof shape)
  WATER: 0.65,        // Fairly distinctive (W shape)
  SUBJECT_I: 0.7,     // Fist + thumb — recognizable
  SUBJECT_HE: 0.7,    // Hitchhiker thumb
  SEE: 0.75,          // V-shape — fairly distinctive but can overlap with WE
  DRINK: 0.85,        // C-shape, can be confused with YOU
  SUBJECT_YOU: 0.85,  // Index point, can be confused with DRINK
  APPLE: 0.9,         // Cupped hand — subtle
  BOOK: 0.9,          // Flat palm — subtle
};

// Audio context singleton — shared across hook instances, never closed
// (closing breaks audio for other mounted instances; browsers manage GC)
let _sharedAudioContext = null;

function getAudioContext() {
  if (!_sharedAudioContext || _sharedAudioContext.state === 'closed') {
    _sharedAudioContext = new (window.AudioContext || window.webkitAudioContext)();
  }
  return _sharedAudioContext;
}

function playSuccessSound() {
  try {
    const audioContext = getAudioContext();

    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    // Pleasant "ding" sound - ascending notes
    oscillator.frequency.setValueAtTime(523.25, audioContext.currentTime); // C5
    oscillator.frequency.setValueAtTime(659.25, audioContext.currentTime + 0.1); // E5
    oscillator.frequency.setValueAtTime(783.99, audioContext.currentTime + 0.2); // G5
    oscillator.type = 'sine';

    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.4);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.4);
  } catch (e) {
    console.warn('Could not play success sound:', e);
  }
}

function playErrorSound() {
  try {
    const audioContext = getAudioContext();

    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 200;
    oscillator.type = 'sawtooth';

    gainNode.gain.setValueAtTime(0.2, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
  } catch (e) {
    console.warn('Could not play error sound:', e);
  }
}

/**
 * Get tense zone from wrist Y position
 */
function getTenseZone(wristY) {
  if (wristY < TENSE_ZONES.FUTURE.max) return 'FUTURE';
  if (wristY < TENSE_ZONES.PAST.min) return 'PRESENT';
  return 'PAST';
}

/**
 * Apply tense modification to a verb
 */
function applyTenseToVerb(grammarId, tense) {
  const lexiconEntry = LEXICON[grammarId];
  if (!lexiconEntry || lexiconEntry.type !== 'VERB') {
    return { grammarId, display: lexiconEntry?.display || grammarId, tense: null };
  }

  const baseDisplay = lexiconEntry.display;

  switch (tense) {
    case 'FUTURE':
      return {
        grammarId,
        display: `will ${baseDisplay}`,
        tense: 'future',
        auxiliary: 'will',
      };
    case 'PAST': {
      // Use IRREGULAR_VERBS from GrammarRules.js (single source of truth)
      // Strip S-form suffix before lookup: "drinks" → "drink", "goes" → "go"
      let lower = baseDisplay.toLowerCase();
      // Check if the base form (without S-form suffix) has an irregular entry
      let irregular = IRREGULAR_VERBS[lower];
      if (!irregular) {
        // Try stripping common S-form suffixes: -es, -s
        const stripped = lower.endsWith('es') ? lower.slice(0, -2)
          : lower.endsWith('s') ? lower.slice(0, -1)
          : null;
        if (stripped && IRREGULAR_VERBS[stripped]) {
          irregular = IRREGULAR_VERBS[stripped];
          lower = stripped;
        }
      }
      // Use the base form (without S-suffix) for past tense construction
      const baseForPast = lower;
      let pastForm;
      if (irregular?.past) {
        pastForm = irregular.past;
      } else if (baseForPast.endsWith('e')) {
        pastForm = baseForPast + 'd';
      } else if (baseForPast.endsWith('op')) {
        pastForm = baseForPast + 'ped'; // stop -> stopped
      } else if (baseForPast.endsWith('ab')) {
        pastForm = baseForPast + 'bed'; // grab -> grabbed
      } else {
        pastForm = baseForPast + 'ed';
      }
      return {
        grammarId,
        display: pastForm,
        tense: 'past',
      };
    }
    case 'PRESENT':
    default:
      return {
        grammarId,
        display: baseDisplay,
        tense: 'present',
      };
  }
}

/**
 * Custom hook for building sentences with gesture input
 * @param {object} options
 * @param {number} options.confidenceFrames — frames required to lock a gesture (from AccessibilityProfile)
 */
export function useSentenceBuilder(options = {}) {
  // Stored in a ref so CognitiveLoadAdapter can update it without re-rendering
  const lockThresholdRef = useRef(options.confidenceFrames || DEFAULT_LOCK_THRESHOLD_FRAMES);
  // Sentence state
  const [sentence, setSentence] = useState([]);

  // Lock state — use a ref alongside state to avoid stale closures in animation loops.
  // The ref is the source of truth; the state drives React re-renders.
  const [isLocked, setIsLocked] = useState(false);
  const isLockedRef = useRef(false);
  const [lockProgress, setLockProgress] = useState(0);

  // Current gesture tracking
  const [currentGesture, setCurrentGesture] = useState(null);
  const [currentTenseZone, setCurrentTenseZone] = useState('PRESENT');

  // Refs for tracking
  const confidenceCounterRef = useRef(0);
  const previousGestureRef = useRef(null);
  const debounceTimeoutRef = useRef(null);
  const neutralTimeoutRef = useRef(null);
  const lastWristYRef = useRef(0.5);

  /**
   * Clear the sentence
   */
  const clearSentence = useCallback(() => {
    setSentence([]);
    setIsLocked(false);
    isLockedRef.current = false;
    setLockProgress(0);
    confidenceCounterRef.current = 0;
    previousGestureRef.current = null;

    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
      debounceTimeoutRef.current = null;
    }
  }, []);

  /**
   * Remove the last word from the sentence
   */
  const undoLastWord = useCallback(() => {
    setSentence(prev => prev.slice(0, -1));
  }, []);

  /**
   * Add a word to the sentence
   */
  const addToSentence = useCallback((grammarId, tense = 'PRESENT') => {
    const lexiconEntry = LEXICON[grammarId];
    if (!lexiconEntry) {
      console.warn(`Unknown grammar ID: ${grammarId}`);
      return false;
    }

    setSentence(prev => {
      // Auto-conjugate verb for subject-verb agreement (present tense only)
      let effectiveGrammarId = grammarId;
      if (lexiconEntry.type === 'VERB' && tense === 'PRESENT' && prev.length > 0) {
        const subject = prev.find(w => w.type === 'SUBJECT');
        if (subject) {
          effectiveGrammarId = getCorrectVerbForm(grammarId, subject.grammar_id);
        }
      }

      const effectiveEntry = LEXICON[effectiveGrammarId] || lexiconEntry;

      // Apply tense modification for verbs
      let wordData;
      if (effectiveEntry.type === 'VERB') {
        const tensedVerb = applyTenseToVerb(effectiveGrammarId, tense);
        wordData = {
          id: Date.now(),
          word: tensedVerb.display,
          type: effectiveEntry.type,
          grammar_id: effectiveGrammarId,
          tense: tensedVerb.tense,
          auxiliary: tensedVerb.auxiliary,
          ...effectiveEntry,
        };
      } else {
        wordData = {
          id: Date.now(),
          word: effectiveEntry.display,
          type: effectiveEntry.type,
          grammar_id: effectiveGrammarId,
          ...effectiveEntry,
        };
      }

      return [...prev, wordData];
    });

    playSuccessSound();

    // Enter debounce period
    isLockedRef.current = true;
    setIsLocked(true);
    debounceTimeoutRef.current = setTimeout(() => {
      isLockedRef.current = false;
      setIsLocked(false);
    }, DEBOUNCE_DURATION_MS);

    // Reset confidence
    confidenceCounterRef.current = 0;
    setLockProgress(0);

    return true;
  }, []);

  /**
   * Process gesture input (called every frame)
   */
  const processGestureInput = useCallback((gesture, wristY = 0.5) => {
    setCurrentGesture(gesture);
    lastWristYRef.current = wristY;

    // Update tense zone based on wrist position
    if (gesture && LEXICON[gesture]?.type === 'VERB') {
      setCurrentTenseZone(getTenseZone(wristY));
    }

    // If locked (in debounce period), ignore input but allow neutral reset.
    // Read from ref (not state) to avoid stale closures in animation loops.
    if (isLockedRef.current) {
      if (!gesture) {
        // User dropped hand - can reset debounce early
        if (neutralTimeoutRef.current) {
          clearTimeout(neutralTimeoutRef.current);
        }
        neutralTimeoutRef.current = setTimeout(() => {
          isLockedRef.current = false;
          setIsLocked(false);
          if (debounceTimeoutRef.current) {
            clearTimeout(debounceTimeoutRef.current);
            debounceTimeoutRef.current = null;
          }
        }, NEUTRAL_TIMEOUT_MS);
      }
      return;
    }

    // Clear neutral timeout if gesture detected
    if (gesture && neutralTimeoutRef.current) {
      clearTimeout(neutralTimeoutRef.current);
      neutralTimeoutRef.current = null;
    }

    // No gesture - reset confidence
    if (!gesture) {
      confidenceCounterRef.current = 0;
      previousGestureRef.current = null;
      setLockProgress(0);
      return;
    }

    // Same gesture as before - increment confidence
    if (gesture === previousGestureRef.current) {
      confidenceCounterRef.current++;
    } else {
      // Different gesture - reset
      confidenceCounterRef.current = 1;
      previousGestureRef.current = gesture;
    }

    // Adaptive threshold: distinctive gestures lock faster
    const gestureMultiplier = GESTURE_LOCK_MULTIPLIER[gesture] || 1.0;
    const effectiveThreshold = Math.max(10, Math.round(lockThresholdRef.current * gestureMultiplier));

    // Update progress
    const progress = Math.min(confidenceCounterRef.current / effectiveThreshold, 1);
    setLockProgress(progress);

    // Check if threshold reached
    if (confidenceCounterRef.current >= effectiveThreshold) {
      // Determine tense for verbs
      const tense = LEXICON[gesture]?.type === 'VERB' ? getTenseZone(wristY) : 'PRESENT';
      addToSentence(gesture, tense);
    }
  }, [addToSentence]);

  /**
   * Force add a word (bypass confidence check)
   */
  const forceAddWord = useCallback((grammarId, tense = 'PRESENT') => {
    if (isLockedRef.current) return false;
    return addToSentence(grammarId, tense);
  }, [addToSentence]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
      if (neutralTimeoutRef.current) {
        clearTimeout(neutralTimeoutRef.current);
      }
    };
  }, []);

  return {
    // State
    sentence,
    isLocked,
    lockProgress,
    currentGesture,
    currentTenseZone,

    // Actions
    processGestureInput,
    addToSentence: forceAddWord,
    clearSentence,
    undoLastWord,

    // Dynamic threshold control (used by CognitiveLoadAdapter)
    setConfidenceThreshold: (frames) => {
      lockThresholdRef.current = Math.max(10, Math.min(120, Math.round(frames)));
    },

    // Utilities
    getTenseZone,
    TENSE_ZONES,
  };
}

export default useSentenceBuilder;
