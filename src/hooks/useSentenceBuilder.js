/**
 * useSentenceBuilder.js
 * Custom React hook for managing sentence construction with gesture input
 * Handles locking, debouncing, confidence tracking, and tense modification
 */

import { useState, useRef, useCallback, useEffect } from 'react';
import { LEXICON, getCorrectVerbForm } from '../utils/GrammarEngine';

// Configuration constants
const LOCK_THRESHOLD_FRAMES = 45; // ~1.5 seconds at 30fps
const DEBOUNCE_DURATION_MS = 2000; // 2 seconds cooldown after adding word
const NEUTRAL_TIMEOUT_MS = 500; // Time with no gesture to reset

// Tense zones based on wrist Y position
const TENSE_ZONES = {
  FUTURE: { min: 0, max: 0.3, label: 'Future', suffix: 'will' },
  PRESENT: { min: 0.3, max: 0.7, label: 'Present', suffix: null },
  PAST: { min: 0.7, max: 1.0, label: 'Past', suffix: 'ed' },
};

// Audio context for sound effects
let audioContext = null;

function playSuccessSound() {
  try {
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

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
    if (!audioContext) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }

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
    case 'PAST':
      // Simple past tense transformation
      let pastForm = baseDisplay;
      if (baseDisplay.endsWith('e')) {
        pastForm = baseDisplay + 'd';
      } else if (baseDisplay.endsWith('op')) {
        pastForm = baseDisplay + 'ped'; // stop -> stopped
      } else if (baseDisplay.endsWith('ab')) {
        pastForm = baseDisplay + 'bed'; // grab -> grabbed
      } else {
        pastForm = baseDisplay + 'ed';
      }
      return {
        grammarId,
        display: pastForm,
        tense: 'past',
      };
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
 */
export function useSentenceBuilder() {
  // Sentence state
  const [sentence, setSentence] = useState([]);

  // Lock state
  const [isLocked, setIsLocked] = useState(false);
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
    setIsLocked(true);
    debounceTimeoutRef.current = setTimeout(() => {
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

    // If locked (in debounce period), ignore input but allow neutral reset
    if (isLocked) {
      if (!gesture) {
        // User dropped hand - can reset debounce early
        if (neutralTimeoutRef.current) {
          clearTimeout(neutralTimeoutRef.current);
        }
        neutralTimeoutRef.current = setTimeout(() => {
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

    // Update progress
    const progress = Math.min(confidenceCounterRef.current / LOCK_THRESHOLD_FRAMES, 1);
    setLockProgress(progress);

    // Check if threshold reached
    if (confidenceCounterRef.current >= LOCK_THRESHOLD_FRAMES) {
      // Determine tense for verbs
      const tense = LEXICON[gesture]?.type === 'VERB' ? getTenseZone(wristY) : 'PRESENT';
      addToSentence(gesture, tense);
    }
  }, [isLocked, addToSentence]);

  /**
   * Force add a word (bypass confidence check)
   */
  const forceAddWord = useCallback((grammarId, tense = 'PRESENT') => {
    if (isLocked) return false;
    return addToSentence(grammarId, tense);
  }, [isLocked, addToSentence]);

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

    // Utilities
    getTenseZone,
    TENSE_ZONES,
  };
}

export default useSentenceBuilder;
