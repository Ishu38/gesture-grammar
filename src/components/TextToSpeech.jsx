/**
 * TextToSpeech.jsx — Speech output for specially-abled users
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * For speech-impaired users, this IS their voice — the system speaks
 * the constructed sentence aloud using the Web Speech API.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

function TextToSpeech({ sentence, isComplete, autoSpeak, enabled }) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [rate, setRate] = useState(0.9);
  const lastSpokenRef = useRef('');

  // Load available voices
  useEffect(() => {
    if (!enabled || !('speechSynthesis' in window)) return;

    function loadVoices() {
      const available = speechSynthesis.getVoices();
      setVoices(available);

      // Default to first English voice
      const englishVoice = available.find(v => v.lang.startsWith('en'));
      if (englishVoice && !selectedVoice) {
        setSelectedVoice(englishVoice);
      }
    }

    loadVoices();
    speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      speechSynthesis.onvoiceschanged = null;
      speechSynthesis.cancel();
    };
  }, [enabled]);

  // Build readable sentence from token array
  const readableSentence = sentence.map(w => w.word || w.display || w.grammar_id).join(' ');

  const speak = useCallback((text) => {
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    if (selectedVoice) utterance.voice = selectedVoice;
    utterance.rate = rate;
    utterance.pitch = 1.0;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    speechSynthesis.speak(utterance);
  }, [selectedVoice, rate]);

  // Auto-speak when sentence is complete
  useEffect(() => {
    if (!enabled || !autoSpeak || !isComplete) return;
    if (readableSentence === lastSpokenRef.current) return;
    if (!readableSentence.trim()) return;

    speak(readableSentence);
    lastSpokenRef.current = readableSentence;
  }, [isComplete, readableSentence, autoSpeak, enabled, speak]);

  const stopSpeaking = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  }, []);

  if (!enabled) return null;

  return (
    <div className="tts-panel" role="region" aria-label="Text to Speech Controls">
      <div className="tts-header">
        <span className="tts-icon" aria-hidden="true">{'\u{1F50A}'}</span>
        <span className="tts-title">Voice Output</span>
      </div>

      <div className="tts-sentence" aria-live="polite">
        {readableSentence || 'Build a sentence to speak...'}
      </div>

      <div className="tts-controls">
        <button
          className={`tts-speak-btn ${isSpeaking ? 'speaking' : ''}`}
          onClick={() => isSpeaking ? stopSpeaking() : speak(readableSentence)}
          disabled={!readableSentence.trim()}
          aria-label={isSpeaking ? 'Stop speaking' : 'Speak sentence'}
        >
          {isSpeaking ? 'Stop' : 'Speak'}
        </button>

        <div className="tts-rate">
          <label htmlFor="tts-rate-slider" className="tts-rate-label">Speed</label>
          <input
            id="tts-rate-slider"
            type="range"
            min="0.5"
            max="1.5"
            step="0.1"
            value={rate}
            onChange={(e) => setRate(parseFloat(e.target.value))}
            aria-label="Speech rate"
          />
        </div>
      </div>
    </div>
  );
}

export default TextToSpeech;
