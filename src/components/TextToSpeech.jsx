/**
 * TextToSpeech.jsx — Speech output for specially-abled users
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * For speech-impaired users, this IS their voice — the system speaks
 * the constructed sentence aloud using the Web Speech API.
 * Prefers high-quality female voices for natural, clear speech.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/** Feminine-sounding English voice name fragments (ordered by quality) */
const FEMALE_VOICE_PATTERNS = [
  'Google UK English Female',
  'Google US English Female',
  'Microsoft Zira',
  'Samantha',
  'Karen',
  'Moira',
  'Fiona',
  'female',
  'Female',
  'woman',
  'Woman',
];

/** Fallback English voice patterns if no female voice found */
const FALLBACK_ENGLISH = ['en-GB', 'en-US', 'en-IN', 'en-AU', 'en'];

function pickBestVoice(voices) {
  if (!voices || voices.length === 0) return null;

  // Try female voices first
  for (const pattern of FEMALE_VOICE_PATTERNS) {
    const match = voices.find(v => v.name.includes(pattern));
    if (match) return match;
  }

  // Fallback: any English voice
  for (const lang of FALLBACK_ENGLISH) {
    const match = voices.find(v => v.lang.startsWith(lang));
    if (match) return match;
  }

  // Last resort: first available voice
  return voices[0] || null;
}

function TextToSpeech({ sentence, isComplete, autoSpeak, enabled }) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [rate, setRate] = useState(0.85); // Slightly slower for clarity
  const [pitch, setPitch] = useState(1.05); // Slightly higher = more feminine
  const [showVoiceSelector, setShowVoiceSelector] = useState(false);
  const lastSpokenRef = useRef('');

  // Load available voices and pick best female voice
  useEffect(() => {
    if (!enabled || !('speechSynthesis' in window)) return;

    function loadVoices() {
      const available = speechSynthesis.getVoices();
      if (available.length === 0) return;
      setVoices(available);

      // Auto-pick best female voice
      const best = pickBestVoice(available);
      if (best && !selectedVoice) {
        setSelectedVoice(best);
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
    if (!text.trim()) return;

    // Cancel any ongoing speech
    speechSynthesis.cancel();

    const utterance = new SpeechSynthesisUtterance(text);
    if (selectedVoice) utterance.voice = selectedVoice;
    utterance.rate = rate;
    utterance.pitch = pitch;
    utterance.volume = 1.0;

    utterance.onstart = () => setIsSpeaking(true);
    utterance.onend = () => setIsSpeaking(false);
    utterance.onerror = () => setIsSpeaking(false);

    speechSynthesis.speak(utterance);
  }, [selectedVoice, rate, pitch]);

  // Auto-speak when sentence is complete
  useEffect(() => {
    if (!enabled || !autoSpeak || !isComplete) return;
    if (readableSentence === lastSpokenRef.current) return;
    if (!readableSentence.trim()) return;

    // Small delay to let the UI settle before speaking
    const timer = setTimeout(() => {
      speak(readableSentence);
    }, 300);
    lastSpokenRef.current = readableSentence;
    return () => clearTimeout(timer);
  }, [isComplete, readableSentence, autoSpeak, enabled, speak]);

  const stopSpeaking = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      speechSynthesis.cancel();
    }
    setIsSpeaking(false);
  }, []);

  if (!enabled) return null;

  const voiceName = selectedVoice ? selectedVoice.name : 'Default';
  const voiceQuality = selectedVoice?.name?.includes('Google')
    ? 'HD' : selectedVoice?.name?.includes('Microsoft') || selectedVoice?.name?.includes('Samantha')
    ? 'Good' : 'Standard';

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
            max="1.3"
            step="0.05"
            value={rate}
            onChange={(e) => setRate(parseFloat(e.target.value))}
            aria-label="Speech rate"
          />
        </div>

        <button
          onClick={() => setShowVoiceSelector(!showVoiceSelector)}
          style={{
            fontSize: '0.55rem', padding: '2px 8px', borderRadius: 4,
            background: 'rgba(255,255,255,0.06)', border: '1px solid rgba(255,255,255,0.1)',
            color: '#94a3b8', cursor: 'pointer',
          }}
        >
          {voiceName.length > 14 ? voiceName.slice(0, 14) + '…' : voiceName}
          <span style={{ marginLeft: 4, color: selectedVoice?.name?.includes('Google') ? '#4ade80' : '#fbbf24', fontSize: '0.5rem' }}>
            {voiceQuality}
          </span>
        </button>
      </div>

      {/* Voice selector dropdown */}
      {showVoiceSelector && (
        <div style={{
          marginTop: '0.5rem', maxHeight: '160px', overflowY: 'auto',
          background: 'rgba(0,0,0,0.6)', borderRadius: 6, border: '1px solid rgba(255,255,255,0.08)',
        }}>
          {voices.filter(v => v.lang.startsWith('en')).map(v => (
            <div
              key={v.name + v.lang}
              onClick={() => { setSelectedVoice(v); setShowVoiceSelector(false); }}
              style={{
                padding: '4px 8px', fontSize: '0.6rem', color: v === selectedVoice ? '#4ade80' : '#94a3b8',
                cursor: 'pointer', background: v === selectedVoice ? 'rgba(74,222,128,0.08)' : 'transparent',
              }}
            >
              {v.name} ({v.lang})
            </div>
          ))}
          {voices.filter(v => v.lang.startsWith('en')).length === 0 && (
            <div style={{ padding: '8px', fontSize: '0.6rem', color: '#64748b', textAlign: 'center' }}>
              No English voices available. Install a TTS engine on your device.
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export default TextToSpeech;
