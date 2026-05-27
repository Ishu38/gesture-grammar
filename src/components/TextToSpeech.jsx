/**
 * TextToSpeech.jsx — Speech output for specially-abled users
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * For speech-impaired users, this IS their voice — the system speaks
 * the constructed sentence aloud using the Web Speech API.
 * Prefers high-quality female voices for natural, clear speech.
 */

import { useState, useEffect, useCallback, useRef } from 'react';

/** Three languages MLAF supports natively. Codes match what the Web Speech
 *  API exposes on every major desktop OS (Chrome / Edge / Firefox).
 *  Bengali ("bn") may be missing on a stock Windows install — the OS-level
 *  TTS engine has to be installed in Settings → Time & language → Speech.
 *  When unavailable the picker shows a helpful empty-state message. */
const LANGUAGES = [
  { code: 'en', label: 'English', defaultLang: 'en-IN', voicePrefixes: ['en'] },
  { code: 'hi', label: 'हिन्दी',  defaultLang: 'hi-IN', voicePrefixes: ['hi'] },
  { code: 'bn', label: 'বাংলা',   defaultLang: 'bn-IN', voicePrefixes: ['bn'] },
];

/** Voice-name fragments that strongly imply a female voice. Pulled from
 *  the Google / Microsoft / Apple TTS catalogues across the three target
 *  languages. The "female" / "woman" generic strings catch most third-
 *  party engines. */
const FEMALE_VOICE_PATTERNS = [
  // English
  'Google UK English Female', 'Google US English Female',
  'Microsoft Zira', 'Microsoft Aria', 'Microsoft Jenny',
  'Samantha', 'Karen', 'Moira', 'Fiona', 'Tessa', 'Veena',
  // Hindi
  'Google हिन्दी', 'Microsoft Heera', 'Microsoft Kalpana', 'Lekha',
  // Bengali
  'Microsoft Tanishaa', 'Microsoft Bashkar', 'Google বাংলা',
  // Generic fallbacks
  'female', 'Female', 'woman', 'Woman',
];

const MALE_VOICE_PATTERNS = [
  'Google UK English Male', 'Google US English Male',
  'Microsoft Mark', 'Microsoft David', 'Microsoft Guy', 'Microsoft Ravi',
  'Microsoft Hemant', 'Microsoft Madhur',
  'Daniel', 'Alex', 'Fred', 'Rishi',
  'male', 'Male', 'man', 'Man',
];

/** Pick the best voice matching (lang, gender). Returns null if none. */
function pickBestVoice(voices, langCode = 'en', gender = 'female') {
  if (!voices || voices.length === 0) return null;
  const langPool = voices.filter(v => v.lang.toLowerCase().startsWith(langCode));
  if (langPool.length === 0) return null;
  const namePatterns = gender === 'male' ? MALE_VOICE_PATTERNS : FEMALE_VOICE_PATTERNS;
  for (const pattern of namePatterns) {
    const match = langPool.find(v => v.name.includes(pattern));
    if (match) return match;
  }
  // No gendered match — return the first voice for that language
  return langPool[0];
}

function TextToSpeech({ sentence, isComplete, autoSpeak, enabled }) {
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [voices, setVoices] = useState([]);
  const [selectedVoice, setSelectedVoice] = useState(null);
  const [rate, setRate] = useState(0.85); // Slightly slower for clarity
  const [pitch, setPitch] = useState(1.05); // Slightly higher = more feminine
  const [showVoiceSelector, setShowVoiceSelector] = useState(false);
  const [language, setLanguage] = useState('en');           // 'en' | 'hi' | 'bn'
  const [gender, setGender]     = useState('female');       // 'female' | 'male'
  const lastSpokenRef = useRef('');

  // Load available voices, and whenever the user changes language or gender,
  // re-pick the best matching voice. Web Speech API populates voices
  // asynchronously on Chrome, hence onvoiceschanged.
  useEffect(() => {
    if (!enabled || !('speechSynthesis' in window)) return;

    function loadVoices() {
      const available = speechSynthesis.getVoices();
      if (available.length === 0) return;
      setVoices(available);
      const best = pickBestVoice(available, language, gender);
      if (best) setSelectedVoice(best);
    }

    loadVoices();
    speechSynthesis.onvoiceschanged = loadVoices;

    return () => {
      speechSynthesis.onvoiceschanged = null;
      speechSynthesis.cancel();
    };
  }, [enabled, language, gender]);

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

      {/* Language + gender pickers — sustainable Web Speech API, no model
          download, works wherever the OS provides voices for the chosen
          language. */}
      <div style={{
        marginTop: '0.5rem', display: 'flex', gap: '0.6rem',
        alignItems: 'center', flexWrap: 'wrap',
      }}>
        <div style={{ display: 'flex', gap: '0.2rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.55rem', color: '#64748b', marginRight: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Lang
          </span>
          {LANGUAGES.map(l => (
            <button
              key={l.code}
              onClick={() => setLanguage(l.code)}
              style={{
                padding: '3px 9px',
                borderRadius: 4,
                fontSize: '0.62rem',
                fontWeight: 600,
                background: language === l.code ? '#FF5252' : 'transparent',
                color: language === l.code ? '#FFF6E6' : '#94a3b8',
                border: `1px solid ${language === l.code ? '#FF5252' : 'rgba(255,255,255,0.12)'}`,
                cursor: 'pointer',
              }}
            >
              {l.label}
            </button>
          ))}
        </div>
        <div style={{ display: 'flex', gap: '0.2rem', alignItems: 'center' }}>
          <span style={{ fontSize: '0.55rem', color: '#64748b', marginRight: 4, textTransform: 'uppercase', letterSpacing: '0.08em' }}>
            Voice
          </span>
          {['female', 'male'].map(g => (
            <button
              key={g}
              onClick={() => setGender(g)}
              style={{
                padding: '3px 9px',
                borderRadius: 4,
                fontSize: '0.62rem',
                fontWeight: 600,
                background: gender === g ? '#A855F7' : 'transparent',
                color: gender === g ? '#FFF6E6' : '#94a3b8',
                border: `1px solid ${gender === g ? '#A855F7' : 'rgba(255,255,255,0.12)'}`,
                cursor: 'pointer',
                textTransform: 'capitalize',
              }}
            >
              {g}
            </button>
          ))}
        </div>
      </div>

      {/* Voice selector dropdown — filtered by selected language */}
      {showVoiceSelector && (() => {
        const langVoices = voices.filter(v => v.lang.toLowerCase().startsWith(language));
        return (
          <div style={{
            marginTop: '0.5rem', maxHeight: '160px', overflowY: 'auto',
            background: 'rgba(0,0,0,0.6)', borderRadius: 6, border: '1px solid rgba(255,255,255,0.08)',
          }}>
            {langVoices.map(v => (
              <div
                key={v.name + v.lang}
                onClick={() => { setSelectedVoice(v); setShowVoiceSelector(false); }}
                style={{
                  padding: '4px 8px', fontSize: '0.6rem',
                  color: v === selectedVoice ? '#4ade80' : '#94a3b8',
                  cursor: 'pointer',
                  background: v === selectedVoice ? 'rgba(74,222,128,0.08)' : 'transparent',
                }}
              >
                {v.name} ({v.lang})
              </div>
            ))}
            {langVoices.length === 0 && (
              <div style={{ padding: '8px', fontSize: '0.6rem', color: '#64748b', textAlign: 'center', lineHeight: 1.5 }}>
                No {LANGUAGES.find(l => l.code === language)?.label} voice installed.<br/>
                Add one in: Windows → Settings → Time &amp; language → Speech → Add voices.
              </div>
            )}
          </div>
        );
      })()}
    </div>
  );
}

export default TextToSpeech;
