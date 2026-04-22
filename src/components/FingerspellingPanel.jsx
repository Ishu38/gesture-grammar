/**
 * FingerspellingPanel.jsx — ISL Fingerspelling Bridge
 * Two modes: "Learn" (practice individual letters) and "Spell" (spell words to add to sentence).
 */

import { useState, useCallback } from 'react';
import { SUPPORTED_LETTERS, getLetterDefinition, letterMasteryColor } from '../core/FingerspellingRecognizer';

// =============================================================================
// LETTER CARD (Learn Mode)
// =============================================================================

function LetterCard({ letter, mastery, isTarget, onSelect }) {
  const def = getLetterDefinition(letter);
  const borderColor = isTarget ? '#60a5fa' : letterMasteryColor(mastery);

  return (
    <button onClick={() => onSelect(letter)} style={{
      width: 48, height: 56, display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center', gap: 2,
      background: isTarget ? 'rgba(96, 165, 250, 0.15)' : '#1a1a2e',
      border: `2px solid ${borderColor}`, borderRadius: 8,
      cursor: 'pointer', transition: 'all 0.2s',
    }}>
      <span style={{ fontSize: 18, fontWeight: 700, color: '#e2e8f0' }}>{letter}</span>
      <span style={{ fontSize: 8, color: '#94a3b8' }}>
        {mastery.mastered ? '★' : mastery.successes > 0 ? '◐' : '○'}
      </span>
    </button>
  );
}

// =============================================================================
// LEARN MODE
// =============================================================================

function LearnMode({ targetLetter, setTargetLetter, detection, recognizer }) {
  const mastery = recognizer.getLetterMastery();
  const def = getLetterDefinition(targetLetter);
  const progress = detection?.candidateProgress || 0;
  const isDetecting = detection?.letter === targetLetter;

  return (
    <div style={{ padding: 12 }}>
      {/* Letter grid */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12, justifyContent: 'center' }}>
        {SUPPORTED_LETTERS.map(l => (
          <LetterCard key={l} letter={l} mastery={mastery[l]} isTarget={l === targetLetter} onSelect={setTargetLetter} />
        ))}
      </div>

      {/* Target letter display */}
      <div style={{
        background: '#1a1a2e', borderRadius: 10, padding: 16,
        border: `1px solid ${isDetecting ? '#4ade80' : '#3a3a5e'}`,
        textAlign: 'center', marginBottom: 12,
        transition: 'border-color 0.3s',
      }}>
        <div style={{ fontSize: 48, fontWeight: 700, color: '#e2e8f0', marginBottom: 4 }}>
          {targetLetter}
        </div>
        <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 8 }}>
          {def?.description || ''}
        </div>

        {/* Detection progress bar */}
        <div style={{ height: 4, background: '#2a2a3e', borderRadius: 2, overflow: 'hidden' }}>
          <div style={{
            width: `${(isDetecting ? progress : 0) * 100}%`,
            height: '100%',
            background: progress >= 0.8 ? '#4ade80' : '#60a5fa',
            borderRadius: 2,
            transition: 'width 0.1s',
          }} />
        </div>

        {detection?.confirmed && detection.letter === targetLetter && (
          <div style={{ color: '#4ade80', fontSize: 13, fontWeight: 700, marginTop: 8 }}>
            Correct!
          </div>
        )}
      </div>

      {/* Detection scores (compact) */}
      {detection?.allScores && Object.keys(detection.allScores).length > 0 && (
        <div style={{ fontSize: 10, color: '#64748b', textAlign: 'center' }}>
          Top: {Object.entries(detection.allScores)
            .sort((a, b) => b[1] - a[1])
            .slice(0, 3)
            .map(([l, s]) => `${l}:${s}`)
            .join(' ')}
        </div>
      )}
    </div>
  );
}

// =============================================================================
// SPELL MODE
// =============================================================================

function SpellMode({ recognizer, detection, onWordMatch }) {
  const buffer = recognizer.getSpellingBuffer();
  const word = recognizer.getSpelledWord();
  const check = word ? recognizer.checkWord() : null;

  const handleSubmitWord = useCallback(() => {
    if (check?.matched) {
      onWordMatch?.(check.grammarId);
      recognizer.clearSpelling();
    }
  }, [check, onWordMatch, recognizer]);

  return (
    <div style={{ padding: 12 }}>
      {/* Spelling buffer display */}
      <div style={{
        background: '#1a1a2e', borderRadius: 10, padding: 16,
        border: '1px solid #3a3a5e', textAlign: 'center', marginBottom: 12,
        minHeight: 60, display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        {buffer.length > 0 ? (
          <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', justifyContent: 'center' }}>
            {buffer.map((l, i) => (
              <span key={i} style={{
                fontSize: 24, fontWeight: 700, color: '#e2e8f0',
                background: '#2a2a3e', borderRadius: 6,
                padding: '4px 8px', minWidth: 32, textAlign: 'center',
              }}>
                {l}
              </span>
            ))}
          </div>
        ) : (
          <span style={{ color: '#64748b', fontSize: 13 }}>
            Show letters to spell a word...
          </span>
        )}
      </div>

      {/* Detection indicator */}
      {detection?.letter && (
        <div style={{
          textAlign: 'center', marginBottom: 8,
          fontSize: 12, color: '#60a5fa',
        }}>
          Detecting: {detection.letter} ({Math.round(detection.confidence * 100)}%)
          {detection.candidateProgress > 0 && (
            <span style={{ color: '#94a3b8' }}>
              {' '}— {Math.round(detection.candidateProgress * 100)}%
            </span>
          )}
        </div>
      )}

      {/* Word check result */}
      {check && (
        <div style={{
          textAlign: 'center', marginBottom: 8, fontSize: 12,
          color: check.matched ? '#4ade80' : check.suggestion ? '#fbbf24' : '#94a3b8',
        }}>
          {check.matched ? `Match: "${word}"` : check.suggestion || `"${word}" — no match`}
        </div>
      )}

      {/* Action buttons */}
      <div style={{ display: 'flex', gap: 8, justifyContent: 'center' }}>
        <button onClick={() => recognizer.backspace()} style={{
          background: '#2a2a3e', border: '1px solid #4a4a6a', borderRadius: 6,
          padding: '6px 12px', color: '#f87171', cursor: 'pointer', fontSize: 11, fontWeight: 600,
        }}>
          ← Delete
        </button>
        <button onClick={() => recognizer.clearSpelling()} style={{
          background: '#2a2a3e', border: '1px solid #4a4a6a', borderRadius: 6,
          padding: '6px 12px', color: '#94a3b8', cursor: 'pointer', fontSize: 11, fontWeight: 600,
        }}>
          Clear
        </button>
        {check?.matched && (
          <button onClick={handleSubmitWord} style={{
            background: '#4ade80', border: 'none', borderRadius: 6,
            padding: '6px 14px', color: '#0f0f1a', cursor: 'pointer', fontSize: 11, fontWeight: 700,
          }}>
            Add to Sentence
          </button>
        )}
      </div>
    </div>
  );
}

// =============================================================================
// MAIN PANEL
// =============================================================================

export default function FingerspellingPanel({ recognizer, detection, onWordMatch, onExit }) {
  const [mode, setMode] = useState('learn'); // 'learn' | 'spell'
  const [targetLetter, setTargetLetter] = useState('A');

  return (
    <div style={{
      background: '#0f0f1a', borderRadius: 12,
      border: '1px solid #3a3a5e', overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        padding: '8px 12px', background: '#1a1a2e', borderBottom: '1px solid #3a3a5e',
      }}>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          <span style={{ color: '#fbbf24', fontWeight: 700, fontSize: 13 }}>Fingerspell</span>
          <div style={{ display: 'flex', gap: 4 }}>
            {['learn', 'spell'].map(m => (
              <button key={m} onClick={() => {
                setMode(m);
                if (m === 'spell') recognizer.startSpelling();
                else recognizer.stopSpelling();
              }} style={{
                background: mode === m ? '#60a5fa' : '#2a2a3e',
                border: 'none', borderRadius: 4,
                padding: '3px 8px', color: mode === m ? '#0f0f1a' : '#94a3b8',
                cursor: 'pointer', fontSize: 10, fontWeight: 600,
                textTransform: 'capitalize',
              }}>
                {m}
              </button>
            ))}
          </div>
        </div>
        <button onClick={onExit} style={{
          background: 'none', border: '1px solid #4a4a6a', borderRadius: 4,
          padding: '3px 8px', color: '#94a3b8', cursor: 'pointer', fontSize: 10,
        }}>
          Exit
        </button>
      </div>

      {/* Content */}
      {mode === 'learn' ? (
        <LearnMode
          targetLetter={targetLetter}
          setTargetLetter={setTargetLetter}
          detection={detection}
          recognizer={recognizer}
        />
      ) : (
        <SpellMode
          recognizer={recognizer}
          detection={detection}
          onWordMatch={onWordMatch}
        />
      )}
    </div>
  );
}
