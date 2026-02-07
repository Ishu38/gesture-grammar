/**
 * SentenceStrip.jsx
 * Visual component for displaying the sentence being built
 * Features color-coded grammar blocks with pop-in animations
 */

import { useEffect, useState } from 'react';

// Block color mapping by type
const BLOCK_COLORS = {
  SUBJECT: {
    bg: 'bg-blue-600',
    border: 'border-blue-400',
    shadow: 'shadow-blue-500/50',
    label: 'text-blue-200',
  },
  VERB: {
    bg: 'bg-red-600',
    border: 'border-red-400',
    shadow: 'shadow-red-500/50',
    label: 'text-red-200',
  },
  OBJECT: {
    bg: 'bg-green-600',
    border: 'border-green-400',
    shadow: 'shadow-green-500/50',
    label: 'text-green-200',
  },
  MODIFIER: {
    bg: 'bg-purple-600',
    border: 'border-purple-400',
    shadow: 'shadow-purple-500/50',
    label: 'text-purple-200',
  },
  UNKNOWN: {
    bg: 'bg-gray-600',
    border: 'border-gray-400',
    shadow: 'shadow-gray-500/50',
    label: 'text-gray-200',
  },
};

/**
 * Individual word block component with animation
 */
function WordBlock({ word, index, isNew }) {
  const [isAnimating, setIsAnimating] = useState(isNew);
  const colors = BLOCK_COLORS[word.type] || BLOCK_COLORS.UNKNOWN;

  useEffect(() => {
    if (isNew) {
      setIsAnimating(true);
      const timer = setTimeout(() => setIsAnimating(false), 500);
      return () => clearTimeout(timer);
    }
  }, [isNew]);

  return (
    <div
      className={`
        word-block-container
        ${isAnimating ? 'animate-pop-in' : ''}
      `}
      style={{ animationDelay: `${index * 50}ms` }}
    >
      <div
        className={`
          word-block
          ${colors.bg}
          ${colors.border}
          ${colors.shadow}
        `}
      >
        <span className={`word-type-label ${colors.label}`}>
          {word.type}
          {word.tense && word.tense !== 'present' && (
            <span className="tense-badge">
              {word.tense}
            </span>
          )}
        </span>
        <span className="word-text">
          {word.word}
        </span>
      </div>
    </div>
  );
}

/**
 * Main SentenceStrip component
 */
function SentenceStrip({ sentence, onClear, onUndo, isLocked, lockProgress }) {
  const [lastLength, setLastLength] = useState(0);

  // Track which blocks are new
  const newBlockIndex = sentence.length > lastLength ? sentence.length - 1 : -1;

  useEffect(() => {
    setLastLength(sentence.length);
  }, [sentence.length]);

  return (
    <div className="sentence-strip-container">
      {/* The Tray/Blackboard ledge */}
      <div className="sentence-strip">
        {/* Lock indicator */}
        {isLocked && (
          <div className="lock-indicator">
            <span className="lock-icon">🔒</span>
            <span className="lock-text">Cooldown...</span>
          </div>
        )}

        {/* Progress bar when building */}
        {lockProgress > 0 && lockProgress < 1 && (
          <div className="lock-progress-bar">
            <div
              className="lock-progress-fill"
              style={{ width: `${lockProgress * 100}%` }}
            />
          </div>
        )}

        {/* Word blocks container */}
        <div className="blocks-container">
          {sentence.length === 0 ? (
            <div className="empty-state">
              <span className="empty-icon">👋</span>
              <span className="empty-text">Perform a gesture to start building...</span>
            </div>
          ) : (
            <>
              {sentence.map((word, index) => (
                <WordBlock
                  key={word.id}
                  word={word}
                  index={index}
                  isNew={index === newBlockIndex}
                />
              ))}

              {/* Cursor indicator */}
              <div className="cursor-indicator">
                <span className="cursor-blink">|</span>
              </div>
            </>
          )}
        </div>

        {/* Action buttons */}
        <div className="strip-actions">
          {sentence.length > 0 && (
            <>
              <button
                className="action-btn undo-btn"
                onClick={onUndo}
                title="Undo last word"
              >
                ↩️
              </button>
              <button
                className="action-btn clear-btn"
                onClick={onClear}
                title="Clear sentence"
              >
                🗑️
              </button>
            </>
          )}
        </div>
      </div>

      {/* Readable sentence preview */}
      {sentence.length > 0 && (
        <div className="sentence-preview">
          "
          {sentence.map((word, i) => (
            <span key={word.id}>
              {i === 0
                ? word.word.charAt(0).toUpperCase() + word.word.slice(1)
                : word.word}
              {i < sentence.length - 1 ? ' ' : ''}
            </span>
          ))}
          {sentence.length >= 2 ? '.' : '...'}
          "
        </div>
      )}
    </div>
  );
}

export default SentenceStrip;
