/**
 * GestureSidebar.jsx
 * Reference guide showing available gestures with real-time highlighting
 */

import { useMemo } from 'react';

// Gesture definitions — kept STRICTLY in sync with src/data/GestureLexicon.json.
// Every entry below corresponds to a real semantic_mapping.grammar_id the
// classifier can actually recognize. The "shape" field must describe the
// SAME handshape the classifier expects (i.e. the lexicon's human_description),
// otherwise users mimic what the guide shows and the system stays at 0%.
// Previous DRINK / BOOK / HOUSE / WATER cards were aspirational stubs and are
// removed — they were never wired into the recognizer and caused users to
// chase shapes the system could not lock.
const GESTURE_CARDS = [
  // Subjects
  {
    id: 'SUBJECT_I',
    category: 'SUBJECT',
    icon: '✊',
    label: 'I',
    description: 'Closed fist',
    shape: 'All fingers curled into palm',
  },
  {
    id: 'SUBJECT_YOU',
    category: 'SUBJECT',
    icon: '☝️',
    label: 'YOU',
    description: 'Point with index finger',
    shape: 'Index extended, others curled',
  },
  {
    id: 'SUBJECT_HE',
    category: 'SUBJECT',
    icon: '🤟',
    label: 'HE / SHE',
    description: 'Three fingers up',
    shape: 'Index + Middle + Ring extended, Pinky curled, Thumb tucked',
  },

  // Verbs
  {
    id: 'GRAB',
    category: 'VERB',
    icon: '🤏',
    label: 'GRAB',
    description: 'Pinch shape',
    shape: 'Thumb tip + Index tip touching, others relaxed',
  },
  {
    id: 'STOP',
    category: 'VERB',
    icon: '✋',
    label: 'STOP',
    description: 'Open palm',
    shape: 'All five fingers spread, palm facing camera',
  },

  // Object
  {
    id: 'APPLE',
    category: 'OBJECT',
    icon: '🤚',
    label: 'APPLE',
    description: 'Flat hand, palm down',
    shape: 'Hand horizontal, palm facing the floor',
  },

  // Modifiers
  {
    id: 'PLURAL',
    category: 'MODIFIER',
    icon: '✌️',
    label: 'PLURAL',
    description: 'Two fingers up (peace sign)',
    shape: 'Index + Middle extended, Thumb / Ring / Pinky curled',
  },
  {
    id: 'AFFIRMATIVE',
    category: 'MODIFIER',
    icon: '👍',
    label: 'YES',
    description: 'Thumbs up',
    shape: 'Thumb extended upward, other fingers curled',
  },

  // Voice markers (dynamic — motion gestures, not static handshape)
  {
    id: 'ACTIVE_VOICE',
    category: 'MODIFIER',
    icon: '➡️',
    label: 'ACTIVE',
    description: 'Swipe right',
    shape: 'Open hand moves left → right across the frame',
  },
  {
    id: 'PASSIVE_VOICE',
    category: 'MODIFIER',
    icon: '⬅️',
    label: 'PASSIVE',
    description: 'Swipe left',
    shape: 'Open hand moves right → left across the frame',
  },
];

// Category colors
const CATEGORY_COLORS = {
  SUBJECT: {
    bg: 'bg-blue-900/50',
    border: 'border-blue-500',
    activeBorder: 'border-yellow-400',
    activeBg: 'bg-blue-700',
    text: 'text-blue-200',
    badge: 'bg-blue-600',
  },
  VERB: {
    bg: 'bg-red-900/50',
    border: 'border-red-500',
    activeBorder: 'border-yellow-400',
    activeBg: 'bg-red-700',
    text: 'text-red-200',
    badge: 'bg-red-600',
  },
  OBJECT: {
    bg: 'bg-green-900/50',
    border: 'border-green-500',
    activeBorder: 'border-yellow-400',
    activeBg: 'bg-green-700',
    text: 'text-green-200',
    badge: 'bg-green-600',
  },
  MODIFIER: {
    bg: 'bg-purple-900/50',
    border: 'border-purple-500',
    activeBorder: 'border-yellow-400',
    activeBg: 'bg-purple-700',
    text: 'text-purple-200',
    badge: 'bg-purple-600',
  },
};

/**
 * Individual gesture card
 */
function GestureCard({ gesture, isActive, lockProgress }) {
  const colors = CATEGORY_COLORS[gesture.category];
  const isBuilding = isActive && lockProgress > 0 && lockProgress < 1;

  return (
    <div
      className={`
        gesture-card
        ${colors.bg}
        ${isActive ? colors.activeBorder : colors.border}
        ${isActive ? colors.activeBg : ''}
        ${isActive ? 'scale-105 shadow-lg' : ''}
        ${isBuilding ? 'pulse-glow' : ''}
      `}
    >
      {/* Progress indicator when building */}
      {isBuilding && (
        <div className="card-progress">
          <div
            className="card-progress-fill"
            style={{ width: `${lockProgress * 100}%` }}
          />
        </div>
      )}

      {/* Icon */}
      <div className={`gesture-icon ${isActive ? 'scale-125' : ''}`}>
        {gesture.icon}
      </div>

      {/* Label */}
      <div className="gesture-info">
        <span className={`gesture-label ${isActive ? 'text-yellow-300 font-bold' : colors.text}`}>
          {gesture.label}
        </span>
        <span className="gesture-shape">
          {gesture.shape}
        </span>
      </div>

      {/* Category badge */}
      <span className={`category-badge ${colors.badge}`}>
        {gesture.category}
      </span>

      {/* Active indicator */}
      {isActive && (
        <div className="active-indicator">
          <span className="active-dot" />
          <span className="active-text">DETECTED</span>
        </div>
      )}
    </div>
  );
}

/**
 * Main GestureSidebar component
 */
function GestureSidebar({ currentGesture, lockProgress = 0, currentTenseZone }) {
  // Group gestures by category — includes MODIFIER for plural, yes/no,
  // and the swipe-direction voice markers so the user can actually see
  // every gesture the classifier recognises.
  const groupedGestures = useMemo(() => {
    const groups = { SUBJECT: [], VERB: [], OBJECT: [], MODIFIER: [] };
    GESTURE_CARDS.forEach(g => {
      if (groups[g.category]) {
        groups[g.category].push(g);
      }
    });
    return groups;
  }, []);

  return (
    <div className="gesture-sidebar">
      <div className="sidebar-header">
        <h3 className="sidebar-title">Gesture Guide</h3>
        {currentGesture && (
          <div className="current-gesture-badge">
            Detecting: <strong>{currentGesture}</strong>
          </div>
        )}
      </div>

      <div className="sidebar-content">
        {/* Subjects */}
        <div className="gesture-category">
          <h4 className="category-title text-blue-400">
            <span className="category-icon">👤</span>
            Subjects
          </h4>
          <div className="gesture-list">
            {groupedGestures.SUBJECT.map(gesture => (
              <GestureCard
                key={gesture.id}
                gesture={gesture}
                isActive={currentGesture === gesture.id}
                lockProgress={currentGesture === gesture.id ? lockProgress : 0}
              />
            ))}
          </div>
        </div>

        {/* Verbs */}
        <div className="gesture-category">
          <h4 className="category-title text-red-400">
            <span className="category-icon">⚡</span>
            Verbs
            {currentTenseZone && currentGesture && GESTURE_CARDS.find(g => g.id === currentGesture)?.category === 'VERB' && (
              <span className="tense-indicator">
                ({currentTenseZone})
              </span>
            )}
          </h4>
          <div className="gesture-list">
            {groupedGestures.VERB.map(gesture => (
              <GestureCard
                key={gesture.id}
                gesture={gesture}
                isActive={currentGesture === gesture.id}
                lockProgress={currentGesture === gesture.id ? lockProgress : 0}
              />
            ))}
          </div>
        </div>

        {/* Objects */}
        <div className="gesture-category">
          <h4 className="category-title text-green-400">
            <span className="category-icon">📦</span>
            Objects
          </h4>
          <div className="gesture-list">
            {groupedGestures.OBJECT.map(gesture => (
              <GestureCard
                key={gesture.id}
                gesture={gesture}
                isActive={currentGesture === gesture.id}
                lockProgress={currentGesture === gesture.id ? lockProgress : 0}
              />
            ))}
          </div>
        </div>

        {/* Modifiers (plural, yes, voice markers) */}
        {groupedGestures.MODIFIER.length > 0 && (
          <div className="gesture-category">
            <h4 className="category-title" style={{ color: '#a855f7' }}>
              <span className="category-icon">✦</span>
              Modifiers
            </h4>
            <div className="gesture-list">
              {groupedGestures.MODIFIER.map(gesture => (
                <GestureCard
                  key={gesture.id}
                  gesture={gesture}
                  isActive={currentGesture === gesture.id}
                  lockProgress={currentGesture === gesture.id ? lockProgress : 0}
                />
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Instructions */}
      <div className="sidebar-footer">
        <p className="instruction-text">
          Hold gesture for ~1.5s to lock in
        </p>
      </div>
    </div>
  );
}

export default GestureSidebar;
