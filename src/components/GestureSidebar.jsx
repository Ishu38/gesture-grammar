/**
 * GestureSidebar.jsx
 * Reference guide showing available gestures with real-time highlighting
 */

import { useMemo } from 'react';

// Gesture definitions with icons and descriptions
const GESTURE_CARDS = [
  // Subjects
  {
    id: 'SUBJECT_I',
    category: 'SUBJECT',
    icon: '👊',
    label: 'I',
    description: 'Thumb pointing at self',
    shape: 'Fist + Thumb inward',
  },
  {
    id: 'SUBJECT_YOU',
    category: 'SUBJECT',
    icon: '👆',
    label: 'YOU',
    description: 'Point at camera',
    shape: 'Index extended forward',
  },
  {
    id: 'SUBJECT_HE',
    category: 'SUBJECT',
    icon: '👍',
    label: 'HE/SHE',
    description: 'Thumb to the side',
    shape: 'Hitchhiker thumb',
  },

  // Verbs
  {
    id: 'GRAB',
    category: 'VERB',
    icon: '🤏',
    label: 'GRAB',
    description: 'Claw / Pinch shape',
    shape: 'All fingertips together',
  },
  {
    id: 'DRINK',
    category: 'VERB',
    icon: '🤙',
    label: 'DRINK',
    description: 'C-shape (holding cup)',
    shape: 'Thumb + Index curved',
  },
  {
    id: 'STOP',
    category: 'VERB',
    icon: '✋',
    label: 'STOP',
    description: 'Open palm (stop sign)',
    shape: 'All fingers extended',
  },

  // Objects
  {
    id: 'APPLE',
    category: 'OBJECT',
    icon: '🍎',
    label: 'APPLE',
    description: 'Cupped hand',
    shape: 'Fingers slightly curved',
  },
  {
    id: 'BOOK',
    category: 'OBJECT',
    icon: '📖',
    label: 'BOOK',
    description: 'Flat palm facing up',
    shape: 'Open hand horizontal',
  },
  {
    id: 'HOUSE',
    category: 'OBJECT',
    icon: '🏠',
    label: 'HOUSE',
    description: 'Roof shape',
    shape: 'Index + Middle tips touch',
  },
  {
    id: 'WATER',
    category: 'OBJECT',
    icon: '💧',
    label: 'WATER',
    description: 'W-shape',
    shape: '3 middle fingers spread',
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
  // Group gestures by category
  const groupedGestures = useMemo(() => {
    const groups = { SUBJECT: [], VERB: [], OBJECT: [] };
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
