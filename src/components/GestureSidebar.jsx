/**
 * GestureSidebar.jsx
 * Reference guide showing available gestures with real-time highlighting
 */

import { useMemo } from 'react';

// Gesture definitions — kept STRICTLY in sync with src/data/GestureLexicon.json.
// Every entry below corresponds to a real semantic_mapping.grammar_id the
// detector (gestureDetection.js) can actually recognize. The "shape" field
// describes the SAME handshape the geometric detector checks, so users mimic
// the guide and the system locks. All 19 gestures are now documented.

const GESTURE_CARDS = [
  // =========================================================================
  // SUBJECTS — Who is doing the action
  // =========================================================================
  {
    id: 'SUBJECT_I', category: 'SUBJECT', icon: '\u270A', label: 'I',
    description: 'Closed fist',
    shape: 'All fingers curled into palm \u2014 every tip tucked in',
  },
  {
    id: 'SUBJECT_YOU', category: 'SUBJECT', icon: '\u261D', label: 'YOU',
    description: 'Point forward',
    shape: 'Index finger extended, all other fingers curled',
  },
  {
    id: 'SUBJECT_HE', category: 'SUBJECT', icon: '\uD83E\uDD1F', label: 'HE',
    description: 'Three fingers up',
    shape: 'Index + Middle + Ring extended, Thumb + Pinky curled',
  },
  {
    id: 'SUBJECT_SHE', category: 'SUBJECT', icon: '\uD83E\uDD1F', label: 'SHE',
    description: 'Three fingers up (same as HE)',
    shape: 'Same shape as HE \u2014 context tells them apart',
  },
  {
    id: 'SUBJECT_WE', category: 'SUBJECT', icon: '\u270C', label: 'WE',
    description: 'Two fingers together',
    shape: 'Index + Middle extended and close together, others curled',
  },
  {
    id: 'SUBJECT_THEY', category: 'SUBJECT', icon: '\uD83D\uDD90', label: 'THEY',
    description: 'All fingers spread wide',
    shape: 'All five fingers extended and spread apart',
  },

  // =========================================================================
  // VERBS — What action is happening
  // =========================================================================
  {
    id: 'GRAB', category: 'VERB', icon: '\uD83E\uDD0F', label: 'GRAB',
    description: 'Pinch shape',
    shape: 'Thumb tip touches Index tip \u2014 others relaxed / curled',
  },
  {
    id: 'DRINK', category: 'VERB', icon: '\uD83E\uDECF', label: 'DRINK',
    description: 'C-shape (holding a cup)',
    shape: 'Thumb + Index spread apart forming a C-gap, Middle/Ring/Pinky curled',
  },
  {
    id: 'EAT', category: 'VERB', icon: '\uD83C\uDF4E', label: 'EAT',
    description: 'Bunched near mouth',
    shape: 'Fingertips bunched together, hand near face level (upper frame)',
  },
  {
    id: 'WANT', category: 'VERB', icon: '\uD83E\uDD1A', label: 'WANT',
    description: 'Open claw reaching',
    shape: 'Fingers partially curled (half-closed), reaching outward',
  },
  {
    id: 'SEE', category: 'VERB', icon: '\uD83D\uDC41', label: 'SEE',
    description: 'V-shape near eyes',
    shape: 'Index + Middle extended and spread (V-sign apart), others curled',
  },
  {
    id: 'GO', category: 'VERB', icon: '\u27A1', label: 'GO',
    description: 'Point forward relaxed',
    shape: 'Index extended, others curled with relaxed thumb \u2014 hand not vertical',
  },
  {
    id: 'STOP', category: 'VERB', icon: '\u270B', label: 'STOP',
    description: 'Open palm facing out',
    shape: 'All five fingers extended, palm facing camera, hand vertical',
  },

  // =========================================================================
  // OBJECTS — What is being acted on
  // =========================================================================
  {
    id: 'APPLE', category: 'OBJECT', icon: '\uD83E\uDD1A', label: 'APPLE',
    description: 'Flat hand, palm down',
    shape: 'Hand horizontal, all fingers extended together, palm facing floor',
  },
  {
    id: 'BALL', category: 'OBJECT', icon: '\u26BD', label: 'BALL',
    description: 'Cupped spread hand',
    shape: 'Fingers curved wide apart like holding a large ball',
  },
  {
    id: 'WATER', category: 'OBJECT', icon: '\uD83D\uDCA7', label: 'WATER',
    description: 'W-shape (three fingers)',
    shape: 'Index + Middle + Ring extended and spread, Thumb + Pinky curled',
  },
  {
    id: 'FOOD', category: 'OBJECT', icon: '\uD83C\uDF5E', label: 'FOOD',
    description: 'Cupped hand (bowl shape)',
    shape: 'All fingers slightly curved, thumb curled in \u2014 like holding a bowl',
  },
  {
    id: 'BOOK', category: 'OBJECT', icon: '\uD83D\uDCD6', label: 'BOOK',
    description: 'Flat palm facing up',
    shape: 'All fingers extended flat, palm up, hand horizontal \u2014 like open book',
  },
  {
    id: 'HOUSE', category: 'OBJECT', icon: '\uD83C\uDFE0', label: 'HOUSE',
    description: 'Roof shape (inverted V)',
    shape: 'Index + Middle extended with tips touching (\u039B), others curled',
  },

  // =========================================================================
  // MODIFIERS
  // =========================================================================
  {
    id: 'PLURAL', category: 'MODIFIER', icon: '\u270C', label: 'PLURAL',
    description: 'Two fingers up (peace sign)',
    shape: 'Index + Middle extended upward, Thumb / Ring / Pinky curled',
  },
  {
    id: 'AFFIRMATIVE', category: 'MODIFIER', icon: '\uD83D\uDC4D', label: 'YES',
    description: 'Thumbs up',
    shape: 'Thumb extended upward, other fingers curled into palm',
  },
  {
    id: 'ACTIVE_VOICE', category: 'MODIFIER', icon: '\u27A1', label: 'ACTIVE',
    description: 'Swipe right',
    shape: 'Open hand moves left \u2192 right across the frame',
  },
  {
    id: 'PASSIVE_VOICE', category: 'MODIFIER', icon: '\u2B05', label: 'PASSIVE',
    description: 'Swipe left',
    shape: 'Open hand moves right \u2192 left across the frame',
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
