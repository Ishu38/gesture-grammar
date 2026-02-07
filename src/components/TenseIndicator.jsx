/**
 * TenseIndicator.jsx
 * Vertical slider showing the current tense zone based on wrist position
 * Only visible when a VERB gesture is being detected
 */

function TenseIndicator({ wristY, currentTenseZone, isVisible }) {
  if (!isVisible) return null;

  // Calculate marker position (inverted because y=0 is top)
  const markerPosition = Math.min(Math.max(wristY * 100, 5), 95);

  return (
    <div className="tense-indicator-container">
      <div className="tense-indicator">
        {/* Header */}
        <div className="tense-header">
          <span className="tense-title">TENSE</span>
          <span className="tense-hint">Move hand up/down</span>
        </div>

        {/* Vertical track */}
        <div className="tense-track">
          {/* Zone labels */}
          <div className={`tense-zone zone-future ${currentTenseZone === 'FUTURE' ? 'active' : ''}`}>
            <span className="zone-label">FUTURE</span>
            <span className="zone-example">will walk</span>
          </div>

          <div className={`tense-zone zone-present ${currentTenseZone === 'PRESENT' ? 'active' : ''}`}>
            <span className="zone-label">PRESENT</span>
            <span className="zone-example">walk/walks</span>
          </div>

          <div className={`tense-zone zone-past ${currentTenseZone === 'PAST' ? 'active' : ''}`}>
            <span className="zone-label">PAST</span>
            <span className="zone-example">walked</span>
          </div>

          {/* Position marker */}
          <div
            className="tense-marker"
            style={{ top: `${markerPosition}%` }}
          >
            <div className="marker-line" />
            <div className="marker-dot" />
            <div className="marker-arrow">◀</div>
          </div>

          {/* Zone boundaries */}
          <div className="zone-boundary" style={{ top: '30%' }} />
          <div className="zone-boundary" style={{ top: '70%' }} />
        </div>

        {/* Current selection */}
        <div className={`current-tense current-${currentTenseZone?.toLowerCase()}`}>
          {currentTenseZone}
        </div>
      </div>
    </div>
  );
}

export default TenseIndicator;
