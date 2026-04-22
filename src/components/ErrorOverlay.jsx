/**
 * ErrorOverlay.jsx — Visual correction feedback overlay
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Draws correction arrows and visual guides on the camera canvas
 * to show users how to adjust their hand position.
 *
 * Color coding:
 *   Green  — constraint within tolerance (correct)
 *   Yellow — close but not quite
 *   Red    — significantly off
 */

import { useRef, useEffect } from 'react';

// =============================================================================
// COLOR CONSTANTS
// =============================================================================

// Polyfill for Canvas roundRect (Safari < 15.4, older browsers)
if (typeof CanvasRenderingContext2D !== 'undefined' && !CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    const radius = typeof r === 'number' ? r : 0;
    this.beginPath();
    this.moveTo(x + radius, y);
    this.lineTo(x + w - radius, y);
    this.quadraticCurveTo(x + w, y, x + w, y + radius);
    this.lineTo(x + w, y + h - radius);
    this.quadraticCurveTo(x + w, y + h, x + w - radius, y + h);
    this.lineTo(x + radius, y + h);
    this.quadraticCurveTo(x, y + h, x, y + h - radius);
    this.lineTo(x, y + radius);
    this.quadraticCurveTo(x, y, x + radius, y);
    this.closePath();
  };
}

const SEVERITY_COLORS = {
  good: { stroke: '#22c55e', fill: 'rgba(34, 197, 94, 0.3)', label: '#4ade80' },
  close: { stroke: '#eab308', fill: 'rgba(234, 179, 8, 0.3)', label: '#facc15' },
  far: { stroke: '#ef4444', fill: 'rgba(239, 68, 68, 0.3)', label: '#f87171' },
};

// =============================================================================
// COMPONENT
// =============================================================================

function ErrorOverlay({ errorData, canvasWidth, canvasHeight, visible, predictableFeedback = false }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (predictableFeedback) return; // Skip canvas drawing in predictable mode
    const canvas = canvasRef.current;
    if (!canvas || !visible) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (!errorData || !errorData.per_constraint_errors) return;

    const overlayArrows = errorData.per_constraint_errors;

    // Draw aggregate error indicator (top-right corner)
    drawAggregateIndicator(ctx, errorData, canvas.width);

    // Draw per-constraint correction indicators
    drawConstraintIndicators(ctx, overlayArrows, canvas.width, canvas.height);

    // Draw correction instruction text
    drawCorrectionText(ctx, overlayArrows, canvas.width, canvas.height);

  }, [errorData, canvasWidth, canvasHeight, visible, predictableFeedback]);

  if (!visible) return null;

  // ASD predictable feedback mode: fixed-position, text-only, no canvas animation
  if (predictableFeedback) {
    const feedbackInfo = getPredictableFeedbackInfo(errorData);
    return (
      <div className="predictable-feedback-zone">
        <div className={`predictable-feedback-message ${feedbackInfo.className}`}>
          {feedbackInfo.message}
        </div>
      </div>
    );
  }

  return (
    <canvas
      ref={canvasRef}
      width={canvasWidth || 640}
      height={canvasHeight || 480}
      className="error-overlay-canvas"
      style={{
        position: 'absolute',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        transform: 'scaleX(-1)',
        zIndex: 10,
      }}
    />
  );
}

/**
 * Extract a single, predictable feedback message for ASD mode.
 * Always returns exactly one message in a fixed format.
 */
function getPredictableFeedbackInfo(errorData) {
  if (!errorData || !errorData.per_constraint_errors) {
    return { message: 'Show your hand to the camera.', className: 'feedback-hint' };
  }

  const errors = errorData.per_constraint_errors;
  const allGood = errors.every(e => e.within_tolerance);

  if (allGood) {
    return { message: 'Great form! Hold steady.', className: 'feedback-success' };
  }

  // Find the single worst error and show its correction
  const worstError = errors
    .filter(e => !e.within_tolerance)
    .sort((a, b) => b.normalized_error - a.normalized_error)[0];

  return {
    message: worstError.correction_instruction || 'Adjust your hand position.',
    className: 'feedback-error',
  };
}

// =============================================================================
// DRAWING FUNCTIONS
// =============================================================================

/**
 * Draw the aggregate error score indicator in the top-right corner.
 */
function drawAggregateIndicator(ctx, errorData, canvasWidth) {
  const x = canvasWidth - 100;
  const y = 20;
  const radius = 30;

  const score = Math.max(0, Math.min(1, 1 - errorData.aggregate_error));
  const color = score > 0.7 ? SEVERITY_COLORS.good
    : score > 0.4 ? SEVERITY_COLORS.close
    : SEVERITY_COLORS.far;

  // Background circle
  ctx.beginPath();
  ctx.arc(x, y + radius, radius, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
  ctx.fill();

  // Progress arc
  ctx.beginPath();
  ctx.arc(x, y + radius, radius - 4, -Math.PI / 2, -Math.PI / 2 + score * Math.PI * 2);
  ctx.strokeStyle = color.stroke;
  ctx.lineWidth = 4;
  ctx.lineCap = 'round';
  ctx.stroke();

  // Score text
  ctx.fillStyle = color.label;
  ctx.font = 'bold 14px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`${Math.round(score * 100)}%`, x, y + radius);

  // Label
  ctx.fillStyle = '#94a3b8';
  ctx.font = '10px Inter, sans-serif';
  ctx.fillText('Accuracy', x, y + radius + radius + 10);
}

/**
 * Draw visual indicators for each constraint.
 */
function drawConstraintIndicators(ctx, errors, canvasWidth, canvasHeight) {
  const barX = 15;
  const barWidth = 8;
  const startY = 80;
  const barSpacing = 40;

  errors.forEach((error, index) => {
    const y = startY + index * barSpacing;
    const severity = getSeverity(error);
    const color = SEVERITY_COLORS[severity];

    // Constraint bar background
    ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
    ctx.fillRect(barX, y, barWidth, 25);

    // Constraint bar fill (inversely proportional to error)
    const fillHeight = Math.max(2, 25 * Math.max(0, 1 - error.normalized_error));
    ctx.fillStyle = color.stroke;
    ctx.fillRect(barX, y + 25 - fillHeight, barWidth, fillHeight);

    // Constraint label
    ctx.fillStyle = color.label;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    const labelText = error.constraint_id.replace(/_/g, ' ');
    ctx.fillText(labelText, barX + barWidth + 6, y + 8);

    // Status icon
    const icon = error.within_tolerance ? '\u2713' : '\u2717';
    ctx.fillText(icon, barX + barWidth + 6, y + 20);
  });
}

/**
 * Draw the most important correction instruction at the bottom.
 */
function drawCorrectionText(ctx, errors, canvasWidth, canvasHeight) {
  // Find the worst error that's not within tolerance
  const worstError = errors
    .filter(e => !e.within_tolerance)
    .sort((a, b) => b.normalized_error - a.normalized_error)[0];

  if (!worstError) {
    // All within tolerance — show success message
    drawTextBox(ctx, 'Great form! Hold steady.', canvasWidth / 2, canvasHeight - 40, SEVERITY_COLORS.good);
    return;
  }

  drawTextBox(ctx, worstError.correction_instruction, canvasWidth / 2, canvasHeight - 40, SEVERITY_COLORS[getSeverity(worstError)]);
}

/**
 * Draw a text box with background.
 */
function drawTextBox(ctx, text, x, y, color) {
  ctx.font = 'bold 13px Inter, sans-serif';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';

  const metrics = ctx.measureText(text);
  const padding = 10;
  const boxWidth = metrics.width + padding * 2;
  const boxHeight = 28;

  // Background
  ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
  ctx.beginPath();
  ctx.roundRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight, 6);
  ctx.fill();

  // Border
  ctx.strokeStyle = color.stroke;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(x - boxWidth / 2, y - boxHeight / 2, boxWidth, boxHeight, 6);
  ctx.stroke();

  // Text
  ctx.fillStyle = color.label;
  ctx.fillText(text, x, y);
}

/**
 * Get severity level from error data.
 */
function getSeverity(error) {
  if (error.within_tolerance) return 'good';
  if (error.normalized_error < 2.0) return 'close';
  return 'far';
}

export default ErrorOverlay;
