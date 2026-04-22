/**
 * ErrorVectorEngine.js — Real-time closed-loop error computation engine
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Computes real-time error vectors between the user's current hand pose
 * and the target gesture, providing directional correction feedback.
 * This is the engine that powers the closed-loop error correction system.
 */

import { computeErrorVector, findClosestGesture, GESTURE_REFERENCE_DB } from './SGRM';

// =============================================================================
// ERROR VECTOR ENGINE
// =============================================================================

export class ErrorVectorEngine {
  /**
   * @param {object} options
   * @param {number} options.toleranceMultiplier — from AccessibilityProfile (default: 1.0)
   * @param {number} options.historySize — how many frames of error history to keep
   */
  constructor(options = {}) {
    this.toleranceMultiplier = options.toleranceMultiplier || 1.0;
    this.historySize = options.historySize || 30;
    this.currentTarget = null;
    this.errorHistory = [];
    this.lastErrorVector = null;
  }

  /**
   * Set the target gesture (what the user SHOULD be doing).
   * @param {string} gestureId — gesture ID from SGRM
   */
  setTarget(gestureId) {
    if (GESTURE_REFERENCE_DB[gestureId]) {
      this.currentTarget = gestureId;
      this.errorHistory = [];
      this.lastErrorVector = null;
    }
  }

  /**
   * Clear the current target.
   */
  clearTarget() {
    this.currentTarget = null;
    this.errorHistory = [];
    this.lastErrorVector = null;
  }

  /**
   * Compute error vector between current hand and target gesture.
   * Returns per-constraint directional errors.
   *
   * @param {Array<{x:number, y:number, z?:number}>} currentLandmarks — 21 landmarks
   * @returns {object|null} error computation result
   */
  compute(currentLandmarks) {
    if (!currentLandmarks || currentLandmarks.length < 21) {
      return null;
    }

    // If no explicit target, find the closest gesture
    const targetId = this.currentTarget;
    const closestResult = targetId ? null : findClosestGesture(currentLandmarks);

    // Compute error against explicit target or closest match
    const errorTarget = targetId || (closestResult ? closestResult.gesture_id : null);
    if (!errorTarget) return null;

    const errorVector = computeErrorVector(currentLandmarks, errorTarget);
    if (!errorVector) return null;

    // Apply tolerance multiplier from accessibility profile
    const adjustedErrors = errorVector.per_constraint_errors.map(e => ({
      ...e,
      normalized_error: e.normalized_error / this.toleranceMultiplier,
      within_tolerance: (e.normalized_error / this.toleranceMultiplier) <= 1.0,
    }));

    const result = {
      timestamp: performance.now(),
      gesture_target: errorTarget,
      is_explicit_target: !!targetId,
      per_constraint_errors: adjustedErrors,
      aggregate_error: errorVector.aggregate_error / this.toleranceMultiplier,
      is_within_tolerance: errorVector.aggregate_error / this.toleranceMultiplier <= 1.0,
      closest_gesture: closestResult ? closestResult.gesture_id : errorTarget,
      constraints_met: adjustedErrors.filter(e => e.within_tolerance).length,
      constraints_total: adjustedErrors.length,
    };

    // Store in history
    this.errorHistory.push(result);
    if (this.errorHistory.length > this.historySize) {
      this.errorHistory.shift();
    }
    this.lastErrorVector = result;

    return result;
  }

  /**
   * Generate visual overlay data for the canvas.
   * Returns arrow vectors to draw on the hand skeleton for correction.
   *
   * @param {object} errorVector — output from compute()
   * @returns {object[]} array of overlay arrow descriptors
   */
  generateOverlayData(errorVector) {
    if (!errorVector) return [];

    const arrows = [];

    for (const error of errorVector.per_constraint_errors) {
      // Determine severity color
      let severity;
      if (error.within_tolerance) {
        severity = 'good';     // Green — within tolerance
      } else if (error.normalized_error < 2.0) {
        severity = 'close';    // Yellow — close but not quite
      } else {
        severity = 'far';      // Red — significantly off
      }

      arrows.push({
        constraint_id: error.constraint_id,
        type: error.type,
        direction: error.direction,
        magnitude: Math.min(error.normalized_error, 3.0), // cap at 3x for display
        severity,
        correction: error.correction_instruction,
        deviation: error.deviation,
      });
    }

    return arrows;
  }

  /**
   * Get trending error data — are the user's errors improving over time?
   *
   * @returns {object} trend analysis
   */
  getErrorTrend() {
    if (this.errorHistory.length < 2) {
      return { trend: 'insufficient_data', samples: this.errorHistory.length };
    }

    const recentHalf = this.errorHistory.slice(-Math.floor(this.errorHistory.length / 2));
    const olderHalf = this.errorHistory.slice(0, Math.floor(this.errorHistory.length / 2));

    const recentAvg = recentHalf.reduce((sum, e) => sum + e.aggregate_error, 0) / recentHalf.length;
    const olderAvg = olderHalf.reduce((sum, e) => sum + e.aggregate_error, 0) / olderHalf.length;

    const improvement = olderAvg - recentAvg;

    return {
      trend: improvement > 0.05 ? 'improving' : improvement < -0.05 ? 'declining' : 'stable',
      recent_avg_error: recentAvg,
      older_avg_error: olderAvg,
      improvement,
      samples: this.errorHistory.length,
    };
  }

  /**
   * Get the most problematic constraints — which ones the user struggles with most.
   *
   * @returns {object[]} top 3 problematic constraints sorted by average error
   */
  getProblematicConstraints() {
    if (this.errorHistory.length === 0) return [];

    const constraintErrors = {};

    for (const frame of this.errorHistory) {
      for (const error of frame.per_constraint_errors) {
        if (!constraintErrors[error.constraint_id]) {
          constraintErrors[error.constraint_id] = {
            id: error.constraint_id,
            totalError: 0,
            count: 0,
            correction: error.correction_instruction,
          };
        }
        constraintErrors[error.constraint_id].totalError += error.normalized_error;
        constraintErrors[error.constraint_id].count++;
      }
    }

    return Object.values(constraintErrors)
      .map(c => ({
        constraint_id: c.id,
        avg_error: c.totalError / c.count,
        correction: c.correction,
      }))
      .sort((a, b) => b.avg_error - a.avg_error)
      .slice(0, 3);
  }

  /**
   * Update the tolerance multiplier (e.g., when accessibility profile changes).
   * @param {number} multiplier
   */
  setToleranceMultiplier(multiplier) {
    this.toleranceMultiplier = multiplier;
  }

  /**
   * Reset the engine state.
   */
  reset() {
    this.currentTarget = null;
    this.errorHistory = [];
    this.lastErrorVector = null;
  }
}
