/**
 * SpatialGrammarMapper.js — AGGME Phase 4: Syntactic Spatial Mapping
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Divides the camera's field of view into grammatical zones along two axes:
 *
 *   X-axis (horizontal) -> Syntactic Role:
 *     LEFT zone   [0.0 - 0.35]  -> Subject / Noun Phrase (NP)
 *     CENTER zone [0.35 - 0.65] -> Verb / Action (VP)
 *     RIGHT zone  [0.65 - 1.0]  -> Object / Receiving Noun Phrase (NP)
 *
 *   Y-axis (vertical) -> Tense:
 *     TOP zone    [0.0 - 0.3]  -> Future
 *     MIDDLE zone [0.3 - 0.7]  -> Present
 *     BOTTOM zone [0.7 - 1.0]  -> Past
 *
 * CP Accessibility Extensions:
 *   - Zone widening: per-profile expansion of zone boundaries to accommodate
 *     imprecise hand positioning (ataxic CP). Overlapping zones are resolved
 *     by proximity to zone center.
 *   - Zone latching: when enabled (ataxic/athetoid CP), the zone is frozen at
 *     gesture onset and held for the duration of the gesture, preventing
 *     spatial drift during execution from changing the grammatical role.
 *
 * AGGME Pipeline Position:
 *   MediaPipe -> RestingBoundaryCalibrator -> LandmarkSmoother -> IntentionalityDetector -> [SpatialGrammarMapper]
 *
 * IMPORTANT: The camera feed is mirrored (scaleX(-1)) in the UI.
 * MediaPipe returns coordinates in the original (unmirrored) frame:
 *   - Raw x=0.0 = user's RIGHT side = screen LEFT
 *   - Raw x=1.0 = user's LEFT side = screen RIGHT
 */

import { ZONE_STRATEGIES } from './AccessibilityProfile.js';

// =============================================================================
// ZONE DEFINITIONS (base boundaries, before widening)
// =============================================================================

const SYNTACTIC_ZONES = {
  SUBJECT_ZONE: { min: 0.00, max: 0.35, label: 'Subject (NP)', role: 'SUBJECT' },
  VERB_ZONE:    { min: 0.35, max: 0.65, label: 'Verb (VP)',    role: 'VERB'    },
  OBJECT_ZONE:  { min: 0.65, max: 1.00, label: 'Object (NP)',  role: 'OBJECT'  },
};

const TENSE_ZONES = {
  FUTURE:  { min: 0.0, max: 0.3, label: 'Future' },
  PRESENT: { min: 0.3, max: 0.7, label: 'Present' },
  PAST:    { min: 0.7, max: 1.0, label: 'Past'    },
};

const WRIST_INDEX = 0;
const VELOCITY_FLOOR = 0.001;

// =============================================================================
// SPATIAL GRAMMAR MAPPER
// =============================================================================

export class SpatialGrammarMapper {
  /**
   * @param {object} [config]
   * @param {string} [config.zoneStrategy='HARD_BOUNDARY'] — HARD_BOUNDARY or LATCHED
   * @param {number} [config.zoneWidening=0] — expansion per zone edge (0–0.15)
   */
  constructor(config = {}) {
    // Previous wrist position for velocity computation
    this._prevWrist = null;

    // Current zone tracking
    this._currentSyntacticZone = null;
    this._currentTenseZone = null;
    this._zoneEntryTimestamp = null;

    // Zone transition tracking
    this._previousSyntacticZone = null;
    this._zoneTransitions = [];

    // Frame counter
    this._frameCount = 0;

    // ── CP Accessibility: Zone latching ──
    this._zoneStrategy = config.zoneStrategy || ZONE_STRATEGIES.HARD_BOUNDARY;
    this._latchedZone = null;          // Frozen zone during gesture
    this._isGestureActive = false;     // Tracks IntentionalityDetector state

    // ── CP Accessibility: Zone widening ──
    this._zoneWidening = Math.max(0, Math.min(0.15, config.zoneWidening || 0));
    this._effectiveZones = this._computeEffectiveZones();
  }

  // ===========================================================================
  // PUBLIC — Core mapping
  // ===========================================================================

  /**
   * Map a frame of smoothed landmarks to syntactic and tense zones.
   *
   * @param {Array} landmarks — 21 smoothed hand landmarks
   * @param {object} [intentResult] — result from IntentionalityDetector (for latching)
   * @returns {SpatialMappingResult}
   */
  map(landmarks, intentResult = null) {
    if (!landmarks || landmarks.length < 21) {
      return this._noResult();
    }

    this._frameCount++;

    const wrist = landmarks[WRIST_INDEX];
    const screenX = 1.0 - wrist.x;
    const screenY = wrist.y;

    // Classify zones from position
    const rawSyntacticZone = this._classifyX(screenX);
    const tenseZone = this._classifyY(screenY);
    const zoneConfidence = this._computeZoneConfidence(screenX, rawSyntacticZone);
    const velocity = this._computeVelocity(wrist);

    // ── Zone latching logic (ataxic/athetoid CP) ──
    let syntacticZone = rawSyntacticZone;

    if (this._zoneStrategy === ZONE_STRATEGIES.LATCHED && intentResult) {
      const wasActive = this._isGestureActive;
      this._isGestureActive = intentResult.intent === 'GESTURE_ACTIVE';

      if (intentResult.onset) {
        // Gesture just started — latch the current zone
        this._latchedZone = rawSyntacticZone;
      }

      if (this._isGestureActive && this._latchedZone) {
        // During gesture, use the latched zone regardless of current position
        syntacticZone = this._latchedZone;
      }

      if (!this._isGestureActive && wasActive) {
        // Gesture ended — release latch
        this._latchedZone = null;
      }
    }

    // Track zone transitions
    let zoneChanged = false;
    if (syntacticZone !== this._currentSyntacticZone) {
      zoneChanged = true;
      this._previousSyntacticZone = this._currentSyntacticZone;
      this._currentSyntacticZone = syntacticZone;
      this._zoneEntryTimestamp = Date.now();

      if (this._previousSyntacticZone !== null) {
        this._zoneTransitions.push({
          from: this._previousSyntacticZone,
          to: syntacticZone,
          timestamp: Date.now(),
        });
        if (this._zoneTransitions.length > 1000) {
          this._zoneTransitions = this._zoneTransitions.slice(-500);
        }
      }
    }

    const tenseChanged = tenseZone !== this._currentTenseZone;
    this._currentTenseZone = tenseZone;

    const durationInZoneMs = this._zoneEntryTimestamp
      ? Date.now() - this._zoneEntryTimestamp
      : 0;

    const zoneInfo = SYNTACTIC_ZONES[syntacticZone];
    const tenseInfo = TENSE_ZONES[tenseZone];

    this._prevWrist = { x: wrist.x, y: wrist.y, z: wrist.z || 0 };

    return {
      syntactic_zone: syntacticZone,
      syntactic_role: zoneInfo.role,
      syntactic_label: zoneInfo.label,
      tense_zone: tenseZone,
      tense_label: tenseInfo.label,
      wrist_position: {
        raw_x: wrist.x,
        raw_y: wrist.y,
        screen_x: screenX,
        screen_y: screenY,
        z: wrist.z || 0,
      },
      zone_confidence: zoneConfidence,
      movement_intensity: velocity,
      duration_in_zone_ms: durationInZoneMs,
      zone_changed: zoneChanged,
      tense_changed: tenseChanged,
      is_latched: this._latchedZone !== null,
      latched_zone: this._latchedZone,
      raw_zone: rawSyntacticZone,
      spatial_token: {
        role: zoneInfo.role,
        tense: tenseZone,
        confidence: zoneConfidence,
        intensity: velocity,
        duration_ms: durationInZoneMs,
      },
    };
  }

  // ===========================================================================
  // PUBLIC — Configuration
  // ===========================================================================

  /**
   * Update the zone strategy at runtime (e.g., when AccessibilityProfile changes).
   * @param {string} strategy — ZONE_STRATEGIES value
   */
  setZoneStrategy(strategy) {
    this._zoneStrategy = strategy;
    if (strategy !== ZONE_STRATEGIES.LATCHED) {
      this._latchedZone = null;
    }
  }

  /**
   * Update the zone widening at runtime.
   * @param {number} widening — 0 to 0.15
   */
  setZoneWidening(widening) {
    this._zoneWidening = Math.max(0, Math.min(0.15, widening));
    this._effectiveZones = this._computeEffectiveZones();
  }

  /**
   * Manually release the zone latch (e.g., when sentence is cleared).
   */
  releaseLatch() {
    this._latchedZone = null;
  }

  // ===========================================================================
  // PUBLIC — Zone transition history
  // ===========================================================================

  getZoneTransitions() {
    return [...this._zoneTransitions];
  }

  hasCompleteSVOPath() {
    const n = this._zoneTransitions.length;
    if (n < 2) return false;

    const recentZones = [];
    let lastZone = null;
    const startIdx = Math.max(0, n - 5);
    if (this._zoneTransitions[startIdx]) {
      recentZones.push(this._zoneTransitions[startIdx].from);
      lastZone = this._zoneTransitions[startIdx].from;
    }
    for (let i = startIdx; i < n; i++) {
      const t = this._zoneTransitions[i];
      if (t.to !== lastZone) {
        recentZones.push(t.to);
        lastZone = t.to;
      }
    }

    for (let i = 0; i <= recentZones.length - 3; i++) {
      if (
        recentZones[i]     === 'SUBJECT_ZONE' &&
        recentZones[i + 1] === 'VERB_ZONE' &&
        recentZones[i + 2] === 'OBJECT_ZONE'
      ) {
        return true;
      }
    }
    return false;
  }

  getCurrentState() {
    return {
      syntactic_zone: this._currentSyntacticZone,
      tense_zone: this._currentTenseZone,
      transitions: this._zoneTransitions.length,
      latched_zone: this._latchedZone,
      zone_strategy: this._zoneStrategy,
      zone_widening: this._zoneWidening,
    };
  }

  reset() {
    this._prevWrist = null;
    this._currentSyntacticZone = null;
    this._currentTenseZone = null;
    this._zoneEntryTimestamp = null;
    this._previousSyntacticZone = null;
    this._zoneTransitions = [];
    this._frameCount = 0;
    this._latchedZone = null;
    this._isGestureActive = false;
  }

  toDescriptor() {
    return {
      syntactic_zones: this._effectiveZones,
      tense_zones: TENSE_ZONES,
      current_syntactic_zone: this._currentSyntacticZone,
      current_tense_zone: this._currentTenseZone,
      total_transitions: this._zoneTransitions.length,
      has_svo_path: this.hasCompleteSVOPath(),
      zone_strategy: this._zoneStrategy,
      zone_widening: this._zoneWidening,
      latched_zone: this._latchedZone,
    };
  }

  // ===========================================================================
  // PRIVATE — Zone classification (with widening support)
  // ===========================================================================

  /**
   * Compute effective zone boundaries after applying widening.
   * Widening expands each zone's boundaries, creating overlap regions.
   * In overlap regions, the zone whose center is nearest wins.
   */
  _computeEffectiveZones() {
    const w = this._zoneWidening;
    return {
      SUBJECT_ZONE: {
        min: Math.max(0, SYNTACTIC_ZONES.SUBJECT_ZONE.min - w),
        max: SYNTACTIC_ZONES.SUBJECT_ZONE.max + w,
        center: (SYNTACTIC_ZONES.SUBJECT_ZONE.min + SYNTACTIC_ZONES.SUBJECT_ZONE.max) / 2,
      },
      VERB_ZONE: {
        min: SYNTACTIC_ZONES.VERB_ZONE.min - w,
        max: SYNTACTIC_ZONES.VERB_ZONE.max + w,
        center: (SYNTACTIC_ZONES.VERB_ZONE.min + SYNTACTIC_ZONES.VERB_ZONE.max) / 2,
      },
      OBJECT_ZONE: {
        min: SYNTACTIC_ZONES.OBJECT_ZONE.min - w,
        max: Math.min(1.0, SYNTACTIC_ZONES.OBJECT_ZONE.max + w),
        center: (SYNTACTIC_ZONES.OBJECT_ZONE.min + SYNTACTIC_ZONES.OBJECT_ZONE.max) / 2,
      },
    };
  }

  _classifyX(screenX) {
    if (this._zoneWidening === 0) {
      // Fast path: no widening, use hard boundaries
      if (screenX < SYNTACTIC_ZONES.SUBJECT_ZONE.max) return 'SUBJECT_ZONE';
      if (screenX < SYNTACTIC_ZONES.VERB_ZONE.max)    return 'VERB_ZONE';
      return 'OBJECT_ZONE';
    }

    // Widened zones: find all zones that contain screenX, pick nearest center
    const ez = this._effectiveZones;
    const candidates = [];

    for (const [name, zone] of Object.entries(ez)) {
      if (screenX >= zone.min && screenX <= zone.max) {
        candidates.push({ name, dist: Math.abs(screenX - zone.center) });
      }
    }

    if (candidates.length === 0) {
      // Edge case: shouldn't happen, fall back to hard boundary
      if (screenX < 0.35) return 'SUBJECT_ZONE';
      if (screenX < 0.65) return 'VERB_ZONE';
      return 'OBJECT_ZONE';
    }

    // Return the zone whose center is nearest
    candidates.sort((a, b) => a.dist - b.dist);
    return candidates[0].name;
  }

  _classifyY(screenY) {
    if (screenY < TENSE_ZONES.FUTURE.max)  return 'FUTURE';
    if (screenY < TENSE_ZONES.PRESENT.max) return 'PRESENT';
    return 'PAST';
  }

  _computeZoneConfidence(screenX, zone) {
    const zoneInfo = SYNTACTIC_ZONES[zone];
    const zoneCenter = (zoneInfo.min + zoneInfo.max) / 2;
    const zoneHalfWidth = (zoneInfo.max - zoneInfo.min) / 2;
    const distFromCenter = Math.abs(screenX - zoneCenter);
    return Math.max(0, 1 - (distFromCenter / zoneHalfWidth));
  }

  _computeVelocity(wrist) {
    if (!this._prevWrist) return 0;

    const dx = wrist.x - this._prevWrist.x;
    const dy = wrist.y - this._prevWrist.y;
    const dz = (wrist.z || 0) - this._prevWrist.z;
    const v = Math.sqrt(dx * dx + dy * dy + dz * dz);

    return v < VELOCITY_FLOOR ? 0 : Math.round(v * 10000) / 10000;
  }

  _noResult() {
    return {
      syntactic_zone: null,
      syntactic_role: null,
      syntactic_label: null,
      tense_zone: null,
      tense_label: null,
      wrist_position: null,
      zone_confidence: 0,
      movement_intensity: 0,
      duration_in_zone_ms: 0,
      zone_changed: false,
      tense_changed: false,
      is_latched: false,
      latched_zone: null,
      raw_zone: null,
      spatial_token: null,
    };
  }
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

export function syntacticZoneColor(zone) {
  switch (zone) {
    case 'SUBJECT_ZONE': return '#60a5fa';
    case 'VERB_ZONE':    return '#f87171';
    case 'OBJECT_ZONE':  return '#4ade80';
    default:             return '#64748b';
  }
}

export function syntacticZoneLabel(zone) {
  switch (zone) {
    case 'SUBJECT_ZONE': return 'S (NP)';
    case 'VERB_ZONE':    return 'V (VP)';
    case 'OBJECT_ZONE':  return 'O (NP)';
    default:             return '--';
  }
}

export { SYNTACTIC_ZONES, TENSE_ZONES };
export default SpatialGrammarMapper;
