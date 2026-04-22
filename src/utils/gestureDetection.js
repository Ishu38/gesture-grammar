/**
 * gestureDetection.js
 * Deterministic gesture detection with directional pronoun recognition,
 * action-based verb detection, and shape-based object detection.
 * Includes confidence lock mechanism for stability.
 *
 * v2: Uses 3D joint angles for finger curl detection (rotation-invariant),
 *     angle-based YOU/DRINK disambiguation, Z-depth fallback to Y-position,
 *     and smoothed tense detection with hysteresis.
 */

import { angleBetweenPoints3D } from './vectorGeometry.js';

// =============================================================================
// LANDMARK INDICES
// =============================================================================

const LANDMARKS = {
  WRIST: 0,
  // Thumb
  THUMB_CMC: 1,
  THUMB_MCP: 2,
  THUMB_IP: 3,
  THUMB_TIP: 4,
  // Index finger
  INDEX_MCP: 5,
  INDEX_PIP: 6,
  INDEX_DIP: 7,
  INDEX_TIP: 8,
  // Middle finger
  MIDDLE_MCP: 9,
  MIDDLE_PIP: 10,
  MIDDLE_DIP: 11,
  MIDDLE_TIP: 12,
  // Ring finger
  RING_MCP: 13,
  RING_PIP: 14,
  RING_DIP: 15,
  RING_TIP: 16,
  // Pinky finger
  PINKY_MCP: 17,
  PINKY_PIP: 18,
  PINKY_DIP: 19,
  PINKY_TIP: 20,
};

// =============================================================================
// CONFIDENCE LOCK STATE
// =============================================================================

export class ConfidenceLock {
  constructor(requiredFrames = 30) {
    this.requiredFrames = requiredFrames;
    this.currentGesture = null;
    this.consecutiveFrames = 0;
    this.lockedGesture = null;
    this.lastLockedGesture = null;
    this.lockCooldown = 0;
    this.cooldownFrames = 15;
  }

  update(detectedGesture) {
    if (this.lockCooldown > 0) {
      this.lockCooldown--;
    }

    if (!detectedGesture) {
      this.currentGesture = null;
      this.consecutiveFrames = 0;
      return {
        gesture: null,
        isLocked: false,
        progress: 0,
        display: null,
      };
    }

    if (detectedGesture === this.currentGesture) {
      this.consecutiveFrames++;
    } else {
      this.currentGesture = detectedGesture;
      this.consecutiveFrames = 1;
    }

    const progress = Math.min(this.consecutiveFrames / this.requiredFrames, 1);

    if (this.consecutiveFrames >= this.requiredFrames) {
      if (detectedGesture !== this.lastLockedGesture || this.lockCooldown === 0) {
        this.lockedGesture = detectedGesture;
        this.lastLockedGesture = detectedGesture;
        this.lockCooldown = this.cooldownFrames;
        this.consecutiveFrames = 0;

        return {
          gesture: this.lockedGesture,
          isLocked: true,
          progress: 1,
          display: detectedGesture,
        };
      }
    }

    return {
      gesture: null,
      isLocked: false,
      progress,
      display: detectedGesture,
    };
  }

  reset() {
    this.currentGesture = null;
    this.consecutiveFrames = 0;
    this.lockedGesture = null;
    this.lastLockedGesture = null;
    this.lockCooldown = 0;
  }

  clearLastLocked() {
    this.lastLockedGesture = null;
    this.lockCooldown = 0;
  }
}

export const confidenceLock = new ConfidenceLock(30);

// =============================================================================
// HELPER FUNCTIONS (3D angle-based — rotation invariant)
// =============================================================================

// Angle thresholds (degrees) for 3D joint-angle finger state detection
const EXTENDED_THRESHOLD = 155;  // Above this = finger extended
const CURLED_THRESHOLD = 130;    // Below this = finger curled
const SLIGHTLY_CURVED_MIN = 100; // Cupped hand: between 100° and 155°
// Thumb uses a more lenient threshold for subject pronoun detection
// (thumb anatomy differs from other fingers — shorter range of motion)
const THUMB_EXTENDED_FOR_SUBJECT = 140;

/**
 * Compute the PIP/IP joint angle for a finger using 3D vectors.
 * For index–pinky: angle at PIP (MCP→PIP→DIP)
 * For thumb: angle at IP (MCP→IP→TIP)
 * Returns degrees [0, 180].
 */
function fingerJointAngle(landmarks, mcpIndex, pipIndex, dipIndex) {
  return angleBetweenPoints3D(
    landmarks[mcpIndex],
    landmarks[pipIndex],
    landmarks[dipIndex]
  );
}

/**
 * Check if a finger is curled using 3D joint angle (rotation-invariant).
 */
function isFingerCurled(landmarks, tipIndex, pipIndex) {
  // Map tip→pip to the correct MCP→PIP→DIP triple
  const joint = TIP_TO_JOINT[tipIndex];
  if (joint) {
    return fingerJointAngle(landmarks, joint.mcp, joint.pip, joint.dip) < CURLED_THRESHOLD;
  }
  // Fallback for thumb (different kinematic chain)
  return landmarks[tipIndex].y > landmarks[pipIndex].y;
}

/**
 * Check if a finger is extended using 3D joint angle (rotation-invariant).
 */
function isFingerExtended(landmarks, tipIndex, pipIndex) {
  const joint = TIP_TO_JOINT[tipIndex];
  if (joint) {
    return fingerJointAngle(landmarks, joint.mcp, joint.pip, joint.dip) > EXTENDED_THRESHOLD;
  }
  return landmarks[tipIndex].y < landmarks[pipIndex].y;
}

// Lookup: tip landmark index → { mcp, pip, dip } for joint angle computation
const TIP_TO_JOINT = {
  [LANDMARKS.THUMB_TIP]:  { mcp: LANDMARKS.THUMB_MCP, pip: LANDMARKS.THUMB_IP,   dip: LANDMARKS.THUMB_TIP },
  [LANDMARKS.INDEX_TIP]:  { mcp: LANDMARKS.INDEX_MCP, pip: LANDMARKS.INDEX_PIP,  dip: LANDMARKS.INDEX_DIP },
  [LANDMARKS.MIDDLE_TIP]: { mcp: LANDMARKS.MIDDLE_MCP, pip: LANDMARKS.MIDDLE_PIP, dip: LANDMARKS.MIDDLE_DIP },
  [LANDMARKS.RING_TIP]:   { mcp: LANDMARKS.RING_MCP,  pip: LANDMARKS.RING_PIP,   dip: LANDMARKS.RING_DIP },
  [LANDMARKS.PINKY_TIP]:  { mcp: LANDMARKS.PINKY_MCP, pip: LANDMARKS.PINKY_PIP,  dip: LANDMARKS.PINKY_DIP },
};

/**
 * Check if finger is slightly curved (joint angle between SLIGHTLY_CURVED_MIN and EXTENDED_THRESHOLD)
 * Used for "cupped" hand detection
 */
function isFingerSlightlyCurved(landmarks, tipIndex, dipIndex, pipIndex) {
  // Use the correct triple for joint angle
  const joint = TIP_TO_JOINT[tipIndex];
  if (joint) {
    const angle = fingerJointAngle(landmarks, joint.mcp, joint.pip, joint.dip);
    return angle > SLIGHTLY_CURVED_MIN && angle < EXTENDED_THRESHOLD;
  }
  // Y-axis fallback
  const tipY = landmarks[tipIndex].y;
  const dipY = landmarks[dipIndex].y;
  const pipY = landmarks[pipIndex].y;
  return tipY > dipY && tipY < pipY;
}

/**
 * Calculate Euclidean distance between two landmarks
 */
function distance(lm1, lm2) {
  const dx = lm1.x - lm2.x;
  const dy = lm1.y - lm2.y;
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate 3D distance including depth
 */
function distance3D(lm1, lm2) {
  const dx = lm1.x - lm2.x;
  const dy = lm1.y - lm2.y;
  const dz = (lm1.z || 0) - (lm2.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Check Z-depth reliability. Returns false if z-values look unreliable
 * (all near zero, which happens on some consumer webcams).
 */
function isZDepthReliable(landmarks) {
  const wristZ = Math.abs(landmarks[LANDMARKS.WRIST].z || 0);
  const indexZ = Math.abs(landmarks[LANDMARKS.INDEX_TIP].z || 0);
  const middleZ = Math.abs(landmarks[LANDMARKS.MIDDLE_TIP].z || 0);
  // If all key z-values are near zero, depth is unreliable.
  // Threshold 0.015 requires meaningful z-variation (avg ~0.005 per landmark).
  return (wristZ + indexZ + middleZ) > 0.015;
}

/**
 * Check if all four fingers (not thumb) are curled (3D angle-based)
 */
function areFourFingersCurled(landmarks) {
  return (
    isFingerCurled(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP) &&
    isFingerCurled(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP) &&
    isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP) &&
    isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP)
  );
}

/**
 * Check if all five fingers are extended (3D angle-based)
 */
function areAllFingersExtended(landmarks) {
  const thumbAngle = fingerJointAngle(landmarks, LANDMARKS.THUMB_MCP, LANDMARKS.THUMB_IP, LANDMARKS.THUMB_TIP);
  const thumbExtended = thumbAngle > EXTENDED_THRESHOLD;
  return (
    thumbExtended &&
    isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP) &&
    isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP) &&
    isFingerExtended(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP) &&
    isFingerExtended(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP)
  );
}

/**
 * Check if all five fingers are curled (3D angle-based)
 */
function areAllFingersCurled(landmarks) {
  const thumbAngle = fingerJointAngle(landmarks, LANDMARKS.THUMB_MCP, LANDMARKS.THUMB_IP, LANDMARKS.THUMB_TIP);
  const thumbCurled = thumbAngle < CURLED_THRESHOLD;
  return thumbCurled && areFourFingersCurled(landmarks);
}

/**
 * Calculate average distance from thumb tip to all other finger tips
 */
function getAverageThumbToFingerDistance(landmarks) {
  const thumbTip = landmarks[LANDMARKS.THUMB_TIP];
  const indexTip = landmarks[LANDMARKS.INDEX_TIP];
  const middleTip = landmarks[LANDMARKS.MIDDLE_TIP];
  const ringTip = landmarks[LANDMARKS.RING_TIP];
  const pinkyTip = landmarks[LANDMARKS.PINKY_TIP];

  const d1 = distance(thumbTip, indexTip);
  const d2 = distance(thumbTip, middleTip);
  const d3 = distance(thumbTip, ringTip);
  const d4 = distance(thumbTip, pinkyTip);

  return (d1 + d2 + d3 + d4) / 4;
}

/**
 * Check if hand is vertical (wrist below finger tips)
 */
function isHandVertical(landmarks) {
  const wristY = landmarks[LANDMARKS.WRIST].y;
  const middleTipY = landmarks[LANDMARKS.MIDDLE_TIP].y;
  return wristY > middleTipY + 0.1; // Wrist is below (higher y value)
}

/**
 * Check if palm is facing up.
 * Uses z-coordinates when reliable, falls back to Y-position heuristic.
 */
function isPalmFacingUp(landmarks) {
  const wrist = landmarks[LANDMARKS.WRIST];
  const middleMCP = landmarks[LANDMARKS.MIDDLE_MCP];
  const middleTip = landmarks[LANDMARKS.MIDDLE_TIP];

  const fingersFlat = Math.abs(middleTip.y - middleMCP.y) < 0.15;

  if (isZDepthReliable(landmarks)) {
    // Z-based: palm surface facing up when MCP z > wrist z
    const palmNormalUp = (middleMCP.z || 0) > (wrist.z || 0) - 0.02;
    return palmNormalUp && fingersFlat;
  }

  // Y-fallback: palm up = hand roughly horizontal, wrist and fingertips at similar Y
  const handHorizontal = Math.abs(wrist.y - middleTip.y) < 0.12;
  return handHorizontal && fingersFlat;
}

// =============================================================================
// SUBJECT (PRONOUN) DETECTION
// =============================================================================

/**
 * Detect subject pronouns based on directional thumb/finger gestures
 * @param {Array} landmarks - 21 hand landmarks
 * @returns {Object|null} { type, value, person, grammar_id } or null
 */
export function detectSubject(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  const wrist = landmarks[LANDMARKS.WRIST];
  const thumbTip = landmarks[LANDMARKS.THUMB_TIP];
  const thumbIP = landmarks[LANDMARKS.THUMB_IP];
  const indexTip = landmarks[LANDMARKS.INDEX_TIP];

  // -------------------------------------------------------------------------
  // 1. 'I' (First Person) - Thumb pointing at self
  // Math Rule: Thumb Tip x > Wrist x (thumb pointing inward)
  //            Four fingers must be curled
  //            Uses 3D angle for thumb extension check
  // -------------------------------------------------------------------------

  const thumbPointingInward = thumbTip.x > wrist.x + 0.05;
  const thumbAngleI = fingerJointAngle(landmarks, LANDMARKS.THUMB_MCP, LANDMARKS.THUMB_IP, LANDMARKS.THUMB_TIP);
  const thumbExtendedForI = thumbAngleI > THUMB_EXTENDED_FOR_SUBJECT;

  if (thumbPointingInward && thumbExtendedForI && areFourFingersCurled(landmarks)) {
    return {
      type: 'SUBJECT',
      value: 'I',
      person: 1,
      number: 'singular',
      grammar_id: 'SUBJECT_I',
    };
  }

  // -------------------------------------------------------------------------
  // 2. 'YOU' (Second Person) - Index finger pointing at camera
  // Math Rule: Index finger extended, pointing forward
  //            Other fingers curled, thumb tucked in (not spread for C-shape)
  // Disambiguate from DRINK: Use the ANGLE at the index MCP between
  //            thumb-tip → index-MCP → index-tip. For YOU the thumb is
  //            tucked alongside the hand (angle > 140°), for DRINK the
  //            thumb spreads out to form a C-gap (angle < 120°).
  // Z-depth fallback: if z unreliable, use Y-position check instead.
  // -------------------------------------------------------------------------

  const indexExtended = isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP);
  // Z-depth check with Y-position fallback for consumer webcams
  const zReliable = isZDepthReliable(landmarks);
  const indexPointingForward = zReliable
    ? (indexTip.z || 0) < (wrist.z || 0) - 0.005  // Relaxed z-threshold
    : indexTip.y < wrist.y - 0.05;                  // Y fallback: index above wrist
  const indexAboveWrist = indexTip.y < wrist.y;
  const middleCurled = isFingerCurled(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP);
  const ringCurled = isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP);
  const pinkyCurled = isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP);
  // Angle-based thumb discrimination (replaces distance-based overlap)
  // Thumb-tip → Index-MCP → Index-tip angle: tight (>130°) means tucked (YOU),
  // wide (<120°) means C-shape (DRINK)
  const thumbIndexAngle = angleBetweenPoints3D(
    thumbTip, landmarks[LANDMARKS.INDEX_MCP], indexTip
  );
  const thumbTuckedAngle = thumbIndexAngle > 130;

  if (indexExtended && indexPointingForward && indexAboveWrist && middleCurled && ringCurled && pinkyCurled && thumbTuckedAngle) {
    return {
      type: 'SUBJECT',
      value: 'YOU',
      person: 2,
      number: 'singular',
      grammar_id: 'SUBJECT_YOU',
    };
  }

  // -------------------------------------------------------------------------
  // 3. 'HE/SHE' (Third Person) - Thumb pointing to the side (Hitchhiker)
  // Math Rule: Thumb Tip x significantly less than Wrist x (pointing away)
  //            Index/Middle/Ring/Pinky are curled
  //            Uses 3D angle for thumb extension check
  // -------------------------------------------------------------------------

  const thumbPointingOutward = thumbTip.x < wrist.x - 0.05;
  const thumbAngleHe = fingerJointAngle(landmarks, LANDMARKS.THUMB_MCP, LANDMARKS.THUMB_IP, LANDMARKS.THUMB_TIP);
  const thumbExtendedForHe = thumbAngleHe > THUMB_EXTENDED_FOR_SUBJECT;

  if (thumbPointingOutward && thumbExtendedForHe && areFourFingersCurled(landmarks)) {
    // Disambiguate HE vs SHE: HE = thumb points right (x < wrist.x),
    // SHE = pinky side points left (thumb tip still outward but hand rotated)
    // Convention: thumb pointing to viewer's LEFT = HE, pinky-led point LEFT = SHE
    // Simpler: if thumb is pointing down-outward (below wrist Y), treat as SHE
    const thumbBelowWrist = thumbTip.y > wrist.y + 0.03;
    if (thumbBelowWrist) {
      return {
        type: 'SUBJECT',
        value: 'SHE',
        person: 3,
        number: 'singular',
        grammar_id: 'SUBJECT_SHE',
      };
    }
    return {
      type: 'SUBJECT',
      value: 'HE',
      person: 3,
      number: 'singular',
      grammar_id: 'SUBJECT_HE',
    };
  }

  // -------------------------------------------------------------------------
  // 4. 'WE' (First Person Plural) - Index + Middle extended, others curled
  // Math Rule: Index and Middle fingers extended and close together
  //            Ring and Pinky curled, thumb curled or tucked
  //            Hand vertical (pointing up)
  // -------------------------------------------------------------------------

  const indexExtWe = isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP);
  const middleExtWe = isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP);
  const ringCurledWe = isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP);
  const pinkyCurledWe = isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP);
  const indexMiddleClose = distance(indexTip, landmarks[LANDMARKS.MIDDLE_TIP]) < 0.06;

  if (indexExtWe && middleExtWe && ringCurledWe && pinkyCurledWe && indexMiddleClose && isHandVertical(landmarks)) {
    return {
      type: 'SUBJECT',
      value: 'WE',
      person: 1,
      number: 'plural',
      grammar_id: 'SUBJECT_WE',
    };
  }

  // -------------------------------------------------------------------------
  // 5. 'THEY' (Third Person Plural) - All 5 fingers extended and spread
  // Math Rule: All fingers extended, fingers spread apart
  //            Hand NOT vertical (distinguishes from STOP)
  //            Palm facing camera (horizontal or slight angle)
  // -------------------------------------------------------------------------

  if (areAllFingersExtended(landmarks) && !isHandVertical(landmarks)) {
    // Check fingers are spread wide
    const idxMidSpread = distance(indexTip, landmarks[LANDMARKS.MIDDLE_TIP]) > 0.05;
    const midRingSpread = distance(landmarks[LANDMARKS.MIDDLE_TIP], landmarks[LANDMARKS.RING_TIP]) > 0.04;
    if (idxMidSpread && midRingSpread) {
      return {
        type: 'SUBJECT',
        value: 'THEY',
        person: 3,
        number: 'plural',
        grammar_id: 'SUBJECT_THEY',
      };
    }
  }

  return null;
}

// =============================================================================
// VERB DETECTION
// =============================================================================

/**
 * Detect verb gestures based on action shapes
 * @param {Array} landmarks - 21 hand landmarks
 * @returns {Object|null} { type, value, transitive, grammar_id } or null
 */
export function detectVerb(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  const thumbTip = landmarks[LANDMARKS.THUMB_TIP];
  const indexTip = landmarks[LANDMARKS.INDEX_TIP];
  const middleTip = landmarks[LANDMARKS.MIDDLE_TIP];
  const ringTip = landmarks[LANDMARKS.RING_TIP];
  const pinkyTip = landmarks[LANDMARKS.PINKY_TIP];

  // -------------------------------------------------------------------------
  // 1. 'GRAB / TAKE' (Transitive Action) - The Claw/Pinch
  // Math Rule: All 5 finger tips brought close together
  //            Average distance from thumb to other 4 tips < 0.05
  // Priority: HIGH (check first as it's most specific)
  // -------------------------------------------------------------------------

  const avgThumbDistance = getAverageThumbToFingerDistance(landmarks);

  if (avgThumbDistance < 0.06) {
    return {
      type: 'VERB',
      value: 'grab',
      display: 'grab',
      transitive: true,
      requires_s_form: false,
      grammar_id: 'GRAB',
      s_form_pair: 'GRABS',
      gesture_name: 'CLAW',
    };
  }

  // -------------------------------------------------------------------------
  // 2. 'DRINK' (Action) - The 'C' shape (holding a cup)
  // Math Rule: Thumb and Index spread apart forming a C-gap
  //            Middle/Ring/Pinky are curled
  //            Thumb-Index opening angle at index-MCP < 120° (C-shape)
  //            (vs YOU where thumb is tucked: angle > 130°)
  // -------------------------------------------------------------------------

  const thumbIndexDist = distance(thumbTip, indexTip);
  const thumbIndexYDiff = Math.abs(thumbTip.y - indexTip.y);
  const middleRingPinkyCurled = (
    isFingerCurled(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP) &&
    isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP) &&
    isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP)
  );

  // Angle-based C-shape: thumb spread away from index (angle < 120° at index MCP)
  const drinkThumbAngle = angleBetweenPoints3D(
    thumbTip, landmarks[LANDMARKS.INDEX_MCP], indexTip
  );
  const isCShape = drinkThumbAngle < 120;

  if (isCShape && thumbIndexDist > 0.05 && thumbIndexDist < 0.25 && thumbIndexYDiff < 0.12 && middleRingPinkyCurled) {
    return {
      type: 'VERB',
      value: 'drink',
      display: 'drink',
      transitive: true,
      requires_s_form: false,
      grammar_id: 'DRINK',
      s_form_pair: 'DRINKS',
      gesture_name: 'C_SHAPE',
    };
  }

  // -------------------------------------------------------------------------
  // 3. 'STOP' (Stative Verb) - Open Palm (Stop sign)
  // Math Rule: All 5 fingers extended (Tip y < PIP y)
  //            Hand is vertical (Wrist y > Finger Tips y)
  // -------------------------------------------------------------------------

  if (areAllFingersExtended(landmarks) && isHandVertical(landmarks)) {
    return {
      type: 'VERB',
      value: 'stop',
      display: 'stop',
      transitive: false,
      requires_s_form: false,
      grammar_id: 'STOP',
      s_form_pair: 'STOPS',
      gesture_name: 'OPEN_PALM',
    };
  }

  // -------------------------------------------------------------------------
  // 4. 'WANT' (Desire) - Claw / half-curl (all fingers partially curled)
  // Math Rule: All fingers in slightly-curved range (100°–155°)
  //            Fingers spread apart (not bunched like GRAB)
  //            Distinguishes from APPLE (WANT = hand vertical, APPLE = relaxed)
  // -------------------------------------------------------------------------

  const wantIndexCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_DIP, LANDMARKS.INDEX_PIP);
  const wantMiddleCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_DIP, LANDMARKS.MIDDLE_PIP);
  const wantRingCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_DIP, LANDMARKS.RING_PIP);
  const wantPinkyCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_DIP, LANDMARKS.PINKY_PIP);
  const wantAvgDist = getAverageThumbToFingerDistance(landmarks);

  if (wantIndexCurved && wantMiddleCurved && wantRingCurved && wantPinkyCurved &&
      wantAvgDist > 0.06 && isHandVertical(landmarks)) {
    return {
      type: 'VERB',
      value: 'want',
      display: 'want',
      transitive: true,
      requires_s_form: false,
      grammar_id: 'WANT',
      s_form_pair: 'WANTS',
      gesture_name: 'CLAW_OPEN',
    };
  }

  // -------------------------------------------------------------------------
  // 5. 'EAT' (Action) - Fingertips bunched together, brought toward mouth
  // Math Rule: All fingertips close together (like GRAB but less tight)
  //            Hand near face level (wrist Y < 0.35, upper portion of frame)
  //            Distinguishes from GRAB by being slightly more open
  // -------------------------------------------------------------------------

  const eatAvgDist = getAverageThumbToFingerDistance(landmarks);
  const handNearFace = landmarks[LANDMARKS.WRIST].y < 0.35;

  if (eatAvgDist >= 0.06 && eatAvgDist < 0.12 && handNearFace && areFourFingersCurled(landmarks)) {
    return {
      type: 'VERB',
      value: 'eat',
      display: 'eat',
      transitive: true,
      requires_s_form: false,
      grammar_id: 'EAT',
      s_form_pair: 'EATS',
      gesture_name: 'BUNCHED_TO_MOUTH',
    };
  }

  // -------------------------------------------------------------------------
  // 6. 'SEE' (Perception) - V-shape: index + middle extended and spread
  // Math Rule: Index and Middle extended and spread apart
  //            Ring and Pinky curled
  //            Hand near eye level, fingers pointing at eyes (like peace sign
  //            near face)
  //            Distinguishes from WE (WE = fingers together, SEE = spread)
  // -------------------------------------------------------------------------

  const seeIndexExt = isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP);
  const seeMiddleExt = isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP);
  const seeRingCurled = isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP);
  const seePinkyCurled = isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP);
  const seeFingerSpread = distance(indexTip, middleTip) > 0.06;

  if (seeIndexExt && seeMiddleExt && seeRingCurled && seePinkyCurled && seeFingerSpread) {
    return {
      type: 'VERB',
      value: 'see',
      display: 'see',
      transitive: true,
      requires_s_form: false,
      grammar_id: 'SEE',
      s_form_pair: 'SEES',
      gesture_name: 'V_SHAPE',
    };
  }

  // -------------------------------------------------------------------------
  // 7. 'GO' (Motion) - Index finger extended pointing forward, flick motion
  // Math Rule: Only index finger extended, pointing forward (z-depth or y)
  //            Other fingers curled, thumb NOT tucked tightly (relaxed)
  //            Distinguishes from YOU: GO has a more relaxed thumb (not tucked)
  //            and hand is NOT vertical
  // -------------------------------------------------------------------------

  const goIndexExt = isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP);
  const goMiddleCurled = isFingerCurled(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP);
  const goRingCurled = isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP);
  const goPinkyCurled = isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP);
  const goHandNotVertical = !isHandVertical(landmarks);
  // GO: thumb relaxed (not tightly tucked) — angle at index MCP between 90° and 130°
  const goThumbAngle = angleBetweenPoints3D(
    thumbTip, landmarks[LANDMARKS.INDEX_MCP], indexTip
  );
  const goThumbRelaxed = goThumbAngle > 90 && goThumbAngle <= 130;

  if (goIndexExt && goMiddleCurled && goRingCurled && goPinkyCurled && goHandNotVertical && goThumbRelaxed) {
    return {
      type: 'VERB',
      value: 'go',
      display: 'go',
      transitive: false,
      requires_s_form: false,
      grammar_id: 'GO',
      s_form_pair: 'GOES',
      gesture_name: 'INDEX_POINT_FORWARD',
    };
  }

  return null;
}

// =============================================================================
// OBJECT DETECTION
// =============================================================================

/**
 * Detect object gestures based on shape mimicry
 * @param {Array} landmarks - 21 hand landmarks
 * @returns {Object|null} { type, value, grammar_id } or null
 */
export function detectObject(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  const indexTip = landmarks[LANDMARKS.INDEX_TIP];
  const middleTip = landmarks[LANDMARKS.MIDDLE_TIP];

  // -------------------------------------------------------------------------
  // 1. 'APPLE' (Small Round Object) - Cupped hand
  // Math Rule: All fingers slightly curved (Tip y > DIP y but Tip y < PIP y)
  //            Looks like a loose fist with open center
  // -------------------------------------------------------------------------

  const indexSlightlyCurved = isFingerSlightlyCurved(
    landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_DIP, LANDMARKS.INDEX_PIP
  );
  const middleSlightlyCurved = isFingerSlightlyCurved(
    landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_DIP, LANDMARKS.MIDDLE_PIP
  );
  const ringSlightlyCurved = isFingerSlightlyCurved(
    landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_DIP, LANDMARKS.RING_PIP
  );
  const pinkySlightlyCurved = isFingerSlightlyCurved(
    landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_DIP, LANDMARKS.PINKY_PIP
  );

  // Check thumb is also curved/cupped
  const thumbCupped = landmarks[LANDMARKS.THUMB_TIP].y > landmarks[LANDMARKS.THUMB_IP].y - 0.05;

  if (indexSlightlyCurved && middleSlightlyCurved && ringSlightlyCurved && pinkySlightlyCurved && thumbCupped) {
    return {
      type: 'OBJECT',
      value: 'apple',
      display: 'apple',
      grammar_id: 'APPLE',
      gesture_name: 'CUPPED_HAND',
    };
  }

  // -------------------------------------------------------------------------
  // 2. 'BOOK' (Flat Object) - Flat hand, palm facing up
  // Math Rule: All fingers extended and flat
  //            Palm normal (z-axis) facing up
  // -------------------------------------------------------------------------

  const allFingersExtended = areAllFingersExtended(landmarks);
  const palmUp = isPalmFacingUp(landmarks);
  const handHorizontal = !isHandVertical(landmarks); // Not vertical

  if (allFingersExtended && palmUp && handHorizontal) {
    return {
      type: 'OBJECT',
      value: 'book',
      display: 'book',
      grammar_id: 'BOOK',
      gesture_name: 'FLAT_PALM_UP',
    };
  }

  // -------------------------------------------------------------------------
  // 3. 'HOUSE' (Structure) - Index and Middle fingertips touching (inverted V)
  // Math Rule: Index and Middle fingers extended and tips touching
  //            Ring and Pinky curled
  //            Forms a "roof" or "tent" shape
  // -------------------------------------------------------------------------

  const indexExtended = isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP);
  const middleExtended = isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP);
  const ringCurled = isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP);
  const pinkyCurled = isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP);

  const indexMiddleTipDistance = distance(indexTip, middleTip);
  const fingertipsTouching = indexMiddleTipDistance < 0.04;

  if (indexExtended && middleExtended && ringCurled && pinkyCurled && fingertipsTouching) {
    return {
      type: 'OBJECT',
      value: 'house',
      display: 'house',
      grammar_id: 'HOUSE',
      gesture_name: 'ROOF_SHAPE',
    };
  }

  // -------------------------------------------------------------------------
  // 4. 'WATER' (Liquid) - W shape with three middle fingers
  // Math Rule: Index, Middle, Ring extended and spread
  //            Thumb and Pinky curled
  // -------------------------------------------------------------------------

  const indexExt = isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP);
  const middleExt = isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP);
  const ringExt = isFingerExtended(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP);
  const pinkyNotExt = isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP);
  const thumbNotExt = landmarks[LANDMARKS.THUMB_TIP].x > landmarks[LANDMARKS.WRIST].x - 0.05;

  // Check fingers are spread (not touching)
  const indexMiddleSpread = distance(indexTip, middleTip) > 0.04;
  const middleRingSpread = distance(middleTip, landmarks[LANDMARKS.RING_TIP]) > 0.04;

  if (indexExt && middleExt && ringExt && pinkyNotExt && thumbNotExt && indexMiddleSpread && middleRingSpread) {
    return {
      type: 'OBJECT',
      value: 'water',
      display: 'water',
      grammar_id: 'WATER',
      gesture_name: 'W_SHAPE',
    };
  }

  // -------------------------------------------------------------------------
  // 5. 'BALL' (Round Object) - Symmetric cupped hand, fingers spread in dome
  // Math Rule: All fingers slightly curved (like APPLE) but MORE spread
  //            Thumb opposed (spread away from fingers)
  //            Distinguishes from APPLE: BALL has wider finger spread
  // -------------------------------------------------------------------------

  const ballIndexCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_DIP, LANDMARKS.INDEX_PIP);
  const ballMiddleCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_DIP, LANDMARKS.MIDDLE_PIP);
  const ballRingCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_DIP, LANDMARKS.RING_PIP);
  const ballPinkyCurved = isFingerSlightlyCurved(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_DIP, LANDMARKS.PINKY_PIP);
  // Ball has wider spread between fingers than apple
  const ballIndexMiddleSpread = distance(landmarks[LANDMARKS.INDEX_TIP], landmarks[LANDMARKS.MIDDLE_TIP]) > 0.05;
  const ballThumbSpread = distance(landmarks[LANDMARKS.THUMB_TIP], landmarks[LANDMARKS.INDEX_TIP]) > 0.1;

  if (ballIndexCurved && ballMiddleCurved && ballRingCurved && ballPinkyCurved && ballIndexMiddleSpread && ballThumbSpread) {
    return {
      type: 'OBJECT',
      value: 'ball',
      display: 'ball',
      grammar_id: 'BALL',
      gesture_name: 'CUPPED_SPREAD',
    };
  }

  // -------------------------------------------------------------------------
  // 6. 'FOOD' (General) - Flat hand, palm down, fingers together
  // Math Rule: All fingers extended and close together (not spread)
  //            Palm facing down (opposite of BOOK which is palm up)
  //            Hand horizontal
  // -------------------------------------------------------------------------

  const foodAllExt = areAllFingersExtended(landmarks);
  const foodNotVertical = !isHandVertical(landmarks);
  const foodPalmNotUp = !isPalmFacingUp(landmarks);
  // Fingers close together (not spread like THEY)
  const foodIdxMidClose = distance(landmarks[LANDMARKS.INDEX_TIP], landmarks[LANDMARKS.MIDDLE_TIP]) < 0.05;
  const foodMidRingClose = distance(landmarks[LANDMARKS.MIDDLE_TIP], landmarks[LANDMARKS.RING_TIP]) < 0.05;

  if (foodAllExt && foodNotVertical && foodPalmNotUp && foodIdxMidClose && foodMidRingClose) {
    return {
      type: 'OBJECT',
      value: 'food',
      display: 'food',
      grammar_id: 'FOOD',
      gesture_name: 'FLAT_PALM_DOWN',
    };
  }

  return null;
}

// =============================================================================
// SPATIAL MODIFIER (TENSE) DETECTION — with hysteresis & EMA smoothing
// =============================================================================

// EMA state for tense smoothing (prevents flipping on noisy boundaries)
// NOTE: Module-level state is shared across all callers within the same session.
// Use resetTenseState() when switching contexts (e.g., new session, HMR reload).
let _tenseEmaValue = 0.5; // 0=future, 0.5=present, 1=past
let _currentTense = 'present';

// Hysteresis thresholds: must cross further to switch, preventing oscillation
const TENSE_HYSTERESIS = {
  futureEnter: 0.25,  // Must go below this to enter future
  futureExit: 0.35,   // Must go above this to leave future
  pastEnter: 0.75,    // Must go above this to enter past
  pastExit: 0.65,     // Must go below this to leave past
};

const TENSE_EMA_ALPHA = 0.15; // Low alpha = more smoothing

/**
 * Detect the spatial modifier (tense) from wrist Y position with EMA smoothing.
 * Uses wrist Y (reliable on all cameras) instead of z-depth.
 * High hand = future, middle = present, low hand = past.
 * @param {Array} landmarks
 * @returns {string} 'past' | 'present' | 'future'
 */
export function detectSpatialModifier(landmarks) {
  if (!landmarks || landmarks.length < 21) return 'present';

  const wristY = landmarks[0].y; // Y: 0=top, 1=bottom

  // EMA smooth the wrist Y position
  _tenseEmaValue = TENSE_EMA_ALPHA * wristY + (1 - TENSE_EMA_ALPHA) * _tenseEmaValue;

  // Apply hysteresis based on current state
  switch (_currentTense) {
    case 'present':
      if (_tenseEmaValue < TENSE_HYSTERESIS.futureEnter) _currentTense = 'future';
      else if (_tenseEmaValue > TENSE_HYSTERESIS.pastEnter) _currentTense = 'past';
      break;
    case 'future':
      if (_tenseEmaValue > TENSE_HYSTERESIS.futureExit) _currentTense = 'present';
      break;
    case 'past':
      if (_tenseEmaValue < TENSE_HYSTERESIS.pastExit) _currentTense = 'present';
      break;
  }

  return _currentTense;
}

/**
 * Reset tense EMA state (call on session start/end).
 */
export function resetTenseState() {
  _tenseEmaValue = 0.5;
  _currentTense = 'present';
}

// MAIN GESTURE DETECTION (with priority)
// =============================================================================

/**
 * Detect gesture from hand landmarks (raw detection, no confidence lock)
 * Priority order prevents misclassification
 * @param {Array} landmarks - 21 hand landmarks
 * @returns {string|null} Grammar ID or null
 */
export function detectGestureRaw(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  // Priority 1: GRAB/CLAW — most specific (all fingertips bunched)
  const avgThumbDist = getAverageThumbToFingerDistance(landmarks);
  if (avgThumbDist < 0.06) {
    return 'GRAB';
  }

  // Priority 2: STOP — open palm, all fingers extended + hand vertical
  if (areAllFingersExtended(landmarks) && isHandVertical(landmarks)) {
    return 'STOP';
  }

  // Priority 3: DRINK — C-shape (thumb-index gap with others curled)
  // Must check BEFORE subjects so C-shape isn't mistaken for YOU
  const verb = detectVerb(landmarks);
  if (verb && verb.grammar_id === 'DRINK') {
    return 'DRINK';
  }

  // Priority 4: SUBJECTS — directional pronouns
  const subject = detectSubject(landmarks);
  if (subject) {
    return subject.grammar_id;
  }

  // Priority 5: OBJECTS — shape-based
  const object = detectObject(landmarks);
  if (object) {
    return object.grammar_id;
  }

  // Priority 6: SEE (V-shape) — check before WE since SEE = spread, WE = together
  if (verb && verb.grammar_id === 'SEE') {
    return 'SEE';
  }

  // Priority 7: Remaining VERBS (EAT, WANT, GO)
  if (verb) {
    return verb.grammar_id;
  }

  // No gesture detected — return null rather than guessing.
  // A closed fist is ambiguous (resting hand, transition state) and
  // should not be interpreted as SUBJECT_I to avoid false positives.
  return null;
}

/**
 * Detect gesture with confidence lock (use this for stable detection)
 * @param {Array} landmarks - 21 hand landmarks
 * @returns {Object} { gesture, isLocked, progress, display }
 */
export function detectGestureWithConfidence(landmarks) {
  const rawGesture = detectGestureRaw(landmarks);
  return confidenceLock.update(rawGesture);
}


// =============================================================================
// DEBUG UTILITIES
// =============================================================================

/**
 * Get debug info about current hand state
 */
export function getHandDebugInfo(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  const wrist = landmarks[LANDMARKS.WRIST];
  const thumbTip = landmarks[LANDMARKS.THUMB_TIP];
  const indexTip = landmarks[LANDMARKS.INDEX_TIP];
  const middleTip = landmarks[LANDMARKS.MIDDLE_TIP];

  const avgThumbDist = getAverageThumbToFingerDistance(landmarks);
  const thumbIndexDist = distance(thumbTip, indexTip);
  const indexMiddleDist = distance(indexTip, middleTip);

  return {
    // Position data
    thumbTipX: thumbTip.x.toFixed(3),
    wristX: wrist.x.toFixed(3),
    thumbDirection: thumbTip.x > wrist.x ? 'INWARD (I)' : 'OUTWARD (HE)',
    thumbDeltaX: (thumbTip.x - wrist.x).toFixed(3),

    // Depth data
    indexTipZ: (indexTip.z || 0).toFixed(3),
    wristZ: (wrist.z || 0).toFixed(3),
    indexPointingForward: (indexTip.z || 0) < (wrist.z || 0) - 0.02,

    // Distance data
    avgThumbToFingers: avgThumbDist.toFixed(3),
    thumbIndexDist: thumbIndexDist.toFixed(3),
    indexMiddleDist: indexMiddleDist.toFixed(3),

    // State checks
    isGrab: avgThumbDist < 0.06,
    isDrink: thumbIndexDist > 0.08 && thumbIndexDist < 0.2,
    isHouse: indexMiddleDist < 0.04,
    fourFingersCurled: areFourFingersCurled(landmarks),
    allFingersCurled: areAllFingersCurled(landmarks),
    allFingersExtended: areAllFingersExtended(landmarks),
    handVertical: isHandVertical(landmarks),
    palmUp: isPalmFacingUp(landmarks),
  };
}
