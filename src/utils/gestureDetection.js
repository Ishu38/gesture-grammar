/**
 * gestureDetection.js
 * Deterministic gesture detection with directional pronoun recognition,
 * action-based verb detection, and shape-based object detection.
 * Includes confidence lock mechanism for stability.
 */

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

class ConfidenceLock {
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
// HELPER FUNCTIONS
// =============================================================================

/**
 * Check if a finger is curled (tip below PIP joint)
 */
function isFingerCurled(landmarks, tipIndex, pipIndex) {
  return landmarks[tipIndex].y > landmarks[pipIndex].y;
}

/**
 * Check if a finger is extended (tip above PIP joint)
 */
function isFingerExtended(landmarks, tipIndex, pipIndex) {
  return landmarks[tipIndex].y < landmarks[pipIndex].y;
}

/**
 * Check if finger is slightly curved (tip below DIP but above PIP)
 * Used for "cupped" hand detection
 */
function isFingerSlightlyCurved(landmarks, tipIndex, dipIndex, pipIndex) {
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
 * Check if all four fingers (not thumb) are curled
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
 * Check if all five fingers are extended
 */
function areAllFingersExtended(landmarks) {
  const thumbExtended = landmarks[LANDMARKS.THUMB_TIP].y < landmarks[LANDMARKS.THUMB_IP].y;
  return (
    thumbExtended &&
    isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP) &&
    isFingerExtended(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP) &&
    isFingerExtended(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP) &&
    isFingerExtended(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP)
  );
}

/**
 * Check if all five fingers are curled
 */
function areAllFingersCurled(landmarks) {
  const thumbCurled = landmarks[LANDMARKS.THUMB_TIP].y > landmarks[LANDMARKS.THUMB_IP].y;
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
 * Check if palm is facing up (using z-coordinates)
 */
function isPalmFacingUp(landmarks) {
  const wrist = landmarks[LANDMARKS.WRIST];
  const middleMCP = landmarks[LANDMARKS.MIDDLE_MCP];
  const middleTip = landmarks[LANDMARKS.MIDDLE_TIP];

  // Palm up: middle finger MCP z > wrist z (palm surface facing up)
  // And fingers are relatively flat (not curled toward palm)
  const palmNormalUp = (middleMCP.z || 0) > (wrist.z || 0) - 0.02;
  const fingersFlat = Math.abs(middleTip.y - middleMCP.y) < 0.15;

  return palmNormalUp && fingersFlat;
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
  // -------------------------------------------------------------------------

  const thumbPointingInward = thumbTip.x > wrist.x + 0.05;
  const thumbExtendedForI = thumbTip.y < thumbIP.y || Math.abs(thumbTip.y - thumbIP.y) < 0.1;

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
  // Math Rule: Index finger extended, pointing forward (z-depth)
  //            Other fingers curled
  // -------------------------------------------------------------------------

  const indexExtended = isFingerExtended(landmarks, LANDMARKS.INDEX_TIP, LANDMARKS.INDEX_PIP);
  const indexPointingForward = (indexTip.z || 0) < (wrist.z || 0) - 0.02;
  const middleCurled = isFingerCurled(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP);
  const ringCurled = isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP);
  const pinkyCurled = isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP);
  const thumbCurledOrNeutral = thumbTip.x < wrist.x + 0.1;

  if (indexExtended && indexPointingForward && middleCurled && ringCurled && pinkyCurled && thumbCurledOrNeutral) {
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
  // -------------------------------------------------------------------------

  const thumbPointingOutward = thumbTip.x < wrist.x - 0.08;
  const thumbExtendedForHe = thumbTip.y < thumbIP.y + 0.05;

  if (thumbPointingOutward && thumbExtendedForHe && areFourFingersCurled(landmarks)) {
    return {
      type: 'SUBJECT',
      value: 'HE',
      person: 3,
      number: 'singular',
      grammar_id: 'SUBJECT_HE',
    };
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
  // Math Rule: Thumb and Index curved but NOT touching (distance > 0.08)
  //            Middle/Ring/Pinky are curled
  //            Thumb y is roughly equal to Index y (horizontal C shape)
  // -------------------------------------------------------------------------

  const thumbIndexDist = distance(thumbTip, indexTip);
  const thumbIndexYDiff = Math.abs(thumbTip.y - indexTip.y);
  const middleRingPinkyCurled = (
    isFingerCurled(landmarks, LANDMARKS.MIDDLE_TIP, LANDMARKS.MIDDLE_PIP) &&
    isFingerCurled(landmarks, LANDMARKS.RING_TIP, LANDMARKS.RING_PIP) &&
    isFingerCurled(landmarks, LANDMARKS.PINKY_TIP, LANDMARKS.PINKY_PIP)
  );

  // C-shape: thumb and index apart but at similar height, others curled
  if (thumbIndexDist > 0.08 && thumbIndexDist < 0.2 && thumbIndexYDiff < 0.08 && middleRingPinkyCurled) {
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

  return null;
}

// =============================================================================
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

  // Priority 1: VERBS - Check action gestures first
  // GRAB/CLAW has highest priority (most specific shape)
  const avgThumbDist = getAverageThumbToFingerDistance(landmarks);
  if (avgThumbDist < 0.06) {
    return 'GRAB';
  }

  // Priority 2: OBJECTS - Check shape-based objects
  const object = detectObject(landmarks);
  if (object) {
    return object.grammar_id;
  }

  // Priority 3: SUBJECTS - Check directional pronouns
  const subject = detectSubject(landmarks);
  if (subject) {
    return subject.grammar_id;
  }

  // Priority 4: Other VERBS
  const verb = detectVerb(landmarks);
  if (verb) {
    return verb.grammar_id;
  }

  // Priority 5: Fallback - Basic gestures
  if (areAllFingersCurled(landmarks)) {
    return 'SUBJECT_I';
  }

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

/**
 * Legacy function for backwards compatibility
 */
export function detectGesture(landmarks) {
  return detectGestureRaw(landmarks);
}

// =============================================================================
// DETAILED DETECTION
// =============================================================================

/**
 * Get detailed gesture detection with all metadata
 * @param {Array} landmarks - 21 hand landmarks
 * @returns {Object|null} Full gesture object or null
 */
export function detectGestureDetailed(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;

  // Check in priority order
  const verb = detectVerb(landmarks);
  if (verb) return verb;

  const object = detectObject(landmarks);
  if (object) return object;

  const subject = detectSubject(landmarks);
  if (subject) return subject;

  // Fallback
  if (areAllFingersCurled(landmarks)) {
    return {
      type: 'SUBJECT',
      value: 'I',
      grammar_id: 'SUBJECT_I',
      person: 1,
      number: 'singular',
      gesture_name: 'FIST',
    };
  }

  return null;
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
