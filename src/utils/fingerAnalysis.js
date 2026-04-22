/**
 * fingerAnalysis.js
 * Angle-based finger state analysis for scale/rotation-invariant gesture detection.
 * Uses PIP/IP joint angles instead of simple y-coordinate heuristics.
 */

import { angleBetweenPoints3D, normalizeToWrist } from './vectorGeometry';

// MediaPipe hand landmark indices
// Thumb:  1(CMC) 2(MCP) 3(IP) 4(TIP)
// Index:  5(MCP) 6(PIP) 7(DIP) 8(TIP)
// Middle: 9(MCP) 10(PIP) 11(DIP) 12(TIP)
// Ring:   13(MCP) 14(PIP) 15(DIP) 16(TIP)
// Pinky:  17(MCP) 18(PIP) 19(DIP) 20(TIP)

const FINGER_JOINTS = {
  thumb:  { mcp: 2, pip: 3, dip: 4 },  // Thumb uses MCP→IP→TIP
  index:  { mcp: 5, pip: 6, dip: 7 },
  middle: { mcp: 9, pip: 10, dip: 11 },
  ring:   { mcp: 13, pip: 14, dip: 15 },
  pinky:  { mcp: 17, pip: 18, dip: 19 },
};

// Aligned with gestureDetection.js EXTENDED_THRESHOLD (155°) so UI display
// and actual gesture detection agree on finger state.
const OPEN_THRESHOLD = 155; // degrees — above this the finger is "extended"

/**
 * Compute the PIP/IP joint angle for a finger.
 * For index–pinky: angle at PIP (MCP→PIP→DIP)
 * For thumb: angle at IP (MCP→IP→TIP) — different kinematic chain
 * @param {Array} landmarks - normalized 21-point hand landmarks
 * @param {string} finger - one of 'thumb','index','middle','ring','pinky'
 * @returns {number} angle in degrees
 */
function fingerJointAngle(landmarks, finger) {
  const { mcp, pip, dip } = FINGER_JOINTS[finger];
  return angleBetweenPoints3D(landmarks[mcp], landmarks[pip], landmarks[dip]);
}

/**
 * Analyze all five fingers and return open/curled boolean states.
 * @param {Array<{x:number, y:number, z?:number}>} landmarks - raw 21-point landmarks
 * @returns {{ thumb:boolean, index:boolean, middle:boolean, ring:boolean, pinky:boolean }}
 */
export function analyzeFingerStates(landmarks) {
  const normalized = normalizeToWrist(landmarks);

  return {
    thumb:  fingerJointAngle(normalized, 'thumb')  > OPEN_THRESHOLD,
    index:  fingerJointAngle(normalized, 'index')  > OPEN_THRESHOLD,
    middle: fingerJointAngle(normalized, 'middle') > OPEN_THRESHOLD,
    ring:   fingerJointAngle(normalized, 'ring')   > OPEN_THRESHOLD,
    pinky:  fingerJointAngle(normalized, 'pinky')  > OPEN_THRESHOLD,
  };
}

/**
 * Detailed analysis returning both raw angles and boolean states.
 * Useful for the debug panel to show exact joint angles.
 * @param {Array<{x:number, y:number, z?:number}>} landmarks - raw 21-point landmarks
 * @returns {{ angles: Record<string,number>, states: Record<string,boolean> }}
 */
export function analyzeFingerStatesDetailed(landmarks) {
  const normalized = normalizeToWrist(landmarks);

  const angles = {};
  const states = {};

  for (const finger of Object.keys(FINGER_JOINTS)) {
    const angle = fingerJointAngle(normalized, finger);
    angles[finger] = angle;
    states[finger] = angle > OPEN_THRESHOLD;
  }

  return { angles, states };
}
