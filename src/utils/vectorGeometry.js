/**
 * vectorGeometry.js
 * Pure 3D vector math primitives for hand landmark analysis.
 * All functions use only Math — no external dependencies.
 */

/**
 * Euclidean distance between two 3D points.
 * @param {{x:number, y:number, z?:number}} p1
 * @param {{x:number, y:number, z?:number}} p2
 * @returns {number}
 */
export function euclideanDistance3D(p1, p2) {
  const dx = p2.x - p1.x;
  const dy = p2.y - p1.y;
  const dz = (p2.z || 0) - (p1.z || 0);
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

/**
 * Angle (in degrees) at vertex b formed by points a → b → c.
 * Uses the dot-product formula: cos(θ) = (BA · BC) / (|BA| |BC|)
 * @param {{x:number, y:number, z?:number}} a
 * @param {{x:number, y:number, z?:number}} b - vertex
 * @param {{x:number, y:number, z?:number}} c
 * @returns {number} angle in degrees [0, 180]
 */
export function angleBetweenPoints3D(a, b, c) {
  const ba = {
    x: a.x - b.x,
    y: a.y - b.y,
    z: (a.z || 0) - (b.z || 0),
  };
  const bc = {
    x: c.x - b.x,
    y: c.y - b.y,
    z: (c.z || 0) - (b.z || 0),
  };

  const dot = ba.x * bc.x + ba.y * bc.y + ba.z * bc.z;
  const magBA = Math.sqrt(ba.x * ba.x + ba.y * ba.y + ba.z * ba.z);
  const magBC = Math.sqrt(bc.x * bc.x + bc.y * bc.y + bc.z * bc.z);

  if (magBA === 0 || magBC === 0) return 0;

  // Clamp to [-1, 1] to guard against floating-point drift
  const cosAngle = Math.max(-1, Math.min(1, dot / (magBA * magBC)));
  return Math.acos(cosAngle) * (180 / Math.PI);
}

/**
 * Normalize landmarks relative to wrist position and max distance.
 * - Translates so wrist is at origin
 * - Scales by max distance from wrist to any landmark
 *   (matches the RF classifier's normalization for consistency)
 * @param {Array<{x:number, y:number, z?:number}>} landmarks - 21 hand landmarks
 * @returns {Array<{x:number, y:number, z:number}>} normalized landmarks
 */
export function normalizeToWrist(landmarks) {
  const wrist = landmarks[0];

  // Center at wrist
  const centered = landmarks.map((lm) => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: (lm.z || 0) - (wrist.z || 0),
  }));

  // Scale by max distance (consistent with RF classifier normalization)
  let maxDist = 0;
  for (let i = 0; i < centered.length; i++) {
    const d = Math.sqrt(centered[i].x ** 2 + centered[i].y ** 2 + centered[i].z ** 2);
    if (d > maxDist) maxDist = d;
  }

  const scale = maxDist > 1e-8 ? maxDist : 1;
  return centered.map((lm) => ({
    x: lm.x / scale,
    y: lm.y / scale,
    z: lm.z / scale,
  }));
}

/**
 * Flatten 21 landmarks into a 63-dimensional configuration vector.
 * @param {Array<{x:number, y:number, z:number}>} landmarks
 * @returns {Float64Array}
 */
export function flattenToConfigVector(landmarks) {
  const vec = new Float64Array(63);
  for (let i = 0; i < 21; i++) {
    vec[i * 3] = landmarks[i].x;
    vec[i * 3 + 1] = landmarks[i].y;
    vec[i * 3 + 2] = landmarks[i].z || 0;
  }
  return vec;
}
