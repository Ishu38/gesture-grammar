/**
 * MLAF — Client-Side CNN Gesture Classifier (ONNX Runtime Web)
 *
 * Loads a trained 1D CNN exported as ONNX and runs inference in the browser.
 * Input: 21 MediaPipe hand landmarks ({x,y,z} objects)
 * Output: { id, label, category, confidence, probabilities }
 *
 * The CNN operates on normalized (21,3) landmark tensors — wrist-origin,
 * unit-scaled — matching the Python training pipeline exactly.
 *
 * Requires: onnxruntime-web (loaded via CDN or npm)
 */

// ---------------------------------------------------------------------------
// Gesture ID mappings (must match training/config.py)
// ---------------------------------------------------------------------------

const GESTURE_IDS = [
  'subject_i', 'subject_you', 'subject_he', 'subject_she',
  'subject_we', 'subject_they',
  'verb_want', 'verb_eat', 'verb_see', 'verb_grab',
  'verb_drink', 'verb_go', 'verb_stop',
  'object_food', 'object_water', 'object_book',
  'object_apple', 'object_ball', 'object_house',
];

const IDX_TO_FRONTEND_ID = {
  0:  'SUBJECT_I',   1:  'SUBJECT_YOU',  2:  'SUBJECT_HE',
  3:  'SUBJECT_SHE', 4:  'SUBJECT_WE',   5:  'SUBJECT_THEY',
  6:  'WANT',        7:  'EAT',          8:  'SEE',
  9:  'GRAB',        10: 'DRINK',        11: 'GO',
  12: 'STOP',
  13: 'FOOD',        14: 'WATER',        15: 'BOOK',
  16: 'APPLE',       17: 'BALL',         18: 'HOUSE',
};

const GESTURE_LABELS = {
  SUBJECT_I: 'I', SUBJECT_YOU: 'You', SUBJECT_HE: 'He',
  SUBJECT_SHE: 'She', SUBJECT_WE: 'We', SUBJECT_THEY: 'They',
  WANT: 'want', EAT: 'eat', SEE: 'see',
  GRAB: 'grab', DRINK: 'drink', GO: 'go', STOP: 'stop',
  FOOD: 'food', WATER: 'water', BOOK: 'book',
  APPLE: 'apple', BALL: 'ball', HOUSE: 'house',
};

const GESTURE_CATEGORIES = {
  SUBJECT_I: 'NP', SUBJECT_YOU: 'NP', SUBJECT_HE: 'NP',
  SUBJECT_SHE: 'NP', SUBJECT_WE: 'NP', SUBJECT_THEY: 'NP',
  WANT: 'VP', EAT: 'VP', SEE: 'VP',
  GRAB: 'VP', DRINK: 'VP', GO: 'VP', STOP: 'VP',
  FOOD: 'NP', WATER: 'NP', BOOK: 'NP',
  APPLE: 'NP', BALL: 'NP', HOUSE: 'NP',
};


// ---------------------------------------------------------------------------
// Normalization (matches Python _normalize_to_wrist and JS normalizeLandmarks)
// ---------------------------------------------------------------------------

function normalizeLandmarks(landmarks) {
  const wrist = landmarks[0];
  const centered = landmarks.map(lm => ({
    x: lm.x - wrist.x,
    y: lm.y - wrist.y,
    z: (lm.z || 0) - (wrist.z || 0),
  }));

  let maxDist = 0;
  for (let i = 0; i < 21; i++) {
    const d = Math.sqrt(centered[i].x ** 2 + centered[i].y ** 2 + centered[i].z ** 2);
    if (d > maxDist) maxDist = d;
  }

  if (maxDist < 1e-8) return centered;
  return centered.map(lm => ({
    x: lm.x / maxDist,
    y: lm.y / maxDist,
    z: lm.z / maxDist,
  }));
}


// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map(l => Math.exp(l - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sum);
}


// ---------------------------------------------------------------------------
// GestureClassifierCNN
// ---------------------------------------------------------------------------

export class GestureClassifierCNN {
  constructor() {
    this.session = null;
    this.loaded = false;
    this.numClasses = GESTURE_IDS.length;
    this._idxToFrontendId = null;
    this._ort = null;          // cached onnxruntime-web module
    this._inputName = 'landmarks';
    this._outputName = 'logits';
  }

  /**
   * Load ONNX model from URL.
   * @param {string} modelUrl - URL to the .onnx file (e.g. '/models/gesture_cnn_latest.onnx')
   * @param {string} [metaUrl] - Optional URL to metadata JSON for class mapping
   */
  async load(modelUrl, metaUrl) {
    // Dynamically import onnxruntime-web (cached for classify calls)
    this._ort = await this._loadOrt();

    this.session = await this._ort.InferenceSession.create(modelUrl, {
      executionProviders: ['wasm'],
    });

    // Discover actual input/output tensor names from the model
    if (this.session.inputNames?.length > 0) {
      this._inputName = this.session.inputNames[0];
    }
    if (this.session.outputNames?.length > 0) {
      this._outputName = this.session.outputNames[0];
    }

    // Load metadata if available
    if (metaUrl) {
      try {
        const resp = await fetch(metaUrl);
        if (resp.ok) {
          const meta = await resp.json();
          const classNames = meta.class_names || meta.gesture_ids;
          if (classNames && Array.isArray(classNames)) {
            this._syncGestureIds(classNames);
          }
        }
      } catch {
        // Metadata is optional — fall back to hardcoded mappings
      }
    }

    this.loaded = true;
  }

  /**
   * Sync frontend gesture ID mappings from model metadata.
   */
  _syncGestureIds(classNames) {
    this._idxToFrontendId = {};
    for (let i = 0; i < classNames.length; i++) {
      const upper = classNames[i].toUpperCase();
      if (upper.startsWith('SUBJECT_')) {
        this._idxToFrontendId[i] = upper;
      } else if (upper.startsWith('VERB_')) {
        this._idxToFrontendId[i] = upper.replace('VERB_', '');
      } else if (upper.startsWith('OBJECT_')) {
        this._idxToFrontendId[i] = upper.replace('OBJECT_', '');
      } else {
        this._idxToFrontendId[i] = upper;
      }
    }
    this.numClasses = classNames.length;
  }

  /**
   * Resolve onnxruntime-web module (cached after first call).
   */
  async _loadOrt() {
    if (this._ort) return this._ort;
    try {
      this._ort = await import('onnxruntime-web');
    } catch {
      if (typeof globalThis.ort !== 'undefined') {
        this._ort = globalThis.ort;
      } else {
        throw new Error('onnxruntime-web not available. Install via npm or load from CDN.');
      }
    }
    return this._ort;
  }

  /**
   * Classify hand landmarks into a gesture.
   * @param {Array} landmarks - 21 {x,y,z} hand landmarks from MediaPipe
   * @param {number} [confidenceThreshold=0.4] - Minimum confidence
   * @returns {Object|null} { id, label, category, confidence, probabilities }
   */
  async classify(landmarks, confidenceThreshold = 0.4) {
    if (!this.loaded || !this._ort || !landmarks || landmarks.length < 21) return null;

    // Normalize landmarks
    const norm = normalizeLandmarks(landmarks);

    // Build (1, 21, 3) Float32Array tensor
    const inputData = new Float32Array(21 * 3);
    for (let i = 0; i < 21; i++) {
      inputData[i * 3 + 0] = norm[i].x;
      inputData[i * 3 + 1] = norm[i].y;
      inputData[i * 3 + 2] = norm[i].z;
    }

    // ONNX Runtime Web inference (uses cached ort + discovered tensor names)
    const inputTensor = new this._ort.Tensor('float32', inputData, [1, 21, 3]);
    const feeds = { [this._inputName]: inputTensor };
    const results = await this.session.run(feeds);

    // Extract logits using discovered output name
    const outputTensor = results[this._outputName];
    if (!outputTensor) return null;

    const logits = Array.from(outputTensor.data);
    const probabilities = softmax(logits);

    // Find best class
    let bestIdx = 0;
    let bestProb = 0;
    for (let c = 0; c < probabilities.length; c++) {
      if (probabilities[c] > bestProb) {
        bestProb = probabilities[c];
        bestIdx = c;
      }
    }

    if (bestProb < confidenceThreshold) return null;

    const idMap = this._idxToFrontendId || IDX_TO_FRONTEND_ID;
    const frontendId = idMap[bestIdx];

    return {
      id: frontendId,
      label: GESTURE_LABELS[frontendId],
      category: GESTURE_CATEGORIES[frontendId],
      confidence: bestProb,
      classIndex: bestIdx,
      gestureId: (bestIdx < GESTURE_IDS.length) ? GESTURE_IDS[bestIdx] : frontendId,
      probabilities: new Float64Array(probabilities),
    };
  }
}
