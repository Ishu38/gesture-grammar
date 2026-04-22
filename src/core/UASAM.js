/**
 * UASAM.js — Unified Acoustic State Analysis Module
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PURPOSE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The UASAM is the acoustic branch of the UMCE's Bayesian late fusion:
 *
 *     P(S | A, V) ∝ P(A | S) · P(V | S) · P(S)
 *
 * This module produces  P(A | S)  — the likelihood of observed acoustic
 * data A given each communicative state S. It runs independently of the
 * visual pipeline (AGGME) and outputs a probability array over the same
 * state space, enabling true bimodal fusion.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ACOUSTIC FEATURE EXTRACTION PIPELINE
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Microphone → AudioContext → AnalyserNode → Per-Frame Feature Vector:
 *
 *   F1: RMS Energy (dB)
 *       E_rms = sqrt( (1/N) Σ x_i² )
 *       E_dB  = 20 · log₁₀(E_rms + ε)
 *       Captures vocalization intensity: silence vs. whisper vs. vocalization.
 *
 *   F2: Spectral Centroid (Hz)
 *       SC = Σ(f_k · |X_k|) / Σ|X_k|
 *       Captures brightness: low grunt vs. high-pitched vocalization.
 *       Maps to arousal/urgency of the communicative attempt.
 *
 *   F3: Spectral Rolloff (Hz)
 *       Frequency below which 85% of spectral energy is concentrated.
 *       SR = f_k where Σ_{j≤k} |X_j|² ≥ 0.85 · Σ |X_j|²
 *       Distinguishes breathy vocalizations from sharp consonant-like bursts.
 *
 *   F4: Zero-Crossing Rate
 *       ZCR = (1/N) Σ |sign(x_i) - sign(x_{i-1})|
 *       High ZCR → fricative/noisy (consonant attempt).
 *       Low ZCR  → periodic/tonal (vowel attempt).
 *
 *   F5: Spectral Flatness (Wiener entropy)
 *       SF = exp( (1/K) Σ ln|X_k| ) / ( (1/K) Σ |X_k| )
 *       SF ≈ 1 → noise-like (breath, ambient).
 *       SF ≈ 0 → tonal (intentional vocalization).
 *
 *   F6: Dominant Frequency / Pitch Estimate (Hz)
 *       Peak bin of the magnitude spectrum in the vocal range [80, 600] Hz.
 *       Rough F0 estimate for vocalization classification.
 *
 *   F7: Harmonic-to-Noise Ratio (simplified)
 *       HNR = E_harmonic / E_noise  (in the vocal band)
 *       High HNR → voiced phonation. Low HNR → breath/noise.
 *
 *   F8: Sub-band Energy Ratios (4 bands)
 *       Band 1 [0–300 Hz]:    fundamental + low vowels
 *       Band 2 [300–1000 Hz]: first formant region (F1)
 *       Band 3 [1000–3000 Hz]: second formant region (F2)
 *       Band 4 [3000–8000 Hz]: fricatives, sibilants
 *       Ratios between bands encode coarse formant structure.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * VOCALIZATION STATE CLASSIFICATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The feature vector is classified into one of 7 vocalization states:
 *
 *   SILENT        — No vocalization (ambient noise floor)
 *   BREATH        — Exhalation without phonation
 *   VOWEL_OPEN    — Open vowel attempt (/a/, /o/) — low F1, low F2
 *   VOWEL_CLOSED  — Closed vowel attempt (/i/, /u/) — high F1 or high F2
 *   CONSONANT_STOP  — Plosive-like burst (/b/, /d/, /g/) — transient energy spike
 *   CONSONANT_FRIC  — Fricative-like noise (/s/, /f/) — high ZCR, high spectral flatness
 *   VOCALIZATION    — General voiced sound (cry, babble, proto-word)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * ACOUSTIC → COMMUNICATIVE STATE MAPPING: P(A | S)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Each communicative state S (gesture vocabulary) has an acoustic profile:
 *   - SUBJECT pronouns (I, YOU, HE) → typically accompanied by vowel-like
 *     vocalizations (/ai/, /ju/, /hi/) — VOWEL_OPEN or VOWEL_CLOSED
 *   - ACTION verbs (GRAB, STOP, EAT) → accompanied by consonant onsets
 *     + vowel nuclei — CONSONANT_STOP or VOCALIZATION
 *   - OBJECTS (APPLE, WATER, BOOK) → longer vocalizations with varied
 *     spectral profiles — VOCALIZATION with specific sub-band ratios
 *
 * The likelihood P(A|S) for each state is computed as:
 *   P(A|S) = Σ_v  P(vocState = v | features) · P(v | S)
 *
 * Where P(v|S) is a pre-defined emission matrix mapping each communicative
 * state to its expected vocalization profile.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * PIPELINE POSITION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   Microphone → AudioContext → AnalyserNode
 *                     ↓
 *              Feature Extraction (F1–F8)
 *                     ↓
 *              Vocalization Classification
 *                     ↓
 *              Acoustic Likelihood: P(A | S)  ──→  UMCE Bayesian Fusion
 *                                                       ↑
 *              AGGME Visual: P(V | S)  ─────────────────┘
 */

import { LEXICON } from '../utils/GrammarEngine';

// =============================================================================
// CONSTANTS
// =============================================================================

/** FFT size for spectral analysis. 2048 gives ~43ms window at 48kHz. */
const FFT_SIZE = 2048;

/** Smoothing time constant for AnalyserNode (0 = no smoothing). */
const SMOOTHING_TIME_CONSTANT = 0.3;

/** Silence threshold in dB — below this = SILENT state. */
const SILENCE_THRESHOLD_DB = -45;

/** Breath threshold in dB — between silence and this = BREATH. */
const BREATH_THRESHOLD_DB = -30;

/** Minimum vocal frequency for pitch detection (Hz). */
const VOCAL_FREQ_MIN = 80;

/** Maximum vocal frequency for pitch detection (Hz). */
const VOCAL_FREQ_MAX = 600;

/** Sub-band boundaries (Hz). */
const SUBBANDS = [
  { name: 'low',    min: 0,    max: 300  },
  { name: 'f1',     min: 300,  max: 1000 },
  { name: 'f2',     min: 1000, max: 3000 },
  { name: 'high',   min: 3000, max: 8000 },
];

/** Spectral rolloff percentage. */
const ROLLOFF_PERCENT = 0.85;

/** ZCR threshold for fricative detection. */
const ZCR_FRICATIVE_THRESHOLD = 0.15;

/** Spectral flatness threshold — above this = noise-like. */
const SPECTRAL_FLATNESS_NOISE_THRESHOLD = 0.5;

/** HNR threshold — above this = voiced phonation. */
const HNR_VOICED_THRESHOLD = 3.0;

/** Feature history window for temporal smoothing (frames). */
const FEATURE_HISTORY_SIZE = 8;

/** Probability floor for acoustic likelihoods. */
const ACOUSTIC_PROB_FLOOR = 0.001;

/** EMA alpha for feature smoothing across frames. */
const FEATURE_SMOOTH_ALPHA = 0.4;

// =============================================================================
// VOCALIZATION STATES
// =============================================================================

const VOC_STATES = {
  SILENT:          'SILENT',
  BREATH:          'BREATH',
  VOWEL_OPEN:      'VOWEL_OPEN',
  VOWEL_CLOSED:    'VOWEL_CLOSED',
  CONSONANT_STOP:  'CONSONANT_STOP',
  CONSONANT_FRIC:  'CONSONANT_FRIC',
  VOCALIZATION:    'VOCALIZATION',
};

// =============================================================================
// EMISSION MATRIX: P(vocalization_state | communicative_state)
// =============================================================================

/**
 * For each gesture/communicative state S, defines the probability of
 * observing each vocalization state. These encode the expected acoustic
 * profile when a child attempts to produce the word associated with
 * the gesture.
 *
 * Rows: communicative states (gesture IDs).
 * Columns: vocalization states.
 * Each row sums to 1.0.
 *
 * Derived from articulatory phonetics of the target English words:
 *   - "I" (/aɪ/) → open vowel diphthong
 *   - "you" (/juː/) → closed vowel
 *   - "he" (/hiː/) → fricative onset + closed vowel
 *   - "grab" (/ɡɹæb/) → stop consonant + open vowel + stop
 *   - "stop" (/stɒp/) → fricative + stop + open vowel + stop
 *   - "eat" (/iːt/) → closed vowel + stop
 *   - "apple" (/æpəl/) → open vowel + stop + vowel
 *   - "water" (/wɔːtə/) → vocalization + open vowel + stop
 *   - etc.
 */
const EMISSION_MATRIX = {
  // ─── SUBJECTS ───────────────────────────────────────────────────────
  //                   SILENT  BREATH  V_OPEN  V_CLOSED  C_STOP  C_FRIC  VOCAL
  SUBJECT_I:          [0.05,  0.05,   0.50,   0.20,     0.02,   0.03,   0.15],
  SUBJECT_YOU:        [0.05,  0.05,   0.10,   0.50,     0.03,   0.07,   0.20],
  SUBJECT_HE:         [0.05,  0.03,   0.08,   0.35,     0.04,   0.25,   0.20],
  SUBJECT_SHE:        [0.05,  0.03,   0.05,   0.20,     0.04,   0.40,   0.23],
  SUBJECT_WE:         [0.05,  0.05,   0.15,   0.40,     0.03,   0.07,   0.25],
  SUBJECT_THEY:       [0.05,  0.03,   0.10,   0.15,     0.07,   0.25,   0.35],

  // ─── VERBS (base form) ─────────────────────────────────────────────
  GRAB:               [0.03,  0.04,   0.25,   0.08,     0.35,   0.05,   0.20],
  GO:                 [0.03,  0.04,   0.30,   0.15,     0.25,   0.03,   0.20],
  EAT:                [0.03,  0.04,   0.10,   0.40,     0.20,   0.03,   0.20],
  WANT:               [0.03,  0.04,   0.20,   0.15,     0.10,   0.08,   0.40],
  STOP:               [0.03,  0.03,   0.15,   0.10,     0.25,   0.24,   0.20],
  DRINK:              [0.03,  0.04,   0.10,   0.15,     0.20,   0.08,   0.40],

  // ─── OBJECTS ────────────────────────────────────────────────────────
  APPLE:              [0.03,  0.04,   0.40,   0.10,     0.20,   0.03,   0.20],
  BALL:               [0.03,  0.04,   0.25,   0.10,     0.30,   0.03,   0.25],
  WATER:              [0.03,  0.04,   0.30,   0.10,     0.15,   0.08,   0.30],
  FOOD:               [0.03,  0.03,   0.15,   0.20,     0.10,   0.24,   0.25],
  BOOK:               [0.03,  0.04,   0.15,   0.20,     0.30,   0.03,   0.25],
  HOUSE:              [0.03,  0.03,   0.20,   0.10,     0.10,   0.24,   0.30],
};

// Vocalization state index order (matches column order in EMISSION_MATRIX)
const VOC_STATE_ORDER = [
  VOC_STATES.SILENT,
  VOC_STATES.BREATH,
  VOC_STATES.VOWEL_OPEN,
  VOC_STATES.VOWEL_CLOSED,
  VOC_STATES.CONSONANT_STOP,
  VOC_STATES.CONSONANT_FRIC,
  VOC_STATES.VOCALIZATION,
];

// =============================================================================
// UASAM CLASS
// =============================================================================

export class UASAM {
  /**
   * @param {object} [config]
   * @param {number} [config.silenceThresholdDb] — override silence threshold (default: -45)
   * @param {number} [config.breathThresholdDb] — override breath threshold (default: -30)
   */
  constructor(config = {}) {
    // ── Per-profile acoustic thresholds (CP accessibility) ──
    this._silenceThresholdDb = config.silenceThresholdDb ?? SILENCE_THRESHOLD_DB;
    this._breathThresholdDb = config.breathThresholdDb ?? BREATH_THRESHOLD_DB;

    // ── Web Audio nodes ──
    this._audioContext = null;
    this._analyser = null;
    this._source = null;
    this._stream = null;

    // ── Buffers (pre-allocated for O(1) per-frame extraction) ──
    this._timeDomainData = null;   // Float32Array for waveform
    this._frequencyData = null;    // Float32Array for magnitude spectrum

    // ── Derived audio parameters ──
    this._sampleRate = 48000;
    this._binWidth = 0;            // Hz per FFT bin

    // ── State ──
    this._isActive = false;
    this._isInitializing = false;
    this._frameCount = 0;

    // ── Per-frame feature vector (smoothed across frames) ──
    this._features = {
      rmsEnergy: 0,
      energyDb: -100,
      spectralCentroid: 0,
      spectralRolloff: 0,
      zeroCrossingRate: 0,
      spectralFlatness: 0,
      dominantFrequency: 0,
      hnr: 0,
      subbandEnergies: [0, 0, 0, 0],
      subbandRatios: { f1_low: 0, f2_f1: 0, high_f2: 0 },
    };

    // ── Previous raw features for EMA smoothing ──
    this._prevFeatures = null;

    // ── Vocalization state probabilities (soft classification) ──
    this._vocStateProbabilities = {};
    for (const vs of VOC_STATE_ORDER) {
      this._vocStateProbabilities[vs] = vs === VOC_STATES.SILENT ? 1.0 : 0.0;
    }

    // ── Vocalization state (hard classification) ──
    this._vocState = VOC_STATES.SILENT;

    // ── Energy onset detector for consonant stops ──
    this._energyHistory = [];
    this._energyHistoryMax = 5;

    // ── Acoustic likelihoods: P(A | S) for each communicative state ──
    this._acousticLikelihoods = {};

    // ── Last frame result ──
    this._lastResult = null;

    // ── All communicative state IDs ──
    this._stateIds = Object.keys(EMISSION_MATRIX);

    // ── Noise floor calibration state ──
    this._noiseFloorCalibrated = false;
    this._calibrationSamples = [];
    this._calibrationInProgress = false;
    this._calibratedNoiseFloorDb = null;
    this._originalSilenceDb = this._silenceThresholdDb;
    this._originalBreathDb = this._breathThresholdDb;
  }

  // ===========================================================================
  // PUBLIC — Lifecycle
  // ===========================================================================

  /**
   * Initialize microphone capture and Web Audio processing graph.
   *
   * AudioContext → MediaStreamSource → AnalyserNode
   *
   * @returns {Promise<boolean>} true if successfully initialized
   */
  async initialize() {
    if (this._isActive || this._isInitializing) return this._isActive;
    this._isInitializing = true;

    try {
      // Request microphone with constraints optimized for speech analysis
      this._stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          sampleRate: { ideal: 48000 },
          channelCount: 1,
        },
      });

      this._audioContext = new (window.AudioContext || window.webkitAudioContext)({
        sampleRate: 48000,
      });

      this._sampleRate = this._audioContext.sampleRate;

      // Create analyser node
      this._analyser = this._audioContext.createAnalyser();
      this._analyser.fftSize = FFT_SIZE;
      this._analyser.smoothingTimeConstant = SMOOTHING_TIME_CONSTANT;

      // Connect: mic → analyser (no output to speakers — analysis only)
      this._source = this._audioContext.createMediaStreamSource(this._stream);
      this._source.connect(this._analyser);

      // Pre-allocate typed arrays
      this._timeDomainData = new Float32Array(this._analyser.fftSize);
      this._frequencyData = new Float32Array(this._analyser.frequencyBinCount);

      // Derived constants
      this._binWidth = this._sampleRate / this._analyser.fftSize;

      this._isActive = true;
      this._isInitializing = false;

      return true;
    } catch (err) {
      console.warn('[UASAM] Microphone initialization failed:', err.message);
      this._isInitializing = false;
      return false;
    }
  }

  /**
   * Release all audio resources.
   */
  destroy() {
    if (this._source) {
      this._source.disconnect();
      this._source = null;
    }
    if (this._stream) {
      this._stream.getTracks().forEach(t => t.stop());
      this._stream = null;
    }
    if (this._audioContext && this._audioContext.state !== 'closed') {
      this._audioContext.close().catch(() => {});
      this._audioContext = null;
    }
    this._analyser = null;
    this._isActive = false;
    this._frameCount = 0;
    this._prevFeatures = null;
    this._lastResult = null;
  }

  /**
   * @returns {boolean} Whether the microphone is active and streaming.
   */
  isActive() {
    return this._isActive;
  }

  // ===========================================================================
  // PUBLIC — Per-Frame Processing
  // ===========================================================================

  /**
   * Process one frame of audio data. Call this at ~30fps (matched to video).
   *
   * Extracts features → classifies vocalization state → computes P(A|S).
   *
   * @returns {UASAMResult} Acoustic analysis result including P(A|S) array
   */
  processFrame() {
    if (!this._isActive || !this._analyser) {
      return this._silentResult();
    }

    this._frameCount++;

    // =====================================================================
    // Step 1: Acquire raw audio data from AnalyserNode
    // =====================================================================
    this._analyser.getFloatTimeDomainData(this._timeDomainData);
    this._analyser.getFloatFrequencyData(this._frequencyData);

    // =====================================================================
    // Step 2: Extract acoustic feature vector (F1–F8)
    // =====================================================================
    const rawFeatures = this._extractFeatures();

    // =====================================================================
    // Step 3: EMA smooth features across frames (temporal stability)
    // =====================================================================
    this._smoothFeatures(rawFeatures);

    // =====================================================================
    // Step 4: Classify vocalization state (soft probabilities)
    // =====================================================================
    this._classifyVocalizationState();

    // =====================================================================
    // Step 5: Compute P(A | S) for each communicative state
    // =====================================================================
    this._computeAcousticLikelihoods();

    // =====================================================================
    // Step 6: Assemble result
    // =====================================================================
    this._lastResult = {
      features: { ...this._features },
      vocalization_state: this._vocState,
      vocalization_probabilities: { ...this._vocStateProbabilities },
      acoustic_likelihoods: { ...this._acousticLikelihoods },
      is_vocalization: this._features.energyDb > this._breathThresholdDb &&
                       this._features.spectralFlatness < SPECTRAL_FLATNESS_NOISE_THRESHOLD,
      frame: this._frameCount,
    };

    return this._lastResult;
  }

  /**
   * Get the acoustic likelihood array P(A|S) from the last processed frame.
   * This is what gets passed into UMCE for Bayesian fusion.
   *
   * @returns {Object} Map<state_id, probability>
   */
  getAcousticLikelihoods() {
    return { ...this._acousticLikelihoods };
  }

  /**
   * Get last frame result.
   * @returns {UASAMResult|null}
   */
  getLastResult() {
    return this._lastResult;
  }

  /**
   * Get current features for debug display.
   * @returns {object}
   */
  getFeatures() {
    return { ...this._features };
  }

  /**
   * Get serializable descriptor for logging/export.
   * @returns {object}
   */
  toDescriptor() {
    return {
      is_active: this._isActive,
      sample_rate: this._sampleRate,
      fft_size: FFT_SIZE,
      frame_count: this._frameCount,
      current_voc_state: this._vocState,
      energy_db: round2(this._features.energyDb),
      spectral_centroid: round2(this._features.spectralCentroid),
      dominant_frequency: round2(this._features.dominantFrequency),
      silence_threshold_db: this._silenceThresholdDb,
      breath_threshold_db: this._breathThresholdDb,
    };
  }

  /**
   * Update acoustic thresholds at runtime (for CP accessibility profiles).
   * Lower thresholds detect quieter vocalizations — critical for children with
   * CP who have respiratory weakness and produce phonation that typical
   * thresholds would classify as SILENT or BREATH.
   *
   * @param {object} thresholds
   * @param {number} [thresholds.silenceDb] — new silence threshold (e.g., -55 for CP)
   * @param {number} [thresholds.breathDb] — new breath threshold (e.g., -38 for CP)
   */
  setAcousticThresholds(thresholds) {
    if (thresholds.silenceDb !== undefined) {
      this._silenceThresholdDb = Math.max(-80, Math.min(-20, thresholds.silenceDb));
    }
    if (thresholds.breathDb !== undefined) {
      this._breathThresholdDb = Math.max(-60, Math.min(-15, thresholds.breathDb));
    }
  }

  /**
   * Get the current acoustic thresholds.
   * @returns {{ silenceDb: number, breathDb: number }}
   */
  getAcousticThresholds() {
    return {
      silenceDb: this._silenceThresholdDb,
      breathDb: this._breathThresholdDb,
    };
  }

  // ===========================================================================
  // PUBLIC — Noise Floor Calibration (Cochlear Implant / Noisy Environments)
  // ===========================================================================

  /**
   * Calibrate the noise floor by sampling ambient room audio.
   *
   * Call this AFTER initialize() succeeds. Samples ~3 seconds of ambient
   * room audio (no one speaking), computes the median energy in dB, and
   * shifts silence/breath thresholds so they are relative to the measured
   * noise floor rather than absolute constants.
   *
   * Algorithm:
   *   1. Collect CALIBRATION_FRAMES frames of energyDb readings
   *   2. Sort and take the median (robust to transient sounds)
   *   3. Set silenceThreshold = noiseFloor + SILENCE_MARGIN_DB
   *   4. Set breathThreshold = noiseFloor + BREATH_MARGIN_DB
   *
   * Margins:
   *   SILENCE_MARGIN = 5 dB above floor (anything below is ambient noise)
   *   BREATH_MARGIN  = 18 dB above floor (typical breath/quiet phonation)
   *
   * In a quiet room (-60 dB floor): silence=-55, breath=-42  (close to defaults)
   * In a noisy classroom (-35 dB floor): silence=-30, breath=-17
   *   → UASAM will only classify sounds clearly above room noise as vocalizations
   *
   * @param {number} [durationMs=3000] — calibration sampling duration
   * @returns {Promise<{ noiseFloorDb: number, silenceDb: number, breathDb: number }>}
   */
  async calibrateNoiseFloor(durationMs = 3000) {
    if (!this._isActive || !this._analyser) {
      throw new Error('[UASAM] Cannot calibrate — microphone not initialized');
    }

    if (this._calibrationInProgress) {
      throw new Error('[UASAM] Calibration already in progress');
    }

    this._calibrationInProgress = true;
    this._calibrationSamples = [];

    const SILENCE_MARGIN_DB = 5;
    const BREATH_MARGIN_DB = 18;
    const SAMPLE_INTERVAL_MS = 50; // sample every 50ms
    const numSamples = Math.ceil(durationMs / SAMPLE_INTERVAL_MS);

    return new Promise((resolve, reject) => {
      let collected = 0;

      const sampleInterval = setInterval(() => {
        if (!this._isActive || !this._analyser) {
          clearInterval(sampleInterval);
          this._calibrationInProgress = false;
          reject(new Error('[UASAM] Microphone lost during calibration'));
          return;
        }

        // Read current energy from analyser
        this._analyser.getFloatTimeDomainData(this._timeDomainData);

        let sumSquares = 0;
        for (let i = 0; i < this._timeDomainData.length; i++) {
          sumSquares += this._timeDomainData[i] * this._timeDomainData[i];
        }
        const rms = Math.sqrt(sumSquares / this._timeDomainData.length);
        const energyDb = rms > 1e-10 ? 20 * Math.log10(rms) : -100;

        this._calibrationSamples.push(energyDb);
        collected++;

        if (collected >= numSamples) {
          clearInterval(sampleInterval);
          this._calibrationInProgress = false;

          // Compute median energy (robust to outliers/transient sounds)
          const sorted = [...this._calibrationSamples].sort((a, b) => a - b);
          const medianIdx = Math.floor(sorted.length / 2);
          const noiseFloorDb = sorted.length % 2 === 0
            ? (sorted[medianIdx - 1] + sorted[medianIdx]) / 2
            : sorted[medianIdx];

          // Shift thresholds relative to measured floor
          const newSilenceDb = noiseFloorDb + SILENCE_MARGIN_DB;
          const newBreathDb = noiseFloorDb + BREATH_MARGIN_DB;

          // Apply (clamped to reasonable bounds)
          this._silenceThresholdDb = Math.max(-80, Math.min(-10, newSilenceDb));
          this._breathThresholdDb = Math.max(-60, Math.min(-5, newBreathDb));
          this._calibratedNoiseFloorDb = noiseFloorDb;
          this._noiseFloorCalibrated = true;

          const result = {
            noiseFloorDb: Math.round(noiseFloorDb * 10) / 10,
            silenceDb: Math.round(this._silenceThresholdDb * 10) / 10,
            breathDb: Math.round(this._breathThresholdDb * 10) / 10,
            samplesCollected: sorted.length,
          };

          resolve(result);
        }
      }, SAMPLE_INTERVAL_MS);
    });
  }

  /**
   * Check if noise floor has been calibrated.
   * @returns {boolean}
   */
  isNoiseFloorCalibrated() {
    return this._noiseFloorCalibrated;
  }

  /**
   * Get the calibrated noise floor info.
   * @returns {{ calibrated: boolean, noiseFloorDb: number|null, silenceDb: number, breathDb: number }}
   */
  getNoiseFloorInfo() {
    return {
      calibrated: this._noiseFloorCalibrated,
      noiseFloorDb: this._calibratedNoiseFloorDb,
      silenceDb: this._silenceThresholdDb,
      breathDb: this._breathThresholdDb,
      originalSilenceDb: this._originalSilenceDb,
      originalBreathDb: this._originalBreathDb,
    };
  }

  /**
   * Reset calibration to original thresholds.
   */
  resetCalibration() {
    this._silenceThresholdDb = this._originalSilenceDb;
    this._breathThresholdDb = this._originalBreathDb;
    this._noiseFloorCalibrated = false;
    this._calibratedNoiseFloorDb = null;
    this._calibrationSamples = [];
  }

  // ===========================================================================
  // PRIVATE — Feature Extraction
  // ===========================================================================

  /**
   * Extract all 8 acoustic features from the current frame's audio data.
   *
   * @returns {object} Raw (unsmoothed) feature vector
   */
  _extractFeatures() {
    const td = this._timeDomainData;
    const fd = this._frequencyData;
    const N = td.length;
    const K = fd.length; // frequencyBinCount = fftSize / 2

    // ─── F1: RMS Energy ───────────────────────────────────────────────
    let sumSq = 0;
    for (let i = 0; i < N; i++) {
      sumSq += td[i] * td[i];
    }
    const rmsEnergy = Math.sqrt(sumSq / N);
    const energyDb = 20 * Math.log10(rmsEnergy + 1e-10);

    // ─── Convert dB spectrum to linear magnitudes for spectral features ─
    const magnitudes = new Float32Array(K);
    let totalMagnitude = 0;
    for (let k = 0; k < K; k++) {
      // fd[k] is in dB; convert to linear: mag = 10^(dB/20)
      magnitudes[k] = Math.pow(10, fd[k] / 20);
      totalMagnitude += magnitudes[k];
    }

    // ─── F2: Spectral Centroid ────────────────────────────────────────
    let weightedFreqSum = 0;
    for (let k = 0; k < K; k++) {
      weightedFreqSum += (k * this._binWidth) * magnitudes[k];
    }
    const spectralCentroid = totalMagnitude > 0
      ? weightedFreqSum / totalMagnitude
      : 0;

    // ─── F3: Spectral Rolloff ─────────────────────────────────────────
    let totalEnergy = 0;
    for (let k = 0; k < K; k++) {
      totalEnergy += magnitudes[k] * magnitudes[k];
    }
    let cumulativeEnergy = 0;
    let spectralRolloff = 0;
    const rolloffTarget = ROLLOFF_PERCENT * totalEnergy;
    for (let k = 0; k < K; k++) {
      cumulativeEnergy += magnitudes[k] * magnitudes[k];
      if (cumulativeEnergy >= rolloffTarget) {
        spectralRolloff = k * this._binWidth;
        break;
      }
    }

    // ─── F4: Zero-Crossing Rate ───────────────────────────────────────
    let zeroCrossings = 0;
    for (let i = 1; i < N; i++) {
      if ((td[i] >= 0 && td[i - 1] < 0) || (td[i] < 0 && td[i - 1] >= 0)) {
        zeroCrossings++;
      }
    }
    const zeroCrossingRate = zeroCrossings / N;

    // ─── F5: Spectral Flatness (Wiener entropy) ──────────────────────
    let logSum = 0;
    let linSum = 0;
    let validBins = 0;
    for (let k = 1; k < K; k++) { // Skip DC bin
      if (magnitudes[k] > 1e-10) {
        logSum += Math.log(magnitudes[k]);
        linSum += magnitudes[k];
        validBins++;
      }
    }
    const spectralFlatness = validBins > 0
      ? Math.exp(logSum / validBins) / (linSum / validBins + 1e-10)
      : 1.0;

    // ─── F6: Dominant Frequency (rough pitch estimate) ────────────────
    const minBin = Math.floor(VOCAL_FREQ_MIN / this._binWidth);
    const maxBin = Math.min(Math.ceil(VOCAL_FREQ_MAX / this._binWidth), K - 1);
    let peakBin = minBin;
    let peakMag = 0;
    for (let k = minBin; k <= maxBin; k++) {
      if (magnitudes[k] > peakMag) {
        peakMag = magnitudes[k];
        peakBin = k;
      }
    }
    // Parabolic interpolation for sub-bin accuracy
    let dominantFrequency = peakBin * this._binWidth;
    if (peakBin > minBin && peakBin < maxBin) {
      const alpha = magnitudes[peakBin - 1];
      const beta = magnitudes[peakBin];
      const gamma = magnitudes[peakBin + 1];
      const denom = alpha - 2 * beta + gamma;
      if (Math.abs(denom) > 1e-10) {
        const correction = 0.5 * (alpha - gamma) / denom;
        dominantFrequency = (peakBin + correction) * this._binWidth;
      }
    }

    // ─── F7: Harmonic-to-Noise Ratio (simplified) ────────────────────
    // Sum energy at harmonic multiples of dominant frequency vs. rest
    let harmonicEnergy = 0;
    let noiseEnergy = 0;
    if (dominantFrequency > VOCAL_FREQ_MIN) {
      const fundamentalBin = Math.round(dominantFrequency / this._binWidth);
      const tolerance = 2; // ±2 bins around each harmonic
      const harmonicBins = new Set();
      for (let h = 1; h <= 6; h++) {
        const hBin = fundamentalBin * h;
        if (hBin >= K) break;
        for (let d = -tolerance; d <= tolerance; d++) {
          const b = hBin + d;
          if (b >= 0 && b < K) harmonicBins.add(b);
        }
      }
      for (let k = minBin; k <= maxBin; k++) {
        const e = magnitudes[k] * magnitudes[k];
        if (harmonicBins.has(k)) {
          harmonicEnergy += e;
        } else {
          noiseEnergy += e;
        }
      }
    }
    const hnr = noiseEnergy > 1e-10 ? harmonicEnergy / noiseEnergy : 0;

    // ─── F8: Sub-band Energies ────────────────────────────────────────
    const subbandEnergies = SUBBANDS.map(band => {
      const lo = Math.floor(band.min / this._binWidth);
      const hi = Math.min(Math.ceil(band.max / this._binWidth), K - 1);
      let energy = 0;
      for (let k = lo; k <= hi; k++) {
        energy += magnitudes[k] * magnitudes[k];
      }
      return energy;
    });

    const totalSubbandEnergy = subbandEnergies.reduce((s, e) => s + e, 0) || 1e-10;
    const subbandRatios = {
      f1_low:  subbandEnergies[0] > 1e-10 ? subbandEnergies[1] / subbandEnergies[0] : 0,
      f2_f1:   subbandEnergies[1] > 1e-10 ? subbandEnergies[2] / subbandEnergies[1] : 0,
      high_f2: subbandEnergies[2] > 1e-10 ? subbandEnergies[3] / subbandEnergies[2] : 0,
    };

    return {
      rmsEnergy,
      energyDb,
      spectralCentroid,
      spectralRolloff,
      zeroCrossingRate,
      spectralFlatness,
      dominantFrequency,
      hnr,
      subbandEnergies,
      subbandRatios,
    };
  }

  /**
   * Apply EMA smoothing to features across frames.
   * P_smooth(t) = α · P_raw(t) + (1 - α) · P_smooth(t-1)
   */
  _smoothFeatures(raw) {
    const a = FEATURE_SMOOTH_ALPHA;

    if (!this._prevFeatures) {
      // First frame: initialize directly
      this._features = { ...raw };
      this._prevFeatures = { ...raw };
      return;
    }

    const prev = this._prevFeatures;
    this._features = {
      rmsEnergy:         a * raw.rmsEnergy         + (1 - a) * prev.rmsEnergy,
      energyDb:          a * raw.energyDb          + (1 - a) * prev.energyDb,
      spectralCentroid:  a * raw.spectralCentroid  + (1 - a) * prev.spectralCentroid,
      spectralRolloff:   a * raw.spectralRolloff   + (1 - a) * prev.spectralRolloff,
      zeroCrossingRate:  a * raw.zeroCrossingRate  + (1 - a) * prev.zeroCrossingRate,
      spectralFlatness:  a * raw.spectralFlatness  + (1 - a) * prev.spectralFlatness,
      dominantFrequency: a * raw.dominantFrequency + (1 - a) * prev.dominantFrequency,
      hnr:               a * raw.hnr               + (1 - a) * prev.hnr,
      subbandEnergies:   raw.subbandEnergies.map((e, i) =>
        a * e + (1 - a) * (prev.subbandEnergies[i] || 0)
      ),
      subbandRatios: {
        f1_low:  a * raw.subbandRatios.f1_low  + (1 - a) * prev.subbandRatios.f1_low,
        f2_f1:   a * raw.subbandRatios.f2_f1   + (1 - a) * prev.subbandRatios.f2_f1,
        high_f2: a * raw.subbandRatios.high_f2 + (1 - a) * prev.subbandRatios.high_f2,
      },
    };

    this._prevFeatures = { ...this._features };
  }

  // ===========================================================================
  // PRIVATE — Vocalization State Classification
  // ===========================================================================

  /**
   * Classify the current feature vector into vocalization state probabilities.
   *
   * Uses a rule-based soft classifier that outputs a probability distribution
   * over the 7 vocalization states, derived from the acoustic features.
   *
   * This is NOT a hard if/else cascade — each state gets a continuous score
   * based on how well the features match its expected profile.
   */
  _classifyVocalizationState() {
    const f = this._features;
    const scores = {};

    // ── SILENT: very low energy ───────────────────────────────────────
    scores[VOC_STATES.SILENT] = sigmoid((this._silenceThresholdDb - f.energyDb) * 0.3);

    // ── BREATH: low energy, high spectral flatness, low HNR ──────────
    scores[VOC_STATES.BREATH] =
      sigmoid((f.energyDb - this._silenceThresholdDb) * 0.2) *
      sigmoid((this._breathThresholdDb - f.energyDb) * 0.2) *
      sigmoid((f.spectralFlatness - 0.3) * 5);

    // ── VOWEL_OPEN: strong energy, low spectral centroid (<800 Hz),
    //    low ZCR, high HNR, dominant F1 region ─────────────────────────
    const isVoiced = sigmoid((f.energyDb - this._breathThresholdDb) * 0.3) *
                     sigmoid((f.hnr - 1.0) * 2);

    scores[VOC_STATES.VOWEL_OPEN] =
      isVoiced *
      sigmoid((800 - f.spectralCentroid) * 0.005) *
      sigmoid((0.10 - f.zeroCrossingRate) * 15) *
      sigmoid((f.subbandRatios.f1_low - 0.3) * 3);

    // ── VOWEL_CLOSED: strong energy, higher spectral centroid (>800 Hz),
    //    low ZCR, high HNR, dominant F2 region ─────────────────────────
    scores[VOC_STATES.VOWEL_CLOSED] =
      isVoiced *
      sigmoid((f.spectralCentroid - 800) * 0.003) *
      sigmoid((0.12 - f.zeroCrossingRate) * 12) *
      sigmoid((f.subbandRatios.f2_f1 - 0.5) * 2);

    // ── CONSONANT_STOP: energy onset (spike), low spectral flatness,
    //    moderate ZCR, transient characteristic ────────────────────────
    this._energyHistory.push(f.rmsEnergy);
    if (this._energyHistory.length > this._energyHistoryMax) {
      this._energyHistory.shift();
    }
    const prevAvgEnergy = this._energyHistory.length > 1
      ? this._energyHistory.slice(0, -1).reduce((s, e) => s + e, 0) / (this._energyHistory.length - 1)
      : f.rmsEnergy;
    const energyOnsetRatio = prevAvgEnergy > 1e-10 ? f.rmsEnergy / prevAvgEnergy : 1;

    scores[VOC_STATES.CONSONANT_STOP] =
      sigmoid((f.energyDb - this._breathThresholdDb) * 0.2) *
      sigmoid((energyOnsetRatio - 1.5) * 3) *
      sigmoid((0.6 - f.spectralFlatness) * 4);

    // ── CONSONANT_FRIC: moderate energy, high ZCR, high spectral flatness,
    //    high spectral centroid, energy in high band ───────────────────
    scores[VOC_STATES.CONSONANT_FRIC] =
      sigmoid((f.energyDb - this._silenceThresholdDb) * 0.2) *
      sigmoid((f.zeroCrossingRate - ZCR_FRICATIVE_THRESHOLD) * 12) *
      sigmoid((f.spectralFlatness - 0.3) * 4) *
      sigmoid((f.spectralCentroid - 2000) * 0.002);

    // ── VOCALIZATION: general voiced sound that doesn't fit neatly
    //    into vowel or consonant categories ────────────────────────────
    scores[VOC_STATES.VOCALIZATION] =
      isVoiced *
      (1 - Math.max(
        scores[VOC_STATES.VOWEL_OPEN],
        scores[VOC_STATES.VOWEL_CLOSED],
        scores[VOC_STATES.CONSONANT_STOP],
        scores[VOC_STATES.CONSONANT_FRIC]
      )) * 0.8 + 0.05;

    // ── Softmax normalization → probability distribution ──────────────
    let maxScore = -Infinity;
    for (const vs of VOC_STATE_ORDER) {
      if (scores[vs] > maxScore) maxScore = scores[vs];
    }
    let sumExp = 0;
    const expScores = {};
    for (const vs of VOC_STATE_ORDER) {
      expScores[vs] = Math.exp((scores[vs] - maxScore) * 5); // Temperature = 0.2
      sumExp += expScores[vs];
    }
    for (const vs of VOC_STATE_ORDER) {
      this._vocStateProbabilities[vs] = sumExp > 0
        ? Math.max(ACOUSTIC_PROB_FLOOR, expScores[vs] / sumExp)
        : 1 / VOC_STATE_ORDER.length;
    }

    // Hard classification (for display)
    let bestState = VOC_STATES.SILENT;
    let bestProb = 0;
    for (const vs of VOC_STATE_ORDER) {
      if (this._vocStateProbabilities[vs] > bestProb) {
        bestProb = this._vocStateProbabilities[vs];
        bestState = vs;
      }
    }
    this._vocState = bestState;
  }

  // ===========================================================================
  // PRIVATE — Acoustic Likelihood Computation
  // ===========================================================================

  /**
   * Compute P(A | S) for each communicative state S.
   *
   * Uses the law of total probability:
   *   P(A | S) = Σ_v  P(vocState = v | acoustic_features) · P(v | S)
   *
   * Where:
   *   P(vocState = v | features) comes from _classifyVocalizationState()
   *   P(v | S) comes from the EMISSION_MATRIX
   *
   * This marginalizes over all possible vocalization states to compute
   * how likely the observed acoustic evidence is under each communicative state.
   */
  _computeAcousticLikelihoods() {
    for (const stateId of this._stateIds) {
      const emissions = EMISSION_MATRIX[stateId];
      if (!emissions) {
        this._acousticLikelihoods[stateId] = ACOUSTIC_PROB_FLOOR;
        continue;
      }

      // P(A|S) = Σ_v P(v|features) · P(v|S)
      let likelihood = 0;
      for (let v = 0; v < VOC_STATE_ORDER.length; v++) {
        const vocState = VOC_STATE_ORDER[v];
        const pVocGivenFeatures = this._vocStateProbabilities[vocState] || 0;
        const pVocGivenState = emissions[v];
        likelihood += pVocGivenFeatures * pVocGivenState;
      }

      this._acousticLikelihoods[stateId] = Math.max(ACOUSTIC_PROB_FLOOR, likelihood);
    }
  }

  // ===========================================================================
  // PRIVATE — Utility
  // ===========================================================================

  /** Result when microphone is inactive. Returns uniform likelihoods. */
  _silentResult() {
    const uniform = 1 / this._stateIds.length;
    const likelihoods = {};
    for (const stateId of this._stateIds) {
      likelihoods[stateId] = uniform;
    }
    return {
      features: { ...this._features },
      vocalization_state: VOC_STATES.SILENT,
      vocalization_probabilities: { [VOC_STATES.SILENT]: 1.0 },
      acoustic_likelihoods: likelihoods,
      is_vocalization: false,
      frame: this._frameCount,
    };
  }
}

// =============================================================================
// MODULE-LEVEL HELPERS
// =============================================================================

/** Sigmoid for soft classification. */
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

/** Round to 2 decimal places. */
function round2(n) {
  return Math.round(n * 100) / 100;
}

// =============================================================================
// DISPLAY HELPERS
// =============================================================================

/**
 * Get color for vocalization state.
 * @param {string} state
 * @returns {string} CSS color
 */
export function vocStateColor(state) {
  switch (state) {
    case VOC_STATES.SILENT:          return '#64748b';
    case VOC_STATES.BREATH:          return '#94a3b8';
    case VOC_STATES.VOWEL_OPEN:      return '#4ade80';
    case VOC_STATES.VOWEL_CLOSED:    return '#34d399';
    case VOC_STATES.CONSONANT_STOP:  return '#f97316';
    case VOC_STATES.CONSONANT_FRIC:  return '#facc15';
    case VOC_STATES.VOCALIZATION:    return '#a78bfa';
    default:                         return '#64748b';
  }
}

/**
 * Get display label for vocalization state.
 * @param {string} state
 * @returns {string}
 */
export function vocStateLabel(state) {
  switch (state) {
    case VOC_STATES.SILENT:          return 'Silent';
    case VOC_STATES.BREATH:          return 'Breath';
    case VOC_STATES.VOWEL_OPEN:      return 'Vowel (Open)';
    case VOC_STATES.VOWEL_CLOSED:    return 'Vowel (Closed)';
    case VOC_STATES.CONSONANT_STOP:  return 'Consonant (Stop)';
    case VOC_STATES.CONSONANT_FRIC:  return 'Consonant (Fric)';
    case VOC_STATES.VOCALIZATION:    return 'Vocalization';
    default:                         return '--';
  }
}

/**
 * Format energy in dB for display.
 * @param {number} db
 * @returns {string}
 */
export function formatEnergyDb(db) {
  return `${db.toFixed(1)} dB`;
}

/**
 * Get color for energy level.
 * @param {number} db
 * @returns {string}
 */
export function energyColor(db) {
  if (db > -10) return '#ef4444';   // Very loud (red)
  if (db > -20) return '#f97316';   // Loud (orange)
  if (db > -30) return '#4ade80';   // Normal vocalization (green)
  if (db > -45) return '#94a3b8';   // Whisper/breath (grey)
  return '#475569';                  // Silent (dark grey)
}

export { VOC_STATES, VOC_STATE_ORDER, EMISSION_MATRIX };
export default UASAM;
