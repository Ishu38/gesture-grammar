/**
 * AccessibilityProfile.js — Adaptive profiles for specially-abled users
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Provides adaptive configuration that adjusts gesture recognition thresholds,
 * feedback modes, and UI adaptations based on the user's disability profile.
 *
 * CP-specific sub-profiles (spastic, athetoid, ataxic, mixed) provide:
 *   - Alternative gesture definitions for restricted motor pathways
 *   - Peak-capture mode for transient gesture windows
 *   - Zone-latching for spatial drift compensation
 *   - Fatigue-aware threshold relaxation
 *   - Per-profile acoustic threshold calibration
 */

// =============================================================================
// ALTERNATIVE GESTURE MAPS — for children who cannot form canonical hand shapes
// =============================================================================

/**
 * Maps canonical gesture IDs to alternative accepted motor patterns.
 * Each alternative specifies which detection strategy to use and the
 * constraints that define the alternative gesture.
 *
 * strategy:
 *   'wrist_rotation'  — Accept wrist supination/pronation instead of finger shape
 *   'partial_extension' — Accept partially extended fingers (wider angle thresholds)
 *   'gross_motion'    — Accept large arm/hand movement direction instead of fine shape
 *   'head_nod'        — Future: accept head gesture as confirmation (requires face landmarks)
 */
export const ALTERNATIVE_GESTURE_MAPS = {
  spastic: {
    SUBJECT_I: [
      { strategy: 'partial_extension', description: 'Partial fist — any curled hand',
        constraints: { minCurledFingers: 3, toleranceMultiplier: 3.0 } },
      { strategy: 'wrist_rotation', description: 'Wrist pronation (palm down)',
        constraints: { wristPalmFacing: 'down', angleTolerance: 45 } },
    ],
    SUBJECT_YOU: [
      { strategy: 'partial_extension', description: 'Any single finger extended',
        constraints: { minExtendedFingers: 1, maxExtendedFingers: 2, toleranceMultiplier: 3.0 } },
      { strategy: 'gross_motion', description: 'Hand push forward',
        constraints: { direction: 'forward', minDisplacement: 0.08 } },
    ],
    SUBJECT_HE: [
      { strategy: 'wrist_rotation', description: 'Wrist supination (palm up) + lateral',
        constraints: { wristPalmFacing: 'up', lateralOffset: 'right', angleTolerance: 45 } },
    ],
    SUBJECT_SHE: [
      { strategy: 'wrist_rotation', description: 'Wrist supination (palm up) + lateral left',
        constraints: { wristPalmFacing: 'up', lateralOffset: 'left', angleTolerance: 45 } },
    ],
    GRAB: [
      { strategy: 'partial_extension', description: 'Partial claw — any finger curl motion',
        constraints: { minCurledFingers: 2, requireCurlMotion: true, toleranceMultiplier: 3.0 } },
      { strategy: 'gross_motion', description: 'Hand closing motion (any grip attempt)',
        constraints: { direction: 'close', minDisplacement: 0.05 } },
    ],
    STOP: [
      { strategy: 'partial_extension', description: 'Any open hand attempt',
        constraints: { minExtendedFingers: 3, toleranceMultiplier: 3.0 } },
      { strategy: 'wrist_rotation', description: 'Palm facing camera',
        constraints: { wristPalmFacing: 'camera', angleTolerance: 50 } },
    ],
    APPLE: [
      { strategy: 'partial_extension', description: 'Cupped hand attempt',
        constraints: { minCurledFingers: 2, maxCurledFingers: 4, toleranceMultiplier: 3.0 } },
    ],
  },

  athetoid: {
    // Athetoid CP: same canonical shapes, but the system must accept them
    // during brief valid windows. No remapping needed — handled by peak-capture DFA mode.
    // This map is intentionally empty; the DFA mode change handles it.
  },

  ataxic: {
    // Ataxic CP: canonical shapes are usually achievable, but spatial positioning
    // is unreliable. Handled by zone-latching in SpatialGrammarMapper.
    // Wider tolerance bands help with imprecise finger positioning.
    SUBJECT_I: [
      { strategy: 'partial_extension', description: 'Relaxed fist tolerance',
        constraints: { minCurledFingers: 3, toleranceMultiplier: 2.5 } },
    ],
    GRAB: [
      { strategy: 'partial_extension', description: 'Relaxed claw tolerance',
        constraints: { minCurledFingers: 2, toleranceMultiplier: 2.5 } },
    ],
  },
};

// =============================================================================
// DFA MODE CONSTANTS
// =============================================================================

export const DFA_MODES = {
  SUSTAINED_HOLD: 'SUSTAINED_HOLD',   // Default: require N consecutive frames
  PEAK_CAPTURE: 'PEAK_CAPTURE',       // Athetoid: lock on first brief valid window
};

// =============================================================================
// ZONE STRATEGY CONSTANTS
// =============================================================================

export const ZONE_STRATEGIES = {
  HARD_BOUNDARY: 'HARD_BOUNDARY',     // Default: fixed X-axis zones
  LATCHED: 'LATCHED',                 // Ataxic: freeze zone at gesture onset
};

// =============================================================================
// PROFILE DEFINITIONS
// =============================================================================

export const ACCESSIBILITY_PROFILES = {
  'speech-impaired': {
    id: 'speech-impaired',
    label: 'Speech Impaired',
    description: 'For non-verbal users — system becomes their voice',
    feedbackModes: ['visual', 'haptic'],
    outputMode: 'text-to-speech',
    gestureSubset: 'full',
    confidenceFrames: 30,
    toleranceMultiplier: 1.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: null,
    alternativeGestureMap: null,
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: true,
      largerText: false,
      highContrast: false,
      screenReaderOptimized: false,
    },
  },
  'hearing-impaired': {
    id: 'hearing-impaired',
    label: 'Hearing Impaired',
    description: 'For deaf/HoH users learning English as L2',
    feedbackModes: ['visual', 'haptic'],
    outputMode: 'visual-text',
    gestureSubset: 'full',
    confidenceFrames: 30,
    toleranceMultiplier: 1.0,
    signLanguageBridge: true,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: null,
    alternativeGestureMap: null,
    uiAdaptations: {
      showTTSControls: false,
      autoSpeak: false,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: false,
      visualAlerts: true,
      noAudioFeedback: true,
    },
  },
  'motor-impaired': {
    id: 'motor-impaired',
    label: 'Motor Impaired',
    description: 'For users with limited hand dexterity (general)',
    feedbackModes: ['visual', 'audio'],
    outputMode: 'text',
    gestureSubset: 'simplified',
    confidenceFrames: 60,
    toleranceMultiplier: 2.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0.05,
    fatigueDetection: true,
    acousticThresholds: { silenceDb: -55, breathDb: -38 },
    alternativeGestureMap: null,
    uiAdaptations: {
      showTTSControls: false,
      autoSpeak: false,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: true,
      largerTouchTargets: true,
      extendedTimeouts: true,
    },
  },

  // =========================================================================
  // CEREBRAL PALSY SUB-PROFILES
  // =========================================================================

  'cp-spastic': {
    id: 'cp-spastic',
    label: 'CP — Spastic',
    description: 'For spastic CP: accepts alternative motor pathways (wrist rotation, partial extension)',
    feedbackModes: ['visual', 'audio', 'haptic'],
    outputMode: 'text-to-speech',
    gestureSubset: 'simplified',
    confidenceFrames: 50,
    toleranceMultiplier: 3.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0.05,
    fatigueDetection: true,
    acousticThresholds: { silenceDb: -55, breathDb: -38 },
    alternativeGestureMap: 'spastic',
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: true,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: false,
      largerTouchTargets: true,
      extendedTimeouts: true,
    },
  },

  'cp-athetoid': {
    id: 'cp-athetoid',
    label: 'CP — Athetoid',
    description: 'For athetoid CP: peak-capture mode locks on brief valid gesture windows (~200ms)',
    feedbackModes: ['visual', 'audio', 'haptic'],
    outputMode: 'text-to-speech',
    gestureSubset: 'simplified',
    confidenceFrames: 6,           // ~200ms at 30fps — the brief valid window
    toleranceMultiplier: 2.5,
    dfaMode: DFA_MODES.PEAK_CAPTURE,
    peakCaptureFrames: 6,          // Lock after just 6 consecutive confident frames
    zoneStrategy: ZONE_STRATEGIES.LATCHED,
    zoneWidening: 0.05,
    fatigueDetection: true,
    acousticThresholds: { silenceDb: -55, breathDb: -38 },
    alternativeGestureMap: 'athetoid',
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: true,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: false,
      largerTouchTargets: true,
      extendedTimeouts: true,
    },
  },

  'cp-ataxic': {
    id: 'cp-ataxic',
    label: 'CP — Ataxic',
    description: 'For ataxic CP: zone-latching prevents spatial drift from changing grammar role',
    feedbackModes: ['visual', 'audio', 'haptic'],
    outputMode: 'text-to-speech',
    gestureSubset: 'simplified',
    confidenceFrames: 45,
    toleranceMultiplier: 2.5,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.LATCHED,
    zoneWidening: 0.10,            // Widen each zone by 10% — significant overlap buffer
    fatigueDetection: true,
    acousticThresholds: { silenceDb: -55, breathDb: -38 },
    alternativeGestureMap: 'ataxic',
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: true,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: false,
      largerTouchTargets: true,
      extendedTimeouts: true,
    },
  },

  'cp-mixed': {
    id: 'cp-mixed',
    label: 'CP — Mixed',
    description: 'For mixed CP: combines peak-capture, zone-latching, and alternative gestures',
    feedbackModes: ['visual', 'audio', 'haptic'],
    outputMode: 'text-to-speech',
    gestureSubset: 'simplified',
    confidenceFrames: 8,
    toleranceMultiplier: 3.0,
    dfaMode: DFA_MODES.PEAK_CAPTURE,
    peakCaptureFrames: 8,          // Slightly longer than athetoid — compromise
    zoneStrategy: ZONE_STRATEGIES.LATCHED,
    zoneWidening: 0.10,
    fatigueDetection: true,
    acousticThresholds: { silenceDb: -58, breathDb: -40 },
    alternativeGestureMap: 'spastic',  // Use spastic alternatives for mixed
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: true,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: false,
      largerTouchTargets: true,
      extendedTimeouts: true,
    },
  },

  // =========================================================================
  // AUTISM SPECTRUM PROFILES
  // =========================================================================

  'asd-low-stimulus': {
    id: 'asd-low-stimulus',
    label: 'ASD — Low Stimulus',
    description: 'For sensory-sensitive autistic users: no animations, predictable feedback, perseveration detection',
    feedbackModes: ['visual'],
    outputMode: 'text',
    gestureSubset: 'full',
    confidenceFrames: 45,
    toleranceMultiplier: 1.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: null,
    alternativeGestureMap: null,
    // ASD-specific
    lowStimulus: true,
    predictableFeedback: true,
    perseverationDetection: true,
    perseverationThreshold: 3,       // flag after 3 identical sequences
    sessionDurationMinutes: 10,      // 10-minute attention window
    microBreakIntervalMinutes: 5,    // suggest break every 5 minutes
    microBreakDurationSeconds: 30,   // 30-second break
    uiAdaptations: {
      showTTSControls: false,
      autoSpeak: false,
      largerText: false,
      highContrast: false,
      screenReaderOptimized: false,
      noAnimations: true,
      noSoundEffects: true,
      fixedFeedbackPosition: true,
      reducedVisualComplexity: true,
    },
  },

  'asd-structured': {
    id: 'asd-structured',
    label: 'ASD — Structured',
    description: 'For ASD with ADHD co-presentation: structured sessions, micro-breaks, minimal distractions',
    feedbackModes: ['visual'],
    outputMode: 'text',
    gestureSubset: 'full',
    confidenceFrames: 45,
    toleranceMultiplier: 1.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: null,
    alternativeGestureMap: null,
    // ASD-specific
    lowStimulus: true,
    predictableFeedback: true,
    perseverationDetection: true,
    perseverationThreshold: 3,
    sessionDurationMinutes: 8,       // shorter window for ADHD co-presentation
    microBreakIntervalMinutes: 4,    // more frequent breaks
    microBreakDurationSeconds: 45,   // slightly longer breaks
    uiAdaptations: {
      showTTSControls: false,
      autoSpeak: false,
      largerText: true,
      highContrast: false,
      screenReaderOptimized: false,
      noAnimations: true,
      noSoundEffects: true,
      fixedFeedbackPosition: true,
      reducedVisualComplexity: true,
    },
  },

  'low-vision': {
    id: 'low-vision',
    label: 'Low Vision',
    description: 'For low-vision users — audio corrections + haptic vibration feedback',
    feedbackModes: ['audio', 'haptic'],
    outputMode: 'text-to-speech',
    gestureSubset: 'full',
    confidenceFrames: 30,
    toleranceMultiplier: 1.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: null,
    alternativeGestureMap: null,
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: true,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: true,
      audioCorrectionFeedback: true,
      hapticFeedback: true,
    },
  },

  'cochlear-implant': {
    id: 'cochlear-implant',
    label: 'Cochlear Implant',
    description: 'For CI users — noise-adaptive acoustic thresholds with ambient calibration',
    feedbackModes: ['visual', 'audio', 'haptic'],
    outputMode: 'text',
    gestureSubset: 'full',
    confidenceFrames: 30,
    toleranceMultiplier: 1.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: { silenceDb: -50, breathDb: -33 },
    noiseFloorCalibration: true,
    alternativeGestureMap: null,
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: false,
      largerText: false,
      highContrast: false,
      screenReaderOptimized: false,
      visualAlerts: true,
    },
  },

  'eye-gaze-aac': {
    id: 'eye-gaze-aac',
    label: 'Eye-Gaze AAC',
    description: 'For non-speaking users who cannot gesture — eye-gaze dwell-click input with phrase bank',
    feedbackModes: ['visual', 'audio', 'haptic'],
    outputMode: 'text-to-speech',
    gestureSubset: 'full',
    confidenceFrames: 30,
    toleranceMultiplier: 1.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: null,
    alternativeGestureMap: null,
    gazeDwellInput: true,
    gazeDwellFrames: 45,
    uiAdaptations: {
      showTTSControls: true,
      autoSpeak: true,
      largerText: true,
      highContrast: true,
      screenReaderOptimized: true,
      audioCorrectionFeedback: true,
      hapticFeedback: true,
      gazeDwellGrid: true,
      phraseBankEnabled: true,
    },
  },

  'default': {
    id: 'default',
    label: 'Standard',
    description: 'Standard L2 learner profile',
    feedbackModes: ['visual', 'audio'],
    outputMode: 'text',
    gestureSubset: 'full',
    confidenceFrames: 45,
    toleranceMultiplier: 1.0,
    dfaMode: DFA_MODES.SUSTAINED_HOLD,
    peakCaptureFrames: null,
    zoneStrategy: ZONE_STRATEGIES.HARD_BOUNDARY,
    zoneWidening: 0,
    fatigueDetection: false,
    acousticThresholds: null,
    alternativeGestureMap: null,
    lowStimulus: false,
    predictableFeedback: false,
    perseverationDetection: false,
    perseverationThreshold: 0,
    sessionDurationMinutes: 0,       // 0 = no limit
    microBreakIntervalMinutes: 0,    // 0 = no breaks
    microBreakDurationSeconds: 0,
    uiAdaptations: {
      showTTSControls: false,
      autoSpeak: false,
      largerText: false,
      highContrast: false,
      screenReaderOptimized: false,
    },
  },
};

// =============================================================================
// SIMPLIFIED GESTURE SUBSET (for motor-impaired users)
// =============================================================================

const SIMPLIFIED_GESTURES = [
  'SUBJECT_I',   // Fist — easy to form
  'SUBJECT_YOU', // Point — universally understood
  'GRAB',        // Claw — gross motor movement
  'STOP',        // Open palm — easy to form
  'APPLE',       // Cupped hand — relaxed position
];

// =============================================================================
// ACCESSIBILITY PROFILE CLASS
// =============================================================================

export class AccessibilityProfile {
  /**
   * @param {string} type — profile ID from ACCESSIBILITY_PROFILES
   */
  constructor(type = 'default') {
    this.type = type;
    this.profile = ACCESSIBILITY_PROFILES[type] || ACCESSIBILITY_PROFILES['default'];
  }

  /**
   * Get the confidence threshold (frames required to lock a gesture).
   * @returns {number} number of frames
   */
  getConfidenceThreshold() {
    return this.profile.confidenceFrames;
  }

  /**
   * Get the set of gestures available for this profile.
   * @returns {string[]|'full'} array of grammar IDs or 'full'
   */
  getGestureSubset() {
    if (this.profile.gestureSubset === 'simplified') {
      return SIMPLIFIED_GESTURES;
    }
    return 'full';
  }

  /**
   * Get feedback modes appropriate for this profile.
   * @returns {string[]} array of feedback mode identifiers
   */
  getFeedbackMode() {
    return this.profile.feedbackModes;
  }

  /**
   * Get tolerance bands multiplier for the SGRM.
   * @returns {number} multiplier for SGRM tolerance bands
   */
  getToleranceBands() {
    return this.profile.toleranceMultiplier;
  }

  /**
   * Get UI adaptations for this profile.
   * @returns {object} UI adaptation flags
   */
  getUIAdaptations() {
    return this.profile.uiAdaptations;
  }

  /**
   * Get the output mode for constructed sentences.
   * @returns {string} 'text', 'text-to-speech', 'visual-text'
   */
  getOutputMode() {
    return this.profile.outputMode;
  }

  /**
   * Check if audio feedback should be suppressed.
   * @returns {boolean}
   */
  shouldSuppressAudio() {
    return this.profile.uiAdaptations.noAudioFeedback === true;
  }

  /**
   * Check if text-to-speech should auto-play on sentence completion.
   * @returns {boolean}
   */
  shouldAutoSpeak() {
    return this.profile.uiAdaptations.autoSpeak === true;
  }

  /**
   * Check if the sign language bridge mode is active.
   * @returns {boolean}
   */
  hasSignLanguageBridge() {
    return this.profile.signLanguageBridge === true;
  }

  // ===========================================================================
  // CP-SPECIFIC ACCESSORS
  // ===========================================================================

  /**
   * Get the DFA gesture lock mode.
   * SUSTAINED_HOLD = default (require N consecutive frames).
   * PEAK_CAPTURE = athetoid mode (lock on first brief valid window).
   * @returns {string} DFA_MODES value
   */
  getDfaMode() {
    return this.profile.dfaMode || DFA_MODES.SUSTAINED_HOLD;
  }

  /**
   * Get the number of frames for peak-capture mode.
   * Only relevant when dfaMode === PEAK_CAPTURE.
   * @returns {number|null}
   */
  getPeakCaptureFrames() {
    return this.profile.peakCaptureFrames || null;
  }

  /**
   * Get the spatial zone strategy.
   * HARD_BOUNDARY = fixed zones.
   * LATCHED = freeze zone at gesture onset (for ataxic drift).
   * @returns {string} ZONE_STRATEGIES value
   */
  getZoneStrategy() {
    return this.profile.zoneStrategy || ZONE_STRATEGIES.HARD_BOUNDARY;
  }

  /**
   * Get zone widening amount (added to each zone boundary).
   * Creates overlap between zones, resolved by proximity to zone center.
   * @returns {number} 0 = no widening, 0.10 = 10% widening per edge
   */
  getZoneWidening() {
    return this.profile.zoneWidening || 0;
  }

  /**
   * Check if fatigue detection should be active.
   * When true, the system monitors declining precision over a session
   * and relaxes thresholds instead of tightening them.
   * @returns {boolean}
   */
  hasFatigueDetection() {
    return this.profile.fatigueDetection === true;
  }

  /**
   * Get per-profile acoustic thresholds for UASAM.
   * Returns null for profiles using default thresholds.
   * @returns {{ silenceDb: number, breathDb: number }|null}
   */
  getAcousticThresholds() {
    return this.profile.acousticThresholds || null;
  }

  /**
   * Get alternative gesture definitions for this profile.
   * Returns null if no alternatives are defined (use canonical gestures).
   * @returns {object|null} Map of gestureId → array of alternative definitions
   */
  getAlternativeGestures() {
    const mapKey = this.profile.alternativeGestureMap;
    if (!mapKey || !ALTERNATIVE_GESTURE_MAPS[mapKey]) return null;
    return ALTERNATIVE_GESTURE_MAPS[mapKey];
  }

  /**
   * Get alternative definitions for a specific gesture.
   * @param {string} gestureId — canonical gesture ID (e.g., 'SUBJECT_I')
   * @returns {Array|null} array of alternative definitions or null
   */
  getAlternativesForGesture(gestureId) {
    const map = this.getAlternativeGestures();
    if (!map || !map[gestureId]) return null;
    return map[gestureId];
  }

  /**
   * Check if this is a CP-specific profile.
   * @returns {boolean}
   */
  isCerebralPalsyProfile() {
    return this.type.startsWith('cp-');
  }

  /**
   * Get the CP subtype if this is a CP profile.
   * @returns {string|null} 'spastic', 'athetoid', 'ataxic', 'mixed', or null
   */
  getCPSubtype() {
    if (!this.isCerebralPalsyProfile()) return null;
    return this.type.replace('cp-', '');
  }

  // ===========================================================================
  // ASD-SPECIFIC ACCESSORS
  // ===========================================================================

  /**
   * Check if low-stimulus mode is active.
   * When true: no animations, no expanding elements, no pulsing/glowing,
   * no sound effects. All feedback is static visual only.
   * @returns {boolean}
   */
  isLowStimulus() {
    return this.profile.lowStimulus === true;
  }

  /**
   * Check if predictable feedback mode is active.
   * When true: corrective feedback always appears in the same screen location,
   * same visual structure, same timing cadence. No variation.
   * @returns {boolean}
   */
  isPredictableFeedback() {
    return this.profile.predictableFeedback === true;
  }

  /**
   * Check if perseveration (script loop) detection is active.
   * @returns {boolean}
   */
  hasPerseverationDetection() {
    return this.profile.perseverationDetection === true;
  }

  /**
   * Get perseveration threshold — how many identical sequences before flagging.
   * @returns {number} 0 = disabled
   */
  getPerseverationThreshold() {
    return this.profile.perseverationThreshold || 0;
  }

  /**
   * Get session duration limit in minutes.
   * @returns {number} 0 = no limit
   */
  getSessionDurationMinutes() {
    return this.profile.sessionDurationMinutes || 0;
  }

  /**
   * Get micro-break interval in minutes.
   * @returns {number} 0 = no breaks
   */
  getMicroBreakIntervalMinutes() {
    return this.profile.microBreakIntervalMinutes || 0;
  }

  /**
   * Get micro-break duration in seconds.
   * @returns {number}
   */
  getMicroBreakDurationSeconds() {
    return this.profile.microBreakDurationSeconds || 30;
  }

  /**
   * Check if this is an ASD-specific profile.
   * @returns {boolean}
   */
  isASDProfile() {
    return this.type.startsWith('asd-');
  }

  /**
   * Check if audio correction feedback is enabled (for low vision).
   * @returns {boolean}
   */
  hasAudioCorrectionFeedback() {
    return this.profile.uiAdaptations?.audioCorrectionFeedback === true ||
           this.profile.feedbackModes?.includes('audio');
  }

  /**
   * Check if haptic vibration feedback is enabled.
   * @returns {boolean}
   */
  hasHapticFeedback() {
    return this.profile.uiAdaptations?.hapticFeedback === true ||
           this.profile.feedbackModes?.includes('haptic');
  }

  /**
   * Check if noise floor calibration is enabled (for cochlear implant).
   * @returns {boolean}
   */
  hasNoiseFloorCalibration() {
    return this.profile.noiseFloorCalibration === true;
  }

  /**
   * Check if gaze-dwell input mode should be auto-activated.
   * @returns {boolean}
   */
  hasGazeDwellInput() {
    return this.profile.gazeDwellInput === true;
  }

  /**
   * Get the gaze dwell frame threshold.
   * @returns {number}
   */
  getGazeDwellFrames() {
    return this.profile.gazeDwellFrames || 45;
  }

  /**
   * Get the complete profile descriptor.
   * @returns {object}
   */
  toDescriptor() {
    return {
      type: this.type,
      label: this.profile.label,
      description: this.profile.description,
      feedbackModes: this.profile.feedbackModes,
      outputMode: this.profile.outputMode,
      confidenceFrames: this.profile.confidenceFrames,
      toleranceMultiplier: this.profile.toleranceMultiplier,
      dfaMode: this.getDfaMode(),
      peakCaptureFrames: this.getPeakCaptureFrames(),
      zoneStrategy: this.getZoneStrategy(),
      zoneWidening: this.getZoneWidening(),
      fatigueDetection: this.hasFatigueDetection(),
      acousticThresholds: this.getAcousticThresholds(),
      hasAlternativeGestures: this.getAlternativeGestures() !== null,
      cpSubtype: this.getCPSubtype(),
      lowStimulus: this.isLowStimulus(),
      predictableFeedback: this.isPredictableFeedback(),
      perseverationDetection: this.hasPerseverationDetection(),
      sessionDurationMinutes: this.getSessionDurationMinutes(),
      microBreakIntervalMinutes: this.getMicroBreakIntervalMinutes(),
      isASD: this.isASDProfile(),
      audioCorrectionFeedback: this.hasAudioCorrectionFeedback(),
      hapticFeedback: this.hasHapticFeedback(),
      noiseFloorCalibration: this.hasNoiseFloorCalibration(),
      gazeDwellInput: this.hasGazeDwellInput(),
      gazeDwellFrames: this.getGazeDwellFrames(),
      uiAdaptations: this.profile.uiAdaptations,
    };
  }
}

// =============================================================================
// LOCAL STORAGE PERSISTENCE
// =============================================================================

const STORAGE_KEY = 'mlaf_accessibility_profile';

/**
 * Save the selected accessibility profile to localStorage.
 * @param {string} profileType
 */
export function saveProfileSelection(profileType) {
  try {
    localStorage.setItem(STORAGE_KEY, profileType);
  } catch {
    // localStorage not available — silently fail
  }
}

/**
 * Load the previously selected accessibility profile from localStorage.
 * @returns {string} profile type ID
 */
export function loadProfileSelection() {
  try {
    return localStorage.getItem(STORAGE_KEY) || 'default';
  } catch {
    return 'default';
  }
}
