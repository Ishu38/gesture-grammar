/**
 * SandboxMode.jsx
 * Main component for the gesture-based sentence building sandbox
 * Integrates camera, gesture detection, sentence builder, error vectors,
 * accessibility profiles, and UI components.
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 */

import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { HandLandmarker, FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { detectGestureRaw, getHandDebugInfo, resetTenseState } from '../utils/gestureDetection';
import { validateSentence, LEXICON } from '../utils/GrammarEngine';
import { analyzeFingerStatesDetailed } from '../utils/fingerAnalysis';
import { classifyFingerState } from '../utils/grammarClassifier';
import { useSentenceBuilder } from '../hooks/useSentenceBuilder';
import { ErrorVectorEngine } from '../core/ErrorVectorEngine';
import { recognize, loadGestureClassifier, getModelLoadStatus } from '../core/SyntacticGesture';
import { GestureClassifierCNN } from '../core/GestureClassifierCNN';
import { PromptTokenInterface } from '../core/PromptTokenInterface';
import { ISLInterferenceDetector } from '../core/ISLInterferenceDetector';
import { AutomaticityTracker, scoreToLabel, trendToArrow, trendToColor } from '../core/AutomaticityTracker';
import { CognitiveLoadAdapter, loadLevelColor, loadLevelDescription } from '../core/CognitiveLoadAdapter';
import { GestureMasteryGate, masteryProgressColor, masteryStatusLabel, CURRICULUM_SEQUENCE } from '../core/GestureMasteryGate';
import { SessionDataLogger, formatAccuracy, accuracyColor } from '../core/SessionDataLogger';
import { RestingBoundaryCalibrator, calibrationStateLabel, calibrationStateColor, CALIBRATION_STATE } from '../core/RestingBoundaryCalibrator';
import { LandmarkSmoother, smoothingIntensityLabel, smoothingIntensityColor } from '../core/LandmarkSmoother';
import { IntentionalityDetector, intentColor, intentLabel } from '../core/IntentionalityDetector';
import { SpatialGrammarMapper, syntacticZoneColor, syntacticZoneLabel } from '../core/SpatialGrammarMapper';
import { GestureLifecycleDFA, DFA_STATES } from '../core/GestureLifecycleDFA';
import { getSemanticTypeSystem } from '../core/SemanticTypeSystem';
import { UMCE, decisionQualityColor, decisionQualityLabel, formatProbability, probabilityColor } from '../core/UMCE';
import { UASAM, vocStateColor, vocStateLabel, formatEnergyDb, energyColor } from '../core/UASAM';
import { EyeGazeTracker } from '../core/EyeGazeTracker';
import { buildMLAFKnowledgeGraph } from '../core/MLAFKnowledgeGraph';
import { GraphRAG } from '../core/GraphRAG';
import { AbductiveFeedbackLoop, ERROR_EVENTS } from '../core/AbductiveFeedbackLoop';
import { CompositionalGeneralization } from '../core/CompositionalGeneralization';
import SentenceStrip from './SentenceStrip';
import GestureSidebar from './GestureSidebar';
import TenseIndicator from './TenseIndicator';
import ParseTreeVisualizer from './ParseTreeVisualizer';
import ErrorOverlay from './ErrorOverlay';
import TextToSpeech from './TextToSpeech';

// ISL Feature imports
import { SpacedRepetitionScheduler, intervalToLabel, easeFactorColor, dueStatusColor } from '../core/SpacedRepetitionScheduler';
import { AchievementSystem, achievementColor, streakColor } from '../core/AchievementSystem';
import { FingerspellingRecognizer } from '../core/FingerspellingRecognizer';
import VisualSentenceSlots from './VisualSentenceSlots';
import ContrastiveDisplay from './ContrastiveDisplay';
import GuidedPracticePanel from './GuidedPracticePanel';
import AchievementPanel from './AchievementPanel';
import AchievementToast from './AchievementToast';
import FingerspellingPanel from './FingerspellingPanel';
import { AccessibleFeedbackEngine } from '../core/AccessibleFeedbackEngine';
import { OBFExporter } from '../core/OBFExporter';
import { PhraseBankManager } from '../core/PhraseBankManager';
import { GazeDwellSelector } from '../core/GazeDwellSelector';

// Gaze state color helper for debug panel
const gazeStateColor = (state) => {
  switch (state) {
    case 'CENTER':      return '#4ade80';
    case 'LEFT':        return '#60a5fa';
    case 'RIGHT':       return '#f472b6';
    case 'UP':          return '#facc15';
    case 'DOWN':        return '#fb923c';
    case 'SACCADE':     return '#c084fc';
    case 'EYES_CLOSED': return '#f87171';
    default:            return '#94a3b8';
  }
};

// Hand connections for drawing skeleton
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

function SandboxMode({ accessibilityProfile, initialMode = 'sandbox', onEndSession }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const streamRef = useRef(null);
  const errorEngineRef = useRef(new ErrorVectorEngine({
    toleranceMultiplier: accessibilityProfile?.getToleranceBands() || 1.0,
  }));
  const promptInterfaceRef = useRef(new PromptTokenInterface({
    model: 'any',
    mode: 'sentence_completion',
  }));
  const islDetectorRef = useRef(new ISLInterferenceDetector());
  const automaticityTrackerRef = useRef(new AutomaticityTracker());
  const prevSentenceLengthRef = useRef(0);
  const cognitiveLoadAdapterRef = useRef(new CognitiveLoadAdapter());
  const masteryGateRef = useRef(new GestureMasteryGate());
  const sessionLoggerRef = useRef(new SessionDataLogger({
    profileType: accessibilityProfile?.type || 'default',
  }));

  // AGGME Pipeline refs
  const calibratorRef = useRef(new RestingBoundaryCalibrator());
  const smootherRef = useRef(new LandmarkSmoother());
  const intentDetectorRef = useRef(new IntentionalityDetector({}));
  const spatialMapperRef = useRef(new SpatialGrammarMapper());
  const gestureDFARef = useRef(null); // initialized after sentence builder hook
  const semanticTypeSystem = useRef(getSemanticTypeSystem());
  const umceRef = useRef(new UMCE());
  const cnnClassifierRef = useRef(new GestureClassifierCNN());
  const cnnResultRef = useRef(null); // cached CNN result for UMCE fusion
  const uasamRef = useRef(new UASAM());
  const gazeTrackerRef = useRef(new EyeGazeTracker());
  const gazeFrameCounter = useRef(0);
  const visionFilesetRef = useRef(null);
  const accessibleFeedbackRef = useRef(new AccessibleFeedbackEngine({
    audioEnabled: accessibilityProfile?.hasAudioCorrectionFeedback?.() || false,
    hapticEnabled: accessibilityProfile?.hasHapticFeedback?.() || false,
  }));

  // AAC systems: OBF export, phrase bank, gaze-dwell input
  const phraseBankRef = useRef(new PhraseBankManager());
  const gazeDwellRef = useRef(new GazeDwellSelector({
    onTokenSelected: null, // wired in useEffect after processGestureInput available
  }));

  // Graph RAG engine (in-browser knowledge graph — 4-layer reasoning)
  // Lazy initialization via factory function passed to useRef
  const graphRAGRef = useRef(null);
  const feedbackLoopRef = useRef(null);
  const compGenRef = useRef(null);
  const graphInitializedRef = useRef(false);

  // ISL Feature refs
  const srsRef = useRef(new SpacedRepetitionScheduler());
  const achievementRef = useRef(new AchievementSystem());
  const fingerspellRef = useRef(new FingerspellingRecognizer());

  // UI State
  const [modelReady, setModelReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTracking, setIsTracking] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [showParseTree, setShowParseTree] = useState(true);
  const [showErrorOverlay, setShowErrorOverlay] = useState(true);
  const [debugInfo, setDebugInfo] = useState(null);
  const [wristY, setWristY] = useState(0.5);
  const [fingerStates, setFingerStates] = useState(null);
  const [grammarToken, setGrammarToken] = useState(null);
  const [errorVectorData, setErrorVectorData] = useState(null);
  const [promptData, setPromptData] = useState(null);
  const [islInterference, setIslInterference] = useState(null);
  const [automaticitySummary, setAutomaticitySummary] = useState(null);
  const [cognitiveLoad, setCognitiveLoad] = useState('LOW');
  const [masteryReport, setMasteryReport] = useState(() => masteryGateRef.current.getMasteryReport());
  const [sessionStats, setSessionStats] = useState(null);
  const [calibrationState, setCalibrationState] = useState('IDLE');
  const [calibrationProgress, setCalibrationProgress] = useState(0);
  const [intentState, setIntentState] = useState('RESTING');
  const [spatialMapping, setSpatialMapping] = useState(null);
  const [umceResult, setUmceResult] = useState(null);
  const [uasamActive, setUasamActive] = useState(false);
  const [uasamResult, setUasamResult] = useState(null);
  const [gazeActive, setGazeActive] = useState(false);
  const [gazeResult, setGazeResult] = useState(null);
  const [dfaState, setDfaState] = useState(DFA_STATES.IDLE);
  const [typeComposition, setTypeComposition] = useState(null);
  const [graphRAGContext, setGraphRAGContext] = useState(null);

  // === RF Gesture Classifier Status ===
  const [rfModelStatus, setRfModelStatus] = useState({ loaded: false, error: null, attempted: false });
  const [gestureConfidence, setGestureConfidence] = useState(null); // { gesture, confidence, isRF }

  // === Grammar Engine (Prolog X-bar) ===
  const [grammarEngineAvailable, setGrammarEngineAvailable] = useState(false);
  const [grammarValidation, setGrammarValidation] = useState(null);
  const [grammarNextCategories, setGrammarNextCategories] = useState(null);

  // ISL Feature state
  const [guidedMode, setGuidedMode] = useState(initialMode === 'guided');
  const [fingerspellMode, setFingerspellMode] = useState(false);
  const [fingerspellDetection, setFingerspellDetection] = useState(null);
  const [achievementReport, setAchievementReport] = useState(() => achievementRef.current.getAchievementReport());
  const [achievementToast, setAchievementToast] = useState(null);

  // AAC state
  const [showPhraseBank, setShowPhraseBank] = useState(false);
  const [savedPhrases, setSavedPhrases] = useState(() => phraseBankRef.current.getAll());
  const [phraseCopiedId, setPhraseCopiedId] = useState(null);
  const [gazeDwellMode, setGazeDwellMode] = useState(false);
  const [gazeDwellGrid, setGazeDwellGrid] = useState(null);
  const [gazeDwellProgress, setGazeDwellProgress] = useState({});
  const [srsReport, setSrsReport] = useState(() => srsRef.current.getReviewReport());

  // Sentence builder hook (with adaptive confidence from accessibility profile)
  const {
    sentence,
    isLocked,
    lockProgress,
    currentGesture,
    currentTenseZone,
    processGestureInput,
    clearSentence,
    undoLastWord,
    setConfidenceThreshold,
  } = useSentenceBuilder({
    confidenceFrames: accessibilityProfile?.getConfidenceThreshold(),
  });

  // Initialize Graph RAG, feedback loop, and compositional generalization (once)
  useEffect(() => {
    if (graphInitializedRef.current) return;
    graphInitializedRef.current = true;
    compGenRef.current = new CompositionalGeneralization();
    try {
      const kg = buildMLAFKnowledgeGraph();
      graphRAGRef.current = new GraphRAG(kg);
      feedbackLoopRef.current = new AbductiveFeedbackLoop(graphRAGRef.current);
    } catch (err) {
      console.error('[MLAF] GraphRAG initialization failed:', err);
    }
  }, []);

  // Wire gaze-dwell progress callback
  useEffect(() => {
    gazeDwellRef.current._onDwellProgress = (cellKey, progress) => {
      setGazeDwellProgress(prev => ({ ...prev, [cellKey]: progress }));
    };
  }, []);

  // Initialize DFA (needs processGestureInput from the hook above)
  useEffect(() => {
    gestureDFARef.current = new GestureLifecycleDFA({
      confirmationFrames: accessibilityProfile?.getConfidenceThreshold() || 30,
      cooldownMs: 1500,
      onStateChange: (newState) => setDfaState(newState),
    });
    return () => gestureDFARef.current?.reset();
  }, [accessibilityProfile]);

  // Update type composition + Graph RAG context whenever the sentence changes
  useEffect(() => {
    const ids = sentence.map(w => w.grammar_id?.toUpperCase?.() || w.grammar_id);
    const composition = semanticTypeSystem.current.composeSentence(ids);
    setTypeComposition(composition);

    // Graph RAG: build full 4-layer context for current sentence state
    if (graphRAGRef.current) {
      const words = sentence.map(w => ({
        type: w.category || LEXICON[w.grammar_id]?.type || 'UNKNOWN',
        grammar_id: w.grammar_id,
        transitive: LEXICON[w.grammar_id]?.transitive,
      }));
      const ctx = graphRAGRef.current.buildLLMContext(ids, words);
      setGraphRAGContext(ctx);
    }
  }, [sentence]);

  // Grammar validation (memoized — only recomputes when sentence changes)
  const validation = useMemo(
    () => validateSentence(sentence.map(w => w.grammar_id)),
    [sentence]
  );

  // Check if current gesture is a verb (for tense indicator)
  const isVerbGesture = currentGesture && LEXICON[currentGesture]?.type === 'VERB';

  // Update error engine tolerance and cognitive load base frames when profile changes
  useEffect(() => {
    if (accessibilityProfile) {
      errorEngineRef.current.setToleranceMultiplier(accessibilityProfile.getToleranceBands());
      cognitiveLoadAdapterRef.current.setBaseFrames(accessibilityProfile.getConfidenceThreshold());
    }
  }, [accessibilityProfile]);

  // === Grammar Engine: Check availability on mount (max 3 retries) ===
  useEffect(() => {
    let alive = true;
    let retryTimer = null;
    let retryCount = 0;
    const MAX_RETRIES = 3;

    const checkEngine = () => {
      fetch('/grammar/health', { signal: AbortSignal.timeout(3000) })
        .then(res => res.json())
        .then(data => {
          if (!alive) return;
          const available = data.engine_loaded === true;
          setGrammarEngineAvailable(available);
          if (!available && retryCount < MAX_RETRIES) {
            retryCount++;
            retryTimer = setTimeout(checkEngine, 10000);
          }
        })
        .catch(() => {
          if (!alive) return;
          setGrammarEngineAvailable(false);
          if (retryCount < MAX_RETRIES) {
            retryCount++;
            retryTimer = setTimeout(checkEngine, 10000);
          }
        });
    };

    checkEngine();

    return () => {
      alive = false;
      if (retryTimer) clearTimeout(retryTimer);
    };
  }, []);

  // Run ISL interference detection reactively on every sentence change
  // Uses Prolog grammar engine when available, falls back to JS detector
  useEffect(() => {
    const abortCtrl = new AbortController();
    const { signal } = abortCtrl;

    if (sentence.length > 0) {
      if (grammarEngineAvailable) {
        const gestureIds = sentence.map(w => w.grammar_id);
        fetch('/grammar/interference', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gestures: gestureIds }),
          signal,
        })
          .then(res => res.ok ? res.json() : null)
          .then(data => {
            if (signal.aborted) return;
            if (data) {
              const report = {
                hasInterference: data.has_interference,
                patterns: data.patterns,
                severity: data.severity,
                sentence_display: sentence.map(w => w.word).join(' '),
                transform_suggestion: data.transform_suggestion,
                timestamp: Date.now(),
              };
              setIslInterference(report.hasInterference ? report : null);
              if (report.hasInterference) {
                sessionLoggerRef.current.logISLInterference(report);
              }
            } else {
              // Engine error, fall back to JS detector
              const report = islDetectorRef.current.analyze(sentence);
              setIslInterference(report.hasInterference ? report : null);
              if (report.hasInterference) sessionLoggerRef.current.logISLInterference(report);
            }
          })
          .catch(err => {
            if (err.name === 'AbortError') return;
            // Engine unreachable, fall back to JS detector
            const report = islDetectorRef.current.analyze(sentence);
            setIslInterference(report.hasInterference ? report : null);
            if (report.hasInterference) sessionLoggerRef.current.logISLInterference(report);
          });

        // Also fetch grammar validation for tense resolution and score
        fetch('/grammar/validate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gestures: gestureIds }),
          signal,
        })
          .then(res => res.ok ? res.json() : null)
          .then(data => { if (!signal.aborted && data) setGrammarValidation(data); })
          .catch(err => { if (err.name !== 'AbortError') setGrammarValidation(null); });
      } else {
        const report = islDetectorRef.current.analyze(sentence);
        setIslInterference(report.hasInterference ? report : null);
        if (report.hasInterference) {
          sessionLoggerRef.current.logISLInterference(report);
        }
      }
    } else {
      setIslInterference(null);
      setGrammarValidation(null);
    }

    return () => abortCtrl.abort();
  }, [sentence, grammarEngineAvailable]);

  // Automaticity: notify tracker when user is ready for next gesture
  useEffect(() => {
    if (!isLocked) {
      automaticityTrackerRef.current.onReadyForNextGesture();
    }
  }, [isLocked]);

  // Automaticity + Mastery + SessionLog: record all metrics when a new word locks
  useEffect(() => {
    if (sentence.length > prevSentenceLengthRef.current && sentence.length > 0) {
      const lastWord = sentence[sentence.length - 1];

      // Automaticity timing
      const prodRecord = automaticityTrackerRef.current.onGestureLocked(lastWord.grammar_id);
      setAutomaticitySummary(automaticityTrackerRef.current.getSessionSummary());

      // Mastery gate
      const masteryResult = masteryGateRef.current.recordProduction(lastWord.grammar_id);
      setMasteryReport(masteryGateRef.current.getMasteryReport());

      // Session data logger — gesture lock event
      sessionLoggerRef.current.logGestureLock(lastWord, {
        productionTime: prodRecord?.total_ms || null,
        cognitiveLoad: cognitiveLoadAdapterRef.current.getLevel(),
        jitter: cognitiveLoadAdapterRef.current.getJitter(),
      });

      // Session data logger — mastery achievement event
      if (masteryResult.isNowMastered && !masteryResult.wasMastered) {
        sessionLoggerRef.current.logMasteryAchieved(lastWord.grammar_id, masteryResult.count);
      }

      // SRS: record successful review on gesture lock
      srsRef.current.recordReview(lastWord.grammar_id, 5);
      setSrsReport(srsRef.current.getReviewReport());

      // Achievement: check mastery unlock
      if (masteryResult.isNowMastered && !masteryResult.wasMastered) {
        const masteryAch = achievementRef.current.onGestureMastered(lastWord.grammar_id);
        if (masteryAch.newlyUnlocked.length > 0) {
          setAchievementToast(masteryAch.newlyUnlocked[0]);
          setAchievementReport(achievementRef.current.getAchievementReport());
        }
      }

      // Abductive Feedback Loop: record successful gesture lock
      if (feedbackLoopRef.current) {
        feedbackLoopRef.current.recordSuccess(lastWord.grammar_id);
      }

      // Compositional Generalization: record practiced sentence
      if (compGenRef.current && sentence.length >= 2) {
        compGenRef.current.recordPracticed(sentence.map(w => w.grammar_id));
      }

      // Update live session stats
      setSessionStats(sessionLoggerRef.current.getSessionSummary());
    }
    prevSentenceLengthRef.current = sentence.length;
  }, [sentence]);

  // SessionLog: record sentence completion when grammar validation passes
  const prevValidationCompleteRef = useRef(false);
  useEffect(() => {
    if (validation.isComplete && !prevValidationCompleteRef.current && sentence.length > 0) {
      sessionLoggerRef.current.logSentenceComplete(sentence, validation);
      setSessionStats(sessionLoggerRef.current.getSessionSummary());

      // Achievement: sentence complete event
      const sentenceAch = achievementRef.current.onSentenceComplete({
        sentence,
        validation,
        islInterference,
      });
      if (sentenceAch.newlyUnlocked.length > 0) {
        setAchievementToast(sentenceAch.newlyUnlocked[0]);
        setAchievementReport(achievementRef.current.getAchievementReport());
      }
    }
    prevValidationCompleteRef.current = validation.isComplete;
  }, [validation.isComplete, sentence, islInterference]);

  // Process landmarks — AGGME Pipeline: Raw → Calibrate → Smooth → Intent → Spatial → Detect
  // Wrapped in try-catch for pipeline resilience — a single frame error must never kill the loop.
  const processLandmarks = useCallback((landmarks) => {
    if (!landmarks || landmarks.length === 0) {
      // Update DFA: no hand detected
      if (gestureDFARef.current) {
        gestureDFARef.current.process({
          handPresent: false,
          intentState: 'RESTING',
          gestureId: null,
        });
      }
      processGestureInput(null, 0.5);
      setDebugInfo(null);
      setFingerStates(null);
      setGrammarToken(null);
      setErrorVectorData(null);
      return;
    }

    try {

    const rawLandmarks = landmarks[0];

    // =========================================================================
    // AGGME Phase 1: Resting Boundary Calibration (feed every frame)
    // =========================================================================
    const calibrator = calibratorRef.current;
    if (calibrator.getState() === 'CALIBRATING') {
      const calResult = calibrator.processFrame(rawLandmarks);
      setCalibrationProgress(calResult.progress);
      if (calResult.state === 'READY') {
        setCalibrationState('READY');
        // Phase 1 complete → configure Phase 2 (smoother) and Phase 3 (intent)
        smootherRef.current.calibrateFromJitter(calibrator.getRestingJitter());
        intentDetectorRef.current.setRestingProfile(calibrator.getRestingProfile());
      } else if (calResult.state === 'FAILED') {
        setCalibrationState('FAILED');
      }
      // During calibration, still show the hand but skip gesture detection
      setWristY(rawLandmarks[0].y);
      if (showDebug) setDebugInfo(getHandDebugInfo(rawLandmarks));
      return;
    }

    // =========================================================================
    // AGGME Phase 2: EMA Signal Smoothing
    // =========================================================================
    const smoothedLandmarks = smootherRef.current.smooth(rawLandmarks);

    // =========================================================================
    // AGGME Phase 3: Intentionality Detection
    // =========================================================================
    let intentResult = null;
    if (calibrator.isReady()) {
      intentResult = intentDetectorRef.current.detect(smoothedLandmarks);
      setIntentState(intentResult.intent);

      // If RESTING (tremor only), suppress gesture detection
      if (intentResult.intent === 'RESTING') {
        // Update DFA: hand present but resting
        if (gestureDFARef.current) {
          gestureDFARef.current.process({
            handPresent: true,
            intentState: 'RESTING',
            gestureId: null,
          });
        }
        processGestureInput(null, smoothedLandmarks[0].y);
        setWristY(smoothedLandmarks[0].y);
        if (showDebug) setDebugInfo(getHandDebugInfo(smoothedLandmarks));
        return;
      }
    }

    // =========================================================================
    // AGGME Phase 4: Spatial Grammar Mapping
    // =========================================================================
    const spatialResult = spatialMapperRef.current.map(smoothedLandmarks);
    if (spatialResult.syntactic_zone) {
      setSpatialMapping(spatialResult);
    }

    // =========================================================================
    // Downstream pipeline (uses smoothed landmarks, not raw)
    // =========================================================================
    const handLandmarks = smoothedLandmarks;
    const wrist = handLandmarks[0];

    // Track wrist Y for tense modification
    setWristY(wrist.y);

    // Update debug info
    if (showDebug) {
      setDebugInfo(getHandDebugInfo(handLandmarks));
    }

    // Angle-based finger analysis
    const detailed = analyzeFingerStatesDetailed(handLandmarks);
    setFingerStates(detailed);
    setGrammarToken(classifyFingerState(detailed.states));

    // =========================================================================
    // UASAM: Acoustic analysis — P(A | S)
    // =========================================================================
    let acousticData = null;
    if (uasamRef.current.isActive()) {
      acousticData = uasamRef.current.processFrame();
      setUasamResult(acousticData);
    }

    // =========================================================================
    // EyeGazeTracker: Gaze analysis — P(G | S) (cached from detection loop)
    // =========================================================================
    const gazeData = gazeTrackerRef.current.getLastResult();

    // =========================================================================
    // UMCE: Bayesian Trimodal Late Fusion
    // P(S | A, V, G) ∝ P(A|S)^w_A · P(V|S)^w_V · P(G|S)^w_G · P(S)
    // =========================================================================
    const rawGesture = detectGestureRaw(handLandmarks);

    // CNN: async inference — fire-and-forget, result cached for next frame
    if (cnnClassifierRef.current.loaded) {
      cnnClassifierRef.current.classify(handLandmarks).then(r => {
        cnnResultRef.current = r;
      }).catch(() => { /* non-critical */ });
    }

    umceRef.current.setSentenceContext(sentence);
    umceRef.current.setMasteryData(masteryGateRef.current.getMasteryReport());
    const fusionResult = umceRef.current.fuse({
      landmarks: handLandmarks,
      spatialResult,
      intentResult,
      cognitiveLoad: {
        level: cognitiveLoadAdapterRef.current.getLevel(),
        jitter: cognitiveLoadAdapterRef.current.getJitter(),
      },
      acousticLikelihoods: acousticData?.acoustic_likelihoods || null,
      acousticActive: uasamRef.current.isActive(),
      vocalizationState: acousticData?.vocalization_state || null,
      gazeLikelihoods: gazeData?.gaze_likelihoods || null,
      gazeActive: gazeTrackerRef.current.isActive(),
      gazeState: gazeData?.gaze_state || null,
      rawGesture,
      cnnResult: cnnResultRef.current,
    });
    setUmceResult(fusionResult);

    // Fingerspelling: detect letters if mode is active
    if (fingerspellMode) {
      const fsResult = fingerspellRef.current.detectLetter(handLandmarks);
      setFingerspellDetection(fsResult);
    }

    // Use UMCE fused classification instead of raw deterministic output
    const gesture = umceRef.current.getClassification(fusionResult) || rawGesture;

    // Track gesture confidence for UI display
    if (gesture) {
      const rfResult = fusionResult?.visual_result;
      setGestureConfidence({
        gesture,
        confidence: rfResult?.confidence || (rawGesture ? 0.85 : 0),
        isRF: !!rfResult?.confidence,
      });
    } else {
      setGestureConfidence(null);
    }

    // DFA: Process gesture through formal finite automaton
    if (gestureDFARef.current) {
      gestureDFARef.current.process({
        handPresent: true,
        intentState: intentResult?.intent || 'GESTURE_ACTIVE',
        gestureId: gesture,
      });
    }

    // Type check: validate gesture slot compatibility before locking.
    // Uses Graph RAG Layer 1 (deductive traversal) for precise valid-next
    // determination including verb agreement, transitivity, and sentence completeness.
    // Abductive Feedback Loop adjusts thresholds based on learner error history.
    let typeValidGesture = gesture;
    if (gesture && graphRAGRef.current) {
      const currentSlots = sentence.map(w => w.grammar_id?.toUpperCase?.() || w.grammar_id);
      const ragResult = graphRAGRef.current.queryValidNext(currentSlots);
      const isValid = ragResult.validNext.some(v => v.grammar_id === gesture);

      // Check if gesture is curriculum-gated by feedback loop
      const isGated = feedbackLoopRef.current?.isGated(gesture);

      if ((!isValid && ragResult.validNext.length > 0) || isGated) {
        typeValidGesture = null; // Block from locking

        // Record type mismatch error in feedback loop
        if (feedbackLoopRef.current && !isValid) {
          feedbackLoopRef.current.recordError(ERROR_EVENTS.TYPE_MISMATCH, {
            recognizedGesture: gesture,
            sentenceTokens: currentSlots,
          });
        }
      }
    } else if (gesture && semanticTypeSystem.current) {
      const currentSlots = sentence.map(w => w.grammar_id?.toUpperCase?.() || w.grammar_id);
      const composition = semanticTypeSystem.current.composeSentence(currentSlots);
      const gestureInfo = semanticTypeSystem.current.getGestureType(gesture);
      if (gestureInfo && !composition.expectsNext.includes(gestureInfo.slot)) {
        typeValidGesture = null;
      }
    }
    processGestureInput(typeValidGesture, wrist.y);

    // Automaticity: record each frame's gesture for onset timing
    automaticityTrackerRef.current.onGestureFrame(gesture);

    // Cognitive load: measure jitter and adjust confidence threshold on level change
    const loadResult = cognitiveLoadAdapterRef.current.update(handLandmarks);
    if (loadResult.levelChanged) {
      setCognitiveLoad(loadResult.level);
      setConfidenceThreshold(loadResult.recommendedFrames);
      sessionLoggerRef.current.logCognitiveLoadChange(
        loadResult.level, loadResult.jitter, loadResult.recommendedFrames
      );
    }

    // Compute error vectors (closed-loop feedback)
    if (showErrorOverlay) {
      const errorResult = errorEngineRef.current.compute(handLandmarks);
      setErrorVectorData(errorResult);

      // Non-visual feedback: TTS corrections + haptic vibration (low vision)
      if (accessibleFeedbackRef.current) {
        accessibleFeedbackRef.current.processFrame(errorResult);
      }
    }

    // Recognize as SyntacticGesture for prompt token interface
    const syntacticGesture = recognize(handLandmarks);
    if (syntacticGesture && gesture) {
      // Store latest prompt data for debug display
      setPromptData({
        gesture_id: syntacticGesture.grammarBinding.grammar_id,
        category: syntacticGesture.grammarBinding.category,
        tense: syntacticGesture.spatialModifier?.zone,
      });
    }

    } catch (err) {
      // Pipeline error in a single frame — log and continue, never kill the loop
      console.warn('[MLAF] processLandmarks frame error:', err.message);
    }
  }, [processGestureInput, showDebug, showErrorOverlay, setConfidenceThreshold, sentence, fingerspellMode]);

  // Draw hand landmarks
  const drawLandmarks = useCallback((landmarks, ctx, width, height) => {
    ctx.clearRect(0, 0, width, height);

    landmarks.forEach((handLandmarks) => {
      // Draw connections
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;

      HAND_CONNECTIONS.forEach(([start, end]) => {
        const startPoint = handLandmarks[start];
        const endPoint = handLandmarks[end];

        ctx.beginPath();
        ctx.moveTo(startPoint.x * width, startPoint.y * height);
        ctx.lineTo(endPoint.x * width, endPoint.y * height);
        ctx.stroke();
      });

      // Draw landmarks
      handLandmarks.forEach((landmark, index) => {
        const x = landmark.x * width;
        const y = landmark.y * height;

        ctx.fillStyle = index === 0 ? '#FFFF00' : // Wrist
                        index === 4 || index === 8 ? '#FF0000' : // Thumb/Index tips
                        '#00FF00';
        ctx.beginPath();
        ctx.arc(x, y, index === 0 ? 10 : 5, 0, 2 * Math.PI);
        ctx.fill();
      });
    });
  }, []);

  // Detection loop
  const detectHands = useCallback(() => {
    // Cancel any previously scheduled frame to prevent duplicate loops
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const handLandmarker = handLandmarkerRef.current;

    if (!video || !canvas || !handLandmarker || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectHands);
      return;
    }

    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;

      try {
        const now = performance.now();
        gazeFrameCounter.current++;

        // Interleave models: gaze runs on frame 3,6,9,... INSTEAD of hand (not alongside)
        // This prevents both GPU models from competing in the same rAF tick.
        const isGazeFrame = gazeFrameCounter.current % 3 === 0 &&
                            gazeTrackerRef.current.isActive();

        if (isGazeFrame) {
          // Gaze-only frame (~10fps): run FaceLandmarker, skip HandLandmarker
          const gazeData = gazeTrackerRef.current.processFrame(video, now);
          setGazeResult(gazeData);

          // Gaze-dwell token selection (AAC eye-gaze input mode)
          if (gazeDwellRef.current?.isActive() && gazeData?.face_detected) {
            const dwellResult = gazeDwellRef.current.processFrame(gazeData);
            if (dwellResult.selected && dwellResult.selectedToken) {
              // Feed selected token into sentence builder as if it were a gesture
              // Gaze-dwell: defaults to PRESENT tense (wristY=0.5) — tense zones are hand-position-based
              processGestureInput(dwellResult.selectedToken, 0.5);
            }
            setGazeDwellGrid(gazeDwellRef.current.getCurrentGrid());
          }
        } else {
          // Hand frame (~20fps): run HandLandmarker, skip FaceLandmarker
          const results = handLandmarker.detectForVideo(video, now);
          const ctx = canvas.getContext('2d');

          if (results.landmarks && results.landmarks.length > 0) {
            processLandmarks(results.landmarks);
            drawLandmarks(results.landmarks, ctx, canvas.width, canvas.height);
          } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            processLandmarks(null);
          }
        }
      } catch (err) {
        // MediaPipe or pipeline error — log and keep the detection loop alive
        console.warn('[MLAF] detectHands frame error:', err.message);
      }
    }

    animationFrameRef.current = requestAnimationFrame(detectHands);
  }, [processLandmarks, drawLandmarks]);

  // Phase 1: Load the ML model on mount (no camera yet)
  useEffect(() => {
    let mounted = true;

    async function loadModel() {
      try {
        setIsLoading(true);

        // Load WASM (local first for speed, CDN fallback)
        let vision;
        try {
          vision = await FilesetResolver.forVisionTasks('/wasm');
        } catch {
          vision = await FilesetResolver.forVisionTasks(
            'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
          );
        }

        // Store vision fileset for shared use by FaceLandmarker
        visionFilesetRef.current = vision;

        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 1
        });

        if (!mounted) {
          handLandmarker.close();
          return;
        }

        handLandmarkerRef.current = handLandmarker;
        setIsLoading(false);
        setModelReady(true);

        // Load RF gesture classifier (non-blocking)
        loadGestureClassifier('/models/gesture_classifier_v1.json').then((ok) => {
          if (mounted) {
            setRfModelStatus(getModelLoadStatus());
          }
        });

        // Load CNN gesture classifier (non-blocking, ONNX Runtime Web)
        cnnClassifierRef.current.load('/models/gesture_cnn.onnx', '/models/gesture_cnn.json')
          .then(() => { if (mounted) console.log('[MLAF] CNN gesture classifier loaded (ONNX)'); })
          .catch((err) => { console.warn('[MLAF] CNN classifier not available:', err.message); });

        // Co-initialize FaceLandmarker for gaze tracking (non-blocking)
        gazeTrackerRef.current.initialize(vision).then((ok) => {
          if (mounted && ok) setGazeActive(true);
        });
      } catch (err) {
        if (mounted) {
          setError(err.message);
          setIsLoading(false);
        }
      }
    }

    loadModel();

    return () => {
      mounted = false;
      resetTenseState(); // Clear module-level tense EMA state for next session
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
      // Release UASAM microphone resources
      uasamRef.current.destroy();
      // Release EyeGazeTracker FaceLandmarker resources
      gazeTrackerRef.current.destroy();
      // Release AccessibleFeedbackEngine TTS resources
      accessibleFeedbackRef.current?.destroy();
    };
  }, []);

  // Phase 2: Start camera (called from user click — needed for browser permissions)
  const startCamera = useCallback(async () => {
    try {
      setError(null);
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: 'user' }
      });
      streamRef.current = stream;
      automaticityTrackerRef.current.initSession();
      sessionLoggerRef.current.logSessionStart();
      achievementRef.current.recordDailyActivity();
      setAchievementReport(achievementRef.current.getAchievementReport());

      // Initialize UASAM (microphone) in parallel — non-blocking
      uasamRef.current.initialize().then(async (active) => {
        setUasamActive(active);
        // Noise floor calibration for cochlear implant profiles
        if (active && accessibilityProfile?.hasNoiseFloorCalibration?.()) {
          try {
            const calResult = await uasamRef.current.calibrateNoiseFloor(3000);
            console.log('[UASAM] Noise floor calibrated:', calResult);
          } catch (e) {
            console.warn('[UASAM] Noise floor calibration failed:', e.message);
          }
        }
      }).catch(() => {
        // Microphone denied or unavailable — visual-only mode
        setUasamActive(false);
      });

      // Auto-activate gaze-dwell input mode for eye-gaze-aac profile
      if (accessibilityProfile?.hasGazeDwellInput?.()) {
        gazeDwellRef.current.setDwellFrames(accessibilityProfile.getGazeDwellFrames());
        gazeDwellRef.current.activate();
        setGazeDwellMode(true);
        setGazeDwellGrid(gazeDwellRef.current.getCurrentGrid());
      }

      setIsTracking(true);
    } catch (err) {
      setError(err.message);
    }
  }, []);

  // Attach camera stream to video element once it renders
  useEffect(() => {
    if (!isTracking || !streamRef.current) return;

    const video = videoRef.current;
    if (video && !video.srcObject) {
      video.srcObject = streamRef.current;
      video.play().then(() => {
        const canvas = canvasRef.current;
        if (canvas) {
          canvas.width = video.videoWidth;
          canvas.height = video.videoHeight;
        }
      });
    }
  }, [isTracking]);

  // Start detection loop
  useEffect(() => {
    if (isTracking) {
      detectHands();
    }
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [isTracking, detectHands]);

  // AGGME: Start resting boundary calibration
  const startCalibration = useCallback(() => {
    calibratorRef.current.startCalibration();
    setCalibrationState('CALIBRATING');
    setCalibrationProgress(0);
  }, []);

  // Wrapped clear handler — logs to session logger before clearing
  const handleClear = useCallback(() => {
    if (sentence.length > 0) {
      sessionLoggerRef.current.logSentenceClear(sentence, 'user_clear');
      setSessionStats(sessionLoggerRef.current.getSessionSummary());
    }
    clearSentence();
  }, [sentence, clearSentence]);

  // Get gesture display name
  const getGestureDisplay = (gesture) => {
    if (!gesture) return null;
    const entry = LEXICON[gesture];
    return entry?.display || gesture;
  };

  // Fingerspelling: when a spelled word matches LEXICON, add to sentence
  const handleFingerspellMatch = useCallback((grammarId) => {
    if (grammarId && LEXICON[grammarId]) {
      processGestureInput(grammarId, 0.5);
    }
  }, [processGestureInput]);

  if (isLoading) {
    return (
      <div className="sandbox-loading">
        <div className="loading-spinner" />
        <p>Loading hand tracking model...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="sandbox-error">
        <p>Error: {error}</p>
        <button onClick={() => window.location.reload()}>Retry</button>
      </div>
    );
  }

  if (modelReady && !isTracking) {
    return (
      <div className="sandbox-loading">
        <p style={{ marginBottom: '1rem', color: '#a0ffa0' }}>Model loaded!</p>
        <button className="start-camera-btn" onClick={startCamera}>
          Start Camera
        </button>
        <p style={{ marginTop: '0.75rem', fontSize: '0.75rem', color: '#888' }}>
          Allow camera access when prompted
        </p>
      </div>
    );
  }

  return (
    <div className="sandbox-mode">
      {/* RF Model Load Warning */}
      {rfModelStatus.attempted && !rfModelStatus.loaded && rfModelStatus.error && (
        <div style={{
          position: 'fixed', top: '0.5rem', left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(251, 146, 60, 0.15)', border: '1px solid rgba(251, 146, 60, 0.4)',
          borderRadius: '8px', padding: '0.5rem 1rem', color: '#fb923c',
          fontSize: '0.75rem', zIndex: 100, maxWidth: '500px', textAlign: 'center',
          backdropFilter: 'blur(8px)',
        }}>
          <span style={{ fontWeight: 600 }}>Heuristic Mode:</span> ML classifier unavailable — using rule-based detection (lower accuracy).
        </div>
      )}

      {/* Left Sidebar - Gesture Guide */}
      <GestureSidebar
        currentGesture={currentGesture}
        lockProgress={lockProgress}
        currentTenseZone={currentTenseZone}
      />

      {/* Main Content */}
      <div className="sandbox-main">
        {/* Sentence Strip at top */}
        <SentenceStrip
          sentence={sentence}
          onClear={handleClear}
          onUndo={undoLastWord}
          isLocked={isLocked}
          lockProgress={lockProgress}
        />

        {/* Grammar validation panel */}
        <div className={`validation-panel ${validation.isComplete ? 'complete' : validation.error ? 'error' : 'building'}`}>
          <span className="validation-status">
            {validation.isComplete ? '✓ Complete' : validation.error ? '✗ Error' : '○ Building'}
          </span>
          <span className="validation-message">
            {validation.error || validation.suggestion}
          </span>
          {validation.parseTree && (
            <button
              className={`parse-tree-toggle ${showParseTree ? 'active' : ''}`}
              onClick={() => setShowParseTree(!showParseTree)}
            >
              <span className="toggle-icon">▶</span>
              Parse Tree
            </button>
          )}
          <span style={{
            fontSize: '0.7rem',
            color: grammarEngineAvailable ? '#8b5cf6' : '#555',
            display: 'inline-flex',
            alignItems: 'center',
            gap: '4px',
            marginLeft: '8px',
          }}>
            <span style={{
              width: '6px',
              height: '6px',
              borderRadius: '50%',
              background: grammarEngineAvailable ? '#8b5cf6' : '#555',
            }} />
            {grammarEngineAvailable ? 'Prolog' : 'JS'}
          </span>
          {grammarValidation && (
            <span style={{ fontSize: '0.7rem', color: '#8b5cf6', marginLeft: '6px' }}>
              Score: {((grammarValidation.grammaticality_score || 0) * 100).toFixed(0)}%
              {grammarValidation.agreement?.inflected_form && (
                <> | {grammarValidation.agreement.inflected_form}</>
              )}
            </span>
          )}
        </div>

        {/* Visual Sentence Slots — SVO structure guide */}
        {!guidedMode && (
          <VisualSentenceSlots sentence={sentence} currentGesture={currentGesture} />
        )}

        {/* ISL Interference Panel — shown when L1 transfer errors are detected */}
        {islInterference && (
          <div className={`isl-interference-panel isl-severity-${islInterference.severity}`}>
            <div className="isl-panel-header">
              <span className="isl-panel-icon">
                {islInterference.severity === 'error' ? '⚠' : '○'}
              </span>
              <span className="isl-panel-title">ISL Transfer Detected</span>
              <span className="isl-pattern-count">
                {islInterference.patterns.length} pattern{islInterference.patterns.length > 1 ? 's' : ''}
              </span>
            </div>
            {islInterference.patterns.map((pattern, i) => (
              <div key={i} className="isl-pattern-block">
                <div className="isl-pattern-name">{pattern.title}</div>
                <div className="isl-pattern-description">{pattern.description}</div>
                <div className="isl-correction">
                  <span className="isl-correction-label">English:</span>
                  {pattern.correction}
                </div>
                <div className="isl-examples">
                  <span className="isl-example-label">ISL pattern:</span>
                  <span className="isl-example-bad">{pattern.example_isl}</span>
                  <span className="isl-example-label">Target:</span>
                  <span className="isl-example-good">{pattern.example_english}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* ISL-English Contrastive Display — visual reordering when ISL transfer detected */}
        {islInterference && (
          <ContrastiveDisplay interferenceReport={islInterference} sentence={sentence} />
        )}

        {/* Parse tree visualization (shown when sentence has a valid parse) */}
        {validation.parseTree && showParseTree && (
          <ParseTreeVisualizer parseTree={validation.parseTree} />
        )}

        {/* Video container */}
        <div className="video-wrapper">
          <div className={`video-container ${isLocked ? 'locked' : ''}`}>
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              style={{ transform: 'scaleX(-1)' }}
            />
            <canvas
              ref={canvasRef}
              className="overlay-canvas"
              style={{ transform: 'scaleX(-1)' }}
            />

            {/* Gesture overlay with confidence */}
            {currentGesture && (
              <div className="gesture-overlay">
                <div className="gesture-name">{getGestureDisplay(currentGesture)}</div>
                {gestureConfidence && (
                  <div style={{
                    fontSize: '0.65rem',
                    color: gestureConfidence.confidence > 0.7 ? '#4ade80'
                         : gestureConfidence.confidence > 0.5 ? '#facc15'
                         : '#f87171',
                    marginTop: '2px',
                    textAlign: 'center',
                  }}>
                    {Math.round(gestureConfidence.confidence * 100)}% {gestureConfidence.isRF ? 'ML' : 'rule'}
                  </div>
                )}
                <div className="gesture-progress">
                  <svg viewBox="0 0 100 100">
                    <circle className="progress-bg" cx="50" cy="50" r="45" />
                    <circle
                      className="progress-fill"
                      cx="50" cy="50" r="45"
                      style={{ strokeDasharray: `${lockProgress * 283} 283` }}
                    />
                  </svg>
                  <span className="progress-percent">{Math.round(lockProgress * 100)}%</span>
                </div>
              </div>
            )}

            {/* Error correction overlay — dimmed when cognitive load is high */}
            <div style={{
              opacity: cognitiveLoadAdapterRef.current.getOverlayOpacity(),
              transition: 'opacity 0.6s ease',
            }}>
              <ErrorOverlay
                errorData={errorVectorData}
                canvasWidth={640}
                canvasHeight={480}
                visible={showErrorOverlay && !!errorVectorData}
                predictableFeedback={accessibilityProfile?.isPredictableFeedback?.()}
              />
            </div>

            {/* Lock indicator */}
            {isLocked && (
              <div className="lock-overlay">
                <span>Cooldown</span>
              </div>
            )}
          </div>

          {/* Control toggles */}
          <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
            <button
              className="debug-toggle"
              onClick={() => setShowDebug(!showDebug)}
            >
              {showDebug ? 'Hide Debug' : 'Debug'}
            </button>
            <button
              className={`debug-toggle ${showErrorOverlay ? 'active' : ''}`}
              onClick={() => setShowErrorOverlay(!showErrorOverlay)}
              style={showErrorOverlay ? { background: 'rgba(34, 197, 94, 0.2)', borderColor: 'rgba(34, 197, 94, 0.4)', color: '#4ade80' } : {}}
            >
              {showErrorOverlay ? 'Errors: ON' : 'Errors: OFF'}
            </button>
            {/* Cognitive load badge — only shown when tracking */}
            {isTracking && (
              <div
                className="cognitive-load-badge"
                style={{ borderColor: loadLevelColor(cognitiveLoad), color: loadLevelColor(cognitiveLoad) }}
                title={`Cognitive load: ${loadLevelDescription(cognitiveLoad)}`}
              >
                <span className="cl-dot" style={{ background: loadLevelColor(cognitiveLoad) }} />
                Load: {cognitiveLoad}
              </div>
            )}
            {/* AGGME Calibrate button */}
            {isTracking && (
              <button
                className="debug-toggle"
                onClick={startCalibration}
                disabled={calibrationState === 'CALIBRATING'}
                style={calibrationState === 'READY'
                  ? { background: 'rgba(74, 222, 128, 0.15)', borderColor: 'rgba(74, 222, 128, 0.4)', color: '#4ade80' }
                  : calibrationState === 'CALIBRATING'
                  ? { background: 'rgba(250, 204, 21, 0.15)', borderColor: 'rgba(250, 204, 21, 0.4)', color: '#facc15' }
                  : {}
                }
              >
                {calibrationState === 'CALIBRATING'
                  ? `Calibrating ${Math.round(calibrationProgress * 100)}%`
                  : calibrationState === 'READY'
                  ? 'Recalibrate'
                  : 'Calibrate'}
              </button>
            )}
            {/* AGGME Intent badge */}
            {isTracking && calibrationState === 'READY' && (
              <div
                className="cognitive-load-badge"
                style={{ borderColor: intentColor(intentState), color: intentColor(intentState) }}
                title={`Intent: ${intentLabel(intentState)}`}
              >
                <span className="cl-dot" style={{ background: intentColor(intentState) }} />
                {intentLabel(intentState)}
              </div>
            )}
            {/* AGGME Spatial zone badge */}
            {isTracking && spatialMapping && (
              <div
                className="cognitive-load-badge"
                style={{ borderColor: syntacticZoneColor(spatialMapping.syntactic_zone), color: syntacticZoneColor(spatialMapping.syntactic_zone) }}
                title={`Zone: ${spatialMapping.syntactic_label}`}
              >
                <span className="cl-dot" style={{ background: syntacticZoneColor(spatialMapping.syntactic_zone) }} />
                {syntacticZoneLabel(spatialMapping.syntactic_zone)}
              </div>
            )}
            {/* Gaze tracking status badge */}
            {isTracking && gazeResult && (
              <div
                className="cognitive-load-badge"
                style={{ borderColor: gazeStateColor(gazeResult.gaze_state), color: gazeStateColor(gazeResult.gaze_state) }}
                title={`Gaze: ${gazeResult.gaze_state} | EAR: ${gazeResult.features?.ear?.toFixed(2) ?? 'N/A'}`}
              >
                <span className="cl-dot" style={{ background: gazeStateColor(gazeResult.gaze_state) }} />
                {gazeResult.gaze_state}
              </div>
            )}
            {/* ISL Feature toggles */}
            {isTracking && (
              <>
                <button
                  className={`debug-toggle ${guidedMode ? 'active' : ''}`}
                  onClick={() => { setGuidedMode(!guidedMode); setFingerspellMode(false); }}
                  style={guidedMode ? { background: 'rgba(96, 165, 250, 0.2)', borderColor: 'rgba(96, 165, 250, 0.4)', color: '#60a5fa' } : {}}
                >
                  {guidedMode ? 'Exit Practice' : 'Guided Practice'}
                </button>
                <button
                  className={`debug-toggle ${fingerspellMode ? 'active' : ''}`}
                  onClick={() => {
                    const next = !fingerspellMode;
                    setFingerspellMode(next);
                    if (next) { fingerspellRef.current.startSpelling(); setGuidedMode(false); }
                    else fingerspellRef.current.stopSpelling();
                  }}
                  style={fingerspellMode ? { background: 'rgba(251, 191, 36, 0.2)', borderColor: 'rgba(251, 191, 36, 0.4)', color: '#fbbf24' } : {}}
                >
                  {fingerspellMode ? 'Exit Spell' : 'Fingerspell'}
                </button>
                {/* AAC Controls */}
                {sentence.length > 0 && (
                  <button
                    className="debug-toggle"
                    onClick={() => {
                      phraseBankRef.current.save(sentence);
                      setSavedPhrases(phraseBankRef.current.getAll());
                    }}
                    style={{ background: 'rgba(59, 130, 246, 0.15)', borderColor: 'rgba(59, 130, 246, 0.4)', color: '#60a5fa' }}
                  >
                    Save Phrase
                  </button>
                )}
                <button
                  className={`debug-toggle ${showPhraseBank ? 'active' : ''}`}
                  onClick={() => {
                    setSavedPhrases(phraseBankRef.current.getAll());
                    setShowPhraseBank(!showPhraseBank);
                  }}
                  style={showPhraseBank ? { background: 'rgba(59, 130, 246, 0.2)', borderColor: 'rgba(59, 130, 246, 0.4)', color: '#60a5fa' } : {}}
                >
                  Phrases ({phraseBankRef.current.count()})
                </button>
                {sentence.length > 0 && (
                  <button
                    className="debug-toggle"
                    onClick={() => {
                      const board = OBFExporter.exportSentenceBoard(sentence);
                      OBFExporter.downloadBoard(board, 'mlaf-sentence.obf');
                    }}
                    style={{ background: 'rgba(16, 185, 129, 0.15)', borderColor: 'rgba(16, 185, 129, 0.4)', color: '#34d399' }}
                  >
                    Export OBF
                  </button>
                )}
                <button
                  className={`debug-toggle ${gazeDwellMode ? 'active' : ''}`}
                  onClick={() => {
                    const next = !gazeDwellMode;
                    setGazeDwellMode(next);
                    if (next) {
                      gazeDwellRef.current.activate();
                      setGazeDwellGrid(gazeDwellRef.current.getCurrentGrid());
                    } else {
                      gazeDwellRef.current.deactivate();
                      setGazeDwellGrid(null);
                    }
                  }}
                  style={gazeDwellMode ? { background: 'rgba(168, 85, 247, 0.2)', borderColor: 'rgba(168, 85, 247, 0.4)', color: '#c084fc' } : {}}
                >
                  {gazeDwellMode ? 'Gaze: ON' : 'Gaze Input'}
                </button>
                {onEndSession && (
                  <button
                    className="debug-toggle"
                    onClick={() => {
                      sessionLoggerRef.current.logSessionEnd();
                      onEndSession({
                        sessionStats: sessionLoggerRef.current.getSessionSummary(),
                        masteryReport: masteryGateRef.current.getMasteryReport(),
                        automaticitySummary: automaticityTrackerRef.current.getSessionSummary(),
                      });
                    }}
                    style={{ background: 'rgba(248, 113, 113, 0.15)', borderColor: 'rgba(248, 113, 113, 0.4)', color: '#f87171' }}
                  >
                    End Session
                  </button>
                )}
              </>
            )}
          </div>
        </div>

        {/* Debug panel */}
        {showDebug && debugInfo && (
          <div className="debug-panel">
            <div className="debug-row">
              <span>Gesture:</span>
              <strong>{currentGesture || 'None'}</strong>
            </div>
            <div className="debug-row">
              <span>Wrist Y:</span>
              <span>{wristY.toFixed(3)}</span>
            </div>
            <div className="debug-row">
              <span>Tense Zone:</span>
              <span>{currentTenseZone}</span>
            </div>
            <div className="debug-row">
              <span>Cognitive Load:</span>
              <strong style={{ color: loadLevelColor(cognitiveLoad) }}>{cognitiveLoad}</strong>
            </div>
            <div className="debug-row">
              <span>Jitter:</span>
              <span>{cognitiveLoadAdapterRef.current.getJitter().toFixed(4)}</span>
            </div>
            <div className="debug-row">
              <span>Avg Thumb Dist:</span>
              <span>{debugInfo.avgThumbToFingers}</span>
            </div>
            <div className="debug-row">
              <span>Lock Progress:</span>
              <span>{(lockProgress * 100).toFixed(0)}%</span>
            </div>
            <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
              <span style={{ color: '#f59e0b', fontWeight: 600 }}>AGGME Pipeline</span>
            </div>
            <div className="debug-row">
              <span>Calibration:</span>
              <strong style={{ color: calibrationStateColor(calibrationState) }}>
                {calibrationStateLabel(calibrationState)}
              </strong>
            </div>
            <div className="debug-row">
              <span>Smoother α:</span>
              <span>
                {smootherRef.current.getAlpha().toFixed(3)}
                {' '}
                <span style={{ color: smoothingIntensityColor(smootherRef.current.getAlpha()) }}>
                  ({smoothingIntensityLabel(smootherRef.current.getAlpha())})
                </span>
              </span>
            </div>
            <div className="debug-row">
              <span>Intent:</span>
              <strong style={{ color: intentColor(intentState) }}>{intentLabel(intentState)}</strong>
            </div>
            <div className="debug-row">
              <span>Displacement:</span>
              <span>{intentDetectorRef.current.getDisplacement().toFixed(3)}</span>
            </div>
            {spatialMapping && (
              <>
                <div className="debug-row">
                  <span>Spatial Zone:</span>
                  <strong style={{ color: syntacticZoneColor(spatialMapping.syntactic_zone) }}>
                    {spatialMapping.syntactic_label}
                  </strong>
                </div>
                <div className="debug-row">
                  <span>Zone Confidence:</span>
                  <span>{(spatialMapping.zone_confidence * 100).toFixed(0)}%</span>
                </div>
                <div className="debug-row">
                  <span>Velocity:</span>
                  <span>{spatialMapping.movement_intensity.toFixed(4)}</span>
                </div>
              </>
            )}
            {/* DFA State & Semantic Type System */}
            <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
              <span style={{ color: '#a78bfa', fontWeight: 600 }}>Formal Systems</span>
            </div>
            <div className="debug-row">
              <span>DFA State:</span>
              <strong style={{ color: dfaState === 'LOCKED' ? '#4ade80' : dfaState === 'CONFIRMING' ? '#facc15' : dfaState === 'COOLDOWN' ? '#f97316' : '#94a3b8' }}>
                {dfaState}
              </strong>
            </div>
            {typeComposition && (
              <>
                <div className="debug-row">
                  <span>Sem. Type:</span>
                  <span style={{ color: typeComposition.complete ? '#4ade80' : '#facc15' }}>
                    {typeComposition.resultTypeString}
                    {typeComposition.complete ? ' (complete)' : ''}
                  </span>
                </div>
                {typeComposition.expectsNext.length > 0 && (
                  <div className="debug-row">
                    <span>Expects:</span>
                    <span style={{ color: '#60a5fa' }}>{typeComposition.expectsNext.join(', ')}</span>
                  </div>
                )}
                {typeComposition.errors.length > 0 && (
                  <div className="debug-row">
                    <span>Type Errors:</span>
                    <span style={{ color: '#f87171' }}>{typeComposition.errors.length}</span>
                  </div>
                )}
              </>
            )}

            {graphRAGContext && (
              <>
                <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
                  <span style={{ color: '#e879f9', fontWeight: 600 }}>Graph RAG (4-Layer)</span>
                </div>
                <div className="debug-row">
                  <span>Valid Next:</span>
                  <span style={{ color: '#4ade80', fontSize: '0.7rem' }}>
                    {graphRAGContext.graph_rag_context.valid_next.tokens.map(t => t.grammar_id).join(', ') || 'none'}
                  </span>
                </div>
                <div className="debug-row">
                  <span>Composition:</span>
                  <span style={{ color: graphRAGContext.graph_rag_context.composition.is_complete ? '#4ade80' : '#facc15' }}>
                    {graphRAGContext.graph_rag_context.composition.current_type}
                    {graphRAGContext.graph_rag_context.composition.is_complete ? ' (complete)' : ''}
                  </span>
                </div>
                {graphRAGContext.graph_rag_context.valid_next.agreement_rule && (
                  <div className="debug-row">
                    <span>Agreement:</span>
                    <span style={{ color: '#60a5fa' }}>{graphRAGContext.graph_rag_context.valid_next.agreement_rule}</span>
                  </div>
                )}
                {graphRAGContext.graph_rag_context.interference?.detected && (
                  <div className="debug-row">
                    <span>L1 Transfer:</span>
                    <span style={{ color: '#f87171' }}>
                      {graphRAGContext.graph_rag_context.interference.patterns.map(p => p.id).join(', ')}
                    </span>
                  </div>
                )}
              </>
            )}

            {umceResult && umceResult.top1 && (
              <>
                <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
                  <span style={{ color: '#38bdf8', fontWeight: 600 }}>
                    UMCE Bayesian Fusion ({
                      umceResult.fusion_mode === 'TRIMODAL' ? 'A+V+G' :
                      umceResult.fusion_mode === 'BIMODAL_VA' ? 'A+V' :
                      umceResult.fusion_mode === 'BIMODAL_VG' ? 'V+G' : 'V only'
                    })
                  </span>
                </div>
                <div className="debug-row">
                  <span>Top-1:</span>
                  <strong style={{ color: probabilityColor(umceResult.top1.probability) }}>
                    {umceResult.top1.gesture_id} ({formatProbability(umceResult.top1.probability)})
                  </strong>
                </div>
                <div className="debug-row">
                  <span>Quality:</span>
                  <strong style={{ color: decisionQualityColor(umceResult.decision_quality) }}>
                    {decisionQualityLabel(umceResult.decision_quality)}
                  </strong>
                </div>
                <div className="debug-row">
                  <span>Margin:</span>
                  <span>{formatProbability(umceResult.margin)}</span>
                </div>
                <div className="debug-row">
                  <span>Entropy:</span>
                  <span>{umceResult.entropy.toFixed(2)} bits</span>
                </div>
                <div className="debug-row">
                  <span>Intent Gate:</span>
                  <span>{(umceResult.intent_gate * 100).toFixed(0)}%</span>
                </div>
                {umceResult.top3.length > 1 && (
                  <div className="debug-row" style={{ gridColumn: '1 / -1' }}>
                    <span>Top-3: </span>
                    <span>
                      {umceResult.top3.map((t, i) => (
                        <span key={t.gesture_id}>
                          {i > 0 && ' · '}
                          <span style={{ color: i === 0 ? probabilityColor(t.probability) : '#888' }}>
                            {t.gesture_id}({formatProbability(t.probability)})
                          </span>
                        </span>
                      ))}
                    </span>
                  </div>
                )}
                {umceResult.raw_gesture_agrees !== null && (
                  <div className="debug-row">
                    <span>Raw Agrees:</span>
                    <span style={{ color: umceResult.raw_gesture_agrees ? '#4ade80' : '#f97316' }}>
                      {umceResult.raw_gesture_agrees ? 'Yes' : 'No'}
                    </span>
                  </div>
                )}
              </>
            )}
            {uasamResult && (
              <>
                <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
                  <span style={{ color: '#f472b6', fontWeight: 600 }}>
                    UASAM Acoustic — P(A|S)
                  </span>
                </div>
                <div className="debug-row">
                  <span>Mic:</span>
                  <span style={{ color: uasamActive ? '#4ade80' : '#f87171' }}>
                    {uasamActive ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="debug-row">
                  <span>Energy:</span>
                  <span style={{ color: energyColor(uasamResult.features.energyDb) }}>
                    {formatEnergyDb(uasamResult.features.energyDb)}
                  </span>
                </div>
                <div className="debug-row">
                  <span>Voc State:</span>
                  <strong style={{ color: vocStateColor(uasamResult.vocalization_state) }}>
                    {vocStateLabel(uasamResult.vocalization_state)}
                  </strong>
                </div>
                <div className="debug-row">
                  <span>Pitch:</span>
                  <span>{uasamResult.features.dominantFrequency.toFixed(0)} Hz</span>
                </div>
                <div className="debug-row">
                  <span>Centroid:</span>
                  <span>{uasamResult.features.spectralCentroid.toFixed(0)} Hz</span>
                </div>
                <div className="debug-row">
                  <span>Flatness:</span>
                  <span>{uasamResult.features.spectralFlatness.toFixed(3)}</span>
                </div>
                <div className="debug-row">
                  <span>ZCR:</span>
                  <span>{uasamResult.features.zeroCrossingRate.toFixed(4)}</span>
                </div>
                <div className="debug-row">
                  <span>HNR:</span>
                  <span>{uasamResult.features.hnr.toFixed(2)}</span>
                </div>
              </>
            )}
            {gazeResult && (
              <>
                <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
                  <span style={{ color: '#a78bfa', fontWeight: 600 }}>
                    EyeGaze — P(G|S)
                  </span>
                </div>
                <div className="debug-row">
                  <span>Gaze:</span>
                  <span style={{ color: gazeTrackerRef.current.isActive() ? '#4ade80' : '#f87171' }}>
                    {gazeTrackerRef.current.isActive() ? 'Active' : 'Inactive'}
                  </span>
                </div>
                <div className="debug-row">
                  <span>State:</span>
                  <strong style={{ color: gazeStateColor(gazeResult.gaze_state) }}>
                    {gazeResult.gaze_state}
                  </strong>
                </div>
                <div className="debug-row">
                  <span>Gaze X:</span>
                  <span>{gazeResult.features.gazeX.toFixed(3)}</span>
                </div>
                <div className="debug-row">
                  <span>Gaze Y:</span>
                  <span>{gazeResult.features.gazeY.toFixed(3)}</span>
                </div>
                <div className="debug-row">
                  <span>EAR:</span>
                  <span style={{ color: (gazeResult.features?.ear || 0) < 0.15 ? '#f87171' : '#4ade80' }}>
                    {gazeResult.features?.ear?.toFixed(3) ?? '—'}
                  </span>
                </div>
                <div className="debug-row">
                  <span>Velocity:</span>
                  <span>{gazeResult.features.velocity.toFixed(4)}</span>
                </div>
                <div className="debug-row">
                  <span>Fixation:</span>
                  <span>{gazeResult.features.fixationDuration} frames</span>
                </div>
                <div className="debug-row">
                  <span>Face:</span>
                  <span style={{ color: gazeResult.face_detected ? '#4ade80' : '#f87171' }}>
                    {gazeResult.face_detected ? 'Detected' : 'None'}
                  </span>
                </div>
              </>
            )}
            {fingerStates && (
              <>
                <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
                  <span style={{ color: '#c084fc', fontWeight: 600 }}>Finger Analysis (3D Angles)</span>
                </div>
                {Object.keys(fingerStates.angles).map((finger) => (
                  <div className="debug-row" key={finger}>
                    <span style={{ textTransform: 'capitalize' }}>{finger}:</span>
                    <span>
                      <strong style={{ color: fingerStates.states[finger] ? '#4ade80' : '#f87171' }}>
                        {fingerStates.states[finger] ? 'OPEN' : 'CURLED'}
                      </strong>
                      {' '}
                      <span style={{ color: '#888' }}>{fingerStates.angles[finger].toFixed(1)}&deg;</span>
                    </span>
                  </div>
                ))}
                <div className="debug-row" style={{ gridColumn: '1 / -1', borderTop: '1px solid rgba(255,255,255,0.1)', paddingTop: '0.4rem', marginTop: '0.2rem' }}>
                  <span>Grammar Token:</span>
                  <strong style={{ color: grammarToken && grammarToken !== 'UNKNOWN' ? '#fbbf24' : '#888' }}>
                    {grammarToken || 'N/A'}
                  </strong>
                </div>
              </>
            )}
          </div>
        )}

        {/* Text-to-Speech for speech-impaired users */}
        <TextToSpeech
          sentence={sentence}
          isComplete={validation.isComplete}
          autoSpeak={accessibilityProfile?.shouldAutoSpeak() || false}
          enabled={accessibilityProfile?.getOutputMode() === 'text-to-speech'}
        />

        {/* Fluency / Automaticity Panel */}
        {showDebug && automaticitySummary && automaticitySummary.session_gestures > 0 && (
          <div className="debug-panel automaticity-panel" style={{ marginTop: '0.5rem' }}>
            <div className="debug-row" style={{ gridColumn: '1 / -1' }}>
              <span style={{ color: '#a78bfa', fontWeight: 600 }}>
                ⚡ Fluency  —  {automaticitySummary.session_gestures} gestures this session
              </span>
            </div>
            {automaticitySummary.sorted_ids.map(id => {
              const g = automaticitySummary.per_gesture[id];
              const score = g.score;
              const barWidth = score !== null ? Math.round(score * 100) : 0;
              const arrow = trendToArrow(g.trend);
              const arrowColor = trendToColor(g.trend);
              return (
                <div key={id} className="automaticity-row">
                  <span className="automaticity-gesture-id">{id.replace('SUBJECT_', '')}</span>
                  <div className="automaticity-bar-wrap">
                    <div
                      className="automaticity-bar-fill"
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                  <span className="automaticity-score-label">
                    {score !== null ? `${Math.round(score * 100)}%` : '--'}
                  </span>
                  <span className="automaticity-label">{scoreToLabel(score)}</span>
                  <span
                    className="automaticity-trend"
                    style={{ color: arrowColor }}
                    title={g.trend}
                  >
                    {arrow}
                  </span>
                  <span className="automaticity-ms" style={{ color: '#475569' }}>
                    {g.session_mean_ms}ms
                  </span>
                </div>
              );
            })}
          </div>
        )}

        {/* Mastery Gate Panel — cumulative curriculum progression (Structured Literacy) */}
        {showDebug && (
          <div className="debug-panel mastery-panel" style={{ marginTop: '0.5rem' }}>
            <div className="debug-row" style={{ gridColumn: '1 / -1' }}>
              <span style={{ color: '#34d399', fontWeight: 600 }}>
                Mastery Gate  —  Stage {masteryReport.currentStage} of {CURRICULUM_SEQUENCE.length}
              </span>
              <span style={{ color: '#64748b', fontSize: '0.65rem' }}>
                {masteryReport.totalMastered} gestures mastered
              </span>
            </div>
            {masteryReport.stages.map(stageData => (
              <div key={stageData.stage} className={`mastery-stage-block${stageData.isComplete ? ' mastery-stage-complete' : stageData.isUnlocked ? ' mastery-stage-active' : ' mastery-stage-locked'}`}>
                <div className="mastery-stage-header">
                  <span className="mastery-stage-icon">
                    {stageData.isComplete ? '✓' : stageData.isUnlocked ? '▶' : '⏸'}
                  </span>
                  <span className="mastery-stage-label">{stageData.label}</span>
                </div>
                {stageData.isUnlocked && (
                  <div className="mastery-gesture-list">
                    {stageData.gestureStatuses.map(g => (
                      <div key={g.id} className="mastery-gesture-row">
                        <span className="mastery-gesture-id">{g.id.replace('SUBJECT_', '')}</span>
                        <div className="mastery-bar-wrap">
                          <div
                            className="mastery-bar-fill"
                            style={{
                              width: `${Math.round(g.progress * 100)}%`,
                              background: masteryProgressColor(g.progress),
                            }}
                          />
                        </div>
                        <span
                          className="mastery-status-label"
                          style={{ color: masteryProgressColor(g.progress) }}
                        >
                          {masteryStatusLabel(g.mastered, g.count, masteryReport.threshold)}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Session Data Logger Panel — diagnostic-prescriptive data */}
        {showDebug && sessionStats && (
          <div className="debug-panel session-log-panel" style={{ marginTop: '0.5rem' }}>
            <div className="debug-row" style={{ gridColumn: '1 / -1', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ color: '#f59e0b', fontWeight: 600 }}>
                Session Log  —  {sessionStats.duration_display}
              </span>
              <button
                className="session-export-btn"
                onClick={() => sessionLoggerRef.current.downloadSessionJSON()}
              >
                Export JSON
              </button>
            </div>
            <div className="session-stats-grid">
              <div className="session-stat">
                <span className="session-stat-value">{sessionStats.total_gestures}</span>
                <span className="session-stat-label">Gestures</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-value">{sessionStats.sentences_completed}</span>
                <span className="session-stat-label">Sentences</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-value" style={{ color: accuracyColor(sessionStats.accuracy_rate) }}>
                  {formatAccuracy(sessionStats.accuracy_rate)}
                </span>
                <span className="session-stat-label">Accuracy</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-value">{sessionStats.gestures_per_minute}</span>
                <span className="session-stat-label">Gest/min</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-value" style={{ color: sessionStats.isl_interferences > 0 ? '#f87171' : '#64748b' }}>
                  {sessionStats.isl_interferences}
                </span>
                <span className="session-stat-label">ISL Xfers</span>
              </div>
              <div className="session-stat">
                <span className="session-stat-value" style={{ color: sessionStats.agreement_errors > 0 ? '#f87171' : '#64748b' }}>
                  {sessionStats.agreement_errors}
                </span>
                <span className="session-stat-label">S-V Errors</span>
              </div>
            </div>
            <div className="session-load-dist">
              <span className="session-load-label">Load:</span>
              <div className="session-load-bar">
                <div className="session-load-seg" style={{ width: `${sessionStats.cognitive_load_distribution.LOW}%`, background: '#4ade80' }} title={`LOW ${sessionStats.cognitive_load_distribution.LOW}%`} />
                <div className="session-load-seg" style={{ width: `${sessionStats.cognitive_load_distribution.MEDIUM}%`, background: '#facc15' }} title={`MED ${sessionStats.cognitive_load_distribution.MEDIUM}%`} />
                <div className="session-load-seg" style={{ width: `${sessionStats.cognitive_load_distribution.HIGH}%`, background: '#f87171' }} title={`HIGH ${sessionStats.cognitive_load_distribution.HIGH}%`} />
              </div>
            </div>
          </div>
        )}

        {/* Guided Practice Panel — lesson-based walkthrough */}
        {guidedMode && (
          <GuidedPracticePanel
            sentence={sentence}
            onClearSentence={clearSentence}
            onExitPractice={() => setGuidedMode(false)}
            masteryReport={masteryReport}
          />
        )}

        {/* Fingerspelling Panel */}
        {fingerspellMode && (
          <FingerspellingPanel
            recognizer={fingerspellRef.current}
            detection={fingerspellDetection}
            onWordMatch={handleFingerspellMatch}
            onExit={() => { setFingerspellMode(false); fingerspellRef.current.stopSpelling(); }}
          />
        )}

        {/* Achievement Panel (in debug area) */}
        {showDebug && (
          <AchievementPanel achievementReport={achievementReport} />
        )}

        {/* SRS Due Gestures (in debug area) */}
        {showDebug && srsReport && srsReport.dueNow > 0 && (
          <div className="debug-panel" style={{ marginTop: '0.5rem' }}>
            <div className="debug-row" style={{ gridColumn: '1 / -1' }}>
              <span style={{ color: '#a78bfa', fontWeight: 600 }}>
                SRS Review — {srsReport.dueNow} due now, {srsReport.dueToday} today
              </span>
            </div>
            {Object.entries(srsReport.perGesture).filter(([, g]) => g.isDue).slice(0, 6).map(([id, g]) => (
              <div key={id} className="debug-row">
                <span>{id.replace('SUBJECT_', '')}</span>
                <span style={{ color: easeFactorColor(g.easeFactor) }}>
                  {intervalToLabel(g.interval)} · EF {g.easeFactor.toFixed(1)}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Achievement Toast — popup notification */}
        {achievementToast && (
          <AchievementToast
            achievement={achievementToast}
            onDismiss={() => setAchievementToast(null)}
            lowStimulus={accessibilityProfile?.isLowStimulus?.()}
          />
        )}

        {/* Phrase Bank Panel */}
        {showPhraseBank && (
          <div style={{
            position: 'fixed', top: 80, right: 20, zIndex: 9000,
            background: 'rgba(15, 15, 26, 0.95)', border: '1px solid rgba(96, 165, 250, 0.3)',
            borderRadius: 12, padding: 16, maxWidth: 360, maxHeight: '60vh', overflowY: 'auto',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
              <span style={{ color: '#60a5fa', fontWeight: 700, fontSize: 14 }}>Saved Phrases</span>
              <button onClick={() => setShowPhraseBank(false)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', fontSize: 16 }}>
                X
              </button>
            </div>
            {savedPhrases.length === 0 ? (
              <div style={{ color: '#64748b', fontSize: 12, textAlign: 'center', padding: 20 }}>
                No saved phrases yet. Build a sentence and tap "Save Phrase".
              </div>
            ) : (
              savedPhrases.map(phrase => (
                <div key={phrase.id} style={{
                  background: 'rgba(30, 41, 59, 0.8)', borderRadius: 8, padding: '10px 12px',
                  marginBottom: 8, border: '1px solid rgba(71, 85, 105, 0.3)',
                }}>
                  <div style={{ color: '#e2e8f0', fontSize: 13, fontWeight: 600, marginBottom: 6 }}>
                    {phrase.sentence}
                  </div>
                  <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    <button onClick={() => phraseBankRef.current.speak(phrase.sentence)}
                      style={{ background: 'rgba(74, 222, 128, 0.15)', border: '1px solid rgba(74, 222, 128, 0.3)',
                        borderRadius: 6, padding: '3px 8px', color: '#4ade80', fontSize: 11, cursor: 'pointer' }}>
                      Speak
                    </button>
                    <button onClick={async () => {
                      const ok = await phraseBankRef.current.copyToClipboard(phrase.sentence);
                      if (ok) { setPhraseCopiedId(phrase.id); setTimeout(() => setPhraseCopiedId(null), 1500); }
                    }}
                      style={{ background: 'rgba(59, 130, 246, 0.15)', border: '1px solid rgba(59, 130, 246, 0.3)',
                        borderRadius: 6, padding: '3px 8px', color: '#60a5fa', fontSize: 11, cursor: 'pointer' }}>
                      {phraseCopiedId === phrase.id ? 'Copied!' : 'Copy'}
                    </button>
                    {phraseBankRef.current.canShare() && (
                      <button onClick={() => phraseBankRef.current.share(phrase.sentence)}
                        style={{ background: 'rgba(168, 85, 247, 0.15)', border: '1px solid rgba(168, 85, 247, 0.3)',
                          borderRadius: 6, padding: '3px 8px', color: '#c084fc', fontSize: 11, cursor: 'pointer' }}>
                        Share
                      </button>
                    )}
                    <button onClick={() => {
                      const board = OBFExporter.exportSentenceBoard(phrase.words.map(w => ({
                        grammar_id: w.grammar_id, word: w.word, display: w.word, category: w.category
                      })));
                      OBFExporter.downloadBoard(board, `mlaf-${phrase.sentence.replace(/\s+/g, '-')}.obf`);
                    }}
                      style={{ background: 'rgba(16, 185, 129, 0.15)', border: '1px solid rgba(16, 185, 129, 0.3)',
                        borderRadius: 6, padding: '3px 8px', color: '#34d399', fontSize: 11, cursor: 'pointer' }}>
                      OBF
                    </button>
                    <button onClick={() => {
                      phraseBankRef.current.delete(phrase.id);
                      setSavedPhrases(phraseBankRef.current.getAll());
                    }}
                      style={{ background: 'rgba(248, 113, 113, 0.1)', border: '1px solid rgba(248, 113, 113, 0.3)',
                        borderRadius: 6, padding: '3px 8px', color: '#f87171', fontSize: 11, cursor: 'pointer', marginLeft: 'auto' }}>
                      Del
                    </button>
                  </div>
                </div>
              ))
            )}
            {savedPhrases.length > 0 && (
              <button onClick={() => {
                const collection = OBFExporter.exportPhraseCollection(savedPhrases);
                OBFExporter.downloadBoard(collection, 'mlaf-phrases.obf');
              }}
                style={{ width: '100%', marginTop: 8, background: 'rgba(16, 185, 129, 0.15)',
                  border: '1px solid rgba(16, 185, 129, 0.3)', borderRadius: 8, padding: '8px 12px',
                  color: '#34d399', fontSize: 12, fontWeight: 600, cursor: 'pointer' }}>
                Export All as OBF Board Set
              </button>
            )}
          </div>
        )}

        {/* Gaze Dwell Selection Grid */}
        {gazeDwellMode && gazeDwellGrid && (
          <div style={{
            position: 'fixed', bottom: 20, left: '50%', transform: 'translateX(-50%)',
            zIndex: 8000, display: 'grid', gridTemplateRows: `repeat(3, 1fr)`,
            gridTemplateColumns: `repeat(3, 1fr)`, gap: 4, padding: 8,
            background: 'rgba(15, 15, 26, 0.9)', borderRadius: 12,
            border: '2px solid rgba(168, 85, 247, 0.4)',
          }}>
            {gazeDwellGrid.flat().map((cell, idx) => {
              const row = Math.floor(idx / 3);
              const col = idx % 3;
              const cellKey = `${row}_${col}`;
              const progress = gazeDwellProgress[cellKey] || 0;
              const bgColor = cell ? (
                cell.type === 'SUBJECT' ? 'rgba(37, 99, 235, 0.3)' :
                cell.type === 'VERB' ? 'rgba(220, 38, 38, 0.3)' :
                'rgba(22, 163, 74, 0.3)'
              ) : 'rgba(30, 41, 59, 0.3)';
              const borderColor = progress > 0.5
                ? `rgba(168, 85, 247, ${0.3 + progress * 0.7})`
                : 'rgba(71, 85, 105, 0.3)';

              return (
                <div key={cellKey} style={{
                  width: 90, height: 60, display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center',
                  background: bgColor, borderRadius: 8,
                  border: `2px solid ${borderColor}`,
                  position: 'relative', overflow: 'hidden',
                }}>
                  {cell && (
                    <>
                      <span style={{ color: '#e2e8f0', fontSize: 14, fontWeight: 700 }}>
                        {cell.display}
                      </span>
                      <span style={{ color: '#94a3b8', fontSize: 9, textTransform: 'uppercase' }}>
                        {cell.type}
                      </span>
                    </>
                  )}
                  {progress > 0 && (
                    <div style={{
                      position: 'absolute', bottom: 0, left: 0, height: 3,
                      width: `${progress * 100}%`,
                      background: '#c084fc', borderRadius: 2,
                    }} />
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* MLAF Prompt Token Debug (shows structured prompt data) */}
        {showDebug && promptData && (
          <div className="debug-panel" style={{ marginTop: '0.5rem' }}>
            <div className="debug-row" style={{ gridColumn: '1 / -1' }}>
              <span style={{ color: '#60a5fa', fontWeight: 600 }}>MLAF Prompt Token</span>
            </div>
            <div className="debug-row">
              <span>Grammar ID:</span>
              <strong style={{ color: '#fbbf24' }}>{promptData.gesture_id}</strong>
            </div>
            <div className="debug-row">
              <span>Category:</span>
              <span>{promptData.category}</span>
            </div>
            <div className="debug-row">
              <span>Tense:</span>
              <span>{promptData.tense || 'N/A'}</span>
            </div>
            {errorVectorData && (
              <>
                <div className="debug-row">
                  <span>Error Score:</span>
                  <strong style={{ color: errorVectorData.is_within_tolerance ? '#4ade80' : '#f87171' }}>
                    {errorVectorData.aggregate_error.toFixed(2)}
                  </strong>
                </div>
                <div className="debug-row">
                  <span>Constraints:</span>
                  <span>{errorVectorData.constraints_met}/{errorVectorData.constraints_total}</span>
                </div>
              </>
            )}
          </div>
        )}
      </div>

      {/* Right side - Tense Indicator (only for verbs) */}
      <TenseIndicator
        wristY={wristY}
        currentTenseZone={currentTenseZone}
        isVisible={isVerbGesture}
      />
    </div>
  );
}

export default SandboxMode;
