/**
 * SandboxMode.jsx
 * Main component for the gesture-based sentence building sandbox
 * Integrates camera, gesture detection, sentence builder, and UI components
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import { detectGestureRaw, getHandDebugInfo } from '../utils/gestureDetection';
import { validateSentence, LEXICON } from '../utils/GrammarEngine';
import { analyzeFingerStatesDetailed } from '../utils/fingerAnalysis';
import { classifyFingerState } from '../utils/grammarClassifier';
import { useSentenceBuilder } from '../hooks/useSentenceBuilder';
import SentenceStrip from './SentenceStrip';
import GestureSidebar from './GestureSidebar';
import TenseIndicator from './TenseIndicator';
import ParseTreeVisualizer from './ParseTreeVisualizer';

// Hand connections for drawing skeleton
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

function SandboxMode() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const streamRef = useRef(null);

  // UI State
  const [modelReady, setModelReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTracking, setIsTracking] = useState(false);
  const [showDebug, setShowDebug] = useState(false);
  const [showParseTree, setShowParseTree] = useState(true);
  const [debugInfo, setDebugInfo] = useState(null);
  const [wristY, setWristY] = useState(0.5);
  const [fingerStates, setFingerStates] = useState(null);
  const [grammarToken, setGrammarToken] = useState(null);

  // Sentence builder hook
  const {
    sentence,
    isLocked,
    lockProgress,
    currentGesture,
    currentTenseZone,
    processGestureInput,
    clearSentence,
    undoLastWord,
  } = useSentenceBuilder();

  // Grammar validation
  const validation = validateSentence(sentence.map(w => w.grammar_id));

  // Check if current gesture is a verb (for tense indicator)
  const isVerbGesture = currentGesture && LEXICON[currentGesture]?.type === 'VERB';

  // Process landmarks
  const processLandmarks = useCallback((landmarks) => {
    if (!landmarks || landmarks.length === 0) {
      processGestureInput(null, 0.5);
      setDebugInfo(null);
      setFingerStates(null);
      setGrammarToken(null);
      return;
    }

    const handLandmarks = landmarks[0];
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

    // Detect gesture and process
    const gesture = detectGestureRaw(handLandmarks);
    processGestureInput(gesture, wrist.y);
  }, [processGestureInput, showDebug]);

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
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const handLandmarker = handLandmarkerRef.current;

    if (!video || !canvas || !handLandmarker || video.readyState < 2) {
      animationFrameRef.current = requestAnimationFrame(detectHands);
      return;
    }

    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;

      const results = handLandmarker.detectForVideo(video, performance.now());
      const ctx = canvas.getContext('2d');

      if (results.landmarks && results.landmarks.length > 0) {
        processLandmarks(results.landmarks);
        drawLandmarks(results.landmarks, ctx, canvas.width, canvas.height);
      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        processLandmarks(null);
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
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      if (handLandmarkerRef.current) {
        handLandmarkerRef.current.close();
      }
      if (videoRef.current?.srcObject) {
        videoRef.current.srcObject.getTracks().forEach(track => track.stop());
      }
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

  // Get gesture display name
  const getGestureDisplay = (gesture) => {
    if (!gesture) return null;
    const entry = LEXICON[gesture];
    return entry?.display || gesture;
  };

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
          onClear={clearSentence}
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
        </div>

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

            {/* Gesture overlay */}
            {currentGesture && (
              <div className="gesture-overlay">
                <div className="gesture-name">{getGestureDisplay(currentGesture)}</div>
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

            {/* Lock indicator */}
            {isLocked && (
              <div className="lock-overlay">
                <span>🔒 Cooldown</span>
              </div>
            )}
          </div>

          {/* Debug toggle */}
          <button
            className="debug-toggle"
            onClick={() => setShowDebug(!showDebug)}
          >
            {showDebug ? 'Hide Debug' : 'Debug'}
          </button>
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
              <span>Avg Thumb Dist:</span>
              <span>{debugInfo.avgThumbToFingers}</span>
            </div>
            <div className="debug-row">
              <span>Lock Progress:</span>
              <span>{(lockProgress * 100).toFixed(0)}%</span>
            </div>
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
