import { useRef, useEffect, useState, useCallback, useMemo } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';
import {
  detectGestureRaw,
  confidenceLock,
  getHandDebugInfo
} from '../utils/gestureDetection';
import { validateSentence, LEXICON } from '../utils/GrammarEngine';

// Hand landmark indices for highlighting
const THUMB_TIP = 4;
const INDEX_FINGER_TIP = 8;

// Hand connections for drawing skeleton
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
  [0, 5], [5, 6], [6, 7], [7, 8],       // Index
  [0, 9], [9, 10], [10, 11], [11, 12],  // Middle
  [0, 13], [13, 14], [14, 15], [15, 16], // Ring
  [0, 17], [17, 18], [18, 19], [19, 20], // Pinky
  [5, 9], [9, 13], [13, 17]              // Palm
];

// Generate a confirmation beep using Web Audio API
function playConfirmationSound() {
  try {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 880;
    oscillator.type = 'sine';

    gainNode.gain.setValueAtTime(0.3, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.2);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.2);
  } catch (e) {
    console.warn('Could not play confirmation sound:', e);
  }
}

// Play error sound
function playErrorSound() {
  try {
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    const oscillator = audioContext.createOscillator();
    const gainNode = audioContext.createGain();

    oscillator.connect(gainNode);
    gainNode.connect(audioContext.destination);

    oscillator.frequency.value = 220;
    oscillator.type = 'sawtooth';

    gainNode.gain.setValueAtTime(0.2, audioContext.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);

    oscillator.start(audioContext.currentTime);
    oscillator.stop(audioContext.currentTime + 0.3);
  } catch (e) {
    console.warn('Could not play error sound:', e);
  }
}

function HandTracker() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [isTracking, setIsTracking] = useState(false);

  // Gesture state
  const [displayGesture, setDisplayGesture] = useState(null);
  const [lockProgress, setLockProgress] = useState(0);
  const [isFlashing, setIsFlashing] = useState(false);
  const [flashType, setFlashType] = useState('success');

  // Debug state
  const [debugInfo, setDebugInfo] = useState(null);
  const [showDebug, setShowDebug] = useState(false);

  // Sentence state
  const [sentence, setSentence] = useState([]);

  // Real-time grammar validation
  const validation = useMemo(() => {
    return validateSentence(sentence);
  }, [sentence]);

  // Handle gesture lock event (called when confidence lock triggers)
  const handleGestureLocked = useCallback((gesture) => {
    // Check if adding this word would create a valid sentence
    const testSentence = [...sentence, gesture];
    const testValidation = validateSentence(testSentence);

    if (testValidation.isValid || sentence.length === 0) {
      // Add word to sentence
      setSentence(testSentence);

      // Play confirmation sound
      playConfirmationSound();

      // Flash the screen green
      setFlashType('success');
      setIsFlashing(true);
      setTimeout(() => setIsFlashing(false), 200);

      // Clear last locked to allow new gestures
      confidenceLock.clearLastLocked();
    } else {
      // Word would create invalid sentence - show error
      playErrorSound();

      // Flash the screen red
      setFlashType('error');
      setIsFlashing(true);
      setTimeout(() => setIsFlashing(false), 300);
    }
  }, [sentence]);

  // Process landmarks - runs every frame
  const processLandmarks = useCallback((landmarks) => {
    if (!landmarks || landmarks.length === 0) {
      confidenceLock.update(null);
      setDisplayGesture(null);
      setLockProgress(0);
      setDebugInfo(null);
      return;
    }

    // Get the first hand's landmarks
    const handLandmarks = landmarks[0];

    // Update debug info
    if (showDebug) {
      setDebugInfo(getHandDebugInfo(handLandmarks));
    }

    // Detect gesture with confidence lock
    const rawGesture = detectGestureRaw(handLandmarks);
    const result = confidenceLock.update(rawGesture);

    // Update display
    setDisplayGesture(result.display);
    setLockProgress(result.progress);

    // Handle lock event
    if (result.isLocked && result.gesture) {
      handleGestureLocked(result.gesture);
    }
  }, [handleGestureLocked, showDebug]);

  // Draw hand landmarks on canvas
  const drawLandmarks = useCallback((landmarks, ctx, width, height) => {
    ctx.clearRect(0, 0, width, height);

    landmarks.forEach((handLandmarks) => {
      // Draw connections (skeleton lines)
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

      // Draw landmark points
      handLandmarks.forEach((landmark, index) => {
        const x = landmark.x * width;
        const y = landmark.y * height;

        // Highlight thumb tip and index tip
        if (index === THUMB_TIP || index === INDEX_FINGER_TIP) {
          ctx.fillStyle = '#FF0000';
          ctx.beginPath();
          ctx.arc(x, y, 8, 0, 2 * Math.PI);
          ctx.fill();
        } else {
          ctx.fillStyle = '#00FF00';
          ctx.beginPath();
          ctx.arc(x, y, 5, 0, 2 * Math.PI);
          ctx.fill();
        }
      });

      // Draw direction indicator for thumb
      const wrist = handLandmarks[0];
      const thumbTip = handLandmarks[THUMB_TIP];

      ctx.strokeStyle = '#FFFF00';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(wrist.x * width, wrist.y * height);
      ctx.lineTo(thumbTip.x * width, thumbTip.y * height);
      ctx.stroke();
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

  // Initialize MediaPipe HandLandmarker
  useEffect(() => {
    let mounted = true;

    async function initializeHandLandmarker() {
      try {
        setIsLoading(true);
        setError(null);

        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        const handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            delegate: 'GPU'
          },
          runningMode: 'VIDEO',
          numHands: 2
        });

        if (!mounted) {
          handLandmarker.close();
          return;
        }

        handLandmarkerRef.current = handLandmarker;

        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' }
        });

        if (!mounted) {
          stream.getTracks().forEach(track => track.stop());
          return;
        }

        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          await video.play();

          const canvas = canvasRef.current;
          if (canvas) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
          }

          setIsLoading(false);
          setIsTracking(true);
        }
      } catch (err) {
        if (mounted) {
          console.error('Error initializing hand tracker:', err);
          setError(err.message || 'Failed to initialize hand tracking');
          setIsLoading(false);
        }
      }
    }

    initializeHandLandmarker();

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

  // Clear sentence
  const clearSentence = () => {
    setSentence([]);
    confidenceLock.reset();
  };

  // Get display text for a word
  const getWordDisplay = (word) => {
    return LEXICON[word]?.display || word;
  };

  // Get word type for styling
  const getWordType = (word) => {
    return LEXICON[word]?.type || 'UNKNOWN';
  };

  // Get gesture display name
  const getGestureDisplayName = (gesture) => {
    if (!gesture) return null;

    const displayNames = {
      // Subjects
      'SUBJECT_I': 'I (Thumb Self)',
      'SUBJECT_YOU': 'YOU (Point)',
      'SUBJECT_HE': 'HE (Thumb Side)',
      // Verbs
      'GRAB': 'GRAB (Claw)',
      'STOP': 'STOP (Open Palm)',
      'DRINK': 'DRINK (C-Shape)',
      // Objects
      'APPLE': 'APPLE (Cupped)',
      'BOOK': 'BOOK (Flat Palm Up)',
      'HOUSE': 'HOUSE (Roof)',
      'WATER': 'WATER (W-Shape)',
    };

    return displayNames[gesture] || gesture;
  };

  return (
    <div className="hand-tracker">
      {/* Grammar Validation Panel */}
      <div className={`grammar-panel ${validation.isComplete ? 'complete' : validation.error ? 'error' : 'pending'}`}>
        <div className="grammar-status">
          <span className={`status-indicator ${validation.isComplete ? 'complete' : validation.error ? 'error' : 'pending'}`}>
            {validation.isComplete ? '✓' : validation.error ? '✗' : '○'}
          </span>
          <span className="status-text">
            {validation.isComplete ? 'Complete' : validation.error ? 'Invalid' : 'Building...'}
          </span>
          <button
            className="debug-toggle"
            onClick={() => setShowDebug(!showDebug)}
          >
            {showDebug ? 'Hide Debug' : 'Debug'}
          </button>
        </div>

        {validation.error && (
          <div className="grammar-error">
            {validation.error}
          </div>
        )}

        <div className="grammar-suggestion">
          {validation.suggestion}
        </div>

        {validation.readableSentence && (
          <div className="readable-sentence">
            "{validation.readableSentence}"
          </div>
        )}
      </div>

      {/* Debug Panel */}
      {showDebug && debugInfo && (
        <div className="debug-panel">
          <div className="debug-title">Hand Debug Info</div>
          <div className="debug-grid">
            <span>Thumb Dir:</span>
            <span className={debugInfo.thumbDirection.includes('I') ? 'debug-highlight-i' : 'debug-highlight-he'}>
              {debugInfo.thumbDirection}
            </span>
            <span>Index→Cam:</span>
            <span>{debugInfo.indexPointingForward ? '✓ YOU' : '✗'}</span>
            <span>Avg Thumb Dist:</span>
            <span className={debugInfo.isGrab ? 'debug-highlight-grab' : ''}>
              {debugInfo.avgThumbToFingers} {debugInfo.isGrab ? '(GRAB)' : ''}
            </span>
            <span>Thumb-Index:</span>
            <span className={debugInfo.isDrink ? 'debug-highlight-drink' : ''}>
              {debugInfo.thumbIndexDist} {debugInfo.isDrink ? '(DRINK?)' : ''}
            </span>
            <span>Index-Middle:</span>
            <span className={debugInfo.isHouse ? 'debug-highlight-house' : ''}>
              {debugInfo.indexMiddleDist} {debugInfo.isHouse ? '(HOUSE)' : ''}
            </span>
            <span>Hand Vertical:</span>
            <span>{debugInfo.handVertical ? '✓ STOP?' : '✗'}</span>
            <span>Palm Up:</span>
            <span>{debugInfo.palmUp ? '✓ BOOK?' : '✗'}</span>
            <span>All Extended:</span>
            <span>{debugInfo.allFingersExtended ? '✓' : '✗'}</span>
          </div>
        </div>
      )}

      {/* Sentence display */}
      <div className="sentence-container">
        <div className="sentence-blocks">
          {sentence.length === 0 ? (
            <span className="sentence-placeholder">Hold a gesture for ~1 second (30 frames) to lock it in...</span>
          ) : (
            sentence.map((word, index) => (
              <span key={index} className={`word-block ${getWordType(word).toLowerCase()}`}>
                <span className="word-type">{getWordType(word)}</span>
                <span className="word-text">{getWordDisplay(word)}</span>
              </span>
            ))
          )}
        </div>
        {sentence.length > 0 && (
          <button className="clear-button" onClick={clearSentence}>
            Clear
          </button>
        )}
      </div>

      <div className={`video-container ${isFlashing ? `flash-${flashType}` : ''}`}>
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

        {/* Gesture display overlay */}
        {displayGesture && (
          <div className="gesture-overlay">
            <span className="gesture-text">{getGestureDisplayName(displayGesture)}</span>
            {/* Progress ring */}
            <svg className="progress-ring" viewBox="0 0 100 100">
              <circle
                className="progress-ring-bg"
                cx="50"
                cy="50"
                r="45"
              />
              <circle
                className="progress-ring-fill"
                cx="50"
                cy="50"
                r="45"
                style={{
                  strokeDasharray: `${lockProgress * 283} 283`
                }}
              />
            </svg>
            <span className="progress-text">{Math.round(lockProgress * 100)}%</span>
          </div>
        )}
      </div>

      {isLoading && (
        <div className="status-message">
          Loading hand tracking model...
        </div>
      )}

      {error && (
        <div className="error-message">
          Error: {error}
        </div>
      )}

      {isTracking && !displayGesture && (
        <div className="tracking-status">
          <strong>Gesture Guide:</strong><br />
          <em>Subjects:</em> Thumb→self=<strong>I</strong> | Point→camera=<strong>YOU</strong> | Thumb→side=<strong>HE</strong><br />
          <em>Verbs:</em> Claw/Pinch=<strong>GRAB</strong> | C-shape=<strong>DRINK</strong> | Open Palm=<strong>STOP</strong><br />
          <em>Objects:</em> Cupped=<strong>APPLE</strong> | Flat palm up=<strong>BOOK</strong> | Roof=<strong>HOUSE</strong>
        </div>
      )}
    </div>
  );
}

export default HandTracker;
