/**
 * GestureRecorder.jsx — Capture & Save Gesture Landmark Data
 * Part of the MLAF (Multimodal Language Acquisition Framework) system.
 *
 * Records MediaPipe hand landmarks for each of the 19 gestures, saves them
 * to the backend as training data in the format expected by preprocess.py:
 *   gesture_id, lm_0_x, lm_0_y, lm_0_z, ..., lm_20_z
 *
 * Flow:
 *   1. User selects a gesture from the vocabulary
 *   2. 3-second countdown starts
 *   3. System records N frames of landmarks (default 30)
 *   4. Landmarks are saved to data/custom/webcam_landmarks.csv
 *   5. User can record more samples or switch to another gesture
 */

import { useRef, useEffect, useState, useCallback } from 'react';
import { HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// All 19 MLAF gestures
const GESTURE_VOCABULARY = [
  { id: 'subject_i', label: 'I', category: 'SUBJECT' },
  { id: 'subject_you', label: 'You', category: 'SUBJECT' },
  { id: 'subject_he', label: 'He', category: 'SUBJECT' },
  { id: 'subject_she', label: 'She', category: 'SUBJECT' },
  { id: 'subject_we', label: 'We', category: 'SUBJECT' },
  { id: 'subject_they', label: 'They', category: 'SUBJECT' },
  { id: 'verb_want', label: 'Want', category: 'VERB' },
  { id: 'verb_eat', label: 'Eat', category: 'VERB' },
  { id: 'verb_see', label: 'See', category: 'VERB' },
  { id: 'verb_grab', label: 'Grab', category: 'VERB' },
  { id: 'verb_drink', label: 'Drink', category: 'VERB' },
  { id: 'verb_go', label: 'Go', category: 'VERB' },
  { id: 'verb_stop', label: 'Stop', category: 'VERB' },
  { id: 'object_food', label: 'Food', category: 'OBJECT' },
  { id: 'object_water', label: 'Water', category: 'OBJECT' },
  { id: 'object_book', label: 'Book', category: 'OBJECT' },
  { id: 'object_apple', label: 'Apple', category: 'OBJECT' },
  { id: 'object_ball', label: 'Ball', category: 'OBJECT' },
  { id: 'object_house', label: 'House', category: 'OBJECT' },
];

const FRAMES_PER_RECORDING = 30;
const COUNTDOWN_SECONDS = 3;
const GRAMMAR_ENGINE_URL = import.meta.env.VITE_GRAMMAR_ENGINE_URL || '/grammar';

// Hand skeleton connections for visualization
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],[5,9],[9,13],[13,17]
];

const categoryColor = (cat) => {
  switch (cat) {
    case 'SUBJECT': return '#60a5fa';
    case 'VERB': return '#4ade80';
    case 'OBJECT': return '#f59e0b';
    default: return '#94a3b8';
  }
};

export default function GestureRecorder({ onBack }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const handLandmarkerRef = useRef(null);
  const animationFrameRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const streamRef = useRef(null);
  const recordingBufferRef = useRef([]);

  const [modelReady, setModelReady] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedGesture, setSelectedGesture] = useState(null);
  const [recordingState, setRecordingState] = useState('idle'); // idle | countdown | recording | saving | done
  const [countdown, setCountdown] = useState(0);
  const [framesRecorded, setFramesRecorded] = useState(0);
  const [handDetected, setHandDetected] = useState(false);
  const [savedCounts, setSavedCounts] = useState({});
  const [totalSaved, setTotalSaved] = useState(0);
  const [errorMsg, setErrorMsg] = useState(null);
  const [lastLandmarks, setLastLandmarks] = useState(null);

  // Initialize MediaPipe + Camera
  useEffect(() => {
    let alive = true;

    async function init() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm'
        );

        const landmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: '/hand_landmarker.task',
            delegate: 'GPU',
          },
          numHands: 1,
          runningMode: 'VIDEO',
          minHandDetectionConfidence: 0.5,
          minHandPresenceConfidence: 0.5,
          minTrackingConfidence: 0.5,
        });

        if (!alive) return;
        handLandmarkerRef.current = landmarker;

        // Start camera
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: 'user' },
        });

        if (!alive) { stream.getTracks().forEach(t => t.stop()); return; }
        streamRef.current = stream;

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          await videoRef.current.play();
        }

        setModelReady(true);
        setIsLoading(false);
      } catch (err) {
        console.error('[GestureRecorder] Init failed:', err);
        setErrorMsg(`Failed to initialize: ${err.message}`);
        setIsLoading(false);
      }
    }

    init();

    return () => {
      alive = false;
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (streamRef.current) streamRef.current.getTracks().forEach(t => t.stop());
      if (saveAbortRef.current) saveAbortRef.current.abort();
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    };
  }, []);

  // Fetch existing saved counts on mount
  useEffect(() => {
    fetch(`${GRAMMAR_ENGINE_URL}/gesture-recordings/stats`)
      .then(r => r.json())
      .then(data => {
        if (data.per_gesture) setSavedCounts(data.per_gesture);
        if (data.total) setTotalSaved(data.total);
      })
      .catch(() => {}); // Server may not have the endpoint yet
  }, []);

  // Detection loop
  useEffect(() => {
    if (!modelReady) return;

    function detect() {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const landmarker = handLandmarkerRef.current;

      if (!video || !canvas || !landmarker || video.readyState < 2) {
        animationFrameRef.current = requestAnimationFrame(detect);
        return;
      }

      if (video.currentTime !== lastVideoTimeRef.current) {
        lastVideoTimeRef.current = video.currentTime;

        const results = landmarker.detectForVideo(video, performance.now());
        const ctx = canvas.getContext('2d');
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (results.landmarks && results.landmarks.length > 0) {
          const hand = results.landmarks[0];
          setHandDetected(true);
          setLastLandmarks(hand);

          // Draw hand skeleton
          drawHand(ctx, hand, canvas.width, canvas.height);

          // If recording, capture frame
          if (recordingState === 'recording' && selectedGesture) {
            recordingBufferRef.current.push(
              hand.map(lm => ({ x: lm.x, y: lm.y, z: lm.z }))
            );
            setFramesRecorded(recordingBufferRef.current.length);

            // Check if done
            if (recordingBufferRef.current.length >= FRAMES_PER_RECORDING) {
              saveRecording();
            }
          }
        } else {
          setHandDetected(false);
          setLastLandmarks(null);
        }
      }

      animationFrameRef.current = requestAnimationFrame(detect);
    }

    detect();
    return () => { if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current); };
  }, [modelReady, recordingState, selectedGesture]);

  // Draw hand landmarks
  function drawHand(ctx, landmarks, w, h) {
    const isRec = recordingState === 'recording';
    const color = isRec ? '#ef4444' : '#4ade80';

    // Connections
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    HAND_CONNECTIONS.forEach(([a, b]) => {
      ctx.beginPath();
      ctx.moveTo(landmarks[a].x * w, landmarks[a].y * h);
      ctx.lineTo(landmarks[b].x * w, landmarks[b].y * h);
      ctx.stroke();
    });

    // Points
    landmarks.forEach((lm, i) => {
      ctx.fillStyle = i === 0 ? '#f59e0b' : color;
      ctx.beginPath();
      ctx.arc(lm.x * w, lm.y * h, i === 0 ? 6 : 4, 0, Math.PI * 2);
      ctx.fill();
    });

    // Recording indicator
    if (isRec) {
      ctx.fillStyle = 'rgba(239, 68, 68, 0.15)';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#ef4444';
      ctx.font = 'bold 18px monospace';
      ctx.fillText(`REC ${recordingBufferRef.current.length}/${FRAMES_PER_RECORDING}`, 12, 28);
    }
  }

  // Start recording flow
  const countdownTimerRef = useRef(null);

  const startRecording = useCallback(() => {
    if (!selectedGesture || !handDetected) return;

    setRecordingState('countdown');
    setFramesRecorded(0);
    recordingBufferRef.current = [];
    setErrorMsg(null);

    let count = COUNTDOWN_SECONDS;
    setCountdown(count);

    // Clear any prior countdown timer
    if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);

    countdownTimerRef.current = setInterval(() => {
      count--;
      setCountdown(count);
      if (count <= 0) {
        clearInterval(countdownTimerRef.current);
        countdownTimerRef.current = null;
        setRecordingState('recording');
      }
    }, 1000);
  }, [selectedGesture, handDetected]);

  // AbortController ref for cancelling in-flight saves on unmount
  const saveAbortRef = useRef(null);

  // Save recorded frames to backend
  const saveRecording = useCallback(async () => {
    setRecordingState('saving');
    const frames = recordingBufferRef.current;
    const gestureId = selectedGesture.id;

    // Cancel any previous in-flight save
    if (saveAbortRef.current) saveAbortRef.current.abort();
    const controller = new AbortController();
    saveAbortRef.current = controller;

    try {
      const response = await fetch(`${GRAMMAR_ENGINE_URL}/gesture-recordings/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        signal: controller.signal,
        body: JSON.stringify({
          gesture_id: gestureId,
          frames: frames.map(hand =>
            hand.map(lm => [lm.x, lm.y, lm.z]).flat()
          ),
        }),
      });

      if (!response.ok) throw new Error(`Server returned ${response.status}`);

      const result = await response.json();

      // Update counts
      setSavedCounts(prev => ({
        ...prev,
        [gestureId]: (prev[gestureId] || 0) + frames.length,
      }));
      setTotalSaved(prev => prev + frames.length);
      setRecordingState('done');
    } catch (err) {
      if (err.name === 'AbortError') return; // Component unmounted, skip state updates
      console.error('[GestureRecorder] Save failed:', err);
      // Fallback: save to localStorage
      const existing = JSON.parse(localStorage.getItem('mlaf_gesture_recordings') || '[]');
      frames.forEach(hand => {
        existing.push({
          gesture_id: gestureId,
          landmarks: hand.map(lm => [lm.x, lm.y, lm.z]).flat(),
          timestamp: Date.now(),
        });
      });
      localStorage.setItem('mlaf_gesture_recordings', JSON.stringify(existing));

      setSavedCounts(prev => ({
        ...prev,
        [gestureId]: (prev[gestureId] || 0) + frames.length,
      }));
      setTotalSaved(prev => prev + frames.length);
      setErrorMsg(`Saved to browser (server unavailable). ${frames.length} frames stored locally.`);
      setRecordingState('done');
    }

    recordingBufferRef.current = [];
  }, [selectedGesture]);

  // Export localStorage recordings as CSV download
  const exportLocalData = useCallback(() => {
    const data = JSON.parse(localStorage.getItem('mlaf_gesture_recordings') || '[]');
    if (data.length === 0) { setErrorMsg('No local recordings to export'); return; }

    // Build CSV header
    const lmCols = [];
    for (let i = 0; i < 21; i++) {
      lmCols.push(`lm_${i}_x`, `lm_${i}_y`, `lm_${i}_z`);
    }
    const header = ['gesture_id', ...lmCols].join(',');

    const rows = data.map(r => {
      const vals = [r.gesture_id, ...r.landmarks.map(v => v.toFixed(6))];
      return vals.join(',');
    });

    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'webcam_landmarks.csv';
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  // Keyboard shortcut: Space to start/restart recording
  useEffect(() => {
    function onKey(e) {
      if (e.code === 'Space' && selectedGesture && recordingState !== 'recording' && recordingState !== 'countdown') {
        e.preventDefault();
        startRecording();
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selectedGesture, recordingState, startRecording]);

  return (
    <div style={{
      display: 'flex', height: '100vh', background: '#0f172a', color: '#e2e8f0',
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
    }}>
      {/* Left: Gesture selector */}
      <div style={{
        width: '280px', borderRight: '1px solid rgba(255,255,255,0.1)',
        display: 'flex', flexDirection: 'column', overflow: 'hidden',
      }}>
        <div style={{
          padding: '1rem', borderBottom: '1px solid rgba(255,255,255,0.1)',
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
        }}>
          <span style={{ fontWeight: 700, fontSize: '1rem' }}>Gesture Recorder</span>
          {onBack && (
            <button onClick={onBack} style={{
              background: 'rgba(255,255,255,0.1)', border: 'none', color: '#94a3b8',
              borderRadius: '6px', padding: '4px 10px', cursor: 'pointer', fontSize: '0.75rem',
            }}>Back</button>
          )}
        </div>

        <div style={{
          padding: '0.75rem', borderBottom: '1px solid rgba(255,255,255,0.1)',
          background: 'rgba(255,255,255,0.03)',
        }}>
          <div style={{ fontSize: '0.7rem', color: '#94a3b8', marginBottom: '4px' }}>
            Total recorded: <strong style={{ color: '#4ade80' }}>{totalSaved}</strong> frames
          </div>
          <div style={{ display: 'flex', gap: '6px' }}>
            <button onClick={exportLocalData} style={{
              flex: 1, background: 'rgba(96,165,250,0.15)', border: '1px solid rgba(96,165,250,0.3)',
              color: '#60a5fa', borderRadius: '6px', padding: '6px', cursor: 'pointer', fontSize: '0.7rem',
            }}>Export CSV</button>
          </div>
        </div>

        <div style={{ flex: 1, overflowY: 'auto', padding: '0.5rem' }}>
          {['SUBJECT', 'VERB', 'OBJECT'].map(cat => (
            <div key={cat} style={{ marginBottom: '0.75rem' }}>
              <div style={{
                fontSize: '0.65rem', fontWeight: 700, color: categoryColor(cat),
                textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '4px',
                padding: '0 4px',
              }}>{cat}S</div>
              {GESTURE_VOCABULARY.filter(g => g.category === cat).map(g => {
                const isSelected = selectedGesture?.id === g.id;
                const count = savedCounts[g.id] || 0;
                return (
                  <button
                    key={g.id}
                    onClick={() => { setSelectedGesture(g); setRecordingState('idle'); setErrorMsg(null); }}
                    style={{
                      display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                      width: '100%', padding: '8px 10px', marginBottom: '2px',
                      background: isSelected ? 'rgba(255,255,255,0.12)' : 'transparent',
                      border: isSelected ? `1px solid ${categoryColor(cat)}` : '1px solid transparent',
                      borderRadius: '8px', color: '#e2e8f0', cursor: 'pointer',
                      fontSize: '0.85rem', textAlign: 'left', transition: 'all 0.15s',
                    }}
                  >
                    <span>{g.label}</span>
                    {count > 0 && (
                      <span style={{
                        fontSize: '0.65rem', background: 'rgba(74,222,128,0.2)',
                        color: '#4ade80', padding: '1px 6px', borderRadius: '10px',
                      }}>{count}</span>
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>
      </div>

      {/* Center: Camera feed */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', position: 'relative' }}>

        {isLoading && (
          <div style={{ textAlign: 'center' }}>
            <div style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>Loading MediaPipe...</div>
            <div style={{ color: '#94a3b8', fontSize: '0.85rem' }}>Initializing hand detection model</div>
          </div>
        )}

        {errorMsg && (
          <div style={{
            position: 'absolute', top: '1rem', left: '50%', transform: 'translateX(-50%)',
            background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)',
            color: '#fca5a5', padding: '8px 16px', borderRadius: '8px', fontSize: '0.8rem',
            zIndex: 10, maxWidth: '500px',
          }}>{errorMsg}</div>
        )}

        <div style={{ position: 'relative', borderRadius: '12px', overflow: 'hidden', boxShadow: '0 0 40px rgba(0,0,0,0.5)' }}>
          <video
            ref={videoRef}
            style={{ width: '640px', height: '480px', objectFit: 'cover', transform: 'scaleX(-1)', display: modelReady ? 'block' : 'none' }}
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            width={640}
            height={480}
            style={{ position: 'absolute', top: 0, left: 0, width: '640px', height: '480px', transform: 'scaleX(-1)' }}
          />

          {/* Countdown overlay */}
          {recordingState === 'countdown' && (
            <div style={{
              position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
              background: 'rgba(0,0,0,0.6)', zIndex: 5,
            }}>
              <div style={{ fontSize: '6rem', fontWeight: 900, color: '#f59e0b' }}>{countdown}</div>
            </div>
          )}

          {/* Recording progress bar */}
          {recordingState === 'recording' && (
            <div style={{
              position: 'absolute', bottom: 0, left: 0, right: 0, height: '6px',
              background: 'rgba(0,0,0,0.5)', zIndex: 5,
            }}>
              <div style={{
                height: '100%', background: '#ef4444',
                width: `${(framesRecorded / FRAMES_PER_RECORDING) * 100}%`,
                transition: 'width 0.05s',
              }} />
            </div>
          )}
        </div>

        {/* Controls below camera */}
        <div style={{ marginTop: '1.5rem', textAlign: 'center' }}>
          {!selectedGesture ? (
            <div style={{ color: '#94a3b8', fontSize: '1rem' }}>Select a gesture from the left panel to start recording</div>
          ) : (
            <>
              <div style={{ marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8' }}>Recording for: </span>
                <span style={{
                  fontSize: '1.2rem', fontWeight: 700,
                  color: categoryColor(selectedGesture.category),
                }}>{selectedGesture.label}</span>
                <span style={{ fontSize: '0.8rem', color: '#64748b', marginLeft: '8px' }}>
                  ({selectedGesture.id})
                </span>
              </div>

              {recordingState === 'idle' && (
                <button
                  onClick={startRecording}
                  disabled={!handDetected}
                  style={{
                    padding: '12px 32px', fontSize: '1rem', fontWeight: 700,
                    background: handDetected ? 'linear-gradient(135deg, #ef4444, #dc2626)' : '#334155',
                    color: handDetected ? '#fff' : '#64748b',
                    border: 'none', borderRadius: '12px', cursor: handDetected ? 'pointer' : 'not-allowed',
                    boxShadow: handDetected ? '0 0 20px rgba(239,68,68,0.3)' : 'none',
                  }}
                >
                  {handDetected ? 'Start Recording (Space)' : 'Show Your Hand First'}
                </button>
              )}

              {recordingState === 'recording' && (
                <div style={{ color: '#ef4444', fontSize: '1.1rem', fontWeight: 700 }}>
                  Recording... {framesRecorded}/{FRAMES_PER_RECORDING} frames
                </div>
              )}

              {recordingState === 'saving' && (
                <div style={{ color: '#f59e0b', fontSize: '1rem' }}>Saving...</div>
              )}

              {recordingState === 'done' && (
                <div>
                  <div style={{ color: '#4ade80', fontSize: '1rem', fontWeight: 600, marginBottom: '1rem' }}>
                    Saved {FRAMES_PER_RECORDING} frames for "{selectedGesture.label}"
                  </div>
                  <button
                    onClick={startRecording}
                    disabled={!handDetected}
                    style={{
                      padding: '10px 24px', fontSize: '0.9rem', fontWeight: 600,
                      background: 'linear-gradient(135deg, #ef4444, #dc2626)',
                      color: '#fff', border: 'none', borderRadius: '10px', cursor: 'pointer',
                      marginRight: '8px',
                    }}
                  >Record Again (Space)</button>
                </div>
              )}

              {/* Hand status */}
              <div style={{
                marginTop: '1rem', fontSize: '0.75rem',
                color: handDetected ? '#4ade80' : '#f87171',
              }}>
                {handDetected ? 'Hand detected' : 'No hand detected — position your hand in view'}
              </div>
            </>
          )}
        </div>
      </div>

      {/* Right: Stats */}
      <div style={{
        width: '220px', borderLeft: '1px solid rgba(255,255,255,0.1)',
        padding: '1rem', overflowY: 'auto',
      }}>
        <div style={{ fontSize: '0.8rem', fontWeight: 700, marginBottom: '1rem', color: '#94a3b8' }}>
          Recording Stats
        </div>

        {GESTURE_VOCABULARY.map(g => {
          const count = savedCounts[g.id] || 0;
          const barWidth = Math.min(count / 60, 1) * 100; // 60 = 2 recordings
          return (
            <div key={g.id} style={{ marginBottom: '6px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7rem', marginBottom: '2px' }}>
                <span style={{ color: categoryColor(g.category) }}>{g.label}</span>
                <span style={{ color: count > 0 ? '#e2e8f0' : '#475569' }}>{count}</span>
              </div>
              <div style={{ height: '3px', background: 'rgba(255,255,255,0.05)', borderRadius: '2px' }}>
                <div style={{
                  height: '100%', borderRadius: '2px',
                  background: count >= 60 ? '#4ade80' : count >= 30 ? '#f59e0b' : '#ef4444',
                  width: `${barWidth}%`, transition: 'width 0.3s',
                }} />
              </div>
            </div>
          );
        })}

        <div style={{
          marginTop: '1.5rem', padding: '0.75rem', background: 'rgba(255,255,255,0.03)',
          borderRadius: '8px', fontSize: '0.7rem', color: '#94a3b8',
        }}>
          <div>Target: 30+ frames per gesture</div>
          <div>Min for training: 540 total</div>
          <div style={{ marginTop: '6px', color: totalSaved >= 540 ? '#4ade80' : '#f59e0b', fontWeight: 600 }}>
            {totalSaved >= 540
              ? 'Ready for training!'
              : `${540 - totalSaved} more frames needed`
            }
          </div>
        </div>
      </div>
    </div>
  );
}
