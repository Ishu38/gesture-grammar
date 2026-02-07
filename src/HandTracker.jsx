import React, { useEffect, useRef, useState } from "react";
import { HandLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { detectGesture } from "./utils/gestureDetection";

export default function HandTracker() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  
  // STATUS LOGGING
  const [status, setStatus] = useState("Step 1: Initializing...");
  const [error, setError] = useState(null);
  
  const [sentence, setSentence] = useState([]); 
  const [currentGesture, setCurrentGesture] = useState("Scan...");
  const [lockTimer, setLockTimer] = useState(0); 
  const LOCK_THRESHOLD = 50; 

  useEffect(() => {
    let handLandmarker = null;
    let animationFrameId = null;
    let lastGesture = "";
    let holdCount = 0;

    const setupVision = async () => {
      try {
        setStatus("Step 2: Loading WASM (Locally)...");
        
        // --- THE FIX IS HERE: We now point to the local folder ---
        const vision = await FilesetResolver.forVisionTasks(
          "/wasm" 
        );
        // ---------------------------------------------------------

        setStatus("Step 3: Loading AI Model file...");
        handLandmarker = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 1
        });

        setStatus("Step 4: Requesting Camera...");
        startWebcam(handLandmarker);
      } catch (err) {
        console.error(err);
        setError("CRITICAL FAILURE: " + err.message);
        setStatus("Failed.");
      }
    };

    const startWebcam = (landmarker) => {
      const constraints = { video: { width: 640, height: 480 } };
      navigator.mediaDevices.getUserMedia(constraints).then((stream) => {
        setStatus("Step 5: Camera Active. Processing...");
        videoRef.current.srcObject = stream;
        videoRef.current.addEventListener("loadeddata", () => {
          predictWebcam(landmarker);
        });
      }).catch(err => {
         setError("CAMERA BLOCKED: " + err.message);
      });
    };

    const predictWebcam = (landmarker) => {
      if (!videoRef.current) return;
      
      const startTimeMs = performance.now();
      const results = landmarker.detectForVideo(videoRef.current, startTimeMs);
      const canvasCtx = canvasRef.current.getContext("2d");
      
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);

      if (results.landmarks && results.landmarks.length > 0) {
        setStatus("Running: Hand Detected");
        const landmarks = results.landmarks[0];
        
        for (const point of landmarks) {
            canvasCtx.beginPath();
            canvasCtx.arc(point.x * 640, point.y * 480, 5, 0, 2 * Math.PI);
            canvasCtx.fillStyle = "#00FF00";
            canvasCtx.fill();
        }

        const detected = detectGesture(landmarks);
        
        if (detected !== "Scan..." && detected === lastGesture) {
            holdCount++;
            setLockTimer((holdCount / LOCK_THRESHOLD) * 100);

            if (holdCount === LOCK_THRESHOLD) {
                addWordToSentence(detected);
                holdCount = 0; 
                setLockTimer(0);
            }
        } else {
            holdCount = 0;
            setLockTimer(0);
            lastGesture = detected;
        }

        setCurrentGesture(detected);
      } else {
        if (status !== "Running: Waiting for Hand...") setStatus("Running: Waiting for Hand...");
        holdCount = 0;
        setLockTimer(0);
        setCurrentGesture("No Hand");
      }

      animationFrameId = requestAnimationFrame(() => predictWebcam(landmarker));
    };

    setupVision();

    return () => {
      if (animationFrameId) cancelAnimationFrame(animationFrameId);
    };
  }, []);

  const addWordToSentence = (word) => {
    const cleanWord = word.split(": ")[1] || word;
    setSentence((prev) => {
        if (prev.length > 0 && prev[prev.length - 1] === cleanWord) return prev;
        return [...prev, cleanWord];
    });
  };

  return (
    <div className="flex flex-col items-center min-h-screen bg-slate-900 text-white p-5">
      
      {error && (
        <div className="bg-red-600 p-4 rounded mb-4 w-full max-w-2xl font-bold border-4 border-red-800 animate-pulse">
            {error}
        </div>
      )}
      
      {!error && (
        <div className="bg-gray-800 p-2 rounded mb-4 text-yellow-300 font-mono">
            STATUS: {status}
        </div>
      )}

      <div className="w-full max-w-2xl min-h-[100px] mb-6 p-4 bg-slate-800 border-2 border-slate-600 rounded-xl flex items-center gap-3 overflow-x-auto shadow-inner">
        {sentence.length === 0 && <span className="text-gray-500 italic">Gestures will appear here...</span>}
        {sentence.map((word, index) => (
            <div key={index} className="px-6 py-3 bg-blue-600 rounded-lg shadow-lg text-xl font-bold border-b-4 border-blue-800 animate-bounce-short">
                {word}
            </div>
        ))}
      </div>

      <div className="relative border-4 border-blue-500 rounded-lg overflow-hidden shadow-2xl">
        <video ref={videoRef} autoPlay playsInline className="absolute w-[640px] h-[480px] opacity-60" style={{ transform: "scaleX(-1)" }}></video>
        <canvas ref={canvasRef} width="640" height="480" className="w-[640px] h-[480px]" style={{ transform: "scaleX(-1)" }}></canvas>
        
        {lockTimer > 0 && (
            <div className="absolute top-0 left-0 h-2 bg-green-500 transition-all duration-75 ease-linear" style={{ width: `${lockTimer}%` }}></div>
        )}

        <div className="absolute bottom-0 left-0 w-full bg-black/80 p-4 text-center">
            <p className="text-3xl font-mono font-bold text-yellow-300">{currentGesture}</p>
            <p className="text-xs text-gray-400 mt-1">Hold gesture to lock it in</p>
        </div>
      </div>
      
      <button 
        onClick={() => setSentence([])}
        className="mt-6 px-8 py-2 bg-red-600 hover:bg-red-700 text-white rounded font-bold transition-colors"
      >
        Clear Sentence
      </button>

    </div>
  );
}
