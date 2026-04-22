"""MLAF Training Pipeline — Webcam Data Collection.

Interactive Python script using OpenCV + MediaPipe to record hand/face/audio
data for all 18 gestures. Captures YOUR specific hand geometry, tremor profile,
and gaze patterns.

Controls:
    SPACE  — Start/stop recording current gesture (captures 30 frames)
    N      — Skip to next gesture
    R      — Re-record current gesture
    Q/ESC  — Quit

Output:
    data/custom/webcam_landmarks.csv   — Hand landmarks (21pt × 3)
    data/custom/webcam_gaze.csv        — Face/gaze landmarks (478pt × 3)
    data/custom/webcam_audio.csv       — Acoustic features per gesture

Usage:
    python -m training.collect_webcam
    python training/collect_webcam.py
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    import sounddevice as sd
    HAS_AUDIO = True
except (ImportError, OSError):
    HAS_AUDIO = False

from .config import (
    CUSTOM_DIR,
    FACE_LANDMARK_DIMS,
    GESTURE_IDS,
    GESTURE_LABEL_MAP,
    NUM_FACE_LANDMARKS,
    NUM_HAND_LANDMARKS,
    WEBCAM_FRAMES_PER_GESTURE,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# MediaPipe solutions
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# ---------------------------------------------------------------------------
# Audio capture helpers
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SEC = 2.0


def _capture_audio_features() -> dict[str, float]:
    """Capture a short audio clip and extract basic acoustic features.

    Returns dict with: rms, zcr (zero crossing rate), spectral_centroid,
    is_voiced (binary), dominant_freq.
    """
    if not HAS_AUDIO:
        return {
            "audio_rms": 0.0,
            "audio_zcr": 0.0,
            "audio_spectral_centroid": 0.0,
            "audio_is_voiced": 0.0,
            "audio_dominant_freq": 0.0,
        }

    try:
        n_samples = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION_SEC)
        audio = sd.rec(n_samples, samplerate=AUDIO_SAMPLE_RATE, channels=1, dtype="float32")
        sd.wait()
        audio = audio.flatten()

        rms = float(np.sqrt(np.mean(audio ** 2)))
        zcr = float(np.mean(np.abs(np.diff(np.sign(audio)))) / 2)

        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / AUDIO_SAMPLE_RATE)
        spectral_centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-8))
        dominant_freq = float(freqs[np.argmax(fft)])
        is_voiced = 1.0 if rms > 0.01 else 0.0

        return {
            "audio_rms": rms,
            "audio_zcr": zcr,
            "audio_spectral_centroid": spectral_centroid,
            "audio_is_voiced": is_voiced,
            "audio_dominant_freq": dominant_freq,
        }
    except Exception as exc:
        logger.warning("Audio capture failed: %s", exc)
        return {
            "audio_rms": 0.0,
            "audio_zcr": 0.0,
            "audio_spectral_centroid": 0.0,
            "audio_is_voiced": 0.0,
            "audio_dominant_freq": 0.0,
        }


# ---------------------------------------------------------------------------
# Gaze estimation from face landmarks
# ---------------------------------------------------------------------------

# Key iris/eye landmark indices in MediaPipe face mesh
_LEFT_IRIS = [468, 469, 470, 471]
_RIGHT_IRIS = [473, 474, 475, 476]
_LEFT_EYE_INNER = 133
_LEFT_EYE_OUTER = 33
_RIGHT_EYE_INNER = 362
_RIGHT_EYE_OUTER = 263


def _estimate_gaze_state(face_landmarks) -> str:
    """Estimate gaze direction from face mesh landmarks.

    Returns one of: 'direct', 'averted', 'tracking_hand', 'tracking_face'.
    Simple heuristic based on iris position relative to eye corners.
    """
    if face_landmarks is None:
        return "averted"

    try:
        lms = face_landmarks.landmark

        # Left eye iris center
        left_iris_x = np.mean([lms[i].x for i in _LEFT_IRIS])
        left_eye_center = (lms[_LEFT_EYE_INNER].x + lms[_LEFT_EYE_OUTER].x) / 2

        # Right eye iris center
        right_iris_x = np.mean([lms[i].x for i in _RIGHT_IRIS])
        right_eye_center = (lms[_RIGHT_EYE_INNER].x + lms[_RIGHT_EYE_OUTER].x) / 2

        # Average horizontal deviation
        left_dev = abs(left_iris_x - left_eye_center)
        right_dev = abs(right_iris_x - right_eye_center)
        avg_dev = (left_dev + right_dev) / 2

        # Vertical — check if looking down (at hands)
        left_iris_y = np.mean([lms[i].y for i in _LEFT_IRIS])
        left_eye_top = lms[159].y  # upper eyelid
        left_eye_bot = lms[145].y  # lower eyelid
        eye_height = abs(left_eye_bot - left_eye_top)
        vert_pos = (left_iris_y - left_eye_top) / (eye_height + 1e-8)

        if avg_dev < 0.015:
            if vert_pos > 0.7:
                return "tracking_hand"
            return "direct"
        elif avg_dev < 0.04:
            return "tracking_face"
        else:
            return "averted"

    except (IndexError, AttributeError):
        return "averted"


# ---------------------------------------------------------------------------
# Collection session
# ---------------------------------------------------------------------------

class WebcamCollector:
    """Interactive webcam data collection session."""

    def __init__(self) -> None:
        self.hand_rows: list[dict] = []
        self.gaze_rows: list[dict] = []
        self.audio_rows: list[dict] = []
        self.gesture_index = 0
        self.recording = False
        self.frame_count = 0
        self.frames_target = WEBCAM_FRAMES_PER_GESTURE

    def run(self) -> None:
        """Open webcam and run interactive collection loop."""
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            logger.error("Cannot open webcam (device 0)")
            sys.exit(1)

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        logger.info("Webcam collection started. Press SPACE to record, Q to quit.")
        self._print_instructions()

        try:
            while cap.isOpened() and self.gesture_index < len(GESTURE_IDS):
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                hand_results = hands.process(rgb)
                face_results = face_mesh.process(rgb)

                # Draw landmarks
                self._draw_overlay(frame, hand_results, face_results)

                # Record if active
                if self.recording and hand_results.multi_hand_landmarks:
                    self._record_frame(hand_results, face_results)

                cv2.imshow("MLAF Webcam Collection", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:  # Q or ESC
                    break
                elif key == ord(" "):  # SPACE
                    self._toggle_recording()
                elif key == ord("n"):  # Next
                    self._next_gesture()
                elif key == ord("r"):  # Re-record
                    self._rerecord_current()

        finally:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()
            face_mesh.close()

        self._save_data()

    def _draw_overlay(self, frame, hand_results, face_results) -> None:
        """Draw landmarks and status info on frame."""
        h, w, _ = frame.shape

        # Draw hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_lms in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lms, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

        # Draw face mesh (tesselation)
        if face_results.multi_face_landmarks:
            for face_lms in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    frame, face_lms,
                    mp.solutions.face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style(),
                )

        # Status bar
        gesture_id = GESTURE_IDS[self.gesture_index]
        label = GESTURE_LABEL_MAP[gesture_id]
        progress = f"{self.gesture_index + 1}/{len(GESTURE_IDS)}"

        # Background bar
        cv2.rectangle(frame, (0, 0), (w, 80), (0, 0, 0), -1)

        # Gesture info
        color = (0, 255, 0) if self.recording else (255, 255, 255)
        cv2.putText(
            frame, f"Gesture: {label} ({gesture_id})",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2,
        )
        cv2.putText(
            frame, f"Progress: {progress}  |  Frames: {self.frame_count}/{self.frames_target}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
        )

        if self.recording:
            cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)  # red dot
            cv2.putText(frame, "REC", (w - 80, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Instructions
        cv2.putText(
            frame, "SPACE=Record  N=Next  R=Redo  Q=Quit",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1,
        )

    def _toggle_recording(self) -> None:
        """Start or stop recording."""
        if self.recording:
            self.recording = False
            logger.info("  Stopped recording. Captured %d frames.", self.frame_count)
            if self.frame_count >= self.frames_target:
                # Also capture audio
                logger.info("  Capturing audio sample …")
                audio_feats = _capture_audio_features()
                gesture_id = GESTURE_IDS[self.gesture_index]
                audio_feats["gesture_id"] = gesture_id
                self.audio_rows.append(audio_feats)
                self._next_gesture()
        else:
            self.recording = True
            self.frame_count = 0
            gesture_id = GESTURE_IDS[self.gesture_index]
            logger.info("  Recording %s — show gesture and press SPACE when done", gesture_id)

    def _record_frame(self, hand_results, face_results) -> None:
        """Record one frame of landmarks."""
        if self.frame_count >= self.frames_target:
            self.recording = False
            logger.info("  Reached %d frames, auto-stopping.", self.frames_target)
            # Capture audio
            audio_feats = _capture_audio_features()
            gesture_id = GESTURE_IDS[self.gesture_index]
            audio_feats["gesture_id"] = gesture_id
            self.audio_rows.append(audio_feats)
            self._next_gesture()
            return

        gesture_id = GESTURE_IDS[self.gesture_index]

        # Hand landmarks
        for hand_lms in hand_results.multi_hand_landmarks:
            row: dict[str, object] = {"gesture_id": gesture_id, "frame": self.frame_count}
            for i, lm in enumerate(hand_lms.landmark):
                row[f"lm_{i}_x"] = lm.x
                row[f"lm_{i}_y"] = lm.y
                row[f"lm_{i}_z"] = lm.z
            self.hand_rows.append(row)
            break  # Use first detected hand only

        # Face/gaze landmarks
        if face_results.multi_face_landmarks:
            face_lms = face_results.multi_face_landmarks[0]
            gaze_state = _estimate_gaze_state(face_lms)

            gaze_row: dict[str, object] = {
                "gesture_id": gesture_id,
                "frame": self.frame_count,
                "gaze_state": gaze_state,
            }
            # Store a subset of key face landmarks (eye + iris regions)
            key_indices = _LEFT_IRIS + _RIGHT_IRIS + [
                _LEFT_EYE_INNER, _LEFT_EYE_OUTER,
                _RIGHT_EYE_INNER, _RIGHT_EYE_OUTER,
                1, 4, 5, 6, 10, 152,  # nose tip, forehead, chin
            ]
            for idx in key_indices:
                lm = face_lms.landmark[idx]
                gaze_row[f"face_{idx}_x"] = lm.x
                gaze_row[f"face_{idx}_y"] = lm.y
                gaze_row[f"face_{idx}_z"] = lm.z

            self.gaze_rows.append(gaze_row)

        self.frame_count += 1

    def _next_gesture(self) -> None:
        """Advance to next gesture."""
        self.gesture_index += 1
        self.frame_count = 0
        self.recording = False
        if self.gesture_index < len(GESTURE_IDS):
            gid = GESTURE_IDS[self.gesture_index]
            label = GESTURE_LABEL_MAP[gid]
            logger.info("Next gesture: %s (%s)  [%d/%d]", label, gid,
                        self.gesture_index + 1, len(GESTURE_IDS))
        else:
            logger.info("All gestures recorded!")

    def _rerecord_current(self) -> None:
        """Discard current gesture data and re-record."""
        gesture_id = GESTURE_IDS[self.gesture_index]
        before_hand = len(self.hand_rows)
        self.hand_rows = [r for r in self.hand_rows if r["gesture_id"] != gesture_id]
        self.gaze_rows = [r for r in self.gaze_rows if r["gesture_id"] != gesture_id]
        self.audio_rows = [r for r in self.audio_rows if r["gesture_id"] != gesture_id]
        removed = before_hand - len(self.hand_rows)
        self.frame_count = 0
        self.recording = False
        logger.info("  Discarded %d frames for %s. Ready to re-record.", removed, gesture_id)

    def _save_data(self) -> None:
        """Save collected data to CSV files."""
        logger.info("Saving collected data …")

        # Hand landmarks
        if self.hand_rows:
            path = CUSTOM_DIR / "webcam_landmarks.csv"
            fieldnames = list(self.hand_rows[0].keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.hand_rows)
            logger.info("  Hand landmarks: %s (%d rows)", path, len(self.hand_rows))

        # Gaze data
        if self.gaze_rows:
            path = CUSTOM_DIR / "webcam_gaze.csv"
            fieldnames = list(self.gaze_rows[0].keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.gaze_rows)
            logger.info("  Gaze data: %s (%d rows)", path, len(self.gaze_rows))

        # Audio features
        if self.audio_rows:
            path = CUSTOM_DIR / "webcam_audio.csv"
            fieldnames = list(self.audio_rows[0].keys())
            with open(path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.audio_rows)
            logger.info("  Audio features: %s (%d rows)", path, len(self.audio_rows))

        total = len(self.hand_rows)
        gestures_recorded = len(set(r["gesture_id"] for r in self.hand_rows))
        logger.info(
            "Collection complete: %d frames across %d gestures",
            total, gestures_recorded,
        )

    def _print_instructions(self) -> None:
        print("\n" + "=" * 60)
        print("  MLAF Webcam Data Collection")
        print("=" * 60)
        print(f"  Gestures to record: {len(GESTURE_IDS)}")
        print(f"  Frames per gesture: {self.frames_target}")
        print()
        print("  Controls:")
        print("    SPACE  — Start/stop recording")
        print("    N      — Skip to next gesture")
        print("    R      — Re-record current gesture")
        print("    Q/ESC  — Quit and save")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    collector = WebcamCollector()
    collector.run()


if __name__ == "__main__":
    main()
