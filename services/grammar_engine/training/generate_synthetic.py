"""MLAF Training Pipeline — Synthetic Landmark Data Generator.

Generates realistic MediaPipe hand landmark data for all 18 MLAF gestures
based on known hand pose configurations. Each gesture has a canonical
21-point hand pose, augmented with rotation, scaling, translation, joint
noise, and finger-angle perturbation to produce diverse training samples.

Design principle: maximize inter-class separation by ensuring each gesture's
canonical pose uses distinct finger configurations. Gestures that were
historically confused (YOU/HE, DRINK/WANT) get maximally different canonical
poses with minimal overlap regions.

Usage:
    python -m training.generate_synthetic
    python training/generate_synthetic.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CUSTOM_DIR,
    GESTURE_IDS,
    GESTURE_LABEL_MAP,
    ID_TO_IDX,
    NUM_HAND_LANDMARKS,
    PROCESSED_DIR,
    RANDOM_SEED,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# MediaPipe hand model topology (21 landmarks)
# ---------------------------------------------------------------------------
#  0: Wrist
#  1-4:   Thumb   (CMC, MCP, IP, TIP)
#  5-8:   Index   (MCP, PIP, DIP, TIP)
#  9-12:  Middle  (MCP, PIP, DIP, TIP)
# 13-16:  Ring    (MCP, PIP, DIP, TIP)
# 17-20:  Pinky   (MCP, PIP, DIP, TIP)
#
# Bone chain: 0 → 1 → 2 → 3 → 4 (thumb)
#             0 → 5 → 6 → 7 → 8 (index)  etc.
# ---------------------------------------------------------------------------

# Base palm geometry (canonical relaxed hand, wrist at origin, fingers up)
_BASE_PALM = np.array([
    # Wrist
    [0.0,   0.0,   0.0],
    # Thumb: angled out to the side
    [0.12, 0.05,  -0.02],   # 1: CMC
    [0.20, 0.15,  -0.03],   # 2: MCP
    [0.25, 0.25,  -0.02],   # 3: IP
    [0.28, 0.35,  -0.01],   # 4: TIP
    # Index
    [0.08, 0.30,   0.0],    # 5: MCP
    [0.08, 0.45,   0.0],    # 6: PIP
    [0.08, 0.55,   0.0],    # 7: DIP
    [0.08, 0.65,   0.0],    # 8: TIP
    # Middle
    [0.0,  0.32,   0.0],    # 9: MCP
    [0.0,  0.48,   0.0],    # 10: PIP
    [0.0,  0.58,   0.0],    # 11: DIP
    [0.0,  0.68,   0.0],    # 12: TIP
    # Ring
    [-0.07, 0.30,  0.0],    # 13: MCP
    [-0.07, 0.44,  0.0],    # 14: PIP
    [-0.07, 0.53,  0.0],    # 15: DIP
    [-0.07, 0.62,  0.0],    # 16: TIP
    # Pinky
    [-0.14, 0.27,  0.0],    # 17: MCP
    [-0.14, 0.38,  0.0],    # 18: PIP
    [-0.14, 0.46,  0.0],    # 19: DIP
    [-0.14, 0.54,  0.0],    # 20: TIP
], dtype=np.float64)


def _curl_finger(landmarks: np.ndarray, finger_start: int, curl_amount: float) -> np.ndarray:
    """Curl a finger by folding PIP/DIP/TIP towards MCP.

    curl_amount: 0.0 = fully extended, 1.0 = fully curled into fist.
    finger_start: MCP index (5 for index, 9 for middle, etc.)
    """
    lm = landmarks.copy()
    mcp = lm[finger_start].copy()
    wrist = lm[0].copy()

    # Direction from MCP towards wrist (curl target)
    curl_dir = wrist - mcp
    curl_dir_norm = curl_dir / (np.linalg.norm(curl_dir) + 1e-8)

    for i, joint_idx in enumerate([finger_start + 1, finger_start + 2, finger_start + 3]):
        # Progressive curl: PIP curls most, DIP next, TIP follows
        factor = curl_amount * (0.3 + 0.25 * i)
        original = lm[joint_idx].copy()
        offset = curl_dir_norm * factor * np.linalg.norm(original - mcp)
        lm[joint_idx] = original + offset
        # Also pull inward (z-axis) to simulate finger wrapping
        lm[joint_idx][2] -= curl_amount * 0.03 * (i + 1)

    return lm


def _curl_thumb(landmarks: np.ndarray, curl_amount: float) -> np.ndarray:
    """Curl thumb toward palm center."""
    lm = landmarks.copy()
    palm_center = np.mean(lm[[5, 9, 13, 17]], axis=0)

    for i, joint_idx in enumerate([2, 3, 4]):
        factor = curl_amount * (0.3 + 0.2 * i)
        direction = palm_center - lm[joint_idx]
        direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
        lm[joint_idx] += direction_norm * factor * 0.15

    return lm


def _point_finger(landmarks: np.ndarray, finger_start: int, direction: np.ndarray) -> np.ndarray:
    """Extend a finger in a specific direction from its MCP."""
    lm = landmarks.copy()
    mcp = lm[finger_start].copy()
    d = direction / (np.linalg.norm(direction) + 1e-8)

    bone_lengths = [0.15, 0.10, 0.10]
    for i, joint_idx in enumerate([finger_start + 1, finger_start + 2, finger_start + 3]):
        lm[joint_idx] = mcp + d * sum(bone_lengths[:i + 1])

    return lm


def _rotate_hand(landmarks: np.ndarray, angle_deg: float, axis: str = "z") -> np.ndarray:
    """Rotate all landmarks around an axis through wrist."""
    lm = landmarks.copy()
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)

    if axis == "z":
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    elif axis == "y":
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "x":
        R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    else:
        R = np.eye(3)

    wrist = lm[0].copy()
    for i in range(NUM_HAND_LANDMARKS):
        lm[i] = wrist + R @ (lm[i] - wrist)
    return lm


# ---------------------------------------------------------------------------
# Canonical pose definitions for all 18 gestures
# ---------------------------------------------------------------------------
# Each returns a 21×3 landmark array. Designed for MAXIMUM inter-class
# separation: gestures that were historically confused get opposite poses.
# ---------------------------------------------------------------------------

def _pose_subject_i() -> np.ndarray:
    """I/ME: Index finger pointing at self (towards camera/chest).
    Index extended pointing down-toward-self, all others curled."""
    lm = _BASE_PALM.copy()
    # Curl all fingers except index
    lm = _curl_thumb(lm, 0.9)
    lm = _curl_finger(lm, 9, 0.95)   # middle curled
    lm = _curl_finger(lm, 13, 0.95)  # ring curled
    lm = _curl_finger(lm, 17, 0.95)  # pinky curled
    # Index points downward (toward chest)
    lm = _point_finger(lm, 5, np.array([0.0, -0.3, -0.8]))
    return lm


def _pose_subject_you() -> np.ndarray:
    """YOU: Index finger pointing forward (away from body).
    Index extended forward, all others curled. DISTINCT from HE by z-axis."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 0.9)
    lm = _curl_finger(lm, 9, 0.95)
    lm = _curl_finger(lm, 13, 0.95)
    lm = _curl_finger(lm, 17, 0.95)
    # Index points forward (positive z = away from camera)
    lm = _point_finger(lm, 5, np.array([0.0, 0.2, 0.9]))
    return lm


def _pose_subject_he() -> np.ndarray:
    """HE: Index pointing to the side (lateral).
    Index extended laterally, all others curled. DISTINCT from YOU by x-axis."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 0.9)
    lm = _curl_finger(lm, 9, 0.95)
    lm = _curl_finger(lm, 13, 0.95)
    lm = _curl_finger(lm, 17, 0.95)
    # Index points laterally (positive x = right)
    lm = _point_finger(lm, 5, np.array([0.9, 0.2, 0.0]))
    return lm


def _pose_subject_she() -> np.ndarray:
    """SHE: Index pointing to opposite side.
    Index extended to the left, all others curled. Mirror of HE."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 0.9)
    lm = _curl_finger(lm, 9, 0.95)
    lm = _curl_finger(lm, 13, 0.95)
    lm = _curl_finger(lm, 17, 0.95)
    # Index points left (negative x)
    lm = _point_finger(lm, 5, np.array([-0.9, 0.2, 0.0]))
    return lm


def _pose_subject_we() -> np.ndarray:
    """WE: Index finger sweeping arc (self to others).
    Index + middle extended, pointing forward-and-down, thumb out, ring+pinky curled."""
    lm = _BASE_PALM.copy()
    lm = _curl_finger(lm, 13, 0.9)
    lm = _curl_finger(lm, 17, 0.9)
    # Index and middle point forward-downish
    lm = _point_finger(lm, 5, np.array([0.1, 0.3, 0.7]))
    lm = _point_finger(lm, 9, np.array([-0.1, 0.3, 0.7]))
    # Thumb out to the side
    lm[4] = lm[1] + np.array([0.2, 0.1, 0.0])
    lm[3] = lm[1] + np.array([0.12, 0.08, 0.0])
    return lm


def _pose_subject_they() -> np.ndarray:
    """THEY: Open hand sweep (all fingers extended, palm forward).
    All five fingers extended and spread, palm facing forward."""
    lm = _BASE_PALM.copy()
    # All fingers already extended in base, just spread them more
    lm = _point_finger(lm, 5, np.array([0.2, 0.8, 0.3]))
    lm = _point_finger(lm, 9, np.array([0.0, 0.9, 0.3]))
    lm = _point_finger(lm, 13, np.array([-0.15, 0.8, 0.3]))
    lm = _point_finger(lm, 17, np.array([-0.3, 0.7, 0.3]))
    lm[4] = lm[1] + np.array([0.3, 0.15, 0.1])
    lm[3] = lm[1] + np.array([0.2, 0.1, 0.05])
    return lm


def _pose_verb_want() -> np.ndarray:
    """WANT: Claw hand pulling toward self.
    All fingers semi-curled (claw shape), palm up, distinct from GRAB (full fist)."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 0.4)
    lm = _curl_finger(lm, 5, 0.5)   # index half-curled
    lm = _curl_finger(lm, 9, 0.5)   # middle half-curled
    lm = _curl_finger(lm, 13, 0.5)  # ring half-curled
    lm = _curl_finger(lm, 17, 0.5)  # pinky half-curled
    # Palm up (rotate around x-axis)
    lm = _rotate_hand(lm, -30, "x")
    return lm


def _pose_verb_eat() -> np.ndarray:
    """EAT: Fingertips bunched together, moving to mouth.
    All fingertips converge to a single point (bunched), distinct from WANT (spread claw)."""
    lm = _BASE_PALM.copy()
    target = np.array([0.05, 0.40, -0.05])  # convergence point above palm
    for finger_start in [5, 9, 13, 17]:
        for i, joint_idx in enumerate([finger_start + 1, finger_start + 2, finger_start + 3]):
            t = (i + 1) / 3.0
            lm[joint_idx] = lm[finger_start] * (1 - t) + target * t
    # Thumb joins
    for i, joint_idx in enumerate([2, 3, 4]):
        t = (i + 1) / 3.0
        lm[joint_idx] = lm[1] * (1 - t) + target * t
    return lm


def _pose_verb_see() -> np.ndarray:
    """SEE: V-shape from eyes (index + middle extended in wide V, others curled).
    Index and middle spread WIDE apart, rest curled tightly. Thumb curled.
    Distinct from HOUSE (narrow inverted-V with tips touching)."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 0.95)
    lm = _curl_finger(lm, 13, 0.98)
    lm = _curl_finger(lm, 17, 0.98)
    # Index and middle spread in WIDE V upward — tips far apart
    lm = _point_finger(lm, 5, np.array([0.5, 0.8, 0.15]))   # index right-up-forward
    lm = _point_finger(lm, 9, np.array([-0.5, 0.8, 0.15]))  # middle left-up-forward
    # Tilt hand forward slightly (V from eyes goes forward)
    lm = _rotate_hand(lm, 15, "x")
    return lm


def _pose_verb_grab() -> np.ndarray:
    """GRAB: Full tight fist (all fingers fully curled, thumb over).
    Complete closure — every finger at max curl. Distinct from WANT (half-curl)."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 1.0)
    lm = _curl_finger(lm, 5, 1.0)
    lm = _curl_finger(lm, 9, 1.0)
    lm = _curl_finger(lm, 13, 1.0)
    lm = _curl_finger(lm, 17, 1.0)
    return lm


def _pose_verb_drink() -> np.ndarray:
    """DRINK: C-hand tilted (thumb + fingers form C-shape around imaginary cup).
    Thumb out, fingers curved but NOT bunched (distinct from EAT).
    Hand tilted (rotated) to simulate tilting cup. Distinct from WANT (no tilt)."""
    lm = _BASE_PALM.copy()
    lm = _curl_finger(lm, 5, 0.6)
    lm = _curl_finger(lm, 9, 0.6)
    lm = _curl_finger(lm, 13, 0.6)
    lm = _curl_finger(lm, 17, 0.6)
    # Thumb stays out to form C
    lm[4] = lm[1] + np.array([0.25, 0.20, 0.0])
    lm[3] = lm[1] + np.array([0.18, 0.15, 0.0])
    lm[2] = lm[1] + np.array([0.10, 0.08, 0.0])
    # Tilt hand (rotate ~40 degrees around z-axis)
    lm = _rotate_hand(lm, 40, "z")
    # Additional tilt backward
    lm = _rotate_hand(lm, -20, "x")
    return lm


def _pose_verb_go() -> np.ndarray:
    """GO: Index finger flicking forward (extended, angled forward+up).
    Only index extended at a steep forward angle. Distinct from YOU (straight forward)
    by having steeper angle and slight upward component."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 0.85)
    lm = _curl_finger(lm, 9, 0.95)
    lm = _curl_finger(lm, 13, 0.95)
    lm = _curl_finger(lm, 17, 0.95)
    # Index flicks forward+up at steep angle
    lm = _point_finger(lm, 5, np.array([0.0, 0.6, 0.7]))
    # Rotate whole hand forward
    lm = _rotate_hand(lm, 25, "x")
    return lm


def _pose_verb_stop() -> np.ndarray:
    """STOP: Flat open palm facing forward (traffic-stop gesture).
    All fingers extended FORWARD (toward camera, z-axis), palm perpendicular.
    Distinct from THEY (fingers spread UP, y-axis) and BOOK (slight fan)."""
    lm = _BASE_PALM.copy()
    # All fingers extended forward toward camera (z-axis dominant)
    lm = _point_finger(lm, 5, np.array([0.1, 0.2, -0.95]))
    lm = _point_finger(lm, 9, np.array([0.0, 0.2, -1.0]))
    lm = _point_finger(lm, 13, np.array([-0.1, 0.2, -0.95]))
    lm = _point_finger(lm, 17, np.array([-0.2, 0.15, -0.9]))
    # Thumb to side, also forward
    lm[4] = lm[1] + np.array([0.25, 0.05, -0.1])
    lm[3] = lm[1] + np.array([0.18, 0.04, -0.07])
    return lm


def _pose_object_food() -> np.ndarray:
    """FOOD: Flat palm facing up (offering gesture).
    All fingers extended, palm rotated to face upward. Distinct from STOP (palm forward)."""
    lm = _BASE_PALM.copy()
    # Extend all fingers
    lm = _point_finger(lm, 5, np.array([0.1, 1.0, 0.0]))
    lm = _point_finger(lm, 9, np.array([0.0, 1.0, 0.0]))
    lm = _point_finger(lm, 13, np.array([-0.1, 1.0, 0.0]))
    lm = _point_finger(lm, 17, np.array([-0.2, 0.95, 0.0]))
    # Rotate palm up (90 degrees around x-axis)
    lm = _rotate_hand(lm, 90, "x")
    return lm


def _pose_object_water() -> np.ndarray:
    """WATER: W-hand (index + middle + ring extended and spread, pinky+thumb curled).
    Three-finger spread. Distinct from SEE (V = 2 fingers) and THEY (5 fingers)."""
    lm = _BASE_PALM.copy()
    lm = _curl_thumb(lm, 0.9)
    lm = _curl_finger(lm, 17, 0.95)  # pinky curled
    # Index, middle, ring extended and spread
    lm = _point_finger(lm, 5, np.array([0.25, 0.9, 0.0]))
    lm = _point_finger(lm, 9, np.array([0.0, 1.0, 0.0]))
    lm = _point_finger(lm, 13, np.array([-0.25, 0.9, 0.0]))
    return lm


def _pose_object_book() -> np.ndarray:
    """BOOK: Two palms opening (single hand simulating open-book angle).
    Fingers close together (not spread), tilted at ~45° angle, palm sideways.
    Distinct from STOP (fingers forward, palm flat) and THEY (spread up)."""
    lm = _BASE_PALM.copy()
    # Fingers close together pointing up-right (like one page of an open book)
    lm = _point_finger(lm, 5, np.array([0.05, 0.95, -0.15]))
    lm = _point_finger(lm, 9, np.array([0.0, 1.0, -0.12]))
    lm = _point_finger(lm, 13, np.array([-0.05, 0.95, -0.10]))
    lm = _point_finger(lm, 17, np.array([-0.08, 0.90, -0.08]))
    # Thumb tucked in (unlike STOP where thumb is out)
    lm = _curl_thumb(lm, 0.6)
    # Rotate hand sideways to simulate book-page angle
    lm = _rotate_hand(lm, 45, "y")
    lm = _rotate_hand(lm, -20, "z")
    return lm


def _pose_object_apple() -> np.ndarray:
    """APPLE: Fist with index knuckle twist at cheek (ASL sign).
    Fist with index protruding significantly and thumb up.
    Distinct from GRAB (tight fist, thumb over) and BALL (open cup)."""
    lm = _BASE_PALM.copy()
    lm = _curl_finger(lm, 5, 0.55)   # index half-curled, protruding
    lm = _curl_finger(lm, 9, 0.95)
    lm = _curl_finger(lm, 13, 0.95)
    lm = _curl_finger(lm, 17, 0.95)
    # Thumb points UP (not curled over)
    lm[4] = lm[1] + np.array([0.05, 0.30, -0.02])
    lm[3] = lm[1] + np.array([0.04, 0.20, -0.01])
    lm[2] = lm[1] + np.array([0.03, 0.10, 0.0])
    # Index PIP protrudes outward strongly
    lm[6] += np.array([0.08, 0.08, -0.12])
    lm[7] += np.array([0.04, 0.04, -0.08])
    # Significant wrist twist
    lm = _rotate_hand(lm, 35, "y")
    return lm


def _pose_object_ball() -> np.ndarray:
    """BALL: Cupped hands (rounded shape, all fingers curved around sphere).
    All fingers semi-curled symmetrically. Distinct from WANT (claw pulls)
    by being more rounded and symmetrical."""
    lm = _BASE_PALM.copy()
    # All fingers curve to form sphere shape (less curl than GRAB, more than WANT)
    lm = _curl_finger(lm, 5, 0.35)
    lm = _curl_finger(lm, 9, 0.35)
    lm = _curl_finger(lm, 13, 0.35)
    lm = _curl_finger(lm, 17, 0.35)
    lm = _curl_thumb(lm, 0.35)
    # Spread fingers outward to form round shape
    for i in [8, 12, 16, 20]:  # tips
        lm[i][2] -= 0.1  # push tips forward (z)
    for i in [4]:
        lm[i][0] += 0.05  # thumb out
        lm[i][2] -= 0.08
    return lm


def _pose_object_house() -> np.ndarray:
    """HOUSE: All five fingers forming steep tent/roof shape.
    ALL fingers converge at a peak above the palm (not just index+middle).
    Distinct from SEE (wide V, only 2 fingers, forward-facing)."""
    lm = _BASE_PALM.copy()
    # All fingers converge to a steep peak above the wrist
    peak = np.array([0.0, 0.75, 0.0])
    # Each finger comes from its MCP and meets at the peak
    for finger_start in [5, 9, 13, 17]:
        mcp = lm[finger_start].copy()
        for i, joint_idx in enumerate([finger_start + 1, finger_start + 2, finger_start + 3]):
            t = (i + 1) / 3.0
            lm[joint_idx] = mcp * (1 - t) + peak * t
    # Thumb also converges
    for i, joint_idx in enumerate([2, 3, 4]):
        t = (i + 1) / 3.0
        lm[joint_idx] = lm[1] * (1 - t) + peak * t
    # Rotate hand upward to be more vertical
    lm = _rotate_hand(lm, -15, "x")
    return lm


# Gesture ID → canonical pose function
POSE_GENERATORS: dict[str, callable] = {
    "subject_i":    _pose_subject_i,
    "subject_you":  _pose_subject_you,
    "subject_he":   _pose_subject_he,
    "subject_she":  _pose_subject_she,
    "subject_we":   _pose_subject_we,
    "subject_they": _pose_subject_they,
    "verb_want":    _pose_verb_want,
    "verb_eat":     _pose_verb_eat,
    "verb_see":     _pose_verb_see,
    "verb_grab":    _pose_verb_grab,
    "verb_drink":   _pose_verb_drink,
    "verb_go":      _pose_verb_go,
    "verb_stop":    _pose_verb_stop,
    "object_food":  _pose_object_food,
    "object_water": _pose_object_water,
    "object_book":  _pose_object_book,
    "object_apple": _pose_object_apple,
    "object_ball":  _pose_object_ball,
    "object_house": _pose_object_house,
}


# ---------------------------------------------------------------------------
# Augmentation pipeline
# ---------------------------------------------------------------------------

def augment_landmarks(
    canonical: np.ndarray,
    n_samples: int,
    *,
    joint_noise_std: float = 0.015,
    rotation_range: float = 25.0,
    scale_range: tuple[float, float] = (0.75, 1.25),
    translation_range: float = 0.08,
) -> np.ndarray:
    """Generate n_samples augmented variants of a canonical 21×3 pose.

    Augmentations (v2 — wider range for better generalization):
      1. Per-joint Gaussian noise (simulates hand tremor / measurement error)
      2. Random 3D rotation (simulates camera angle + hand orientation variation)
      3. Random uniform scaling (simulates hand-camera distance)
      4. Random translation (simulates hand position in frame)
      5. Random per-finger curl perturbation (simulates natural hand variation)
      6. Z-depth noise (simulates webcam depth unreliability)
    """
    samples = np.zeros((n_samples, NUM_HAND_LANDMARKS, 3), dtype=np.float64)

    for i in range(n_samples):
        lm = canonical.copy()

        # 1. Joint noise (varied intensity)
        noise_scale = np.random.uniform(0.5, 1.5) * joint_noise_std
        noise = np.random.normal(0, noise_scale, lm.shape)
        noise[0] *= 0.3  # Less noise on wrist
        lm += noise

        # 2. Random rotation (all three axes, wider range)
        for axis in ("x", "y", "z"):
            angle = np.random.uniform(-rotation_range, rotation_range)
            lm = _rotate_hand(lm, angle, axis)

        # 3. Random scale
        scale = np.random.uniform(*scale_range)
        wrist = lm[0].copy()
        lm = wrist + (lm - wrist) * scale

        # 4. Random translation
        offset = np.random.uniform(-translation_range, translation_range, 3)
        lm += offset

        # 5. Per-finger curl perturbation (randomly adjust individual fingers slightly)
        for finger_start in [5, 9, 13, 17]:
            curl_perturb = np.random.uniform(-0.08, 0.08)
            if abs(curl_perturb) > 0.02:
                lm = _curl_finger(lm, finger_start, curl_perturb)

        # 6. Z-depth noise (webcams have ~3x more z noise than x/y)
        z_noise = np.random.normal(0, joint_noise_std * 2.5, NUM_HAND_LANDMARKS)
        lm[:, 2] += z_noise

        samples[i] = lm

    return samples


def normalize_to_wrist(landmarks: np.ndarray) -> np.ndarray:
    """Normalize: translate wrist to origin, scale max distance to 1."""
    lm = landmarks.copy()
    wrist = lm[0].copy()
    lm -= wrist
    max_dist = np.max(np.linalg.norm(lm, axis=1))
    if max_dist > 1e-8:
        lm /= max_dist
    return lm


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(samples_per_gesture: int = 200) -> pd.DataFrame:
    """Generate synthetic landmark dataset for all 18 gestures.

    Args:
        samples_per_gesture: Number of augmented samples per class.

    Returns:
        DataFrame with columns: gesture_id, lm_0_x, ..., lm_20_z, source
    """
    logger.info("Generating synthetic dataset: %d samples × %d gestures = %d total",
                samples_per_gesture, len(GESTURE_IDS), samples_per_gesture * len(GESTURE_IDS))

    all_rows: list[dict] = []

    for gesture_id in GESTURE_IDS:
        pose_fn = POSE_GENERATORS.get(gesture_id)
        if pose_fn is None:
            logger.warning("No pose generator for %s — skipping", gesture_id)
            continue

        canonical = pose_fn()
        augmented = augment_landmarks(canonical, samples_per_gesture)

        for i in range(samples_per_gesture):
            lm = normalize_to_wrist(augmented[i])
            row: dict = {"gesture_id": gesture_id}
            for j in range(NUM_HAND_LANDMARKS):
                row[f"lm_{j}_x"] = float(lm[j, 0])
                row[f"lm_{j}_y"] = float(lm[j, 1])
                row[f"lm_{j}_z"] = float(lm[j, 2])
            row["source"] = "synthetic"
            all_rows.append(row)

        logger.info("  %s (%s): %d samples", gesture_id, GESTURE_LABEL_MAP[gesture_id], samples_per_gesture)

    df = pd.DataFrame(all_rows)
    logger.info("Total synthetic samples: %d", len(df))
    return df


# ---------------------------------------------------------------------------
# Inter-class separation analysis
# ---------------------------------------------------------------------------

def analyze_separation(df: pd.DataFrame) -> dict:
    """Compute pairwise inter-class distances to verify no collisions.

    Returns dict with min/mean distances and collision warnings.
    """
    logger.info("Analyzing inter-class separation …")

    lm_cols = [f"lm_{i}_{a}" for i in range(NUM_HAND_LANDMARKS) for a in ("x", "y", "z")]

    # Compute class centroids
    centroids: dict[str, np.ndarray] = {}
    for gid in GESTURE_IDS:
        mask = df["gesture_id"] == gid
        if mask.sum() > 0:
            centroids[gid] = df.loc[mask, lm_cols].values.mean(axis=0)

    # Pairwise distances
    gesture_list = list(centroids.keys())
    n = len(gesture_list)
    distances = np.zeros((n, n))
    min_dist = float("inf")
    min_pair = ("", "")

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(centroids[gesture_list[i]] - centroids[gesture_list[j]])
            distances[i, j] = d
            distances[j, i] = d
            if d < min_dist:
                min_dist = d
                min_pair = (gesture_list[i], gesture_list[j])

    # Compute intra-class spread (std of distances from centroid)
    spreads: dict[str, float] = {}
    for gid in GESTURE_IDS:
        mask = df["gesture_id"] == gid
        if mask.sum() > 0:
            data = df.loc[mask, lm_cols].values
            centroid = centroids[gid]
            dists = np.linalg.norm(data - centroid, axis=1)
            spreads[gid] = float(np.std(dists))

    # Check for potential collisions
    max_spread = max(spreads.values()) if spreads else 0
    collision_threshold = 2.5 * max_spread  # classes closer than 2.5× spread may collide
    collisions = []

    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] < collision_threshold:
                collisions.append({
                    "pair": (gesture_list[i], gesture_list[j]),
                    "distance": float(distances[i, j]),
                    "threshold": collision_threshold,
                })

    result = {
        "min_inter_class_distance": min_dist,
        "closest_pair": min_pair,
        "mean_inter_class_distance": float(np.mean(distances[distances > 0])),
        "max_intra_class_spread": max_spread,
        "separation_ratio": min_dist / (max_spread + 1e-8),
        "potential_collisions": collisions,
    }

    logger.info("  Min inter-class distance: %.4f (between %s and %s)",
                min_dist, min_pair[0], min_pair[1])
    logger.info("  Max intra-class spread: %.4f", max_spread)
    logger.info("  Separation ratio: %.2f (>2.0 is good)", result["separation_ratio"])

    if collisions:
        logger.warning("  Potential collisions detected:")
        for c in collisions:
            logger.warning("    %s ↔ %s (dist=%.4f < threshold=%.4f)",
                          c["pair"][0], c["pair"][1], c["distance"], c["threshold"])
    else:
        logger.info("  No collisions detected — all classes well-separated!")

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    """Generate synthetic dataset and analyze separation."""
    logger.info("MLAF Training Pipeline — Synthetic Data Generation")

    # Generate with enough samples for robust training (500 for >90% target accuracy)
    df = generate_dataset(samples_per_gesture=500)

    # Save
    output_path = PROCESSED_DIR / "synthetic_landmarks.csv"
    df.to_csv(output_path, index=False)
    logger.info("Saved: %s", output_path)

    # Analyze separation
    separation = analyze_separation(df)

    # Save analysis
    analysis_path = PROCESSED_DIR / "separation_analysis.json"
    with open(analysis_path, "w") as f:
        json.dump(separation, f, indent=2, default=str)
    logger.info("Separation analysis saved: %s", analysis_path)

    stats = {
        "total_samples": len(df),
        "gestures": len(GESTURE_IDS),
        "samples_per_gesture": 500,
        "separation": separation,
    }
    return stats


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)
