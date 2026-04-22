"""MLAF Training Pipeline — Emission Matrix Training.

Learns empirical emission probabilities from webcam-captured data:
  - UASAM: P(vocalization_state | gesture) matrix
  - EyeGaze: P(gaze_state | gesture) matrix
  - Fusion weights: optimal (w_V, w_A, w_G) via grid search

Replaces hand-guessed probability matrices with data-driven values.

Usage:
    python -m training.train_emission_matrices
    python training/train_emission_matrices.py
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    CUSTOM_DIR,
    EXPERIMENT_REGISTRY_PATH,
    GAZE_STATES,
    GESTURE_IDS,
    GESTURE_LABEL_MAP,
    ID_TO_IDX,
    IDX_TO_ID,
    INSTITUTION,
    LOGS_DIR,
    MODELS_DIR,
    MODALITY_WEIGHTS_GRID,
    NUM_GESTURE_CLASSES,
    PROJECT_NAME,
    RANDOM_SEED,
    SPLITS_DIR,
    UASAM_STATES,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Laplace smoothing parameter
ALPHA = 1.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_gaze_data() -> pd.DataFrame | None:
    """Load webcam gaze CSV (from collect_webcam.py)."""
    path = CUSTOM_DIR / "webcam_gaze.csv"
    if not path.exists():
        logger.warning("No gaze data at %s — run collect_webcam.py first", path)
        return None
    df = pd.read_csv(path)
    logger.info("Loaded gaze data: %d rows", len(df))
    return df


def _load_audio_data() -> pd.DataFrame | None:
    """Load webcam audio CSV (from collect_webcam.py)."""
    path = CUSTOM_DIR / "webcam_audio.csv"
    if not path.exists():
        logger.warning("No audio data at %s — run collect_webcam.py first", path)
        return None
    df = pd.read_csv(path)
    logger.info("Loaded audio data: %d rows", len(df))
    return df


# ---------------------------------------------------------------------------
# UASAM emission matrix: P(vocState | gesture)
# ---------------------------------------------------------------------------

def _classify_vocalization_state(audio_row: dict) -> str:
    """Classify a single audio sample into a UASAM vocalization state.

    States: silent, vocalization, speech_attempt, clear_speech
    Based on RMS energy and spectral features.
    """
    rms = audio_row.get("audio_rms", 0.0)
    zcr = audio_row.get("audio_zcr", 0.0)
    spectral_centroid = audio_row.get("audio_spectral_centroid", 0.0)

    if rms < 0.005:
        return "silent"
    elif rms < 0.02:
        return "vocalization"
    elif spectral_centroid < 1500:
        return "speech_attempt"
    else:
        return "clear_speech"


def train_uasam_emission(audio_df: pd.DataFrame) -> dict:
    """Learn UASAM emission matrix from audio data.

    Returns:
        emission_matrix: dict[gesture_id, dict[state, probability]]
    """
    logger.info("=== Training UASAM Emission Matrix ===")

    # Classify each audio sample
    audio_df = audio_df.copy()
    audio_df["voc_state"] = audio_df.apply(
        lambda row: _classify_vocalization_state(row.to_dict()), axis=1,
    )

    # Count occurrences: gesture × vocalization state
    num_states = len(UASAM_STATES)
    state_idx = {s: i for i, s in enumerate(UASAM_STATES)}

    counts = np.zeros((NUM_GESTURE_CLASSES, num_states), dtype=np.float64)

    for _, row in audio_df.iterrows():
        gid = row.get("gesture_id")
        state = row.get("voc_state")
        if gid in ID_TO_IDX and state in state_idx:
            counts[ID_TO_IDX[gid], state_idx[state]] += 1

    # Laplace smoothing + normalize
    counts += ALPHA
    emission = counts / counts.sum(axis=1, keepdims=True)

    # Format as dict
    matrix: dict[str, dict[str, float]] = {}
    for i, gid in enumerate(GESTURE_IDS):
        matrix[gid] = {
            state: float(emission[i, j])
            for j, state in enumerate(UASAM_STATES)
        }

    logger.info("  UASAM emission matrix: %d gestures × %d states", NUM_GESTURE_CLASSES, num_states)

    # Log sample
    for gid in GESTURE_IDS[:3]:
        logger.info("    %s: %s", gid, {k: f"{v:.3f}" for k, v in matrix[gid].items()})

    return matrix


# ---------------------------------------------------------------------------
# EyeGaze emission matrix: P(gazeState | gesture)
# ---------------------------------------------------------------------------

def train_gaze_emission(gaze_df: pd.DataFrame) -> dict:
    """Learn eye gaze emission matrix from gaze data.

    Returns:
        emission_matrix: dict[gesture_id, dict[state, probability]]
    """
    logger.info("=== Training EyeGaze Emission Matrix ===")

    num_states = len(GAZE_STATES)
    state_idx = {s: i for i, s in enumerate(GAZE_STATES)}

    counts = np.zeros((NUM_GESTURE_CLASSES, num_states), dtype=np.float64)

    for _, row in gaze_df.iterrows():
        gid = row.get("gesture_id")
        state = row.get("gaze_state")
        if gid in ID_TO_IDX and state in state_idx:
            counts[ID_TO_IDX[gid], state_idx[state]] += 1

    # Laplace smoothing + normalize
    counts += ALPHA
    emission = counts / counts.sum(axis=1, keepdims=True)

    matrix: dict[str, dict[str, float]] = {}
    for i, gid in enumerate(GESTURE_IDS):
        matrix[gid] = {
            state: float(emission[i, j])
            for j, state in enumerate(GAZE_STATES)
        }

    logger.info("  Gaze emission matrix: %d gestures × %d states", NUM_GESTURE_CLASSES, num_states)

    for gid in GESTURE_IDS[:3]:
        logger.info("    %s: %s", gid, {k: f"{v:.3f}" for k, v in matrix[gid].items()})

    return matrix


# ---------------------------------------------------------------------------
# Fusion weight optimization
# ---------------------------------------------------------------------------

def optimize_fusion_weights(
    gesture_probs: np.ndarray,
    uasam_matrix: np.ndarray,
    gaze_matrix: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Grid search over (w_V, w_A, w_G) to maximize validation accuracy.

    Args:
        gesture_probs: (N, C) visual gesture classifier probabilities
        uasam_matrix: (C, S_a) UASAM emission matrix
        gaze_matrix: (C, S_g) gaze emission matrix
        y_true: (N,) true labels

    Returns:
        dict with best weights and accuracy
    """
    logger.info("=== Fusion Weight Optimization ===")

    w_visual_opts = MODALITY_WEIGHTS_GRID["w_visual"]
    w_acoustic_opts = MODALITY_WEIGHTS_GRID["w_acoustic"]
    w_gaze_opts = MODALITY_WEIGHTS_GRID["w_gaze"]

    best_acc = 0.0
    best_weights = {"w_visual": 0.5, "w_acoustic": 0.25, "w_gaze": 0.25}
    results: list[dict] = []

    for w_v in w_visual_opts:
        for w_a in w_acoustic_opts:
            for w_g in w_gaze_opts:
                # Normalize weights to sum to 1
                total = w_v + w_a + w_g
                if total < 1e-8:
                    continue
                wv, wa, wg = w_v / total, w_a / total, w_g / total

                # Fused score: weighted sum of modality contributions
                # For audio/gaze, use uniform prior (each sample gets same emission)
                fused = wv * gesture_probs

                # Add acoustic prior (uniform across states → marginal gesture prob)
                acoustic_prior = uasam_matrix.sum(axis=1)
                acoustic_prior /= acoustic_prior.sum() + 1e-8
                fused += wa * np.tile(acoustic_prior, (len(gesture_probs), 1))

                # Add gaze prior
                gaze_prior = gaze_matrix.sum(axis=1)
                gaze_prior /= gaze_prior.sum() + 1e-8
                fused += wg * np.tile(gaze_prior, (len(gesture_probs), 1))

                y_pred = fused.argmax(axis=1)
                acc = (y_pred == y_true).mean()

                results.append({
                    "w_visual": wv, "w_acoustic": wa, "w_gaze": wg,
                    "accuracy": float(acc),
                })

                if acc > best_acc:
                    best_acc = acc
                    best_weights = {"w_visual": wv, "w_acoustic": wa, "w_gaze": wg}

    logger.info("  Best fusion weights: %s → accuracy %.4f",
                {k: f"{v:.3f}" for k, v in best_weights.items()}, best_acc)

    return {
        "best_weights": best_weights,
        "best_accuracy": best_acc,
        "grid_search_results": results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    """Train emission matrices and optimize fusion weights."""
    logger.info("MLAF Training Pipeline — Emission Matrices")

    results: dict = {
        "timestamp": datetime.datetime.now().isoformat(),
        "project": PROJECT_NAME,
        "institution": INSTITUTION,
    }

    # Load webcam data
    gaze_df = _load_gaze_data()
    audio_df = _load_audio_data()

    if audio_df is not None and len(audio_df) > 0:
        uasam_matrix = train_uasam_emission(audio_df)
        results["uasam_emission"] = uasam_matrix

        # Save as numpy array too
        uasam_np = np.array([
            [uasam_matrix[gid][state] for state in UASAM_STATES]
            for gid in GESTURE_IDS
        ])
        np.save(MODELS_DIR / "uasam_emission.npy", uasam_np)
    else:
        logger.warning("No audio data — generating uniform UASAM prior")
        uasam_matrix = {
            gid: {state: 1.0 / len(UASAM_STATES) for state in UASAM_STATES}
            for gid in GESTURE_IDS
        }
        results["uasam_emission"] = uasam_matrix
        results["uasam_note"] = "uniform prior (no audio data)"
        uasam_np = np.ones((NUM_GESTURE_CLASSES, len(UASAM_STATES))) / len(UASAM_STATES)

    if gaze_df is not None and len(gaze_df) > 0:
        gaze_matrix = train_gaze_emission(gaze_df)
        results["gaze_emission"] = gaze_matrix

        gaze_np = np.array([
            [gaze_matrix[gid][state] for state in GAZE_STATES]
            for gid in GESTURE_IDS
        ])
        np.save(MODELS_DIR / "gaze_emission.npy", gaze_np)
    else:
        logger.warning("No gaze data — generating uniform gaze prior")
        gaze_matrix = {
            gid: {state: 1.0 / len(GAZE_STATES) for state in GAZE_STATES}
            for gid in GESTURE_IDS
        }
        results["gaze_emission"] = gaze_matrix
        results["gaze_note"] = "uniform prior (no gaze data)"
        gaze_np = np.ones((NUM_GESTURE_CLASSES, len(GAZE_STATES))) / len(GAZE_STATES)

    # Fusion weight optimization (if gesture classifier exists)
    val_path = SPLITS_DIR / "val.csv"
    gesture_model_paths = sorted(MODELS_DIR.glob("gesture_rf_*.joblib"))

    if val_path.exists() and gesture_model_paths:
        import joblib
        from sklearn.metrics import accuracy_score

        logger.info("Loading gesture model for fusion optimization …")
        model = joblib.load(gesture_model_paths[-1])

        val_df = pd.read_csv(val_path)
        meta_cols = {"gesture_id", "gesture_label_raw", "source", "class_idx", "frame"}
        feat_cols = [c for c in val_df.columns if c not in meta_cols and val_df[c].dtype in (np.float64, np.float32, np.int64)]
        X_val = val_df[feat_cols].values.astype(np.float32)
        X_val = np.nan_to_num(X_val, nan=0.0)
        y_val = val_df["gesture_id"].map(ID_TO_IDX).values.astype(np.int64)

        try:
            gesture_probs = model.predict_proba(X_val)
            fusion_results = optimize_fusion_weights(gesture_probs, uasam_np, gaze_np, y_val)
            results["fusion_weights"] = fusion_results
        except Exception as exc:
            logger.warning("Fusion optimization failed: %s", exc)
            results["fusion_weights"] = {
                "best_weights": {"w_visual": 0.6, "w_acoustic": 0.2, "w_gaze": 0.2},
                "note": f"default weights (optimization failed: {exc})",
            }
    else:
        logger.info("No gesture model/val data — using default fusion weights")
        results["fusion_weights"] = {
            "best_weights": {"w_visual": 0.6, "w_acoustic": 0.2, "w_gaze": 0.2},
            "note": "default weights (no gesture model available)",
        }

    # Save combined results
    output_path = MODELS_DIR / "emission_matrices.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Emission matrices saved: %s", output_path)

    # Save training log
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    log_path = LOGS_DIR / f"emission_log_{timestamp}.json"
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Emission log saved: %s", log_path)

    return results


if __name__ == "__main__":
    result = main()
    weights = result.get("fusion_weights", {}).get("best_weights", {})
    print(f"\nFusion weights: V={weights.get('w_visual', '?')}, "
          f"A={weights.get('w_acoustic', '?')}, G={weights.get('w_gaze', '?')}")
    sys.exit(0)
