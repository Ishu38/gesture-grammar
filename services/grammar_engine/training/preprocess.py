"""MLAF Training Pipeline — Preprocessing.

Converts all data sources (Zenodo CSV, HaGRID annotations, webcam captures)
into a unified landmark CSV schema, engineers features, and creates
stratified train/val/test splits.

Unified schema:
    gesture_id, lm_0_x, lm_0_y, lm_0_z, ..., lm_20_z, [engineered features]

Usage:
    python -m training.preprocess
    python training/preprocess.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    CUSTOM_DIR,
    FINGER_MCP_INDICES,
    FINGER_NAMES,
    FINGERTIP_INDICES,
    GESTURE_IDS,
    HAND_FEATURE_DIM,
    ID_TO_IDX,
    NUM_HAND_LANDMARKS,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    SPLIT_RATIOS,
    SPLITS_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ---------------------------------------------------------------------------
# Landmark column names
# ---------------------------------------------------------------------------

def _landmark_columns() -> list[str]:
    """Return 63 column names: lm_0_x, lm_0_y, lm_0_z, ..., lm_20_z."""
    cols = []
    for i in range(NUM_HAND_LANDMARKS):
        for axis in ("x", "y", "z"):
            cols.append(f"lm_{i}_{axis}")
    return cols


LANDMARK_COLS = _landmark_columns()


# ---------------------------------------------------------------------------
# Source 1: Zenodo CSV
# ---------------------------------------------------------------------------

def _load_zenodo(path: Path) -> pd.DataFrame | None:
    """Load Zenodo hand-gestures.csv and normalize to unified schema."""
    if not path.exists():
        logger.warning("Zenodo CSV not found at %s", path)
        return None

    logger.info("Loading Zenodo CSV: %s", path)
    df = pd.read_csv(path)

    # Detect landmark columns — expect 63 numeric columns + 1 label
    numeric_cols = [c for c in df.columns if df[c].dtype in (np.float64, np.float32, np.int64)]
    label_col = [c for c in df.columns if c not in numeric_cols]

    if len(numeric_cols) < HAND_FEATURE_DIM:
        logger.warning(
            "Zenodo CSV has %d numeric cols (expected >= %d), skipping",
            len(numeric_cols), HAND_FEATURE_DIM,
        )
        return None

    # Take first 63 numeric columns as landmarks
    lm_data = df[numeric_cols[:HAND_FEATURE_DIM]].values

    # Normalize landmarks relative to wrist (landmark 0)
    lm_data = _normalize_to_wrist(lm_data)

    result = pd.DataFrame(lm_data, columns=LANDMARK_COLS)

    # Map labels
    if label_col:
        result["gesture_label_raw"] = df[label_col[0]].values
        result["gesture_id"] = result["gesture_label_raw"].apply(_map_external_label)
    else:
        result["gesture_id"] = "unknown"

    result["source"] = "zenodo"
    logger.info("  Zenodo: %d samples loaded", len(result))
    return result


# ---------------------------------------------------------------------------
# Source 2: HaGRID annotations
# ---------------------------------------------------------------------------

def _load_hagrid(hagrid_dir: Path) -> pd.DataFrame | None:
    """Load HaGRID annotation JSONs containing hand landmark data."""
    if not hagrid_dir.exists():
        logger.warning("HaGRID directory not found at %s", hagrid_dir)
        return None

    json_files = sorted(hagrid_dir.glob("*.json"))
    if not json_files:
        logger.warning("No JSON files in %s", hagrid_dir)
        return None

    all_rows: list[dict] = []

    for jf in json_files:
        logger.info("  Loading HaGRID: %s", jf.name)
        try:
            with open(jf) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("    Failed to parse %s: %s", jf.name, exc)
            continue

        # Extract gesture class from filename: train_val_<class>.json
        gesture_class = jf.stem.replace("train_val_", "")

        if isinstance(data, dict):
            for _key, entry in data.items():
                landmarks = entry.get("landmarks") or entry.get("hand_landmarks")
                if landmarks and isinstance(landmarks, list):
                    # Flatten list of [x, y, z] triples
                    flat = _flatten_landmarks(landmarks)
                    if flat is not None and len(flat) == HAND_FEATURE_DIM:
                        row = {LANDMARK_COLS[i]: flat[i] for i in range(HAND_FEATURE_DIM)}
                        row["gesture_label_raw"] = gesture_class
                        row["gesture_id"] = _map_external_label(gesture_class)
                        row["source"] = "hagrid"
                        all_rows.append(row)
        elif isinstance(data, list):
            for entry in data:
                landmarks = entry.get("landmarks") or entry.get("hand_landmarks")
                if landmarks and isinstance(landmarks, list):
                    flat = _flatten_landmarks(landmarks)
                    if flat is not None and len(flat) == HAND_FEATURE_DIM:
                        row = {LANDMARK_COLS[i]: flat[i] for i in range(HAND_FEATURE_DIM)}
                        row["gesture_label_raw"] = gesture_class
                        row["gesture_id"] = _map_external_label(gesture_class)
                        row["source"] = "hagrid"
                        all_rows.append(row)

    if not all_rows:
        logger.warning("No landmark data extracted from HaGRID")
        return None

    result = pd.DataFrame(all_rows)
    logger.info("  HaGRID: %d samples loaded", len(result))
    return result


# ---------------------------------------------------------------------------
# Source 3: Webcam custom data
# ---------------------------------------------------------------------------

def _load_webcam(custom_dir: Path) -> pd.DataFrame | None:
    """Load webcam-captured landmark data.

    Supports two layouts:
      1. Sharded: data/custom/landmarks/{gesture_id}.csv  (25GB+ scalable)
      2. Legacy:  data/custom/webcam_landmarks.csv         (single file)

    For sharded layout, reads each CSV in chunks to stay memory-efficient.
    """
    frames: list[pd.DataFrame] = []

    # --- Layout 1: Sharded per-gesture CSVs ---
    landmarks_dir = custom_dir / "landmarks"
    if landmarks_dir.is_dir():
        shard_files = sorted(landmarks_dir.glob("*.csv"))
        if shard_files:
            logger.info("Loading sharded webcam data from %s (%d shards)", landmarks_dir, len(shard_files))
            for shard in shard_files:
                if shard.stat().st_size == 0:
                    continue
                try:
                    # Chunked reading — keeps peak memory low even for multi-GB shards
                    chunks = pd.read_csv(shard, chunksize=50_000)
                    for chunk in chunks:
                        if "gesture_id" in chunk.columns:
                            frames.append(chunk)
                except Exception as exc:
                    logger.warning("Failed to read shard %s: %s", shard.name, exc)

    # --- Layout 2: Legacy single-file ---
    legacy_csv = custom_dir / "webcam_landmarks.csv"
    if legacy_csv.exists() and legacy_csv.stat().st_size > 0:
        logger.info("Loading legacy webcam data: %s", legacy_csv)
        try:
            chunks = pd.read_csv(legacy_csv, chunksize=50_000)
            for chunk in chunks:
                if "gesture_id" in chunk.columns:
                    frames.append(chunk)
        except Exception as exc:
            logger.warning("Failed to read legacy CSV: %s", exc)

    if not frames:
        logger.info("No webcam data found in %s", custom_dir)
        return None

    df = pd.concat(frames, ignore_index=True)

    # Validate landmark columns
    for col in LANDMARK_COLS:
        if col not in df.columns:
            logger.warning("Webcam data missing column: %s", col)
            return None

    df["source"] = "webcam"
    df["gesture_label_raw"] = df["gesture_id"]
    logger.info("  Webcam: %d samples loaded", len(df))
    return df


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

# Map external dataset labels → our 18 gesture IDs where possible
_EXTERNAL_LABEL_MAP: dict[str, str] = {
    # HaGRID classes → MLAF gestures
    "stop": "verb_stop",
    "stop_inverted": "verb_stop",
    "fist": "verb_grab",
    "palm": "verb_stop",
    "one": "subject_i",
    "like": "subject_you",
    "call": "subject_you",
    "peace": "object_ball",
    "ok": "object_apple",
    "mute": "verb_drink",
    "rock": "verb_go",
    # Direct matches (if Zenodo labels match ours)
    "i": "subject_i",
    "you": "subject_you",
    "he": "subject_he",
    "she": "subject_she",
    "we": "subject_we",
    "they": "subject_they",
    "want": "verb_want",
    "eat": "verb_eat",
    "see": "verb_see",
    "grab": "verb_grab",
    "drink": "verb_drink",
    "go": "verb_go",
    "food": "object_food",
    "water": "object_water",
    "book": "object_book",
    "apple": "object_apple",
    "ball": "object_ball",
    "house": "object_house",
}


def _map_external_label(raw_label: str) -> str:
    """Map an external dataset label to an MLAF gesture ID."""
    normalized = str(raw_label).strip().lower().replace(" ", "_").replace("-", "_")
    return _EXTERNAL_LABEL_MAP.get(normalized, "unknown")


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _normalize_to_wrist(landmarks: np.ndarray) -> np.ndarray:
    """Normalize 63-D landmark vectors relative to wrist (landmark 0).

    Translates so wrist = origin, scales so max distance = 1.
    """
    n = landmarks.shape[0]
    result = landmarks.copy()

    for i in range(n):
        row = result[i].reshape(NUM_HAND_LANDMARKS, 3)
        wrist = row[0].copy()
        row -= wrist  # translate to wrist origin

        max_dist = np.max(np.linalg.norm(row, axis=1))
        if max_dist > 1e-8:
            row /= max_dist  # scale to unit

        result[i] = row.flatten()

    return result


def _flatten_landmarks(landmarks: list) -> np.ndarray | None:
    """Flatten nested landmark list [[x,y,z], ...] to flat array."""
    try:
        flat = []
        if isinstance(landmarks[0], (list, tuple)):
            for pt in landmarks:
                flat.extend(pt[:3])
        else:
            flat = list(landmarks)
        return np.array(flat, dtype=np.float32)
    except (IndexError, TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features: finger angles, distances, thumb ratios."""
    logger.info("Engineering features …")
    lm_data = df[LANDMARK_COLS].values.reshape(-1, NUM_HAND_LANDMARKS, 3)
    n = lm_data.shape[0]

    features: dict[str, np.ndarray] = {}

    # 1. Inter-finger distances (all pairs of fingertips) — 10 features
    tips = list(FINGERTIP_INDICES.values())
    for i_idx in range(len(tips)):
        for j_idx in range(i_idx + 1, len(tips)):
            name_i = FINGER_NAMES[i_idx]
            name_j = FINGER_NAMES[j_idx]
            dists = np.linalg.norm(
                lm_data[:, tips[i_idx]] - lm_data[:, tips[j_idx]], axis=1
            )
            features[f"dist_{name_i}_{name_j}"] = dists

    # 2. Finger curl angles (tip-MCP-wrist angle) — 5 features
    wrist = lm_data[:, 0]
    for fname in FINGER_NAMES:
        tip = lm_data[:, FINGERTIP_INDICES[fname]]
        mcp = lm_data[:, FINGER_MCP_INDICES[fname]]

        v1 = tip - mcp
        v2 = wrist - mcp
        cos_angle = np.sum(v1 * v2, axis=1) / (
            np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-8
        )
        cos_angle = np.clip(cos_angle, -1, 1)
        angles = np.arccos(cos_angle)
        features[f"curl_{fname}"] = angles

    # 3. Thumb-to-finger distance ratios — 4 features
    thumb_tip = lm_data[:, FINGERTIP_INDICES["thumb"]]
    for fname in FINGER_NAMES[1:]:  # skip thumb
        other_tip = lm_data[:, FINGERTIP_INDICES[fname]]
        dist = np.linalg.norm(thumb_tip - other_tip, axis=1)
        palm_span = np.linalg.norm(
            lm_data[:, FINGERTIP_INDICES["index"]] - lm_data[:, FINGERTIP_INDICES["pinky"]],
            axis=1,
        ) + 1e-8
        features[f"thumb_ratio_{fname}"] = dist / palm_span

    # 4. Hand spread (max distance between any two landmarks) — 1 feature
    spreads = np.zeros(n)
    for i in range(n):
        dists = np.linalg.norm(lm_data[i][:, None] - lm_data[i][None, :], axis=2)
        spreads[i] = np.max(dists)
    features["hand_spread"] = spreads

    # 5. Center of mass offset from wrist — 3 features
    com = np.mean(lm_data, axis=1)
    features["com_x"] = com[:, 0]
    features["com_y"] = com[:, 1]
    features["com_z"] = com[:, 2]

    feat_df = pd.DataFrame(features, index=df.index)
    logger.info("  Added %d engineered features", len(features))
    return pd.concat([df, feat_df], axis=1)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------

def create_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Create stratified train/val/test splits."""
    # Filter to known gesture IDs
    known = df[df["gesture_id"].isin(GESTURE_IDS)].copy()
    unknown = df[~df["gesture_id"].isin(GESTURE_IDS)]

    if len(unknown) > 0:
        logger.info("  Dropping %d samples with unknown gesture IDs", len(unknown))

    if len(known) < 10:
        logger.warning("Too few known samples (%d) for splitting", len(known))
        return {"train": known, "val": known, "test": known}

    # Encode labels for stratification
    known["class_idx"] = known["gesture_id"].map(ID_TO_IDX)

    # First split: train+val vs test
    test_ratio = SPLIT_RATIOS["test"]
    val_ratio = SPLIT_RATIOS["val"] / (1 - test_ratio)

    train_val, test = train_test_split(
        known,
        test_size=test_ratio,
        stratify=known["class_idx"],
        random_state=RANDOM_SEED,
    )

    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        stratify=train_val["class_idx"],
        random_state=RANDOM_SEED,
    )

    logger.info("  Splits: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    """Run full preprocessing pipeline. Returns dataset statistics."""
    logger.info("MLAF Training Pipeline — Preprocessing")

    # 1. Load all sources
    frames: list[pd.DataFrame] = []

    zenodo_df = _load_zenodo(RAW_DIR / "zenodo_hand_landmarks.csv")
    if zenodo_df is not None:
        frames.append(zenodo_df)

    hagrid_df = _load_hagrid(RAW_DIR / "hagrid")
    if hagrid_df is not None:
        frames.append(hagrid_df)

    webcam_df = _load_webcam(CUSTOM_DIR)
    if webcam_df is not None:
        frames.append(webcam_df)

    # Load synthetic data (generated by generate_synthetic.py)
    synthetic_path = PROCESSED_DIR / "synthetic_landmarks.csv"
    if synthetic_path.exists():
        logger.info("Loading synthetic data: %s", synthetic_path)
        syn_df = pd.read_csv(synthetic_path)
        syn_df["gesture_label_raw"] = syn_df["gesture_id"]
        frames.append(syn_df)
        logger.info("  Synthetic: %d samples loaded", len(syn_df))

    if not frames:
        logger.error("No data loaded! Run generate_synthetic.py, download_datasets.py, or collect_webcam.py first.")
        return {"error": "no data"}

    # 2. Concatenate
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d samples", len(combined))

    # 3. Engineer features
    combined = engineer_features(combined)

    # 4. Save processed dataset
    processed_path = PROCESSED_DIR / "unified_landmarks.csv"
    combined.to_csv(processed_path, index=False)
    logger.info("Saved processed dataset: %s", processed_path)

    # 5. Create splits
    splits = create_splits(combined)

    for split_name, split_df in splits.items():
        path = SPLITS_DIR / f"{split_name}.csv"
        split_df.to_csv(path, index=False)
        logger.info("Saved %s split: %s (%d samples)", split_name, path, len(split_df))

    # 6. Stats
    stats = {
        "total_samples": len(combined),
        "sources": dict(combined["source"].value_counts()),
        "gesture_distribution": dict(combined["gesture_id"].value_counts()),
        "num_features": len(combined.columns),
        "splits": {k: len(v) for k, v in splits.items()},
        "known_gesture_samples": int(combined["gesture_id"].isin(GESTURE_IDS).sum()),
        "unknown_samples": int((~combined["gesture_id"].isin(GESTURE_IDS)).sum()),
    }

    logger.info("Dataset stats: %s", json.dumps(stats, indent=2, default=str))
    return stats


if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2, default=str))
    sys.exit(0)
