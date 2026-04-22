"""MLAF Training Pipeline — Configuration.

Centralizes paths, hyperparameters, gesture ID mappings, and dataset URLs
for the entire training pipeline.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent          # grammar_engine/
TRAINING_DIR = BASE_DIR / "training"
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CUSTOM_DIR = DATA_DIR / "custom"
SPLITS_DIR = DATA_DIR / "splits"
LOGS_DIR = BASE_DIR / "logs"
MODELS_DIR = BASE_DIR / "models"

for _d in (RAW_DIR, PROCESSED_DIR, CUSTOM_DIR, SPLITS_DIR, LOGS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Dataset URLs
# ---------------------------------------------------------------------------

ZENODO_HAND_LANDMARKS_URL = (
    "https://zenodo.org/records/18108472/files/hand-gestures.csv?download=1"
)

HAGRID_ANNOTATIONS_ZIP_URL = (
    "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/"
    "datasets/hagrid_v2/annotations_with_landmarks/annotations.zip"
)

# HaGRID gesture classes present in the annotation ZIP
HAGRID_GESTURE_CLASSES: list[str] = [
    "call", "dislike", "fist", "four", "grabbing", "grip",
    "like", "mute", "ok", "one", "palm",
    "peace", "peace_inverted", "point",
    "rock", "stop", "stop_inverted",
    "three", "three2", "two_up", "two_up_inverted",
]

# ---------------------------------------------------------------------------
# Gesture ID mappings (canonical 18 gestures)
# ---------------------------------------------------------------------------

GESTURE_IDS: list[str] = [
    "subject_i",
    "subject_you",
    "subject_he",
    "subject_she",
    "subject_we",
    "subject_they",
    "verb_want",
    "verb_eat",
    "verb_see",
    "verb_grab",
    "verb_drink",
    "verb_go",
    "verb_stop",
    "object_food",
    "object_water",
    "object_book",
    "object_apple",
    "object_ball",
    "object_house",
]

GESTURE_LABEL_MAP: dict[str, str] = {
    "subject_i":    "I",
    "subject_you":  "You",
    "subject_he":   "He",
    "subject_she":  "She",
    "subject_we":   "We",
    "subject_they": "They",
    "verb_want":    "Want",
    "verb_eat":     "Eat",
    "verb_see":     "See",
    "verb_grab":    "Grab",
    "verb_drink":   "Drink",
    "verb_go":      "Go",
    "verb_stop":    "Stop",
    "object_food":  "Food",
    "object_water": "Water",
    "object_book":  "Book",
    "object_apple": "Apple",
    "object_ball":  "Ball",
    "object_house": "House",
}

NUM_GESTURE_CLASSES = len(GESTURE_IDS)

# Map from integer class index ↔ gesture string ID
ID_TO_IDX: dict[str, int] = {gid: i for i, gid in enumerate(GESTURE_IDS)}
IDX_TO_ID: dict[int, str] = {i: gid for i, gid in enumerate(GESTURE_IDS)}

# ---------------------------------------------------------------------------
# MediaPipe landmark configuration
# ---------------------------------------------------------------------------

NUM_HAND_LANDMARKS = 21
HAND_LANDMARK_DIMS = 3  # x, y, z
HAND_FEATURE_DIM = NUM_HAND_LANDMARKS * HAND_LANDMARK_DIMS  # 63

NUM_FACE_LANDMARKS = 478
FACE_LANDMARK_DIMS = 3
FACE_FEATURE_DIM = NUM_FACE_LANDMARKS * FACE_LANDMARK_DIMS  # 1434

# ---------------------------------------------------------------------------
# Engineered feature names (computed in preprocess.py)
# ---------------------------------------------------------------------------

FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]

# Landmark indices for each fingertip (MediaPipe hand model)
FINGERTIP_INDICES = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20,
}

FINGER_MCP_INDICES = {
    "thumb": 2,
    "index": 5,
    "middle": 9,
    "ring": 13,
    "pinky": 17,
}

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

# Random Forest (tuned for >90% accuracy)
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = None
RF_PARAM_GRID: dict = {
    "n_estimators": [200, 300, 500],
    "max_depth": [None, 25, 35, 50],
    "min_samples_split": [2, 3, 5],
    "min_samples_leaf": [1, 2],
    "max_features": ["sqrt", "log2", None],
}

# MLP (PyTorch) — wider architecture for better representation
MLP_HIDDEN_LAYERS = [256, 128, 64]
MLP_DROPOUT = 0.25
MLP_LEARNING_RATE = 5e-4
MLP_EPOCHS = 150
MLP_BATCH_SIZE = 64
MLP_EARLY_STOPPING_PATIENCE = 15

# Data splits
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# Webcam collection
WEBCAM_FRAMES_PER_GESTURE = 30
WEBCAM_MIN_SAMPLES = 540  # 30 × 18

# ---------------------------------------------------------------------------
# Emission matrix states
# ---------------------------------------------------------------------------

UASAM_STATES = ["silent", "vocalization", "speech_attempt", "clear_speech"]
GAZE_STATES = ["direct", "averted", "tracking_hand", "tracking_face"]
MODALITY_WEIGHTS_GRID = {
    "w_visual": [0.4, 0.5, 0.6],
    "w_acoustic": [0.1, 0.2, 0.3],
    "w_gaze": [0.1, 0.2, 0.3],
}

# ---------------------------------------------------------------------------
# Research / logging
# ---------------------------------------------------------------------------

EXPERIMENT_REGISTRY_PATH = LOGS_DIR / "experiment_registry.json"
PROJECT_NAME = "MLAF - Multimodal Language Acquisition Framework"
INSTITUTION = "Research collaboration — IIT Patna & IIT Kharagpur"
