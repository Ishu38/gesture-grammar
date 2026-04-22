"""MLAF Training Pipeline — 1D CNN Gesture Classifier.

Trains a lightweight 1D Convolutional Neural Network on MediaPipe hand
landmarks (21 joints x 3 axes) and exports to ONNX for browser inference
via ONNX Runtime Web.

Architecture:
    Input: (batch, 21, 3) — 21 landmarks as spatial sequence, 3 channels
    → Conv1D(3→32, k=3) → BN → ReLU → Conv1D(32→64, k=3) → BN → ReLU → Pool
    → Conv1D(64→128, k=3) → BN → ReLU → GlobalAvgPool
    → FC(128→64) → ReLU → Dropout → FC(64→19)

Total params: ~50K — runs <1ms on CPU, <100KB ONNX file.

The CNN operates on the raw 21×3 normalized landmark tensor rather than
the 86-feature engineered vector. This lets the convolution filters learn
spatial patterns across neighboring joints (wrist→thumb→index→…) directly,
which is more expressive than hand-crafted inter-finger distances.

Usage:
    python -m training.train_cnn_classifier
    python training/train_cnn_classifier.py
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (
    EXPERIMENT_REGISTRY_PATH,
    GESTURE_IDS,
    ID_TO_IDX,
    IDX_TO_ID,
    INSTITUTION,
    LOGS_DIR,
    MODELS_DIR,
    NUM_GESTURE_CLASSES,
    NUM_HAND_LANDMARKS,
    HAND_LANDMARK_DIMS,
    PROJECT_NAME,
    RANDOM_SEED,
    SPLITS_DIR,
)
from .preprocess import LANDMARK_COLS

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

CNN_LEARNING_RATE = 3e-4
CNN_EPOCHS = 200
CNN_BATCH_SIZE = 64
CNN_EARLY_STOPPING_PATIENCE = 20
CNN_WEIGHT_DECAY = 1e-4

# Data augmentation
AUG_NOISE_STD = 0.01        # Gaussian noise on landmark coords
AUG_SCALE_RANGE = (0.85, 1.15)  # Random scale factor
AUG_ROTATION_DEG = 15       # Random rotation around Z axis


# ---------------------------------------------------------------------------
# Data loading — landmarks only (no engineered features)
# ---------------------------------------------------------------------------

def _load_landmarks(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a split and return raw normalized landmarks as (N, 21, 3) and labels."""
    path = SPLITS_DIR / f"{split_name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}. Run preprocess.py first.")

    df = pd.read_csv(path)

    # Extract only the 63 landmark columns
    lm_data = df[LANDMARK_COLS].values.astype(np.float32)
    X = lm_data.reshape(-1, NUM_HAND_LANDMARKS, HAND_LANDMARK_DIMS)

    y = df["gesture_id"].map(ID_TO_IDX).values.astype(np.int64)

    # Handle NaN
    nan_mask = np.isnan(X)
    if nan_mask.any():
        logger.warning("  %d NaN values in %s, replacing with 0", nan_mask.sum(), split_name)
        X = np.nan_to_num(X, nan=0.0)

    return X, y


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def augment_batch(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply random augmentations to a batch of (B, 21, 3) landmarks.

    Augmentations are rotation-invariant-friendly:
    - Gaussian noise on coordinates
    - Random uniform scaling
    - Random Z-axis rotation (simulates different camera angles)
    """
    B = X.shape[0]
    X_aug = X.copy()

    # 1. Gaussian noise
    X_aug += rng.normal(0, AUG_NOISE_STD, X_aug.shape).astype(np.float32)

    # 2. Random scale
    scales = rng.uniform(*AUG_SCALE_RANGE, size=(B, 1, 1)).astype(np.float32)
    X_aug *= scales

    # 3. Random Z-axis rotation
    angles = rng.uniform(-AUG_ROTATION_DEG, AUG_ROTATION_DEG, size=B)
    angles_rad = np.deg2rad(angles).astype(np.float32)
    cos_a = np.cos(angles_rad)
    sin_a = np.sin(angles_rad)

    x_rot = X_aug[:, :, 0] * cos_a[:, None] - X_aug[:, :, 1] * sin_a[:, None]
    y_rot = X_aug[:, :, 0] * sin_a[:, None] + X_aug[:, :, 1] * cos_a[:, None]
    X_aug[:, :, 0] = x_rot
    X_aug[:, :, 1] = y_rot

    return X_aug


# ---------------------------------------------------------------------------
# CNN Model Definition
# ---------------------------------------------------------------------------

def _build_model():
    """Build the 1D CNN model using PyTorch."""
    import torch
    import torch.nn as nn

    class GestureCNN(nn.Module):
        """Lightweight 1D CNN for hand gesture classification.

        Input: (batch, 21, 3) → permuted to (batch, 3, 21) for Conv1d
        Output: (batch, 19) logits
        """
        def __init__(self, num_classes: int = NUM_GESTURE_CLASSES):
            super().__init__()

            # Conv blocks: 3→32→64→128 with batch norm
            self.features = nn.Sequential(
                # Block 1: (3, 21) → (32, 19)
                nn.Conv1d(HAND_LANDMARK_DIMS, 32, kernel_size=3, padding=0),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),

                # Block 2: (32, 19) → (64, 17)
                nn.Conv1d(32, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2),  # (64, 8)

                # Block 3: (64, 8) → (128, 6)
                nn.Conv1d(64, 128, kernel_size=3, padding=0),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),

                # Global average pooling → (128,)
                nn.AdaptiveAvgPool1d(1),
            )

            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(64, num_classes),
            )

        def forward(self, x):
            # x: (batch, 21, 3) → (batch, 3, 21) for Conv1d
            x = x.permute(0, 2, 1)
            x = self.features(x)
            x = self.classifier(x)
            return x

    return GestureCNN()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_cnn(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[object, dict]:
    """Train 1D CNN gesture classifier with data augmentation."""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.error("PyTorch not installed — cannot train CNN")
        return None, {"model": "CNN", "error": "torch not installed"}

    logger.info("=== Training 1D CNN Gesture Classifier ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Device: %s", device)

    model = _build_model().to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("  Parameters: %d total, %d trainable", num_params, num_trainable)
    logger.info("  Model:\n%s", model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CNN_LEARNING_RATE,
        weight_decay=CNN_WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CNN_EPOCHS)

    # Validation loader (no augmentation)
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    val_loader = DataLoader(val_ds, batch_size=CNN_BATCH_SIZE)

    rng = np.random.default_rng(RANDOM_SEED)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    t0 = time.perf_counter()

    for epoch in range(CNN_EPOCHS):
        # --- Train with augmentation ---
        model.train()
        # Shuffle and augment training data each epoch
        perm = rng.permutation(len(X_train))
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]

        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for start in range(0, len(X_shuffled), CNN_BATCH_SIZE):
            end = min(start + CNN_BATCH_SIZE, len(X_shuffled))
            X_batch_np = augment_batch(X_shuffled[start:end], rng)
            y_batch_np = y_shuffled[start:end]

            X_batch = torch.tensor(X_batch_np, dtype=torch.float32).to(device)
            y_batch = torch.tensor(y_batch_np, dtype=torch.long).to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                val_correct += (logits.argmax(1) == y_batch).sum().item()
                val_total += len(y_batch)

        train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total
        val_loss = val_loss_sum / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "  Epoch %3d/%d  |  loss=%.4f  acc=%.4f  |  val_loss=%.4f  val_acc=%.4f  |  lr=%.6f",
                epoch + 1, CNN_EPOCHS, train_loss, train_acc, val_loss, val_acc, current_lr,
            )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= CNN_EARLY_STOPPING_PATIENCE:
                logger.info("  Early stopping at epoch %d", epoch + 1)
                break

    train_time = time.perf_counter() - t0

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final validation metrics
    model.eval()
    all_preds = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            all_preds.extend(logits.argmax(1).cpu().numpy())

    y_val_pred = np.array(all_preds)

    from sklearn.metrics import (
        accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
    )

    val_acc_final = accuracy_score(y_val, y_val_pred)
    val_f1_final = f1_score(y_val, y_val_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )
    cm = confusion_matrix(y_val, y_val_pred, labels=list(range(NUM_GESTURE_CLASSES)))

    metrics = {
        "model": "CNN_1D",
        "architecture": "Conv1d(3→32→64→128) + GAP + FC(128→64→19)",
        "num_params": num_params,
        "num_trainable_params": num_trainable,
        "augmentation": {
            "noise_std": AUG_NOISE_STD,
            "scale_range": list(AUG_SCALE_RANGE),
            "rotation_deg": AUG_ROTATION_DEG,
        },
        "learning_rate": CNN_LEARNING_RATE,
        "weight_decay": CNN_WEIGHT_DECAY,
        "batch_size": CNN_BATCH_SIZE,
        "epochs_run": len(history["train_loss"]),
        "train_time_sec": train_time,
        "val_accuracy": val_acc_final,
        "val_f1_macro": val_f1_final,
        "best_val_accuracy": best_val_acc,
        "per_class": {
            IDX_TO_ID.get(i, f"class_{i}"): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(NUM_GESTURE_CLASSES)
            if support[i] > 0
        },
        "confusion_matrix": cm.tolist(),
        "training_curves": history,
    }

    logger.info("  CNN val accuracy: %.4f  |  F1 macro: %.4f", val_acc_final, val_f1_final)

    return model, metrics


# ---------------------------------------------------------------------------
# ONNX Export
# ---------------------------------------------------------------------------

def export_to_onnx(model, output_path: Path) -> Path:
    """Export trained CNN to ONNX format for ONNX Runtime Web inference.

    The exported model expects input shape (1, 21, 3) — a single hand's
    normalized landmarks — and outputs (1, 19) logits.
    """
    import torch

    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, NUM_HAND_LANDMARKS, HAND_LANDMARK_DIMS)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=["landmarks"],
        output_names=["logits"],
        dynamic_axes={
            "landmarks": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
    )

    size_kb = output_path.stat().st_size / 1024
    logger.info("  ONNX model saved: %s (%.1f KB)", output_path, size_kb)
    return output_path


def export_metadata_json(metrics: dict, onnx_path: Path) -> Path:
    """Export model metadata as JSON for the browser-side loader.

    This file sits alongside the ONNX model and tells the JS inference
    module the class mapping, input shape, and normalization params.
    """
    meta = {
        "model_type": "CNN_1D",
        "onnx_file": onnx_path.name,
        "input_shape": [1, NUM_HAND_LANDMARKS, HAND_LANDMARK_DIMS],
        "output_shape": [1, NUM_GESTURE_CLASSES],
        "num_classes": NUM_GESTURE_CLASSES,
        "gesture_ids": GESTURE_IDS,
        "class_names": GESTURE_IDS,
        "normalization": "wrist_origin_unit_scale",
        "val_accuracy": metrics.get("val_accuracy"),
        "val_f1_macro": metrics.get("val_f1_macro"),
    }

    meta_path = onnx_path.with_suffix(".json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("  Metadata saved: %s", meta_path)
    return meta_path


# ---------------------------------------------------------------------------
# Experiment logging (reuses infrastructure from train_gesture_classifier)
# ---------------------------------------------------------------------------

def _new_experiment_id() -> str:
    if EXPERIMENT_REGISTRY_PATH.exists():
        with open(EXPERIMENT_REGISTRY_PATH) as f:
            registry = json.load(f)
        n = len(registry.get("experiments", []))
    else:
        n = 0
    return f"EXP_{n + 1:03d}"


def _register_experiment(exp_id: str, description: str, log_file: str, status: str) -> None:
    if EXPERIMENT_REGISTRY_PATH.exists():
        with open(EXPERIMENT_REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = {"project": PROJECT_NAME, "institution": INSTITUTION, "experiments": []}

    registry["experiments"].append({
        "id": exp_id,
        "date": datetime.datetime.now().isoformat(),
        "description": description,
        "log_file": log_file,
        "status": status,
    })

    with open(EXPERIMENT_REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)


def _save_training_log(log: dict, exp_id: str) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"cnn_training_log_{timestamp}_{exp_id}.json"
    path = LOGS_DIR / filename
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    logger.info("Training log saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    """Run CNN training pipeline: load data → train → export ONNX."""
    logger.info("MLAF Training Pipeline — 1D CNN Gesture Classifier")

    exp_id = _new_experiment_id()
    logger.info("Experiment: %s", exp_id)

    # Load data (raw landmarks only, reshaped to 21×3)
    X_train, y_train = _load_landmarks("train")
    X_val, y_val = _load_landmarks("val")
    X_test, y_test = _load_landmarks("test")

    logger.info("Data: train=%d, val=%d, test=%d  |  shape=%s",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape)

    # Class distribution
    dataset_stats = {
        "train_samples": int(X_train.shape[0]),
        "val_samples": int(X_val.shape[0]),
        "test_samples": int(X_test.shape[0]),
        "input_shape": list(X_train.shape[1:]),
        "num_classes": NUM_GESTURE_CLASSES,
        "class_distribution_train": {
            IDX_TO_ID.get(i, f"class_{i}"): int((y_train == i).sum())
            for i in range(NUM_GESTURE_CLASSES)
        },
    }

    training_log = {
        "experiment_id": exp_id,
        "project": PROJECT_NAME,
        "model_type": "CNN_1D",
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset_stats,
        "hyperparameters": {
            "learning_rate": CNN_LEARNING_RATE,
            "epochs": CNN_EPOCHS,
            "batch_size": CNN_BATCH_SIZE,
            "weight_decay": CNN_WEIGHT_DECAY,
            "early_stopping_patience": CNN_EARLY_STOPPING_PATIENCE,
            "augmentation": {
                "noise_std": AUG_NOISE_STD,
                "scale_range": list(AUG_SCALE_RANGE),
                "rotation_deg": AUG_ROTATION_DEG,
            },
        },
    }

    # ---- Train CNN ----
    model, cnn_metrics = train_cnn(X_train, y_train, X_val, y_val)
    training_log["cnn_metrics"] = cnn_metrics

    if model is None:
        training_log["status"] = "failed"
        _save_training_log(training_log, exp_id)
        return training_log

    # ---- Test evaluation ----
    logger.info("=== Final Test Evaluation (CNN) ===")

    import torch
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

    model.eval()
    model.cpu()
    with torch.no_grad():
        X_t = torch.tensor(X_test, dtype=torch.float32)
        logits = model(X_t)
        y_test_pred = logits.argmax(1).numpy()

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )
    test_cm = confusion_matrix(y_test, y_test_pred, labels=list(range(NUM_GESTURE_CLASSES)))

    training_log["test_evaluation"] = {
        "test_accuracy": test_acc,
        "test_f1_macro": test_f1,
        "per_class": {
            IDX_TO_ID.get(i, f"class_{i}"): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(NUM_GESTURE_CLASSES)
            if support[i] > 0
        },
        "confusion_matrix": test_cm.tolist(),
    }

    logger.info("  Test accuracy: %.4f  |  F1 macro: %.4f", test_acc, test_f1)

    # ---- Export to ONNX ----
    logger.info("=== Exporting to ONNX ===")

    onnx_path = MODELS_DIR / f"gesture_cnn_{exp_id}.onnx"
    export_to_onnx(model, onnx_path)

    meta_path = export_metadata_json(cnn_metrics, onnx_path)

    # Also save a "latest" copy for the frontend
    latest_onnx = MODELS_DIR / "gesture_cnn_latest.onnx"
    latest_meta = MODELS_DIR / "gesture_cnn_latest.json"

    import shutil
    shutil.copy2(onnx_path, latest_onnx)
    shutil.copy2(meta_path, latest_meta)
    logger.info("  Latest copies: %s, %s", latest_onnx.name, latest_meta.name)

    # Save PyTorch checkpoint
    pt_path = MODELS_DIR / f"gesture_cnn_{exp_id}.pt"
    torch.save(model.state_dict(), pt_path)

    training_log["model_artifacts"] = {
        "onnx": str(onnx_path),
        "onnx_latest": str(latest_onnx),
        "metadata": str(meta_path),
        "pytorch_checkpoint": str(pt_path),
    }
    training_log["status"] = "completed"

    # ---- Save log & register ----
    log_path = _save_training_log(training_log, exp_id)
    _register_experiment(
        exp_id,
        f"CNN gesture classifier — test acc {test_acc:.4f}, F1 {test_f1:.4f}",
        str(log_path),
        "completed",
    )

    logger.info("Training complete. Experiment: %s", exp_id)

    # Print summary
    print("\n" + "=" * 60)
    print("  CNN Training Summary")
    print("=" * 60)
    print(f"  Experiment:     {exp_id}")
    print(f"  Val accuracy:   {cnn_metrics['val_accuracy']:.4f}")
    print(f"  Test accuracy:  {test_acc:.4f}")
    print(f"  Test F1 macro:  {test_f1:.4f}")
    print(f"  Parameters:     {cnn_metrics['num_params']:,}")
    print(f"  ONNX model:     {onnx_path} ({onnx_path.stat().st_size / 1024:.1f} KB)")
    print(f"  Metadata:       {meta_path}")
    print("=" * 60)

    return training_log


if __name__ == "__main__":
    result = main()
    sys.exit(0 if result.get("status") == "completed" else 1)
