"""MLAF Training Pipeline — Gesture Classifier Training.

Two-stage training:
  Stage A: scikit-learn Random Forest + Gradient Boosted Trees (baseline)
  Stage B: PyTorch MLP (if RF < target accuracy)

Produces:
  - Trained model artifacts in models/
  - Detailed training log JSON in logs/

Usage:
    python -m training.train_gesture_classifier
    python training/train_gesture_classifier.py
"""

from __future__ import annotations

import datetime
import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

from .config import (
    EXPERIMENT_REGISTRY_PATH,
    GESTURE_IDS,
    ID_TO_IDX,
    IDX_TO_ID,
    INSTITUTION,
    LOGS_DIR,
    MLP_BATCH_SIZE,
    MLP_DROPOUT,
    MLP_EARLY_STOPPING_PATIENCE,
    MLP_EPOCHS,
    MLP_HIDDEN_LAYERS,
    MLP_LEARNING_RATE,
    MODELS_DIR,
    NUM_GESTURE_CLASSES,
    PROJECT_NAME,
    RANDOM_SEED,
    RF_N_ESTIMATORS,
    RF_PARAM_GRID,
    SPLITS_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Target accuracy — switch to MLP if RF is below this
RF_TARGET_ACCURACY = 0.90


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a data split CSV and return (X, y) arrays."""
    path = SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}. Run preprocess.py first.")

    df = pd.read_csv(path)

    # Feature columns = all numeric except class_idx, gesture_id, source, etc.
    meta_cols = {"gesture_id", "gesture_label_raw", "source", "class_idx", "frame"}
    feature_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in (np.float64, np.float32, np.int64)]

    X = df[feature_cols].values.astype(np.float32)
    y = df["gesture_id"].map(ID_TO_IDX).values.astype(np.int64)

    # Handle NaN
    nan_mask = np.isnan(X)
    if nan_mask.any():
        logger.warning("  Found %d NaN values in %s, replacing with 0", nan_mask.sum(), name)
        X = np.nan_to_num(X, nan=0.0)

    return X, y


def _get_feature_names() -> list[str]:
    """Get feature column names from the train split."""
    path = SPLITS_DIR / "train.csv"
    df = pd.read_csv(path, nrows=0)
    meta_cols = {"gesture_id", "gesture_label_raw", "source", "class_idx", "frame"}
    return [c for c in df.columns if c not in meta_cols and c not in ("gesture_id",)]


# ---------------------------------------------------------------------------
# Hardware / environment info
# ---------------------------------------------------------------------------

def _system_info() -> dict:
    info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not installed"
        info["cuda_available"] = False

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        info["git_hash"] = result.stdout.strip() if result.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        info["git_hash"] = "unknown"

    return info


# ---------------------------------------------------------------------------
# Experiment logging
# ---------------------------------------------------------------------------

def _new_experiment_id() -> str:
    """Generate experiment ID like EXP_001, EXP_002, ..."""
    if EXPERIMENT_REGISTRY_PATH.exists():
        with open(EXPERIMENT_REGISTRY_PATH) as f:
            registry = json.load(f)
        n = len(registry.get("experiments", []))
    else:
        n = 0
    return f"EXP_{n + 1:03d}"


def _register_experiment(exp_id: str, description: str, log_file: str, status: str) -> None:
    """Add experiment to the master registry."""
    if EXPERIMENT_REGISTRY_PATH.exists():
        with open(EXPERIMENT_REGISTRY_PATH) as f:
            registry = json.load(f)
    else:
        registry = {
            "project": PROJECT_NAME,
            "institution": INSTITUTION,
            "experiments": [],
        }

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
    """Save per-run training log to JSON."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    filename = f"training_log_{timestamp}_{exp_id}.json"
    path = LOGS_DIR / filename
    with open(path, "w") as f:
        json.dump(log, f, indent=2, default=str)
    logger.info("Training log saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Stage A: scikit-learn classifiers
# ---------------------------------------------------------------------------

def train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[RandomForestClassifier, dict]:
    """Train Random Forest with GridSearchCV hyperparameter optimization."""
    logger.info("=== Stage A: Random Forest ===")

    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    logger.info("Running GridSearchCV (%d parameter combinations) …",
                np.prod([len(v) for v in RF_PARAM_GRID.values()]))

    grid = GridSearchCV(
        rf,
        RF_PARAM_GRID,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    t0 = time.perf_counter()
    grid.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    best_rf: RandomForestClassifier = grid.best_estimator_

    # Evaluate on validation set
    y_val_pred = best_rf.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )

    cm = confusion_matrix(y_val, y_val_pred, labels=list(range(NUM_GESTURE_CLASSES)))

    # Feature importances
    feature_importances = best_rf.feature_importances_.tolist()

    metrics = {
        "model": "RandomForest",
        "best_params": grid.best_params_,
        "train_time_sec": train_time,
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1,
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
        "feature_importances": feature_importances,
        "cv_results_summary": {
            "mean_test_score": float(grid.cv_results_["mean_test_score"].max()),
            "std_test_score": float(
                grid.cv_results_["std_test_score"][grid.cv_results_["mean_test_score"].argmax()]
            ),
        },
    }

    logger.info("  RF val accuracy: %.4f  |  F1 macro: %.4f", val_acc, val_f1)
    logger.info("  Best params: %s", grid.best_params_)
    return best_rf, metrics


def train_gradient_boosting(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[GradientBoostingClassifier, dict]:
    """Train Gradient Boosted Trees as secondary baseline."""
    logger.info("=== Stage A (alt): Gradient Boosted Trees ===")

    gbt = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=RANDOM_SEED,
    )

    t0 = time.perf_counter()
    gbt.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    y_val_pred = gbt.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average="macro")

    logger.info("  GBT val accuracy: %.4f  |  F1 macro: %.4f", val_acc, val_f1)

    metrics = {
        "model": "GradientBoostedTrees",
        "train_time_sec": train_time,
        "val_accuracy": val_acc,
        "val_f1_macro": val_f1,
    }

    return gbt, metrics


# ---------------------------------------------------------------------------
# Stage B: PyTorch MLP
# ---------------------------------------------------------------------------

def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> tuple[object, dict]:
    """Train PyTorch MLP gesture classifier.

    Architecture: input → 128 → 64 → 18 (ReLU, dropout 0.3).
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        logger.warning("PyTorch not installed — skipping MLP training")
        return None, {"model": "MLP", "error": "torch not installed"}

    logger.info("=== Stage B: PyTorch MLP ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("  Device: %s", device)

    input_dim = X_train.shape[1]

    # Build model
    layers = []
    prev_dim = input_dim
    for hidden_dim in MLP_HIDDEN_LAYERS:
        layers.extend([
            nn.Linear(prev_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(MLP_DROPOUT),
        ])
        prev_dim = hidden_dim
    layers.append(nn.Linear(prev_dim, NUM_GESTURE_CLASSES))

    model = nn.Sequential(*layers).to(device)
    logger.info("  Model: %s", model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=MLP_LEARNING_RATE)

    # Data loaders
    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    val_ds = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long),
    )
    train_loader = DataLoader(train_ds, batch_size=MLP_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=MLP_BATCH_SIZE)

    # Training loop
    history: dict[str, list[float]] = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
    }
    best_val_acc = 0.0
    patience_counter = 0
    best_state = None

    t0 = time.perf_counter()

    for epoch in range(MLP_EPOCHS):
        # Train
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(y_batch)
            train_correct += (logits.argmax(1) == y_batch).sum().item()
            train_total += len(y_batch)

        # Validate
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

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                "  Epoch %3d/%d  |  train_loss=%.4f  train_acc=%.4f  |  val_loss=%.4f  val_acc=%.4f",
                epoch + 1, MLP_EPOCHS, train_loss, train_acc, val_loss, val_acc,
            )

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= MLP_EARLY_STOPPING_PATIENCE:
                logger.info("  Early stopping at epoch %d (patience=%d)", epoch + 1, MLP_EARLY_STOPPING_PATIENCE)
                break

    train_time = time.perf_counter() - t0

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Final validation metrics
    model.eval()
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    y_val_pred = np.array(all_preds)
    y_val_probs = np.array(all_probs)

    val_acc_final = accuracy_score(y_val, y_val_pred)
    val_f1_final = f1_score(y_val, y_val_pred, average="macro")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_val_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )
    cm = confusion_matrix(y_val, y_val_pred, labels=list(range(NUM_GESTURE_CLASSES)))

    metrics = {
        "model": "MLP",
        "architecture": f"{input_dim} → {' → '.join(map(str, MLP_HIDDEN_LAYERS))} → {NUM_GESTURE_CLASSES}",
        "dropout": MLP_DROPOUT,
        "learning_rate": MLP_LEARNING_RATE,
        "batch_size": MLP_BATCH_SIZE,
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
        "val_probabilities": y_val_probs.tolist(),
    }

    logger.info("  MLP val accuracy: %.4f  |  F1 macro: %.4f", val_acc_final, val_f1_final)

    return model, metrics


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> dict:
    """Run full training pipeline. Returns training log dict."""
    logger.info("MLAF Training Pipeline — Gesture Classifier")

    exp_id = _new_experiment_id()
    logger.info("Experiment: %s", exp_id)

    # Load data
    X_train, y_train = _load_split("train")
    X_val, y_val = _load_split("val")
    X_test, y_test = _load_split("test")

    logger.info("Data: train=%d, val=%d, test=%d, features=%d",
                X_train.shape[0], X_val.shape[0], X_test.shape[0], X_train.shape[1])

    # Dataset stats
    dataset_stats = {
        "train_samples": X_train.shape[0],
        "val_samples": X_val.shape[0],
        "test_samples": X_test.shape[0],
        "num_features": X_train.shape[1],
        "num_classes": NUM_GESTURE_CLASSES,
        "class_distribution_train": {
            IDX_TO_ID.get(i, f"class_{i}"): int((y_train == i).sum())
            for i in range(NUM_GESTURE_CLASSES)
        },
    }

    # Initialize training log
    training_log: dict = {
        "experiment_id": exp_id,
        "project": PROJECT_NAME,
        "institution": INSTITUTION,
        "timestamp": datetime.datetime.now().isoformat(),
        "dataset": dataset_stats,
        "system_info": _system_info(),
        "stages": {},
    }

    # ---- Stage A: Random Forest ----
    rf_model, rf_metrics = train_random_forest(X_train, y_train, X_val, y_val)
    training_log["stages"]["random_forest"] = rf_metrics

    # Save RF model
    rf_path = MODELS_DIR / f"gesture_rf_{exp_id}.joblib"
    joblib.dump(rf_model, rf_path)
    logger.info("RF model saved: %s", rf_path)

    # Also train GBT for comparison
    gbt_model, gbt_metrics = train_gradient_boosting(X_train, y_train, X_val, y_val)
    training_log["stages"]["gradient_boosted_trees"] = gbt_metrics

    gbt_path = MODELS_DIR / f"gesture_gbt_{exp_id}.joblib"
    joblib.dump(gbt_model, gbt_path)

    # ---- Stage B: MLP (if RF below target) ----
    best_model = rf_model
    best_model_name = "RandomForest"

    if rf_metrics["val_accuracy"] < RF_TARGET_ACCURACY:
        logger.info("RF accuracy %.4f < target %.4f — training MLP …",
                     rf_metrics["val_accuracy"], RF_TARGET_ACCURACY)
        mlp_model, mlp_metrics = train_mlp(X_train, y_train, X_val, y_val)
        training_log["stages"]["mlp"] = mlp_metrics

        if mlp_model is not None:
            # Save PyTorch model
            try:
                import torch
                mlp_path = MODELS_DIR / f"gesture_mlp_{exp_id}.pt"
                torch.save(mlp_model.state_dict(), mlp_path)
                logger.info("MLP model saved: %s", mlp_path)

                if mlp_metrics.get("val_accuracy", 0) > rf_metrics["val_accuracy"]:
                    best_model = mlp_model
                    best_model_name = "MLP"
            except ImportError:
                pass
    else:
        logger.info("RF accuracy %.4f ≥ target %.4f — skipping MLP", rf_metrics["val_accuracy"], RF_TARGET_ACCURACY)

    # ---- Final test evaluation with best model ----
    logger.info("=== Final Test Evaluation (%s) ===", best_model_name)

    if best_model_name == "RandomForest":
        y_test_pred = best_model.predict(X_test)
        try:
            y_test_probs = best_model.predict_proba(X_test)
        except Exception:
            y_test_probs = None
    else:
        import torch
        best_model.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32)
            logits = best_model(X_t)
            y_test_pred = logits.argmax(1).numpy()
            y_test_probs = torch.softmax(logits, dim=1).numpy()

    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average="macro")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_test_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )
    test_cm = confusion_matrix(y_test, y_test_pred, labels=list(range(NUM_GESTURE_CLASSES)))

    test_metrics = {
        "best_model": best_model_name,
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

    if y_test_probs is not None:
        test_metrics["test_probabilities"] = y_test_probs.tolist()

    training_log["test_evaluation"] = test_metrics
    training_log["best_model"] = best_model_name
    training_log["model_artifacts"] = {
        "random_forest": str(rf_path),
        "gradient_boosted_trees": str(gbt_path),
    }

    logger.info("  Test accuracy: %.4f  |  F1 macro: %.4f", test_acc, test_f1)

    # Save training log
    log_path = _save_training_log(training_log, exp_id)
    _register_experiment(
        exp_id,
        f"Gesture classifier ({best_model_name}) — test acc {test_acc:.4f}",
        str(log_path),
        "completed",
    )

    logger.info("Training complete. Experiment: %s", exp_id)
    return training_log


if __name__ == "__main__":
    result = main()
    print(f"\nBest model: {result['best_model']}")
    print(f"Test accuracy: {result['test_evaluation']['test_accuracy']:.4f}")
    print(f"Test F1 macro: {result['test_evaluation']['test_f1_macro']:.4f}")
    sys.exit(0)
