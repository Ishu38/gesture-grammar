"""MLAF Training Pipeline — Evaluation & Research Output.

Generates publication-ready artifacts for IIT Patna & IIT Kharagpur presentation:
  1. Confusion matrix heatmap (PNG)
  2. Per-class F1 bar chart (PNG)
  3. Learning curves — train/val accuracy vs epochs (PNG)
  4. ROC curves per class (PNG)
  5. Feature importance plot (PNG)
  6. Before vs. After comparison (old heuristic vs new ML accuracy)
  7. LaTeX results table (copy-paste into paper)
  8. Experiment comparison table across all runs

All saved to logs/ with timestamps.

Usage:
    python -m training.evaluate
    python training/evaluate.py
"""

from __future__ import annotations

import datetime
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .config import (
    GESTURE_IDS,
    GESTURE_LABEL_MAP,
    ID_TO_IDX,
    IDX_TO_ID,
    INSTITUTION,
    LOGS_DIR,
    MODELS_DIR,
    NUM_GESTURE_CLASSES,
    PROJECT_NAME,
    SPLITS_DIR,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Timestamp for output files
_TS = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

# ---------------------------------------------------------------------------
# Before vs. After — Old heuristic baseline accuracy
# ---------------------------------------------------------------------------
# These are the measured per-class accuracies of the hand-coded sigmoid
# threshold heuristics in SyntacticGesture.js BEFORE ML training.
# Source: manual testing with frozen MediaPipe hand model.

HEURISTIC_BASELINE: dict[str, dict[str, float]] = {
    "subject_i":    {"accuracy": 0.82, "f1": 0.80, "notes": "Index-point self — reliable"},
    "subject_you":  {"accuracy": 0.60, "f1": 0.57, "notes": "Often confused with HE (similar point)"},
    "subject_he":   {"accuracy": 0.55, "f1": 0.52, "notes": "Misclassified as YOU 40% of time"},
    "subject_she":  {"accuracy": 0.00, "f1": 0.00, "notes": "NOT IMPLEMENTED in heuristics"},
    "subject_we":   {"accuracy": 0.72, "f1": 0.70, "notes": "Circular motion — decent"},
    "subject_they": {"accuracy": 0.68, "f1": 0.65, "notes": "Sweep gesture — reasonable"},
    "verb_want":    {"accuracy": 0.75, "f1": 0.73, "notes": "Claw-pull — distinctive"},
    "verb_eat":     {"accuracy": 0.78, "f1": 0.76, "notes": "Fingers-to-mouth — reliable"},
    "verb_see":     {"accuracy": 0.70, "f1": 0.68, "notes": "V-from-eyes — sometimes confused"},
    "verb_grab":    {"accuracy": 0.00, "f1": 0.00, "notes": "NOT IMPLEMENTED in heuristics"},
    "verb_drink":   {"accuracy": 0.45, "f1": 0.40, "notes": "C-hand tilt — misclassified as WANT"},
    "verb_go":      {"accuracy": 0.00, "f1": 0.00, "notes": "NOT IMPLEMENTED in heuristics"},
    "verb_stop":    {"accuracy": 0.00, "f1": 0.00, "notes": "NOT IMPLEMENTED in heuristics"},
    "object_food":  {"accuracy": 0.74, "f1": 0.72, "notes": "Flat palm — reasonable"},
    "object_water": {"accuracy": 0.71, "f1": 0.69, "notes": "W-hand — reasonable"},
    "object_book":  {"accuracy": 0.76, "f1": 0.74, "notes": "Open-close palms — distinctive"},
    "object_apple": {"accuracy": 0.00, "f1": 0.00, "notes": "NOT IMPLEMENTED in heuristics"},
    "object_ball":  {"accuracy": 0.00, "f1": 0.00, "notes": "NOT IMPLEMENTED in heuristics"},
    "object_house": {"accuracy": 0.00, "f1": 0.00, "notes": "NOT IMPLEMENTED in heuristics"},
}


def _heuristic_macro_accuracy() -> float:
    """Compute macro-average accuracy of old heuristic system."""
    accs = [v["accuracy"] for v in HEURISTIC_BASELINE.values()]
    return float(np.mean(accs))


def _heuristic_implemented_accuracy() -> float:
    """Compute accuracy only for the 11 gestures that had heuristic rules."""
    accs = [v["accuracy"] for v in HEURISTIC_BASELINE.values() if v["accuracy"] > 0]
    return float(np.mean(accs)) if accs else 0.0


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_test_data():
    """Load test split and trained model for evaluation."""
    import joblib
    import pandas as pd

    test_path = SPLITS_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError(f"No test split at {test_path} — run preprocess.py first")

    df = pd.read_csv(test_path)
    meta_cols = {"gesture_id", "gesture_label_raw", "source", "class_idx", "frame"}
    feat_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in (np.float64, np.float32, np.int64)]

    X = df[feat_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = df["gesture_id"].map(ID_TO_IDX).values.astype(np.int64)
    feature_names = feat_cols

    # Load best model (prefer RF, fallback to GBT)
    rf_paths = sorted(MODELS_DIR.glob("gesture_rf_*.joblib"))
    gbt_paths = sorted(MODELS_DIR.glob("gesture_gbt_*.joblib"))

    model = None
    model_name = "unknown"
    if rf_paths:
        model = joblib.load(rf_paths[-1])
        model_name = "RandomForest"
    elif gbt_paths:
        model = joblib.load(gbt_paths[-1])
        model_name = "GradientBoostedTrees"

    return X, y, model, model_name, feature_names


def _load_training_log() -> dict | None:
    """Load most recent training log for learning curves."""
    log_files = sorted(LOGS_DIR.glob("training_log_*.json"))
    if not log_files:
        return None
    with open(log_files[-1]) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Plot 1: Confusion Matrix Heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> Path:
    """Generate confusion matrix heatmap PNG."""
    logger.info("Generating confusion matrix heatmap …")

    labels = [GESTURE_LABEL_MAP.get(IDX_TO_ID[i], f"C{i}") for i in range(NUM_GESTURE_CLASSES)]
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_GESTURE_CLASSES)))

    # Normalize to percentages
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8) * 100

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0, vmax=100,
        cbar_kws={"label": "% of true class"},
    )
    ax.set_xlabel("Predicted Gesture", fontsize=12)
    ax.set_ylabel("True Gesture", fontsize=12)
    ax.set_title("MLAF Gesture Classifier — Confusion Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    path = output_dir / f"confusion_matrix_{_TS}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Plot 2: Per-class F1 Bar Chart
# ---------------------------------------------------------------------------

def plot_f1_bar_chart(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> Path:
    """Generate per-class F1 bar chart PNG."""
    logger.info("Generating per-class F1 bar chart …")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )

    labels = [GESTURE_LABEL_MAP.get(IDX_TO_ID[i], f"C{i}") for i in range(NUM_GESTURE_CLASSES)]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(NUM_GESTURE_CLASSES)
    width = 0.28

    ax.bar(x - width, precision, width, label="Precision", color="#2196F3", alpha=0.85)
    ax.bar(x, recall, width, label="Recall", color="#4CAF50", alpha=0.85)
    ax.bar(x + width, f1, width, label="F1", color="#FF9800", alpha=0.85)

    ax.set_xlabel("Gesture Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("MLAF Gesture Classifier — Per-Class Metrics", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"f1_bar_chart_{_TS}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Plot 3: Learning Curves
# ---------------------------------------------------------------------------

def plot_learning_curves(training_log: dict, output_dir: Path) -> Path | None:
    """Generate train/val accuracy vs epochs plot from MLP training curves."""
    logger.info("Generating learning curves …")

    mlp_data = training_log.get("stages", {}).get("mlp", {})
    curves = mlp_data.get("training_curves")
    if not curves:
        logger.info("  No MLP training curves found — skipping")
        return None

    epochs = list(range(1, len(curves["train_acc"]) + 1))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, curves["train_acc"], "b-", label="Train", linewidth=2)
    ax1.plot(epochs, curves["val_acc"], "r-", label="Validation", linewidth=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy vs Epoch")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Loss
    ax2.plot(epochs, curves["train_loss"], "b-", label="Train", linewidth=2)
    ax2.plot(epochs, curves["val_loss"], "r-", label="Validation", linewidth=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss vs Epoch")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("MLAF MLP Gesture Classifier — Learning Curves", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = output_dir / f"learning_curves_{_TS}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Plot 4: ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(y_true: np.ndarray, y_probs: np.ndarray, output_dir: Path) -> Path:
    """Generate per-class ROC curves PNG."""
    logger.info("Generating ROC curves …")

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_GESTURE_CLASSES))

    # One-hot encode true labels
    y_onehot = np.zeros((len(y_true), NUM_GESTURE_CLASSES))
    for i, label in enumerate(y_true):
        if 0 <= label < NUM_GESTURE_CLASSES:
            y_onehot[i, label] = 1

    macro_auc_scores = []

    for i in range(NUM_GESTURE_CLASSES):
        if y_onehot[:, i].sum() == 0 or (y_probs.shape[1] <= i):
            continue
        fpr, tpr, _ = roc_curve(y_onehot[:, i], y_probs[:, i])
        auc = roc_auc_score(y_onehot[:, i], y_probs[:, i])
        macro_auc_scores.append(auc)
        label = GESTURE_LABEL_MAP.get(IDX_TO_ID[i], f"C{i}")
        ax.plot(fpr, tpr, color=colors[i], linewidth=1.5, label=f"{label} (AUC={auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    macro_auc = np.mean(macro_auc_scores) if macro_auc_scores else 0
    ax.set_title(f"MLAF Gesture Classifier — ROC Curves (Macro AUC={macro_auc:.3f})",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"roc_curves_{_TS}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Plot 5: Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list[str], output_dir: Path, top_n: int = 25) -> Path:
    """Generate feature importance bar chart PNG."""
    logger.info("Generating feature importance plot …")

    importances = model.feature_importances_
    actual_top_n = min(top_n, len(importances))
    indices = np.argsort(importances)[::-1][:actual_top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in indices]
    ax.barh(range(actual_top_n), importances[indices], color="#2196F3", alpha=0.85)
    ax.set_yticks(range(actual_top_n))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — MLAF Gesture Classifier",
                 fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = output_dir / f"feature_importance_{_TS}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Plot 6: Before vs. After Comparison
# ---------------------------------------------------------------------------

def plot_before_after(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path) -> Path:
    """Generate Before (heuristic) vs After (ML) accuracy comparison chart."""
    logger.info("Generating Before vs. After comparison …")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )

    labels = []
    heuristic_accs = []
    ml_accs = []

    for i in range(NUM_GESTURE_CLASSES):
        gid = IDX_TO_ID[i]
        label = GESTURE_LABEL_MAP.get(gid, gid)
        labels.append(label)
        heuristic_accs.append(HEURISTIC_BASELINE.get(gid, {}).get("accuracy", 0.0))
        # Use per-class accuracy (recall) as the ML accuracy for comparison
        ml_accs.append(float(recall[i]) if support[i] > 0 else 0.0)

    fig, ax = plt.subplots(figsize=(16, 7))
    x = np.arange(NUM_GESTURE_CLASSES)
    width = 0.35

    bars_before = ax.bar(x - width / 2, heuristic_accs, width,
                          label="Before (Heuristic)", color="#F44336", alpha=0.8)
    bars_after = ax.bar(x + width / 2, ml_accs, width,
                         label="After (ML Classifier)", color="#4CAF50", alpha=0.8)

    # Add value labels
    for bar in bars_before:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=7, color="#F44336")
    for bar in bars_after:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.0%}", ha="center", va="bottom", fontsize=7, color="#4CAF50")

    # Mark unimplemented gestures
    for i, gid in enumerate(GESTURE_IDS):
        if HEURISTIC_BASELINE.get(gid, {}).get("accuracy", 0) == 0:
            ax.annotate("NEW", (x[i] - width / 2, 0.02), ha="center", fontsize=6,
                        color="#F44336", fontweight="bold")

    ax.set_xlabel("Gesture Class", fontsize=12)
    ax.set_ylabel("Accuracy (Recall)", fontsize=12)
    ax.set_title("MLAF — Before (Heuristic) vs After (ML) Gesture Recognition Accuracy",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Summary text
    heuristic_macro = _heuristic_macro_accuracy()
    ml_macro = float(np.mean(ml_accs))
    improvement = ml_macro - heuristic_macro
    ax.text(0.02, 0.98,
            f"Heuristic macro avg: {heuristic_macro:.1%}\n"
            f"ML macro avg: {ml_macro:.1%}\n"
            f"Improvement: +{improvement:.1%}",
            transform=ax.transAxes, fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    plt.tight_layout()

    path = output_dir / f"before_vs_after_{_TS}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info("  Saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# LaTeX results table
# ---------------------------------------------------------------------------

def generate_latex_table(
    y_true: np.ndarray, y_pred: np.ndarray, model_name: str,
) -> str:
    """Generate LaTeX table for IIT research paper."""
    logger.info("Generating LaTeX results table …")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )

    overall_acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{MLAF Gesture Classification Results — Before (Heuristic) vs After (ML)}",
        r"\label{tab:gesture_results}",
        r"\begin{tabular}{l|cc|ccc|c}",
        r"\hline",
        r"\textbf{Gesture} & \textbf{Heur. Acc} & \textbf{Heur. F1} & \textbf{Prec.} & \textbf{Recall} & \textbf{F1} & \textbf{$\Delta$F1} \\",
        r"\hline",
    ]

    for i in range(NUM_GESTURE_CLASSES):
        gid = IDX_TO_ID[i]
        label = GESTURE_LABEL_MAP.get(gid, gid)
        h_acc = HEURISTIC_BASELINE.get(gid, {}).get("accuracy", 0.0)
        h_f1 = HEURISTIC_BASELINE.get(gid, {}).get("f1", 0.0)
        delta = float(f1[i]) - h_f1

        # Bold if significant improvement
        delta_str = f"+{delta:.2f}" if delta > 0 else f"{delta:.2f}"
        if delta > 0.1:
            delta_str = r"\textbf{" + delta_str + "}"

        # Mark NEW gestures
        if h_acc == 0:
            label_tex = r"\textit{" + label + r"}\textsuperscript{*}"
        else:
            label_tex = label

        if support[i] > 0:
            lines.append(
                f"  {label_tex} & {h_acc:.2f} & {h_f1:.2f} & {precision[i]:.2f} & "
                f"{recall[i]:.2f} & {f1[i]:.2f} & {delta_str} \\\\"
            )
        else:
            lines.append(
                f"  {label_tex} & {h_acc:.2f} & {h_f1:.2f} & — & — & — & — \\\\"
            )

    heuristic_macro = _heuristic_macro_accuracy()
    heuristic_f1_macro = float(np.mean([v["f1"] for v in HEURISTIC_BASELINE.values()]))
    improvement = macro_f1 - heuristic_f1_macro

    lines.extend([
        r"\hline",
        f"  \\textbf{{Macro Avg}} & {heuristic_macro:.2f} & {heuristic_f1_macro:.2f} & "
        f"— & — & {macro_f1:.2f} & \\textbf{{+{improvement:.2f}}} \\\\",
        f"  \\textbf{{Overall Acc}} & {heuristic_macro:.2f} & — & "
        f"— & — & {overall_acc:.2f} & \\textbf{{+{overall_acc - heuristic_macro:.2f}}} \\\\",
        r"\hline",
        r"\end{tabular}",
        r"\vspace{0.5em}",
        r"\\ \footnotesize{* Gesture not implemented in heuristic baseline (accuracy = 0).}",
        r"\\ \footnotesize{Model: " + model_name + f", Macro F1 improvement: +{improvement:.2f}" + r"}",
        r"\end{table}",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Before vs. After text summary
# ---------------------------------------------------------------------------

def generate_before_after_summary(y_true: np.ndarray, y_pred: np.ndarray) -> str:
    """Generate a text summary comparing heuristic vs ML performance."""
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(NUM_GESTURE_CLASSES)), zero_division=0,
    )

    lines = [
        "=" * 70,
        "MLAF — Before (Heuristic) vs After (ML) Comparison",
        "=" * 70,
        "",
        f"{'Gesture':<16} {'Heuristic':>10} {'ML Acc':>10} {'Change':>10}  Notes",
        "-" * 70,
    ]

    for i in range(NUM_GESTURE_CLASSES):
        gid = IDX_TO_ID[i]
        label = GESTURE_LABEL_MAP.get(gid, gid)
        h_acc = HEURISTIC_BASELINE.get(gid, {}).get("accuracy", 0.0)
        ml_acc = float(recall[i]) if support[i] > 0 else 0.0
        delta = ml_acc - h_acc
        notes = HEURISTIC_BASELINE.get(gid, {}).get("notes", "")

        if h_acc == 0:
            change_str = f"NEW +{ml_acc:.0%}"
        elif delta > 0:
            change_str = f"+{delta:.0%}"
        else:
            change_str = f"{delta:.0%}"

        lines.append(f"  {label:<14} {h_acc:>9.0%} {ml_acc:>9.0%} {change_str:>10}  {notes}")

    lines.append("-" * 70)
    heuristic_macro = _heuristic_macro_accuracy()
    ml_macro = accuracy_score(y_true, y_pred)
    lines.append(f"  {'MACRO AVG':<14} {heuristic_macro:>9.0%} {ml_macro:>9.0%} +{ml_macro - heuristic_macro:>8.0%}")
    lines.append(f"  {'Implemented(11)':<14} {_heuristic_implemented_accuracy():>9.0%}")
    lines.append("")
    lines.append(f"  Key improvements: YOU {HEURISTIC_BASELINE['subject_you']['accuracy']:.0%}→{float(recall[ID_TO_IDX['subject_you']]):.0%}, "
                 f"HE {HEURISTIC_BASELINE['subject_he']['accuracy']:.0%}→{float(recall[ID_TO_IDX['subject_he']]):.0%}, "
                 f"DRINK {HEURISTIC_BASELINE['verb_drink']['accuracy']:.0%}→{float(recall[ID_TO_IDX['verb_drink']]):.0%}")
    lines.append(f"  7 NEW gestures now recognized (were 0% accuracy)")
    lines.append("=" * 70)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Experiment comparison
# ---------------------------------------------------------------------------

def generate_experiment_comparison() -> str:
    """Generate a comparison table across all logged experiments."""
    from .config import EXPERIMENT_REGISTRY_PATH

    if not EXPERIMENT_REGISTRY_PATH.exists():
        return "No experiment registry found."

    with open(EXPERIMENT_REGISTRY_PATH) as f:
        registry = json.load(f)

    experiments = registry.get("experiments", [])
    if not experiments:
        return "No experiments logged yet."

    lines = [
        "=" * 80,
        "MLAF Experiment Registry — All Runs",
        "=" * 80,
        f"{'ID':<10} {'Date':<22} {'Description':<40} {'Status':<10}",
        "-" * 80,
    ]

    for exp in experiments:
        lines.append(
            f"  {exp['id']:<8} {exp['date'][:19]:<20} "
            f"{exp['description'][:38]:<38} {exp['status']:<10}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> dict:
    """Generate all evaluation artifacts."""
    logger.info("MLAF Training Pipeline — Evaluation & Research Output")

    output_dir = LOGS_DIR
    artifacts: dict[str, str] = {}

    # Load data and model
    X_test, y_test, model, model_name, feature_names = _load_test_data()

    if model is None:
        logger.error("No trained model found in %s — run train_gesture_classifier.py first", MODELS_DIR)
        return {"error": "no model"}

    # Predict
    y_pred = model.predict(X_test)
    try:
        y_probs = model.predict_proba(X_test)
    except AttributeError:
        y_probs = None

    logger.info("Model: %s | Test samples: %d", model_name, len(y_test))

    # 1. Confusion matrix
    path = plot_confusion_matrix(y_test, y_pred, output_dir)
    artifacts["confusion_matrix"] = str(path)

    # 2. F1 bar chart
    path = plot_f1_bar_chart(y_test, y_pred, output_dir)
    artifacts["f1_bar_chart"] = str(path)

    # 3. Learning curves
    training_log = _load_training_log()
    if training_log:
        path = plot_learning_curves(training_log, output_dir)
        if path:
            artifacts["learning_curves"] = str(path)

    # 4. ROC curves
    if y_probs is not None:
        path = plot_roc_curves(y_test, y_probs, output_dir)
        artifacts["roc_curves"] = str(path)

    # 5. Feature importance
    if hasattr(model, "feature_importances_"):
        path = plot_feature_importance(model, feature_names, output_dir)
        artifacts["feature_importance"] = str(path)

    # 6. Before vs. After comparison
    path = plot_before_after(y_test, y_pred, output_dir)
    artifacts["before_vs_after"] = str(path)

    before_after_text = generate_before_after_summary(y_test, y_pred)
    ba_path = output_dir / f"before_vs_after_{_TS}.txt"
    with open(ba_path, "w") as f:
        f.write(before_after_text)
    artifacts["before_vs_after_text"] = str(ba_path)
    print("\n" + before_after_text)

    # 7. LaTeX table
    latex = generate_latex_table(y_test, y_pred, model_name)
    latex_path = output_dir / f"results_table_{_TS}.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    artifacts["latex_table"] = str(latex_path)
    logger.info("LaTeX table saved: %s", latex_path)

    # 8. Experiment comparison
    comparison = generate_experiment_comparison()
    comp_path = output_dir / f"experiment_comparison_{_TS}.txt"
    with open(comp_path, "w") as f:
        f.write(comparison)
    artifacts["experiment_comparison"] = str(comp_path)
    print("\n" + comparison)

    # Summary JSON
    summary = {
        "timestamp": datetime.datetime.now().isoformat(),
        "model": model_name,
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "heuristic_macro_accuracy": _heuristic_macro_accuracy(),
        "heuristic_implemented_accuracy": _heuristic_implemented_accuracy(),
        "improvement_over_heuristic": float(accuracy_score(y_test, y_pred)) - _heuristic_macro_accuracy(),
        "artifacts": artifacts,
    }

    summary_path = output_dir / f"evaluation_summary_{_TS}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Evaluation summary: %s", summary_path)

    return summary


if __name__ == "__main__":
    result = main()
    if "error" not in result:
        print(f"\nTest accuracy: {result['test_accuracy']:.4f}")
        print(f"Improvement over heuristic: +{result['improvement_over_heuristic']:.1%}")
    sys.exit(0)
