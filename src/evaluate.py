"""
evaluate.py
-----------
Evaluation utilities: ROC/AUC computation, plotting, and results summary.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, classification_report
from pathlib import Path


def compute_roc(y_true: np.ndarray, scores: np.ndarray) -> tuple:
    """
    Compute ROC curve and AUC.

    Parameters
    ----------
    y_true  : ground-truth labels (0 = healthy, 1 = tumor)
    scores  : anomaly scores (higher = more likely tumor)

    Returns
    -------
    fpr, tpr, thresholds, roc_auc
    """
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, thresholds, roc_auc


def plot_roc(
    y_true: np.ndarray,
    scores_dict: dict,
    title: str = "ROC Curves",
    save_path: str = None,
):
    """
    Plot one or more ROC curves on a single figure.

    Parameters
    ----------
    y_true      : ground-truth labels
    scores_dict : {label: scores_array} — one entry per model/condition
    title       : figure title
    save_path   : if provided, saves the figure to this path
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Random baseline")

    for label, scores in scores_dict.items():
        fpr, tpr, _, roc_auc = compute_roc(y_true, scores)
        ax.plot(fpr, tpr, label=f"{label} (AUC = {roc_auc:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()
    return fig


def print_summary(results: dict):
    """
    Print a formatted AUC summary table.

    Parameters
    ----------
    results : {model_name: {"clean": auc_val, "defect": auc_val}}
    """
    print(f"\n{'Model':<35} {'Clean AUC':>10} {'Defect AUC':>12}")
    print("-" * 60)
    for model, aucs in results.items():
        clean = f"{aucs.get('clean', float('nan')):.3f}"
        defect = f"{aucs.get('defect', float('nan')):.3f}"
        print(f"{model:<35} {clean:>10} {defect:>12}")
    print()


def reconstruction_error_distribution(
    errors_healthy: np.ndarray,
    errors_tumor: np.ndarray,
    title: str = "Reconstruction Error Distribution",
    save_path: str = None,
):
    """Plot histogram of reconstruction errors for healthy vs tumor samples."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(errors_healthy, bins=15, alpha=0.6, label="Healthy", color="steelblue")
    ax.hist(errors_tumor, bins=15, alpha=0.6, label="Tumor", color="darkorange")
    ax.set_xlabel("Reconstruction MSE")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Saved: {save_path}")

    plt.show()
    return fig
