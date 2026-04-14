"""
train.py
--------
CLI entry point to train and evaluate all models.

Usage
-----
  python src/train.py --data_dir data/ --results_dir results/ --model all
  python src/train.py --data_dir data/ --model deep_ae
  python src/train.py --data_dir data/ --model svm

Models: shallow_ae | deep_ae | svm | iforest | all
"""

import argparse
import numpy as np
from pathlib import Path

from preprocessing import load_dataset, flatten, get_healthy
from models import (
    build_shallow_autoencoder,
    build_deep_matrix_autoencoder,
    OneClassSVMWrapper,
    IsolationForestWrapper,
    reconstruction_error,
)
from evaluate import compute_roc, plot_roc, print_summary


EPOCHS = 50
BATCH_SIZE = 4


def train_autoencoder(model, X_healthy_flat, X_all_flat, y, label, results_dir):
    print(f"\nTraining {label}...")
    model.fit(
        X_healthy_flat, X_healthy_flat,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    scores = reconstruction_error(model, X_all_flat)
    _, _, _, roc_auc = compute_roc(y, scores)
    print(f"  AUC: {roc_auc:.3f}")

    plot_roc(
        y,
        {label: scores},
        title=f"ROC – {label}",
        save_path=str(results_dir / f"roc_{label.lower().replace(' ', '_')}.png"),
    )
    return roc_auc


def train_classical(wrapper, X_healthy_flat, X_all_flat, y, label, results_dir):
    print(f"\nTraining {label}...")
    wrapper.fit(X_healthy_flat)
    scores = wrapper.anomaly_score(X_all_flat)
    _, _, _, roc_auc = compute_roc(y, scores)
    print(f"  AUC: {roc_auc:.3f}")

    plot_roc(
        y,
        {label: scores},
        title=f"ROC – {label}",
        save_path=str(results_dir / f"roc_{label.lower().replace(' ', '_')}.png"),
    )
    return roc_auc


def main(args):
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    X_clean, X_defect, y = load_dataset(
        clean_path=f"{args.data_dir}/X_clean.npy",
        defect_path=f"{args.data_dir}/X_defect.npy",
        labels_path=f"{args.data_dir}/y.npy",
    )
    print(f"  Samples: {len(y)} | Healthy: {(y==0).sum()} | Tumor: {(y==1).sum()}")

    X_clean_flat  = flatten(X_clean)
    X_defect_flat = flatten(X_defect)
    X_healthy_clean  = get_healthy(X_clean_flat, y)
    X_healthy_defect = get_healthy(X_defect_flat, y)

    summary = {}

    if args.model in ("shallow_ae", "all"):
        auc_c = train_autoencoder(
            build_shallow_autoencoder(), X_healthy_clean, X_clean_flat,
            y, "Shallow AE (clean)", results_dir,
        )
        auc_d = train_autoencoder(
            build_shallow_autoencoder(), X_healthy_defect, X_defect_flat,
            y, "Shallow AE (defect)", results_dir,
        )
        summary["Shallow Autoencoder"] = {"clean": auc_c, "defect": auc_d}

    if args.model in ("deep_ae", "all"):
        auc_c = train_autoencoder(
            build_deep_matrix_autoencoder(), X_healthy_clean, X_clean_flat,
            y, "Deep AE (clean)", results_dir,
        )
        auc_d = train_autoencoder(
            build_deep_matrix_autoencoder(), X_healthy_defect, X_defect_flat,
            y, "Deep AE (defect)", results_dir,
        )
        summary["Deep Matrix Autoencoder"] = {"clean": auc_c, "defect": auc_d}

    if args.model in ("svm", "all"):
        auc_c = train_classical(
            OneClassSVMWrapper(), X_healthy_clean, X_clean_flat,
            y, "One-Class SVM (clean)", results_dir,
        )
        auc_d = train_classical(
            OneClassSVMWrapper(), X_healthy_defect, X_defect_flat,
            y, "One-Class SVM (defect)", results_dir,
        )
        summary["One-Class SVM"] = {"clean": auc_c, "defect": auc_d}

    if args.model in ("iforest", "all"):
        auc_c = train_classical(
            IsolationForestWrapper(), X_healthy_clean, X_clean_flat,
            y, "Isolation Forest (clean)", results_dir,
        )
        auc_d = train_classical(
            IsolationForestWrapper(), X_healthy_defect, X_defect_flat,
            y, "Isolation Forest (defect)", results_dir,
        )
        summary["Isolation Forest"] = {"clean": auc_c, "defect": auc_d}

    print_summary(summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CSI anomaly detection models")
    parser.add_argument("--data_dir",    default="data",    help="Path to .npy files")
    parser.add_argument("--results_dir", default="results", help="Where to save ROC plots")
    parser.add_argument(
        "--model", default="all",
        choices=["shallow_ae", "deep_ae", "svm", "iforest", "all"],
        help="Which model to train",
    )
    main(parser.parse_args())
