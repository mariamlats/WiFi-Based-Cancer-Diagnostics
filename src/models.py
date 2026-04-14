"""
models.py
---------
All model architectures used in the Wi-Fi CSI cancer-detection study.

Models
------
  - build_shallow_autoencoder   : baseline dense AE (128-dim bottleneck)
  - build_deep_matrix_autoencoder : best-performing model (512-256-128 bottleneck)
  - build_1d_cnn                : supervised convolutional classifier
  - build_lstm                  : supervised recurrent classifier
  - OneClassSVMWrapper          : sklearn OC-SVM with a unified .fit/.score API
  - IsolationForestWrapper      : sklearn Isolation Forest with same API

All Keras models expect flattened input (N, 78052) except CNN and LSTM,
which expect (N, 1501, 52).
"""

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


INPUT_DIM = 1501 * 52   # 78 052
PACKETS    = 1501
SUBCARRIERS = 52


# ── Autoencoders ──────────────────────────────────────────────────────────────

def build_shallow_autoencoder(input_dim: int = INPUT_DIM) -> keras.Model:
    """
    Single hidden-layer autoencoder.
    Encoder: input_dim -> 128
    Decoder: 128       -> input_dim
    """
    inputs = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(128, activation="relu")(inputs)
    decoded = layers.Dense(input_dim, activation="sigmoid")(encoded)
    model = keras.Model(inputs, decoded, name="shallow_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_deep_matrix_autoencoder(input_dim: int = INPUT_DIM) -> keras.Model:
    """
    Deep fully-connected autoencoder — best model in the study.
    Encoder: input_dim -> 512 -> 256 -> 128
    Decoder: 128       -> 256 -> 512 -> input_dim
    """
    inputs = keras.Input(shape=(input_dim,))

    # Encoder
    x = layers.Dense(512, activation="relu")(inputs)
    x = layers.Dense(256, activation="relu")(x)
    encoded = layers.Dense(128, activation="relu")(x)

    # Decoder
    x = layers.Dense(256, activation="relu")(encoded)
    x = layers.Dense(512, activation="relu")(x)
    decoded = layers.Dense(input_dim, activation="sigmoid")(x)

    model = keras.Model(inputs, decoded, name="deep_matrix_autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


# ── Supervised neural classifiers ────────────────────────────────────────────

def build_1d_cnn(packets: int = PACKETS, subcarriers: int = SUBCARRIERS) -> keras.Model:
    """
    1D-CNN classifier for CSI matrices shaped (packets, subcarriers).
    Two conv layers -> global average pooling -> dense head.
    """
    inputs = keras.Input(shape=(packets, subcarriers))
    x = layers.Conv1D(32, kernel_size=3, activation="relu", padding="same")(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)
    x = layers.Conv1D(64, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="cnn_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def build_lstm(packets: int = PACKETS, subcarriers: int = SUBCARRIERS) -> keras.Model:
    """
    Stacked LSTM classifier for CSI time-series (packets, subcarriers).
    Two LSTM layers -> dense output.
    """
    inputs = keras.Input(shape=(packets, subcarriers))
    x = layers.LSTM(64, return_sequences=True)(inputs)
    x = layers.LSTM(32)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="lstm_classifier")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ── Classical anomaly detectors ───────────────────────────────────────────────

class OneClassSVMWrapper:
    """
    One-Class SVM wrapper with a unified fit / anomaly_score interface.
    Higher score => more anomalous (tumor-like).
    """
    def __init__(self, kernel="rbf", nu=0.1, gamma="scale"):
        self.model = OneClassSVM(kernel=kernel, nu=nu, gamma=gamma)

    def fit(self, X_healthy: np.ndarray):
        self.model.fit(X_healthy)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: positive = inlier, negative = outlier
        # negate so higher = more anomalous
        return -self.model.decision_function(X)


class IsolationForestWrapper:
    """
    Isolation Forest wrapper with a unified fit / anomaly_score interface.
    Higher score => more anomalous (tumor-like).
    """
    def __init__(self, n_estimators=200, contamination=0.1, random_state=42):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def fit(self, X_healthy: np.ndarray):
        self.model.fit(X_healthy)
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(X)


# ── Anomaly score for autoencoders ───────────────────────────────────────────

def reconstruction_error(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """
    Compute per-sample MSE reconstruction error.
    Used as the anomaly score for autoencoder models.
    """
    X_hat = model.predict(X, verbose=0)
    return np.mean((X - X_hat) ** 2, axis=1)
