"""
preprocessing.py
----------------
CSI amplitude preprocessing pipeline for Wi-Fi-based cancer diagnostics.

Steps:
  1. Interpolation & alignment  -> fixed shape (1501, 52)
  2. IQR-based amplitude sanitization (packet-wise outlier removal)
  3. Per-sample min-max normalization to [0, 1]
  4. Optional defect-aware masking (reintroduces hardware-fault regions)
"""

import numpy as np
from scipy.interpolate import interp1d


TARGET_PACKETS = 1501
TARGET_SUBCARRIERS = 52


def interpolate_and_align(matrix: np.ndarray) -> np.ndarray:
    """Resize a CSI matrix to (TARGET_PACKETS, TARGET_SUBCARRIERS) via linear interpolation."""
    n_packets, n_sub = matrix.shape

    if n_sub != TARGET_SUBCARRIERS:
        raise ValueError(f"Expected {TARGET_SUBCARRIERS} subcarriers, got {n_sub}.")

    if n_packets == TARGET_PACKETS:
        return matrix.copy()

    old_idx = np.linspace(0, 1, n_packets)
    new_idx = np.linspace(0, 1, TARGET_PACKETS)
    interpolated = np.zeros((TARGET_PACKETS, n_sub), dtype=matrix.dtype)

    for col in range(n_sub):
        f = interp1d(old_idx, matrix[:, col], kind="linear")
        interpolated[:, col] = f(new_idx)

    return interpolated


def iqr_sanitize(matrix: np.ndarray, iqr_factor: float = 1.5) -> np.ndarray:
    """
    Packet-wise IQR outlier removal.
    Replaces values outside [Q1 - k*IQR, Q3 + k*IQR] with the row median.
    """
    sanitized = matrix.copy()
    for i in range(sanitized.shape[0]):
        row = sanitized[i]
        q1, q3 = np.percentile(row, 25), np.percentile(row, 75)
        iqr = q3 - q1
        lower, upper = q1 - iqr_factor * iqr, q3 + iqr_factor * iqr
        median = np.median(row)
        mask = (row < lower) | (row > upper)
        sanitized[i, mask] = median
    return sanitized


def minmax_normalize(matrix: np.ndarray) -> np.ndarray:
    """Per-sample min-max normalization to [0, 1]."""
    mn, mx = matrix.min(), matrix.max()
    if mx - mn < 1e-10:
        return np.zeros_like(matrix, dtype=np.float32)
    return ((matrix - mn) / (mx - mn)).astype(np.float32)


def apply_defect_mask(matrix: np.ndarray, mask: np.ndarray, fill_value: float = -1.0) -> np.ndarray:
    """
    Reintroduce hardware-fault regions using a boolean mask.
    Masked positions are set to fill_value to simulate packet-loss defects.
    """
    defect = matrix.copy()
    defect[mask] = fill_value
    return defect


def preprocess_sample(
    matrix: np.ndarray,
    defect_mask: np.ndarray = None,
    iqr_factor: float = 1.5,
) -> np.ndarray:
    """
    Full preprocessing pipeline for a single CSI matrix.

    Parameters
    ----------
    matrix      : raw CSI amplitude matrix, shape (n_packets, 52)
    defect_mask : optional boolean mask, same shape as output; if provided,
                  produces the defect-aware version
    iqr_factor  : IQR multiplier for outlier clipping

    Returns
    -------
    Preprocessed matrix of shape (1501, 52), dtype float32
    """
    x = interpolate_and_align(matrix)
    x = iqr_sanitize(x, iqr_factor=iqr_factor)
    x = minmax_normalize(x)
    if defect_mask is not None:
        x = apply_defect_mask(x, defect_mask)
    return x


def load_dataset(
    clean_path: str,
    defect_path: str,
    labels_path: str,
) -> tuple:
    """
    Load preprocessed .npy arrays from disk.

    Returns
    -------
    X_clean  : (N, 1501, 52) float32
    X_defect : (N, 1501, 52) float32
    y        : (N,) int   — 0 = healthy, 1 = tumor
    """
    X_clean = np.load(clean_path).astype(np.float32)
    X_defect = np.load(defect_path).astype(np.float32)
    y = np.load(labels_path).astype(int)
    return X_clean, X_defect, y


def flatten(X: np.ndarray) -> np.ndarray:
    """Flatten (N, 1501, 52) -> (N, 78052) for dense models."""
    return X.reshape(X.shape[0], -1)


def get_healthy(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Return only healthy samples (label == 0)."""
    return X[y == 0]
