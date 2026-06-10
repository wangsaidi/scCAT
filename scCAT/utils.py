import random
from typing import Iterable, List

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unique_in_order(items: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(items))


def safe_minmax(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values.astype(np.float32)
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < eps:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - vmin) / (vmax - vmin + eps)).astype(np.float32)


def center_rows_for_pearson(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float32)
    centered = matrix - matrix.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1).astype(np.float32)
    return centered.astype(np.float32), norms


def pearson_corr_to_many(
    centered_matrix: np.ndarray,
    norms: np.ndarray,
    anchor_idx: int,
    candidate_indices: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.size == 0:
        return np.zeros(0, dtype=np.float32)

    anchor_vec = centered_matrix[anchor_idx]
    anchor_norm = float(norms[anchor_idx])
    candidate_vecs = centered_matrix[candidate_indices]
    candidate_norms = norms[candidate_indices]

    numerators = candidate_vecs @ anchor_vec
    denominators = (candidate_norms * anchor_norm) + eps
    corr = numerators / denominators
    corr = np.clip(corr, -1.0, 1.0)
    return corr.astype(np.float32)


def pearson_corr_pair(
    centered_matrix: np.ndarray,
    norms: np.ndarray,
    i: int,
    j: int,
    eps: float = 1e-12,
) -> float:
    return float(pearson_corr_to_many(centered_matrix, norms, i, np.array([j], dtype=np.int64), eps=eps)[0])


def euclidean_distance_to_many(
    matrix: np.ndarray,
    anchor_idx: int,
    candidate_indices: np.ndarray,
) -> np.ndarray:
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if candidate_indices.size == 0:
        return np.zeros(0, dtype=np.float32)
    diff = matrix[candidate_indices] - matrix[anchor_idx]
    dist = np.linalg.norm(diff, axis=1)
    return dist.astype(np.float32)


def total_loss_is_stable(
    loss_history: list[float],
    min_epochs_before_early_stop: int,
    stable_window: int,
    stable_relative_tolerance: float,
) -> bool:
    if len(loss_history) < max(min_epochs_before_early_stop, stable_window):
        return False
    recent = np.asarray(loss_history[-stable_window:], dtype=np.float64)
    span = float(recent.max() - recent.min())
    reference = max(1.0, abs(float(recent.mean())))
    return span <= (stable_relative_tolerance * reference)
