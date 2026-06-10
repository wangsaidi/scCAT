from typing import Any, Dict

import numpy as np
from sklearn.decomposition import PCA

from .config import Config


def library_size_normalize(matrix: np.ndarray, target_sum: float) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    cell_sums = matrix.sum(axis=1, keepdims=True).astype(np.float32)
    cell_sums = np.where(cell_sums <= 0.0, 1.0, cell_sums)
    normalized = matrix / cell_sums
    normalized = normalized * float(target_sum)
    return normalized.astype(np.float32)


def log1p_transform(matrix: np.ndarray) -> np.ndarray:
    return np.log1p(np.asarray(matrix, dtype=np.float32)).astype(np.float32)


def select_hvg_by_variance(
    matrix: np.ndarray,
    n_hvg: int,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, list[str] | None]:
    matrix = np.asarray(matrix, dtype=np.float32)
    actual_n_hvg = min(int(n_hvg), matrix.shape[1])
    variances = matrix.var(axis=0)
    top_idx = np.argsort(variances)[::-1][:actual_n_hvg]
    selected = matrix[:, top_idx].astype(np.float32)

    if feature_names is None:
        selected_names = None
    else:
        selected_names = [feature_names[i] for i in top_idx.tolist()]
    return selected, selected_names


def batch_z_scale(
    matrix: np.ndarray,
    batch_labels: np.ndarray,
    eps: float,
) -> np.ndarray:
    """
    Z-scale features within each batch independently.

    Parameters
    ----------
    matrix
        Cell-by-feature matrix after norm+log1p+HVG.
    batch_labels
        Batch label for each cell.
    eps
        Numerical stability term for zero-variance features.
    """
    matrix = np.asarray(matrix, dtype=np.float32)
    batch_labels = np.asarray(batch_labels).astype(str)

    scaled = np.empty_like(matrix, dtype=np.float32)
    unique_batches = list(dict.fromkeys(batch_labels.tolist()))

    for batch_name in unique_batches:
        batch_mask = batch_labels == batch_name
        batch_matrix = matrix[batch_mask]

        if batch_matrix.shape[0] == 0:
            continue

        batch_mean = batch_matrix.mean(axis=0, keepdims=True).astype(np.float32)
        batch_std = batch_matrix.std(axis=0, keepdims=True).astype(np.float32)
        batch_std = np.where(batch_std <= eps, 1.0, batch_std)

        scaled[batch_mask] = ((batch_matrix - batch_mean) / batch_std).astype(np.float32)

    return scaled.astype(np.float32)


def run_pca(matrix: np.ndarray, n_pca: int, seed: int) -> tuple[np.ndarray, PCA]:
    matrix = np.asarray(matrix, dtype=np.float32)
    max_components = min(int(n_pca), matrix.shape[0], matrix.shape[1])
    if max_components < 1:
        raise ValueError("PCA requires at least one component.")
    pca = PCA(n_components=max_components, random_state=seed)
    transformed = pca.fit_transform(matrix).astype(np.float32)
    return transformed, pca


def prepare_inputs(
    expression_matrix: np.ndarray,
    feature_names: list[str],
    batch_labels: np.ndarray,
    input_data_state: str,
    config: Config,
) -> Dict[str, Any]:
    """
    input_data_state:
        - "raw":
            expression_matrix is the raw expression matrix.
            Triplet space:
                norm + log1p + HVG + batch-wise z-scale + PCA
            Encoder input space:
                norm + log1p + HVG + batch-wise z-scale + PCA

        - "preprocessed_hvg":
            expression_matrix is already the processed HVG matrix
            (typically after norm + log1p + HVG).
            Triplet space:
                input HVG + batch-wise z-scale + PCA
            Encoder input space:
                input HVG + batch-wise z-scale + PCA
    """
    expression_matrix = np.asarray(expression_matrix, dtype=np.float32)
    batch_labels = np.asarray(batch_labels).astype(str)

    if input_data_state == "raw":
        print("[Preprocess] Input mode = raw")
        print("[Preprocess] Running normalization -> log1p -> HVG ...")
        normalized = library_size_normalize(expression_matrix, config.target_sum)
        logged = log1p_transform(normalized)
        hvg_matrix, selected_names = select_hvg_by_variance(logged, config.n_hvg, feature_names)

        print("[Preprocess] Building triplet space: HVG -> batch-wise z-scale -> PCA ...")
        hvg_matrix_batch_scaled = batch_z_scale(hvg_matrix, batch_labels, config.eps)
        triplet_input, triplet_pca_model = run_pca(hvg_matrix_batch_scaled, config.n_pca, config.seed)

        print("[Preprocess] Building encoder input space: HVG -> batch-wise z-scale -> PCA ...")
        model_input, model_pca_model = run_pca(hvg_matrix_batch_scaled, config.n_pca, config.seed)

        return {
            "model_input": model_input,
            "triplet_input": triplet_input,
            "hvg_matrix": hvg_matrix,
            "hvg_matrix_batch_scaled": hvg_matrix_batch_scaled,
            "selected_feature_names": selected_names,
            "model_pca_model": model_pca_model,
            "triplet_pca_model": triplet_pca_model,
        }

    if input_data_state == "preprocessed_hvg":
        print("[Preprocess] Input mode = preprocessed_hvg")
        print("[Preprocess] Skipping normalization/log1p/HVG selection ...")
        hvg_matrix = expression_matrix.astype(np.float32)

        print("[Preprocess] Building triplet space: HVG -> batch-wise z-scale -> PCA ...")
        hvg_matrix_batch_scaled = batch_z_scale(hvg_matrix, batch_labels, config.eps)
        triplet_input, triplet_pca_model = run_pca(hvg_matrix_batch_scaled, config.n_pca, config.seed)

        print("[Preprocess] Building encoder input space: HVG -> batch-wise z-scale -> PCA ...")
        model_input, model_pca_model = run_pca(hvg_matrix_batch_scaled, config.n_pca, config.seed)

        return {
            "model_input": model_input,
            "triplet_input": triplet_input,
            "hvg_matrix": hvg_matrix,
            "hvg_matrix_batch_scaled": hvg_matrix_batch_scaled,
            "selected_feature_names": feature_names,
            "model_pca_model": model_pca_model,
            "triplet_pca_model": triplet_pca_model,
        }

    raise ValueError(
        "input_data_state must be either 'raw' or 'preprocessed_hvg'. "
        "Supported modes are: raw input, or preprocessed HVG input."
    )
