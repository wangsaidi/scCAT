from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from sklearn.neighbors import NearestNeighbors

from .config import Config
from .utils import (
    center_rows_for_pearson,
    euclidean_distance_to_many,
    pearson_corr_to_many,
    safe_minmax,
    unique_in_order,
)


@dataclass
class TripletBundle:
    triplets: np.ndarray
    omega: np.ndarray
    rho_bar: np.ndarray
    same_batch_knn_edges: np.ndarray
    rare_flags: np.ndarray
    batch_codes_per_cell: np.ndarray
    batch_names: List[str]
    summary: Dict[str, int]
    # local_knn_indices: np.ndarray


def density_similarity_scalar(rho_i: float, rho_j: float, eps: float) -> float:
    return float(np.exp(-abs(rho_i - rho_j) / (rho_i + rho_j + eps)))


def density_similarity_to_many(rho_i: float, rho_values: np.ndarray, eps: float) -> np.ndarray:
    rho_values = np.asarray(rho_values, dtype=np.float32)
    if rho_values.size == 0:
        return np.zeros(0, dtype=np.float32)
    values = np.exp(-np.abs(rho_i - rho_values) / (rho_i + rho_values + eps))
    return values.astype(np.float32)


def build_same_batch_knn(
    model_input: np.ndarray,
    batch_labels: np.ndarray,
    config: Config,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray]:
    print("[Triplets] Building same-batch KNN graph ...")
    batch_names = unique_in_order(batch_labels.tolist())
    same_batch_knn: dict[int, np.ndarray] = {}
    same_batch_knn_distances: dict[int, np.ndarray] = {}
    same_batch_knn_edges: list[list[int]] = []

    for batch_name in batch_names:
        batch_indices = np.where(batch_labels == batch_name)[0]
        print(f"[Triplets]   Batch={batch_name}, cells={len(batch_indices)}")

        if len(batch_indices) <= 1:
            for global_idx in batch_indices.tolist():
                same_batch_knn[global_idx] = np.zeros(0, dtype=np.int64)
                same_batch_knn_distances[global_idx] = np.zeros(0, dtype=np.float32)
            continue

        n_neighbors = min(config.knn_k + 1, len(batch_indices))
        nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        nn_model.fit(model_input[batch_indices])
        distances, neighbors = nn_model.kneighbors(model_input[batch_indices], return_distance=True)

        for row_idx, global_idx in enumerate(batch_indices.tolist()):
            local_neighbors = neighbors[row_idx]
            local_distances = distances[row_idx]

            # Drop self; on self-query, it should be the first with distance 0.
            local_neighbors = local_neighbors[1:]
            local_distances = local_distances[1:]

            global_neighbors = batch_indices[local_neighbors]
            same_batch_knn[global_idx] = global_neighbors.astype(np.int64)
            same_batch_knn_distances[global_idx] = local_distances.astype(np.float32)

            for neighbor_global in global_neighbors.tolist():
                same_batch_knn_edges.append([global_idx, int(neighbor_global)])

    if len(same_batch_knn_edges) == 0:
        edge_array = np.zeros((0, 2), dtype=np.int64)
    else:
        edge_array = np.asarray(same_batch_knn_edges, dtype=np.int64)

    return same_batch_knn, same_batch_knn_distances, edge_array


def compute_local_density(
    same_batch_knn_distances: dict[int, np.ndarray],
    n_cells: int,
    eps: float,
) -> np.ndarray:
    print("[Triplets] Computing local density rho_i ...")
    rho = np.zeros(n_cells, dtype=np.float32)
    for i in range(n_cells):
        distances = same_batch_knn_distances.get(i, np.zeros(0, dtype=np.float32))
        if distances.size == 0:
            rho[i] = 0.0
        else:
            mean_distance = float(distances.mean())
            rho[i] = 1.0 / (mean_distance + eps)
    return rho.astype(np.float32)


def build_cross_batch_mnn(
    model_input: np.ndarray,
    batch_labels: np.ndarray,
    config: Config,
) -> tuple[dict[int, list[int]], dict[tuple[int, int], float]]:
    print("[Triplets] Building cross-batch MNN graph ...")
    n_cells = model_input.shape[0]
    batch_names = unique_in_order(batch_labels.tolist())
    cross_batch_positives: dict[int, list[int]] = {i: [] for i in range(n_cells)}
    raw_mnn_scores: dict[tuple[int, int], float] = {}

    for batch_idx_a in range(len(batch_names)):
        for batch_idx_b in range(batch_idx_a + 1, len(batch_names)):
            batch_a = batch_names[batch_idx_a]
            batch_b = batch_names[batch_idx_b]
            idx_a = np.where(batch_labels == batch_a)[0]
            idx_b = np.where(batch_labels == batch_b)[0]

            print(
                f"[Triplets]   MNN between batch {batch_a} (n={len(idx_a)}) "
                f"and batch {batch_b} (n={len(idx_b)})"
            )

            if len(idx_a) == 0 or len(idx_b) == 0:
                continue

            k_ab = min(config.mnn_k, len(idx_b))
            k_ba = min(config.mnn_k, len(idx_a))
            if k_ab == 0 or k_ba == 0:
                continue

            nn_ab = NearestNeighbors(n_neighbors=k_ab, metric="euclidean")
            nn_ab.fit(model_input[idx_b])
            _, nbr_ab_local = nn_ab.kneighbors(model_input[idx_a], return_distance=True)
            nbr_ab_global = idx_b[nbr_ab_local]

            nn_ba = NearestNeighbors(n_neighbors=k_ba, metric="euclidean")
            nn_ba.fit(model_input[idx_a])
            _, nbr_ba_local = nn_ba.kneighbors(model_input[idx_b], return_distance=True)
            nbr_ba_global = idx_a[nbr_ba_local]

            reverse_rank_lookup: dict[tuple[int, int], int] = {}
            for row_b, global_b in enumerate(idx_b.tolist()):
                for rank, global_a in enumerate(nbr_ba_global[row_b].tolist(), start=1):
                    reverse_rank_lookup[(global_b, global_a)] = rank

            pair_count = 0
            for row_a, global_a in enumerate(idx_a.tolist()):
                for rank_ab, global_b in enumerate(nbr_ab_global[row_a].tolist(), start=1):
                    rank_ba = reverse_rank_lookup.get((global_b, global_a))
                    if rank_ba is None:
                        continue

                    score = 0.5 * ((1.0 / float(rank_ab)) + (1.0 / float(rank_ba)))
                    raw_mnn_scores[(global_a, global_b)] = score
                    raw_mnn_scores[(global_b, global_a)] = score
                    cross_batch_positives[global_a].append(global_b)
                    cross_batch_positives[global_b].append(global_a)
                    pair_count += 1

            print(f"[Triplets]     Directed MNN positives found: {pair_count * 2}")

    # Deduplicate while keeping deterministic order
    for i in range(n_cells):
        if len(cross_batch_positives[i]) == 0:
            continue
        cross_batch_positives[i] = list(dict.fromkeys(cross_batch_positives[i]))

    if len(raw_mnn_scores) == 0:
        normalized_scores: dict[tuple[int, int], float] = {}
    else:
        keys = list(raw_mnn_scores.keys())
        values = np.asarray([raw_mnn_scores[k] for k in keys], dtype=np.float32)
        values_norm = safe_minmax(values, eps=config.eps)
        normalized_scores = {k: float(v) for k, v in zip(keys, values_norm.tolist())}

    return cross_batch_positives, normalized_scores


def construct_triplets(
    model_input: np.ndarray,
    hvg_matrix: np.ndarray,
    batch_labels: np.ndarray,
    config: Config,
) -> TripletBundle:
    n_cells = model_input.shape[0]
    batch_labels = np.asarray(batch_labels).astype(str)
    batch_names = unique_in_order(batch_labels.tolist())
    batch_to_indices = {b: np.where(batch_labels == b)[0] for b in batch_names}
    batch_to_code = {b: idx for idx, b in enumerate(batch_names)}
    batch_codes_per_cell = np.asarray([batch_to_code[b] for b in batch_labels.tolist()], dtype=np.int64)

    same_batch_knn, same_batch_knn_distances, same_batch_knn_edges = build_same_batch_knn(
        model_input=model_input,
        batch_labels=batch_labels,
        config=config,
    )



    # # 新增：
    # max_k = config.knn_k + 1
    # local_knn_indices = np.zeros((n_cells, max_k), dtype=np.int64)
    # for i in range(n_cells):
    #     neighbors = same_batch_knn.get(i, np.zeros(0, dtype=np.int64))
    #     padded = np.concatenate(([i], neighbors))  # 包含自己
    #     if len(padded) > max_k:
    #         padded = padded[:max_k]
    #     elif len(padded) < max_k:
    #         padded = np.pad(padded, (0, max_k - len(padded)), mode='edge')
    #     local_knn_indices[i] = padded





    rho = compute_local_density(
        same_batch_knn_distances=same_batch_knn_distances,
        n_cells=n_cells,
        eps=config.eps,
    )
    rho_norm = safe_minmax(rho, eps=config.eps)

    cross_batch_positives, mnn_scores = build_cross_batch_mnn(
        model_input=model_input,
        batch_labels=batch_labels,
        config=config,
    )

    rare_flags = np.asarray(
        [1.0 if len(cross_batch_positives[i]) == 0 else 0.0 for i in range(n_cells)],
        dtype=np.float32,
    )

    centered_hvg, hvg_norms = center_rows_for_pearson(hvg_matrix)




    # # 新增
    # print("[Triplets] Computing continuous rare flags...")
    # rare_flags = np.ones(n_cells, dtype=np.float32)
    # alpha_sum = float(config.alpha1 + config.alpha2 + config.alpha3)
    #
    # for anchor in range(n_cells):
    #     cb_pos_list = cross_batch_positives.get(anchor, [])
    #     if len(cb_pos_list) == 0:
    #         continue
    #
    #     # 批量计算该锚点与其跨批次MNN的HVG相似度
    #     cb_pos_array = np.asarray(cb_pos_list, dtype=np.int64)
    #     cb_pos_corr = pearson_corr_to_many(centered_hvg, hvg_norms, anchor, cb_pos_array, eps=config.eps)
    #     cb_s_hvg = (1.0 + cb_pos_corr) / 2.0
    #
    #     max_c_pos_norm = 0.0
    #     for pos_idx, p in enumerate(cb_pos_list):
    #         s_density = density_similarity_scalar(rho[anchor], rho[p], config.eps)
    #         s_hvg = float(cb_s_hvg[pos_idx])
    #         s_mnn = float(mnn_scores.get((anchor, p), 0.0))
    #
    #         c_pos = config.alpha1 * s_mnn + config.alpha2 * s_density + config.alpha3 * s_hvg
    #         c_pos_norm = c_pos / alpha_sum
    #         if c_pos_norm > max_c_pos_norm:
    #             max_c_pos_norm = float(c_pos_norm)
    #
    #     # 稀有度 = 1.0 - 其最强跨批次连接的置信度
    #     rare_flags[anchor] = 1.0 - max_c_pos_norm





    other_batch_pool: dict[str, np.ndarray] = {}
    for batch_name in batch_names:
        others = [batch_to_indices[other] for other in batch_names if other != batch_name]
        other_batch_pool[batch_name] = (
            np.concatenate(others).astype(np.int64) if len(others) > 0 else np.zeros(0, dtype=np.int64)
        )

    print("[Triplets] Constructing triplets ...")
    triplets: list[list[int]] = []
    omegas: list[float] = []
    rho_bars: list[float] = []

    total_same_batch_positive_pairs = 0
    total_cross_batch_positive_pairs = 0

    # --- 新增：过滤统计计数器 ---
    count_raw_pos = 0  # 原始正样本数（满足KNN/MNN的候选对）
    count_passed_pos = 0  # 过滤后保留的正样本数
    count_raw_neg = 0  # 原始负样本/三元组数（为保留的正样本匹配的所有候选负样本）
    count_passed_neg = 0  # 过滤后保留的负样本/三元组数


    for anchor in range(n_cells):
        if (
            anchor == 0
            or (anchor + 1) % config.triplet_log_every_anchors == 0
            or (anchor + 1) == n_cells
        ):
            print(f"[Triplets]   Anchor progress: {anchor + 1}/{n_cells}")

        anchor_batch = batch_labels[anchor]
        same_batch_positive_array = same_batch_knn.get(anchor, np.zeros(0, dtype=np.int64))
        cross_batch_positive_array = np.asarray(cross_batch_positives.get(anchor, []), dtype=np.int64)

        total_same_batch_positive_pairs += int(len(same_batch_positive_array))
        total_cross_batch_positive_pairs += int(len(cross_batch_positive_array))

        # Same-batch negative pool:
        # Remove self and same-batch neighbors first, then apply the 70%-90% distance-rank rule.
        same_batch_candidates = batch_to_indices[anchor_batch]
        same_batch_candidates = same_batch_candidates[same_batch_candidates != anchor]
        if same_batch_positive_array.size > 0:
            same_batch_candidates = same_batch_candidates[~np.isin(same_batch_candidates, same_batch_positive_array)]

        same_window_candidates = np.zeros(0, dtype=np.int64)
        same_window_distances = np.zeros(0, dtype=np.float32)
        same_window_negative_confidence = np.zeros(0, dtype=np.float32)

        if same_batch_candidates.size > 0:
            candidate_distances = euclidean_distance_to_many(model_input, anchor, same_batch_candidates)
            sort_order = np.argsort(candidate_distances)
            same_batch_candidates = same_batch_candidates[sort_order]
            candidate_distances = candidate_distances[sort_order]

            low_rank = int(np.floor(config.same_batch_negative_rank_low * len(same_batch_candidates)))
            high_rank = int(np.ceil(config.same_batch_negative_rank_high * len(same_batch_candidates)))
            high_rank = max(high_rank, low_rank + 1)
            high_rank = min(high_rank, len(same_batch_candidates))

            same_window_candidates = same_batch_candidates[low_rank:high_rank].astype(np.int64)
            same_window_distances = candidate_distances[low_rank:high_rank].astype(np.float32)

            if same_window_candidates.size > 0:
                same_s_density = density_similarity_to_many(
                    rho[anchor],
                    rho[same_window_candidates],
                    config.eps,
                )
                same_corr = pearson_corr_to_many(
                    centered_hvg,
                    hvg_norms,
                    anchor,
                    same_window_candidates,
                    eps=config.eps,
                )
                same_s_hvg = (1.0 + same_corr) / 2.0
                same_window_negative_confidence = (
                    config.beta1 * (1.0 - same_s_density)
                    + config.beta2 * (1.0 - same_s_hvg)
                ).astype(np.float32)

        # Other-batch negative pool:
        # Not anchor's cross-batch MNN; prioritize large negative confidence c^-,
        # use distance as the secondary key.
        other_batch_candidates = other_batch_pool[anchor_batch]
        if other_batch_candidates.size > 0 and cross_batch_positive_array.size > 0:
            other_batch_candidates = other_batch_candidates[
                ~np.isin(other_batch_candidates, cross_batch_positive_array)
            ]

        selected_other_batch_negatives = np.zeros(0, dtype=np.int64)
        selected_other_batch_negative_confidence = np.zeros(0, dtype=np.float32)

        if other_batch_candidates.size > 0:
            other_distances = euclidean_distance_to_many(model_input, anchor, other_batch_candidates)
            other_s_density = density_similarity_to_many(
                rho[anchor],
                rho[other_batch_candidates],
                config.eps,
            )
            other_corr = pearson_corr_to_many(
                centered_hvg,
                hvg_norms,
                anchor,
                other_batch_candidates,
                eps=config.eps,
            )
            other_s_hvg = (1.0 + other_corr) / 2.0
            other_negative_confidence = (
                config.beta1 * (1.0 - other_s_density)
                + config.beta2 * (1.0 - other_s_hvg)
            ).astype(np.float32)

            # Primary: larger c^- ; Secondary: smaller distance
            select_order = np.lexsort((other_distances, -other_negative_confidence))
            select_order = select_order[: config.num_neg_other_batch]

            selected_other_batch_negatives = other_batch_candidates[select_order].astype(np.int64)
            selected_other_batch_negative_confidence = other_negative_confidence[select_order].astype(np.float32)

        # Same-batch positive pairs
        if same_batch_positive_array.size > 0:
            same_pos_corr = pearson_corr_to_many(
                centered_hvg,
                hvg_norms,
                anchor,
                same_batch_positive_array,
                eps=config.eps,
            )
            same_pos_s_hvg = ((1.0 + same_pos_corr) / 2.0).astype(np.float32)

            # 原--不过滤
            # for pos_idx, positive in enumerate(same_batch_positive_array.tolist()):
            #     positive = int(positive)
            #     s_density = density_similarity_scalar(rho[anchor], rho[positive], config.eps)
            #     s_hvg = float(same_pos_s_hvg[pos_idx])
            #
            #     # Same-batch positive confidence exactly follows the PDF:
            #     # c_ij^+ = alpha1 * s_density + alpha2 * s_hvg
            #     c_pos = float(config.alpha1 * s_density + config.alpha2 * s_hvg)
            #     pos_distance = float(np.linalg.norm(model_input[anchor] - model_input[positive]))
            #
            #     eligible_same_batch_idx = np.where(same_window_distances > pos_distance)[0]
            #     eligible_same_batch_idx = eligible_same_batch_idx[: config.num_neg_same_batch]
            #
            #     for neg_pool_idx in eligible_same_batch_idx.tolist():
            #         negative = int(same_window_candidates[neg_pool_idx])
            #         c_neg = float(same_window_negative_confidence[neg_pool_idx])
            #         omega = float(c_pos * c_neg)
            #
            #         triplets.append([anchor, positive, negative])
            #         omegas.append(omega)
            #         rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))
            #
            #     for neg_idx, negative in enumerate(selected_other_batch_negatives.tolist()):
            #         c_neg = float(selected_other_batch_negative_confidence[neg_idx])
            #         omega = float(c_pos * c_neg)
            #
            #         triplets.append([anchor, positive, int(negative)])
            #         omegas.append(omega)
            #         rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))

            # 新--过滤
            for pos_idx, positive in enumerate(same_batch_positive_array.tolist()):
                count_raw_pos += 1  # 记录原始正样本数

                positive = int(positive)
                s_density = density_similarity_scalar(rho[anchor], rho[positive], config.eps)
                s_hvg = float(same_pos_s_hvg[pos_idx])

                # 1. 计算原始正样本置信度
                c_pos = float(config.alpha1 * s_density + config.alpha2 * s_hvg)

                # 2. 归一化用于过滤 (同批次只有 alpha1 和 alpha2)
                c_pos_norm = c_pos / float(config.alpha1 + config.alpha2)
                if c_pos_norm < config.min_c_pos:
                    continue

                count_passed_pos += 1  # 记录过滤后正样本数

                pos_distance = float(np.linalg.norm(model_input[anchor] - model_input[positive]))
                eligible_same_batch_idx = np.where(same_window_distances > pos_distance)[0]
                eligible_same_batch_idx = eligible_same_batch_idx[: config.num_neg_same_batch]

                # 记录针对该正样本准备评估的所有候选负样本数
                count_raw_neg += len(eligible_same_batch_idx) + len(selected_other_batch_negatives)

                # --- 负样本循环 (同批次) ---
                for neg_pool_idx in eligible_same_batch_idx.tolist():
                    negative = int(same_window_candidates[neg_pool_idx])
                    c_neg = float(same_window_negative_confidence[neg_pool_idx])

                    # 归一化用于过滤
                    c_neg_norm = c_neg / float(config.beta1 + config.beta2)
                    if c_neg_norm < config.min_c_neg:
                        continue

                    # 3. 原始值用于权重，移除 omega 过滤
                    omega = float(c_pos * c_neg)
                    triplets.append([anchor, positive, negative])
                    omegas.append(omega)
                    rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))
                    count_passed_neg += 1  # 记录保留的三元组/负样本数

                # --- 负样本循环 (跨批次) ---
                for neg_idx, negative in enumerate(selected_other_batch_negatives.tolist()):
                    c_neg = float(selected_other_batch_negative_confidence[neg_idx])

                    c_neg_norm = c_neg / float(config.beta1 + config.beta2)
                    if c_neg_norm < config.min_c_neg:
                        continue

                    omega = float(c_pos * c_neg)
                    triplets.append([anchor, positive, int(negative)])
                    omegas.append(omega)
                    rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))
                    count_passed_neg += 1  # 记录保留的三元组/负样本数



        # Cross-batch positive pairs
        if cross_batch_positive_array.size > 0:
            cross_pos_corr = pearson_corr_to_many(
                centered_hvg,
                hvg_norms,
                anchor,
                cross_batch_positive_array,
                eps=config.eps,
            )
            cross_pos_s_hvg = ((1.0 + cross_pos_corr) / 2.0).astype(np.float32)

            # 原--不过滤
            # for pos_idx, positive in enumerate(cross_batch_positive_array.tolist()):
            #     positive = int(positive)
            #     s_density = density_similarity_scalar(rho[anchor], rho[positive], config.eps)
            #     s_hvg = float(cross_pos_s_hvg[pos_idx])
            #     s_mnn = float(mnn_scores.get((anchor, positive), 0.0))
            #
            #     # Cross-batch positive confidence exactly follows the PDF:
            #     # c_ij^+ = alpha1 * s_mnn + alpha2 * s_density + alpha3 * s_hvg
            #     c_pos = float(config.alpha1 * s_mnn + config.alpha2 * s_density + config.alpha3 * s_hvg)
            #     pos_distance = float(np.linalg.norm(model_input[anchor] - model_input[positive]))
            #
            #     eligible_same_batch_idx = np.where(same_window_distances > pos_distance)[0]
            #     eligible_same_batch_idx = eligible_same_batch_idx[: config.num_neg_same_batch]
            #
            #     for neg_pool_idx in eligible_same_batch_idx.tolist():
            #         negative = int(same_window_candidates[neg_pool_idx])
            #         c_neg = float(same_window_negative_confidence[neg_pool_idx])
            #         omega = float(c_pos * c_neg)
            #
            #         triplets.append([anchor, positive, negative])
            #         omegas.append(omega)
            #         rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))
            #
            #     for neg_idx, negative in enumerate(selected_other_batch_negatives.tolist()):
            #         c_neg = float(selected_other_batch_negative_confidence[neg_idx])
            #         omega = float(c_pos * c_neg)
            #
            #         triplets.append([anchor, positive, int(negative)])
            #         omegas.append(omega)
            #         rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))

            # 新--过滤
            for pos_idx, positive in enumerate(cross_batch_positive_array.tolist()):
                count_raw_pos += 1  # 记录原始正样本数

                positive = int(positive)
                s_density = density_similarity_scalar(rho[anchor], rho[positive], config.eps)
                s_hvg = float(cross_pos_s_hvg[pos_idx])
                s_mnn = float(mnn_scores.get((anchor, positive), 0.0))

                # 1. 计算原始正样本置信度
                c_pos = float(config.alpha1 * s_mnn + config.alpha2 * s_density + config.alpha3 * s_hvg)

                # 2. 归一化用于过滤 (跨批次有 alpha1, alpha2, alpha3)
                c_pos_norm = c_pos / float(config.alpha1 + config.alpha2 + config.alpha3)
                if c_pos_norm < config.min_c_pos:
                    continue

                count_passed_pos += 1  # 记录过滤后正样本数

                pos_distance = float(np.linalg.norm(model_input[anchor] - model_input[positive]))
                eligible_same_batch_idx = np.where(same_window_distances > pos_distance)[0]
                eligible_same_batch_idx = eligible_same_batch_idx[: config.num_neg_same_batch]

                # 记录针对该正样本准备评估的所有候选负样本数
                count_raw_neg += len(eligible_same_batch_idx) + len(selected_other_batch_negatives)

                # --- 负样本循环 (同批次) ---
                for neg_pool_idx in eligible_same_batch_idx.tolist():
                    negative = int(same_window_candidates[neg_pool_idx])
                    c_neg = float(same_window_negative_confidence[neg_pool_idx])

                    c_neg_norm = c_neg / float(config.beta1 + config.beta2)
                    if c_neg_norm < config.min_c_neg:
                        continue

                    # 3. 原始值用于权重
                    omega = float(c_pos * c_neg)
                    triplets.append([anchor, positive, negative])
                    omegas.append(omega)
                    rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))
                    count_passed_neg += 1  # 记录保留的三元组/负样本数

                # --- 负样本循环 (跨批次) ---
                for neg_idx, negative in enumerate(selected_other_batch_negatives.tolist()):
                    c_neg = float(selected_other_batch_negative_confidence[neg_idx])

                    c_neg_norm = c_neg / float(config.beta1 + config.beta2)
                    if c_neg_norm < config.min_c_neg:
                        continue

                    omega = float(c_pos * c_neg)
                    triplets.append([anchor, positive, int(negative)])
                    omegas.append(omega)
                    rho_bars.append(float((rho_norm[anchor] + rho_norm[positive]) / 2.0))
                    count_passed_neg += 1  # 记录保留的三元组/负样本数

    # ---------------------------------------------------------
    # 打印过滤统计信息面板
    # ---------------------------------------------------------
    print("\n" + "=" * 50)
    print("[Triplets] FILTERING STATISTICS SUMMARY")
    print("=" * 50)
    print(f"[Triplets] Positive Pairs : {count_raw_pos} (Raw) -> {count_passed_pos} (Passed)")
    print(f"[Triplets] Negative Pairs : {count_raw_neg} (Raw) -> {count_passed_neg} (Passed)")
    print(f"[Triplets] Total Triplets : {count_raw_neg} (Raw) -> {len(triplets)} (Passed)")
    reduction_rate = (1.0 - (len(triplets) / max(1, count_raw_neg))) * 100
    print(f"[Triplets] Noise Reduction  : {reduction_rate:.2f}% of possible triplets filtered")
    print("=" * 50 + "\n")
    # ---------------------------------------------------------

    if len(triplets) == 0:
        raise ValueError("No triplets were constructed. Please check the input data.")

    triplet_array = np.asarray(triplets, dtype=np.int64)
    omega_array = np.asarray(omegas, dtype=np.float32)
    rho_bar_array = np.asarray(rho_bars, dtype=np.float32)

    summary = {
        "n_cells": int(n_cells),
        "n_batches": int(len(batch_names)),
        "same_batch_positive_pairs": int(total_same_batch_positive_pairs),
        "cross_batch_positive_pairs": int(total_cross_batch_positive_pairs),
        "total_triplets": int(len(triplets)),
        "rare_cells": int(rare_flags.sum()),
        "same_batch_knn_edges": int(len(same_batch_knn_edges)),
    }

    print("[Triplets] Construction summary:")
    for key, value in summary.items():
        print(f"[Triplets]   {key}: {value}")

    return TripletBundle(
        triplets=triplet_array,
        omega=omega_array,
        rho_bar=rho_bar_array,
        same_batch_knn_edges=same_batch_knn_edges.astype(np.int64),
        rare_flags=rare_flags.astype(np.float32),
        batch_codes_per_cell=batch_codes_per_cell.astype(np.int64),
        batch_names=batch_names,
        summary=summary,
        # local_knn_indices=local_knn_indices,   # 新增
    )
