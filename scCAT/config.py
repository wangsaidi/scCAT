from dataclasses import dataclass


@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Preprocessing
    target_sum: float = 1e4
    n_hvg: int = 1000
    n_pca: int = 50

    # Encoder
    hidden_dim: int = 128
    latent_dim: int = 64

    # Graph / triplet construction
    mnn_k: int = 15
    knn_k: int = 5
    num_neg_same_batch: int = 5
    num_neg_other_batch: int = 3
    same_batch_negative_rank_low: float = 0.50
    same_batch_negative_rank_high: float = 0.70

    # cross_batch_mnn_positive_conf_percentile: float = 40.0
    # same_batch_mnn_positive_conf_percentile: float = 20.0
    # same_batch_knn_positive_conf_percentile: float = 40.0

    # 过滤阈值配置
    min_c_pos: float = 0.7  # 归一化后的正样本最低置信度
    min_c_neg: float = 0.3  # 归一化后的负样本最低置信度


    # Confidence weights (kept exactly following the PDF formulas)
    # Cross-batch positive: alpha1 * s_mnn + alpha2 * s_density + alpha3 * s_hvg
    # Same-batch positive: alpha1 * s_density + alpha2 * s_hvg
    alpha1: float = 1.0
    alpha2: float = 1.0
    alpha3: float = 1.0

    # Negative confidence: beta1 * (1 - s_density) + beta2 * (1 - s_hvg)
    beta1: float = 1.0
    beta2: float = 1.0

    # Weighted triplet loss
    gamma: float = 0.5
    tau: float = 1.0  # 越小惩罚负样本太近越激烈

    # Adaptive margin
    m0: float = 0.5
    lambda_rho: float = 0.3
    lambda_b_margin: float = 0.3
    eta: float = 2.0

    # Total loss
    mu_rare: float = 0.3

    # Optimization
    lr: float = 1e-3
    weight_decay: float = 1e-5
    max_epochs: int = 300

    # Stable early stop on total loss
    min_epochs_before_early_stop: int = 20
    stable_window: int = 10
    stable_relative_tolerance: float = 1e-4

    # Exact full-batch computation can still be chunked internally for memory,
    # while keeping one optimizer step over the whole dataset.
    triplet_loss_chunk_size: int = 50000

    # Logging
    triplet_log_every_anchors: int = 100

    # Numerical stability
    eps: float = 1e-12
