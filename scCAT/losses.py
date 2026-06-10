import torch
import torch.nn.functional as F

from .config import Config
from .triplets import TripletBundle


def compute_batch_statistics(
    z: torch.Tensor,
    batch_codes_per_cell: torch.Tensor,
    n_batches: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    mus = []
    sigmas = []

    for batch_code in range(n_batches):
        batch_mask = batch_codes_per_cell == batch_code
        z_batch = z[batch_mask]
        mu_b = z_batch.mean(dim=0)
        sigma_b = torch.sqrt(((z_batch - mu_b) ** 2).sum(dim=1).mean() + eps)
        mus.append(mu_b)
        sigmas.append(sigma_b)

    mu_matrix = torch.stack(mus, dim=0)
    sigma_vector = torch.stack(sigmas, dim=0)
    return mu_matrix, sigma_vector


def compute_b_norm_matrix(
    mu_matrix: torch.Tensor,
    sigma_vector: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    mu_diff = torch.cdist(mu_matrix, mu_matrix, p=2)
    denom = sigma_vector[:, None] + sigma_vector[None, :] + eps
    b_matrix = mu_diff / denom

    b_min = torch.min(b_matrix)
    b_max = torch.max(b_matrix)
    b_norm_matrix = (b_matrix - b_min) / (b_max - b_min + eps)
    return b_norm_matrix


def weighted_triplet_loss(
    z: torch.Tensor,
    triplet_bundle: TripletBundle,
    batch_codes_per_cell: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    device = z.device

    triplets = torch.as_tensor(triplet_bundle.triplets, dtype=torch.long, device=device)
    omega = torch.as_tensor(triplet_bundle.omega, dtype=torch.float32, device=device)
    rho_bar = torch.as_tensor(triplet_bundle.rho_bar, dtype=torch.float32, device=device)

    anchor_batch_codes = batch_codes_per_cell[triplets[:, 0]]
    positive_batch_codes = batch_codes_per_cell[triplets[:, 1]]

    mu_matrix, sigma_vector = compute_batch_statistics(
        z=z,
        batch_codes_per_cell=batch_codes_per_cell,
        n_batches=len(triplet_bundle.batch_names),
        eps=config.eps,
    )
    b_norm_matrix = compute_b_norm_matrix(mu_matrix, sigma_vector, config.eps)

    loss_sum = torch.zeros((), device=device)

    for start in range(0, triplets.shape[0], config.triplet_loss_chunk_size):
        end = min(start + config.triplet_loss_chunk_size, triplets.shape[0])
        batch_triplets = triplets[start:end]

        anchors = z[batch_triplets[:, 0]]
        positives = z[batch_triplets[:, 1]]
        negatives = z[batch_triplets[:, 2]]

        d_ij = torch.norm(anchors - positives, dim=1)
        d_ik = torch.norm(anchors - negatives, dim=1)

        rho_bar_chunk = rho_bar[start:end]
        omega_chunk = omega[start:end]

        b_norm_chunk = b_norm_matrix[
            anchor_batch_codes[start:end],
            positive_batch_codes[start:end],
        ]

        # m_ij = m0 + lambda_rho * ((1 - rho_bar_ij) ** eta) + lambda_b * b_ij_norm
        density_term = torch.pow(torch.clamp(1.0 - rho_bar_chunk, min=0.0), config.eta)
        margin = (
            config.m0
            + config.lambda_rho * density_term
            + config.lambda_b_margin * b_norm_chunk
        )

        # (1 + gamma * omega_ijk) * log(1 + exp((d_ij - d_ik + m_ij) / tau))
        logits = (d_ij - d_ik + margin) / config.tau
        # weighted_softplus = (1.0 + config.gamma * omega_chunk) * F.softplus(logits)
        weighted_softplus = (1.0 + config.gamma * omega_chunk) * F.relu(logits)
        loss_sum = loss_sum + weighted_softplus.sum()

    return loss_sum


# def weighted_triplet_loss(
#         z: torch.Tensor,
#         triplet_bundle: TripletBundle,
#         batch_codes_per_cell: torch.Tensor,
#         config: Config,
# ) -> torch.Tensor:
#     device = z.device
#
#     triplets = torch.as_tensor(triplet_bundle.triplets, dtype=torch.long, device=device)
#     omega = torch.as_tensor(triplet_bundle.omega, dtype=torch.float32, device=device)
#     rho_bar = torch.as_tensor(triplet_bundle.rho_bar, dtype=torch.float32, device=device)
#
#     # 1. 直接计算所有细胞的局部均值和方差
#     knn_idx = torch.as_tensor(triplet_bundle.local_knn_indices, dtype=torch.long, device=device)
#     z_knn = z[knn_idx]  # shape: (N, K, D)
#     mu_local = z_knn.mean(dim=1)  # shape: (N, D)
#     # 计算局部方差 (对每个细胞的K个邻居求距离平方的均值)
#     sigma_local = torch.sqrt(((z_knn - mu_local.unsqueeze(1)) ** 2).sum(dim=2).mean(dim=1) + config.eps)  # shape: (N,)
#
#     loss_sum = torch.zeros((), device=device)
#
#     for start in range(0, triplets.shape[0], config.triplet_loss_chunk_size):
#         end = min(start + config.triplet_loss_chunk_size, triplets.shape[0])
#         batch_triplets = triplets[start:end]
#
#         anchors = z[batch_triplets[:, 0]]
#         positives = z[batch_triplets[:, 1]]
#         negatives = z[batch_triplets[:, 2]]
#
#         d_ij = torch.norm(anchors - positives, dim=1)
#         d_ik = torch.norm(anchors - negatives, dim=1)
#
#         rho_bar_chunk = rho_bar[start:end]
#         omega_chunk = omega[start:end]
#
#         # 2. 动态计算局部的 batch_norm_matrix (b_ij)
#         mu_a = mu_local[batch_triplets[:, 0]]
#         mu_p = mu_local[batch_triplets[:, 1]]
#         sigma_a = sigma_local[batch_triplets[:, 0]]
#         sigma_p = sigma_local[batch_triplets[:, 1]]
#
#         b_ij_raw = torch.norm(mu_a - mu_p, dim=1) / (sigma_a + sigma_p + config.eps)
#
#         # 在当前 chunk 内归一化 b_ij
#         b_min = b_ij_raw.min()
#         b_max = b_ij_raw.max()
#         b_norm_chunk = (b_ij_raw - b_min) / (b_max - b_min + config.eps)
#
#         # 3. 组合 margin
#         density_term = torch.pow(torch.clamp(1.0 - rho_bar_chunk, min=0.0), config.eta)
#         margin = (
#                 config.m0
#                 + config.lambda_rho * density_term
#                 + config.lambda_b_margin * b_norm_chunk
#         )
#
#         logits = (d_ij - d_ik + margin) / config.tau
#         weighted_softplus = (1.0 + config.gamma * omega_chunk) * F.relu(logits)
#         loss_sum = loss_sum + weighted_softplus.sum()
#
#     return loss_sum



def rare_loss(
    z: torch.Tensor,
    triplet_bundle: TripletBundle,
) -> torch.Tensor:
    device = z.device
    edges = torch.as_tensor(triplet_bundle.same_batch_knn_edges, dtype=torch.long, device=device)
    rare_flags = torch.as_tensor(triplet_bundle.rare_flags, dtype=torch.float32, device=device)

    if edges.numel() == 0:
        return torch.zeros((), device=device)

    anchors = edges[:, 0]
    neighbors = edges[:, 1]
    squared_dist = ((z[anchors] - z[neighbors]) ** 2).sum(dim=1)

    weights = rare_flags[anchors]
    return (weights * squared_dist).sum()


def total_loss(
    z: torch.Tensor,
    triplet_bundle: TripletBundle,
    batch_codes_per_cell: torch.Tensor,
    config: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    l_triplet = weighted_triplet_loss(
        z=z,
        triplet_bundle=triplet_bundle,
        batch_codes_per_cell=batch_codes_per_cell,
        config=config,
    )
    l_rare = rare_loss(
        z=z,
        triplet_bundle=triplet_bundle,
    )

    total = l_triplet + (config.mu_rare * l_rare)

    parts = {
        "triplet": l_triplet,
        "rare": l_rare,
        "total": total,
    }
    return total, parts
