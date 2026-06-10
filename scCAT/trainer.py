import copy
import time
from typing import Any, Dict

import numpy as np
import torch

from .config import Config
from .losses import total_loss
from .model import MLPEncoder
from .triplets import TripletBundle
from .utils import set_seed, total_loss_is_stable


def train_full_batch(
    model_input: np.ndarray,
    triplet_bundle: TripletBundle,
    config: Config,
    device: str,
) -> tuple[MLPEncoder, np.ndarray, list[dict[str, float]]]:
    set_seed(config.seed)

    x = torch.as_tensor(model_input, dtype=torch.float32, device=device)
    batch_codes_per_cell = torch.as_tensor(
        triplet_bundle.batch_codes_per_cell,
        dtype=torch.long,
        device=device,
    )

    model = MLPEncoder(
        input_dim=model_input.shape[1],
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )

    best_total_value = float("inf")
    best_state_dict = copy.deepcopy(model.state_dict())
    history: list[dict[str, float]] = []
    total_loss_history: list[float] = []

    print("[Train] Starting full-batch training ...")
    print(f"[Train] Device: {device}")
    print(f"[Train] Cells: {model_input.shape[0]}")
    print(f"[Train] Input dimension: {model_input.shape[1]}")
    print(f"[Train] Latent dimension: {config.latent_dim}")
    print(f"[Train] Total triplets: {triplet_bundle.summary['total_triplets']}")
    print(f"[Train] Rare cells: {triplet_bundle.summary['rare_cells']}")

    for epoch in range(1, config.max_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        optimizer.zero_grad()

        z = model(x)
        total, parts = total_loss(
            z=z,
            triplet_bundle=triplet_bundle,
            batch_codes_per_cell=batch_codes_per_cell,
            config=config,
        )
        total.backward()
        optimizer.step()

        total_value = float(parts["total"].detach().cpu().item())
        triplet_value = float(parts["triplet"].detach().cpu().item())
        rare_value = float(parts["rare"].detach().cpu().item())

        elapsed = time.time() - epoch_start_time

        epoch_record = {
            "epoch": float(epoch),
            "triplet_loss": triplet_value,
            "rare_loss": rare_value,
            "total_loss": total_value,
            "epoch_seconds": elapsed,
        }
        history.append(epoch_record)
        total_loss_history.append(total_value)

        if total_value < best_total_value:
            best_total_value = total_value
            best_state_dict = copy.deepcopy(model.state_dict())

        print(
            f"[Train] Epoch {epoch:04d} | "
            f"triplet={triplet_value:.6f} | "
            f"rare={rare_value:.6f} | "
            f"total={total_value:.6f} | "
            f"best_total={best_total_value:.6f} | "
            f"time={elapsed:.2f}s"
        )

        if total_loss_is_stable(
            loss_history=total_loss_history,
            min_epochs_before_early_stop=config.min_epochs_before_early_stop,
            stable_window=config.stable_window,
            stable_relative_tolerance=config.stable_relative_tolerance,
        ):
            recent = np.asarray(total_loss_history[-config.stable_window:], dtype=np.float64)
            span = float(recent.max() - recent.min())
            ref = max(1.0, abs(float(recent.mean())))
            print(
                f"[Train] Early stop triggered because total loss is stable. "
                f"Window={config.stable_window}, span={span:.6e}, "
                f"relative_span={span / ref:.6e}"
            )
            break

    print("[Train] Restoring best model parameters by total loss ...")
    model.load_state_dict(best_state_dict)
    model.eval()

    with torch.no_grad():
        final_embedding = model(x).detach().cpu().numpy().astype(np.float32)

    return model, final_embedding, history
