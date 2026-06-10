# scCAT hyperparameter reference

Every field of `Config` (in `scCAT/config.py`) is documented below, with its
**default value**, **search range tested in Suppl Fig. S10 + Suppl Table S13**
and a **qualitative description of its effect**.

The defaults are the values used throughout the paper unless stated
otherwise; sensitivity analysis on the HDC dataset shows scCAT is robust to
all of them within their tested ranges (IB range < 0.05 per parameter).

---

## Reproducibility

| Parameter | Default | Description |
|---|---:|---|
| `seed` | 42 | RNG seed for `random`, `numpy`, `torch`. Set via `set_seed(config.seed)` before preprocessing for full reproducibility. |

---

## Preprocessing

| Parameter | Default | Tested range | Effect |
|---|---:|---|---|
| `target_sum` | 1e4 | — | Library-size normalisation target (cells normalised to `target_sum` total counts before log1p). Standard scanpy default. |
| `n_hvg` | 1000 | 500 – 5000 | Number of highly variable genes used as features. Higher = more biological signal but more noise; capped automatically at `n_genes`. |
| `n_pca` | 50 | 30 – 100 | Dimensionality of PCA applied to z-scaled HVGs. Used for both the encoder input and the triplet space. |

---

## Encoder architecture

| Parameter | Default | Tested range | Effect |
|---|---:|---|---|
| `hidden_dim` | 128 | 64 – 256 | Width of the MLP encoder hidden layer. |
| `latent_dim` | 64 | 16 – 128 | Output embedding dimensionality. 32 / 64 work well for most datasets; smaller = forces more compression. |

---

## Triplet construction — graph

| Parameter | Default | Tested range | Effect |
|---|---:|---|---|
| `mnn_k` | 15 | 5 – 30 | Number of cross-batch MNN candidates per cell. Higher = more candidate positives (more recall, more noise). |
| `knn_k` | 5 | 3 – 20 | Number of same-batch KNN neighbours; also used by the BSP regulariser. |
| `num_neg_same_batch` | 5 | 2 – 10 | Negatives sampled from the same batch per anchor. |
| `num_neg_other_batch` | 3 | 1 – 8 | Negatives sampled from other batches per anchor. |
| `same_batch_negative_rank_low` | 0.50 | 0.30 – 0.60 | Lower rank fraction for same-batch negative sampling (skip very-near neighbours). |
| `same_batch_negative_rank_high` | 0.70 | 0.60 – 0.85 | Upper rank fraction for same-batch negative sampling (skip very-far cells). |

---

## Triplet construction — confidence filtering

| Parameter | Default | Tested range | Effect |
|---|---:|---|---|
| `min_c_pos` | **0.7** | 0.3 – 0.9 | Drop positive pairs with confidence below this threshold. Lowering relaxes filtering (more triplets, more noise); raising tightens it (fewer triplets, higher purity). One of the most important knobs. |
| `min_c_neg` | 0.3 | 0.1 – 0.6 | Drop negative pairs with confidence below this threshold. |

---

## Confidence-weight composition

| Parameter | Default | Tested range | Effect |
|---|---:|---|---|
| `alpha1` | 1.0 | 0 – 2 | Weight on `s_mnn` (cross-batch) / `s_density` (same-batch) in positive confidence. |
| `alpha2` | 1.0 | 0 – 2 | Weight on `s_density` (cross-batch) / `s_hvg` (same-batch). |
| `alpha3` | 1.0 | 0 – 2 | Weight on `s_hvg` (cross-batch only). Set to 0 to ignore HVG similarity in cross-batch confidence. |
| `beta1` | 1.0 | 0 – 2 | Weight on `(1 - s_density)` in negative confidence. |
| `beta2` | 1.0 | 0 – 2 | Weight on `(1 - s_hvg)` in negative confidence. |

The default `(α₁, α₂, α₃) = (1, 1, 1)` and `(β₁, β₂) = (1, 1)` weight every
signal equally; biased choices were tested in sensitivity analysis and
showed marginal effect on Integration Balance.

---

## Adaptive margin

| Parameter | Default | Tested range | Effect |
|---|---:|---|---|
| `gamma` | **0.5** | 0 – 2 | Confidence-weight scaling: `w = exp(γ · (c⁺ + c⁻))`. γ = 0 disables confidence weighting (this is the `-noConf` ablation). |
| `tau` | 1.0 | 0.5 – 2 | Hinge sharpness: `max(0, ...)^τ`. τ > 1 makes the penalty more aggressive at near-violations. |
| `m0` | **0.5** | 0.1 – 1.0 | Base margin in the triplet loss. |
| `lambda_rho` | 0.3 | 0 – 0.6 | Margin boost for low-density cells. |
| `lambda_b_margin` | 0.3 | 0 – 0.6 | Margin boost for cross-batch-spanning triplets. |
| `eta` | 2.0 | 1 – 4 | Density-normalisation exponent used internally during margin computation. |

Setting `lambda_rho = lambda_b_margin = 0` reduces to fixed-margin triplet
loss (this is the `-fixedMargin` ablation).

---

## Batch-specific protection (BSP)

| Parameter | Default | Tested range | Effect |
|---|---:|---|---|
| `mu_rare` | **0.3** | 0 – 1.0 | Strength of the BSP regulariser that pulls rare cells toward their same-batch KNN. `mu_rare = 0` disables BSP (this is the `-noBSP` ablation). The BSP module is **directly responsible for preventing overcorrection of batch-specific cells** — see Fig. 4d, OCI almost zero with default vs. > 35× higher without BSP. |

---

## Optimisation

| Parameter | Default | Description |
|---|---:|---|
| `lr` | 1e-3 | Adam learning rate. |
| `weight_decay` | 1e-5 | Adam weight decay (L2). |
| `max_epochs` | 300 | Hard upper bound on training epochs; early stop usually fires first. |

---

## Early stopping

| Parameter | Default | Description |
|---|---:|---|
| `min_epochs_before_early_stop` | 20 | Train at least this many epochs before checking stability. |
| `stable_window` | 10 | Look-back window of epochs for the stability test. |
| `stable_relative_tolerance` | 1e-4 | Relative change in total loss below which training is considered stable. |

---

## Internal / numerical

| Parameter | Default | Description |
|---|---:|---|
| `triplet_loss_chunk_size` | 50000 | Triplets are processed in chunks of this size inside the loss computation to bound memory. Lower if you OOM on a large dataset. |
| `triplet_log_every_anchors` | 100 | Print a progress message every N anchors during triplet construction. |
| `eps` | 1e-12 | Numerical-stability epsilon for divisions. |

---

## Recommended tweaks for common scenarios

| Scenario | Tweak |
|---|---|
| Very small dataset (< 1 000 cells) | `n_hvg = 500`, `mnn_k = 5`, `knn_k = 3` — fewer parameters, less risk of overfitting |
| Atlas-level (> 100 000 cells, GPU) | `latent_dim = 32`, `triplet_loss_chunk_size = 100000`, use GPU |
| Strong batch effect, ample shared cells | `min_c_pos = 0.5` (more triplets, accept some noise) |
| Suspected overcorrection of rare cells | Increase `mu_rare` to 0.5; tighten `min_c_pos` to 0.8 |
| Reproduce paper results | **leave all defaults** |
