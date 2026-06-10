# The scCAT algorithm

A more compact recap of the algorithm than what is in Methods of the paper,
oriented toward people who want to read the code.

---

## 1. Inputs

| Symbol | Meaning | scCAT variable |
|---|---|---|
| `X ∈ ℝ^(N×G)` | Raw or pre-processed expression matrix (cells × genes) | `expression_matrix` |
| `b ∈ {0,…,B-1}^N` | Per-cell batch labels | `batch_labels` |
| `g₁,…,g_G` | Gene names | `feature_names` |

## 2. Pipeline overview

```
                 ┌──────────────────────────────────────────────┐
                 │ 1. PREPROCESS                                │
                 │    • normalise → log1p → HVG                 │
                 │    • PCA on z-scaled HVG (encoder input)     │
                 │    • PCA on batch-z-scaled HVG (triplet sp.) │
                 └─────────────┬────────────────────────────────┘
                               │
                 ┌─────────────▼────────────────────────────────┐
                 │ 2. TRIPLET CONSTRUCTION (the heart of scCAT) │
                 │    • cross-batch MNN     (cands. & confidences)
                 │    • same-batch KNN      (cands. & confidences)
                 │    • local density ρ_i                        │
                 │    • HVG similarity      s_hvg                │
                 │    • positive / negative confidence scores    │
                 │    • filter on (c⁺, c⁻) thresholds            │
                 │    • mark rare cells for BSP regulariser      │
                 └─────────────┬────────────────────────────────┘
                               │
                 ┌─────────────▼────────────────────────────────┐
                 │ 3. TRAINING (MLP encoder, full-batch SGD)    │
                 │    • weighted adaptive-margin triplet loss   │
                 │    • + BSP same-batch KNN regulariser        │
                 └─────────────┬────────────────────────────────┘
                               │
                 ┌─────────────▼────────────────────────────────┐
                 │ 4. OUTPUT: embedding ∈ ℝ^(N × latent_dim)    │
                 │    → UMAP / Leiden / your downstream         │
                 └──────────────────────────────────────────────┘
```

## 3. The three mechanisms

scCAT's design is built on three coupled mechanisms. Each is independently
ablatable (see Figure 4 and Suppl Table S7).

### 3.1 Confidence-weighted triplet construction

For every candidate (anchor *i*, positive *j*) pair we compute a
**positive confidence**:

- **Cross-batch positive**:
  `c⁺ = α₁·s_mnn + α₂·s_density + α₃·s_hvg`
  where `s_mnn` is the bidirectional MNN ranking score,
  `s_density = exp(-|ρ_i - ρ_j| / (ρ_i + ρ_j + ε))` measures local-density
  consistency, and `s_hvg` is Pearson correlation on z-scaled HVGs.
- **Same-batch positive**:
  `c⁺ = α₁·s_density + α₂·s_hvg`

A symmetric **negative confidence** is computed for candidate (anchor *i*,
negative *k*) pairs:

- `c⁻ = β₁·(1 - s_density) + β₂·(1 - s_hvg)`

Both are min-max normalised within the dataset to lie in [0, 1]. Pairs
with `c⁺ < min_c_pos` (default 0.7) or `c⁻ < min_c_neg` (default 0.3) are
**filtered out** before training — they never contribute to the loss.

### 3.2 Adaptive margin

Every retained triplet `(i, j⁺, k⁻)` gets its own margin:

```
m(i) = m₀ + λ_ρ · (1 - ρ̄_i / max ρ̄) + λ_b · 𝟙[b_j ≠ b_k]
```

where `ρ̄_i` is the average local density at anchor *i*. Effects:

- Low-density anchors (rare / transition cells) get a **larger margin**,
  forcing the encoder to separate them more aggressively from negatives.
- Triplets that span different batches get an extra margin term,
  giving more pull to the cross-batch anchor structure.

### 3.3 Batch-specific local-topology protection (BSP)

For every cell *i* that has **no confident cross-batch positive partner**
(`c⁺_max < min_c_pos`), `rare_flag_i = 1`. These cells are not pulled toward
any other batch. Instead, the BSP regulariser pulls them toward their
same-batch k-NN neighbours:

```
L_BSP = μ_rare · ∑_{i: rare_flag_i = 1} ∑_{j ∈ KNN_same(i)} ||z_i - z_j||²
```

This is what prevents overcorrection of batch-specific cell types — they
are constrained only by what looks like them within their own batch, never
forced to align with anything in other batches.

## 4. The total loss

```
L_total = L_triplet(weighted, adaptive margin) + μ_rare · L_BSP
```

with the weighted triplet loss:

```
L_triplet = (1 / |T|) · ∑_{(i,j,k) ∈ T} w(i,j,k) ·
            max(0, ‖z_i - z_j‖² - ‖z_i - z_k‖² + m(i))^τ
```

where the per-triplet weight is `w = exp(γ · (c⁺_ij + c⁻_ik))` (the
high-confidence pairs get amplified gradients) and `τ` raises the
hinge to soften / sharpen the penalty on near-violations.

## 5. Training

- Encoder: 2-layer MLP (input → `hidden_dim` → `latent_dim`, ReLU, L2
  normalisation on output).
- Optimiser: Adam, `lr = 1e-3`, `weight_decay = 1e-5`.
- Schedule: full-batch SGD over all triplets per epoch.
- Stopping: early stop after `min_epochs_before_early_stop` epochs if the
  total loss has been stable (relative change < `stable_relative_tolerance`)
  for `stable_window` consecutive epochs.

The returned embedding is `z = Encoder(x)` for every cell.

## 6. Why does it help with overcorrection?

The standard failure mode of MNN-based methods is that they create
"phantom" anchors between cells that should not be aligned (e.g. a
batch-specific cell type whose nearest neighbour in the other batch is in
a totally different population). scCAT addresses this in three
complementary ways:

1. **Filter** the phantom anchors out before training (confidence threshold).
2. **Down-weight** any that survive (confidence-weighted loss).
3. **Protect** the cells that have no anchor at all (BSP regulariser keeps
   them where the same-batch KNN says they should be).

The combined effect is shown quantitatively in Figure 4 — removing any of
the three modules causes a visible drop in OCI (overcorrection index) or
Integration Balance.

## 7. Where each piece lives in the code

| Mechanism | Module | Key function |
|---|---|---|
| Preprocessing (HVG, PCA) | `scCAT/preprocess.py` | `prepare_inputs` |
| Confidence-weighted triplet construction | `scCAT/triplets.py` | `construct_triplets`, `build_cross_batch_mnn`, `build_same_batch_knn` |
| Adaptive margin & loss | `scCAT/losses.py` | `total_loss` |
| Encoder | `scCAT/model.py` | `MLPEncoder` |
| Training loop | `scCAT/trainer.py` | `train_full_batch` |
| Defaults / hyperparameters | `scCAT/config.py` | `Config` |
