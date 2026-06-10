"""scCAT — Confidence-weighted Adaptive Triplet learning for single-cell
batch integration.

Public top-level API
--------------------
    from scCAT import Config, prepare_inputs, construct_triplets, train_full_batch, set_seed

Typical usage::

    from scCAT import Config, prepare_inputs, construct_triplets, train_full_batch, set_seed

    config = Config()                # default hyperparameters
    set_seed(config.seed)

    prepared = prepare_inputs(
        expression_matrix=X,         # np.ndarray (n_cells, n_genes)
        feature_names=gene_names,    # list[str]
        batch_labels=batch_labels,   # np.ndarray (n_cells,)
        input_data_state="raw",      # or "preprocessed_hvg"
        config=config,
    )

    triplets = construct_triplets(
        model_input=prepared["triplet_input"],
        hvg_matrix=prepared["hvg_matrix"],
        batch_labels=batch_labels,
        config=config,
    )

    model, embedding, history = train_full_batch(
        model_input=prepared["model_input"],
        triplet_bundle=triplets,
        config=config,
        device="cpu",                # or "cuda"
    )

The returned ``embedding`` is an ``(n_cells, latent_dim)`` matrix that can be
fed directly into UMAP / clustering / downstream analysis.

Integration-quality metrics
---------------------------
The metrics used to evaluate batch integration in the paper - including the
novel Overcorrection Index (OCI), Batch-Specific Retention Score (BSRS) and
Integration Balance - live in the ``scCAT.metrics`` submodule and can be called
independently::

    from scCAT.metrics import compute_oci, compute_bsrs, integration_balance

``scCAT.metrics`` depends on scikit-learn and is intentionally *not* imported by
``import scCAT``, so the core training API stays dependency-light.
"""

from .config import Config
from .preprocess import prepare_inputs
from .triplets import construct_triplets, TripletBundle
from .trainer import train_full_batch
from .losses import total_loss
from .model import MLPEncoder
from .utils import set_seed

__version__ = "1.0.0"

__all__ = [
    "Config",
    "prepare_inputs",
    "construct_triplets",
    "TripletBundle",
    "train_full_batch",
    "total_loss",
    "MLPEncoder",
    "set_seed",
    "__version__",
]
